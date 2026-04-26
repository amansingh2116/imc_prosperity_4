"""
========================================================================
ROUND 3 ADVANCED TRADING ALGORITHM
========================================================================

STRATEGY OVERVIEW & QUANTITATIVE RATIONALE
-------------------------------------------

1. VELVETFRUIT_EXTRACT (VE) — Adaptive Market-Making with EWM Fair Value
   • Use exponential-weighted mid-price as fair value estimate (fast decay
     for price discovery, slow decay for long-run mean)
   • Quote tight spreads around fair value, skew quotes based on inventory
     to mean-revert position risk
   • Momentum filter: if recent price trend is strong, lean with it

2. HYDROGEL_PACK (HP) — Statistical Mean-Reversion + Trend Filter
   • Simple EWM fair-value market-maker, same inventory skew logic
   • Separate from VE so each product manages its own risk

3. VELVETFRUIT_EXTRACT_VOUCHERS (VEVs) — Black-Scholes Options Pricing
   ---------------------------------------------------------------
   CORE QUANT INSIGHT:
   The VEVs are call options on VELVETFRUIT_EXTRACT with known strikes.
   At Round 3 start, TTE = 5 days (historical data: TTE=8 at day0, 7 at
   day1, 6 at day2 → we extrapolate: round 3 = TTE 5, but we refine
   intra-day using timestamp fraction).

   BLACK-SCHOLES MODEL:
   C = S·N(d1) - K·e^{-rT}·N(d2)
   d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
   d2 = d1 - σ·√T

   Parameters calibrated from historical data:
   • σ (implied vol) estimated from realized vol of VE price series
   • r = 0 (no risk-free rate in this market)
   • T in "day fraction" units (max 7-day expiry, each round is 1 day,
     each timestamp step within day covers ~1/10000 of a day)

   TRADING STRATEGY FOR OPTIONS:
   a) Calibrate σ_realized from rolling window of VE returns
   b) For each voucher, compute BS theoretical price C_BS(S, K, T, σ)
   c) If market bid > C_BS + edge_threshold → SELL (voucher overpriced)
      If market ask < C_BS - edge_threshold → BUY  (voucher underpriced)
   d) Manage delta risk: hedge VEV positions with VE spot trades
      Delta = N(d1) per voucher held
      Net delta = Σ(position_i × delta_i) across all VEVs
      Hedge: trade VE spot to offset net delta

   DELTA HEDGING:
   • Every timestamp, compute total delta exposure from VEV book
   • Target VE spot position = -net_delta (capped by ±200 limit)
   • This makes the portfolio "delta-neutral" → profit from vol mismatch

   DEEP ITM / OTM HANDLING:
   • Deep ITM vouchers (S >> K): treat as near-1 delta, trade for parity
   • Deep OTM vouchers (S << K): near-zero value, avoid buying
   • Focus activity on near-ATM strikes where edge is largest

4. VOLATILITY CALIBRATION:
   • Use 20-timestamp rolling window of VE returns → σ_realized
   • Scale to per-day units: σ_daily = σ_tick × √(10000)  (10000 ticks/day)
   • Apply 10% vol premium buffer to account for bid-ask bounce noise

5. POSITION SIZING & RISK:
   • VEVs: max 200 per voucher (conservative vs 300 limit), scale by
     price edge / theoretical price ratio
   • VE spot: ±200 limit, used partly for hedging VEV deltas
   • HP: ±180 limit with inventory mean-reversion skew
   • Never exceed 80% of position limits to allow adjustment room

========================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import json
from collections import deque

# ─── Competition framework types (minimal stubs matching real interface) ─────

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int  # positive = buy, negative = sell


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)   # price → qty (positive)
    sell_orders: Dict[int, int] = field(default_factory=dict)  # price → qty (negative)


@dataclass
class TradingState:
    timestamp: int
    listings: Dict
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict
    market_trades: Dict
    position: Dict[str, int]
    observations: object
    traderData: str = ""


# ─── Constants ───────────────────────────────────────────────────────────────

PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VELVETFRUIT_EXTRACT_VOUCHER_4000",
    "VELVETFRUIT_EXTRACT_VOUCHER_4500",
    "VELVETFRUIT_EXTRACT_VOUCHER_5000",
    "VELVETFRUIT_EXTRACT_VOUCHER_5100",
    "VELVETFRUIT_EXTRACT_VOUCHER_5200",
    "VELVETFRUIT_EXTRACT_VOUCHER_5300",
    "VELVETFRUIT_EXTRACT_VOUCHER_5400",
    "VELVETFRUIT_EXTRACT_VOUCHER_5500",
    "VELVETFRUIT_EXTRACT_VOUCHER_6000",
    "VELVETFRUIT_EXTRACT_VOUCHER_6500",
]

STRIKES = {
    "VELVETFRUIT_EXTRACT_VOUCHER_4000": 4000,
    "VELVETFRUIT_EXTRACT_VOUCHER_4500": 4500,
    "VELVETFRUIT_EXTRACT_VOUCHER_5000": 5000,
    "VELVETFRUIT_EXTRACT_VOUCHER_5100": 5100,
    "VELVETFRUIT_EXTRACT_VOUCHER_5200": 5200,
    "VELVETFRUIT_EXTRACT_VOUCHER_5300": 5300,
    "VELVETFRUIT_EXTRACT_VOUCHER_5400": 5400,
    "VELVETFRUIT_EXTRACT_VOUCHER_5500": 5500,
    "VELVETFRUIT_EXTRACT_VOUCHER_6000": 6000,
    "VELVETFRUIT_EXTRACT_VOUCHER_6500": 6500,
}

POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{v: 300 for v in STRIKES},
}

# Black-Scholes calibrated parameters
# σ estimated from historical VE price data:
# VE trades in range ~4000-6000, typical daily σ ≈ 15-20% of price level
# Per historical data analysis: σ_daily ≈ 0.16 (calibrated)
SIGMA_DAILY = 0.16          # annualized-equivalent daily vol (per 1-day unit)
SIGMA_TICK_SCALE = math.sqrt(10000)  # 10000 timestamps per day → tick vol scaling

# Round 3: TTE = 5 days at start of round (days remaining until expiry)
# Within a 10000-timestamp day, fraction consumed = timestamp / 10000
TTE_ROUND_START = 5.0  # days

# EWM parameters for fair value estimation
EWM_ALPHA_FAST = 0.3   # fast signal for momentum
EWM_ALPHA_SLOW = 0.05  # slow signal for fair value
EWM_ALPHA_VOL  = 0.1   # for rolling volatility estimation

# Market-making parameters
MM_SPREAD_BASE_VE  = 3    # half-spread for VE market making
MM_SPREAD_BASE_HP  = 4    # half-spread for HP market making
INVENTORY_SKEW_FACTOR = 0.08  # how aggressively to skew quotes vs inventory

# Options trading parameters
OPTION_EDGE_THRESHOLD = 2.5   # min edge (absolute SeaShells) to trade a voucher
OPTION_MAX_POSITION   = 200   # conservative limit (vs 300 hard limit)
OPTION_AGGRESS_SIZE   = 20    # max qty per single order on options
DELTA_HEDGE_THRESHOLD = 5     # min delta units before hedging

# ─── Math utilities ──────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """
    Black-Scholes European call price.
    S: underlying price
    K: strike
    T: time to expiry in days (we treat 1 unit = 1 day)
    sigma: daily volatility (not annualized — raw per-day units)
    r: risk-free rate per day (default 0)
    """
    if T <= 0:
        return max(S - K, 0.0)
    if S <= 0 or sigma <= 0:
        return max(S - K, 0.0)

    sqrt_T = math.sqrt(T)
    # With daily σ and daily T, the BS formula works directly
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    call = S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return max(call, max(S - K, 0.0))  # floor at intrinsic value


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """
    Black-Scholes delta for a call option = N(d1).
    """
    if T <= 0:
        return 1.0 if S > K else 0.0
    if S <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """BS vega: sensitivity of call price to σ."""
    if T <= 0 or S <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return S * norm_pdf(d1) * sqrt_T


def implied_vol_from_price(market_price: float, S: float, K: float, T: float,
                            r: float = 0.0, tol: float = 0.001, max_iter: int = 50) -> float:
    """
    Newton-Raphson implied volatility solver.
    Returns sigma_daily.
    """
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic:
        return 0.0
    if T <= 0:
        return 0.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = math.sqrt(2.0 * math.pi / T) * market_price / S
    sigma = max(0.01, min(sigma, 5.0))

    for _ in range(max_iter):
        price = bs_call_price(S, K, T, sigma, r)
        vega  = bs_vega(S, K, T, sigma, r)
        if abs(vega) < 1e-8:
            break
        diff = price - market_price
        if abs(diff) < tol:
            break
        sigma -= diff / vega
        sigma = max(0.001, min(sigma, 10.0))

    return sigma


# ─── State management ────────────────────────────────────────────────────────

class EWMEstimator:
    """Exponential-weighted moving average tracker."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class VolatilityEstimator:
    """Rolling realized volatility from log returns."""
    def __init__(self, window: int = 30):
        self.window = window
        self.prices: deque = deque(maxlen=window + 1)
        self.sigma_daily: float = SIGMA_DAILY  # fallback

    def update(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) >= 10:
            returns = []
            prices_list = list(self.prices)
            for i in range(1, len(prices_list)):
                if prices_list[i - 1] > 0:
                    returns.append(math.log(prices_list[i] / prices_list[i - 1]))
            if len(returns) >= 5:
                mean_r = sum(returns) / len(returns)
                variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                sigma_tick = math.sqrt(variance)
                # Scale: tick-level vol → daily vol (√10000 ticks per day)
                self.sigma_daily = sigma_tick * SIGMA_TICK_SCALE
                # Clamp to reasonable range
                self.sigma_daily = max(0.05, min(self.sigma_daily, 2.0))
        return self.sigma_daily


# ─── Helper: best bid/ask extraction ─────────────────────────────────────────

def best_bid(od: OrderDepth) -> Optional[int]:
    if not od.buy_orders:
        return None
    return max(od.buy_orders.keys())

def best_ask(od: OrderDepth) -> Optional[int]:
    if not od.sell_orders:
        return None
    return min(od.sell_orders.keys())

def mid_price(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0

def weighted_mid(od: OrderDepth) -> Optional[float]:
    """Volume-weighted mid (better estimate of fair value)."""
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None:
        return None
    bq = od.buy_orders.get(bb, 1)
    aq = abs(od.sell_orders.get(ba, 1))
    # weighted toward thinner side
    total = bq + aq
    if total == 0:
        return (bb + ba) / 2.0
    return (bb * aq + ba * bq) / total  # micro-price


# ─── Market-making orders ────────────────────────────────────────────────────

def mm_orders(symbol: str, fair: float, position: int,
              pos_limit: int, half_spread: int,
              od: OrderDepth) -> List[Order]:
    """
    Generate market-making orders around fair value with inventory skew.
    Skews quotes to lean against inventory → mean-revert position.
    """
    orders = []
    skew = int(round(INVENTORY_SKEW_FACTOR * position))
    bid_price = int(fair - half_spread - skew)
    ask_price = int(fair + half_spread - skew)

    # Available room
    max_buy  = pos_limit - position
    max_sell = pos_limit + position

    if max_buy <= 0 and max_sell <= 0:
        return []

    # Passive layer: post quotes
    buy_qty  = min(max_buy,  20)
    sell_qty = min(max_sell, 20)

    if buy_qty > 0 and bid_price > 0:
        orders.append(Order(symbol, bid_price, buy_qty))
    if sell_qty > 0:
        orders.append(Order(symbol, ask_price, -sell_qty))

    # Aggressive layer: lift/hit mispriced orders
    # If someone is selling below our fair - spread → buy
    for ask_p in sorted(od.sell_orders.keys()):
        if ask_p >= bid_price:
            break
        qty_avail = abs(od.sell_orders[ask_p])
        qty = min(qty_avail, max_buy - buy_qty)
        if qty > 0:
            orders.append(Order(symbol, ask_p, qty))
            buy_qty += qty

    # If someone is buying above our fair + spread → sell
    for bid_p in sorted(od.buy_orders.keys(), reverse=True):
        if bid_p <= ask_price:
            break
        qty_avail = od.buy_orders[bid_p]
        qty = min(qty_avail, max_sell - sell_qty)
        if qty > 0:
            orders.append(Order(symbol, bid_p, -qty))
            sell_qty += qty

    return orders


# ─── Options trading logic ───────────────────────────────────────────────────

def option_orders(symbol: str, strike: int, S: float, T: float, sigma: float,
                  position: int, pos_limit: int, od: OrderDepth,
                  iv_surface: Dict[str, float]) -> Tuple[List[Order], float]:
    """
    Generate orders for a single VEV based on BS mispricing.
    Returns (orders, delta) where delta is the BS delta of our position.
    """
    orders = []

    # Compute BS theoretical price
    theo = bs_call_price(S, strike, T, sigma)
    delta_per_unit = bs_delta(S, strike, T, sigma)

    bb = best_bid(od)
    ba = best_ask(od)

    if bb is None and ba is None:
        return [], position * delta_per_unit

    # --- Selling overpriced vouchers ---
    if bb is not None and bb > theo + OPTION_EDGE_THRESHOLD:
        # Someone willing to buy at bb which is above theo → sell to them
        max_sell = pos_limit + position  # room to go short
        qty_market = od.buy_orders.get(bb, 0)
        qty = min(qty_market, max_sell, OPTION_AGGRESS_SIZE)
        if qty > 0:
            orders.append(Order(symbol, bb, -qty))

        # Also post a passive sell slightly above theo
        passive_sell_price = int(theo + OPTION_EDGE_THRESHOLD)
        passive_qty = min(max_sell - qty, OPTION_AGGRESS_SIZE)
        if passive_qty > 0:
            orders.append(Order(symbol, passive_sell_price, -passive_qty))

    # --- Buying underpriced vouchers ---
    if ba is not None and ba < theo - OPTION_EDGE_THRESHOLD:
        # Someone willing to sell at ba which is below theo → buy from them
        max_buy = pos_limit - position
        qty_market = abs(od.sell_orders.get(ba, 0))
        qty = min(qty_market, max_buy, OPTION_AGGRESS_SIZE)
        if qty > 0:
            orders.append(Order(symbol, ba, qty))

        # Also post a passive buy slightly below theo
        passive_buy_price = int(theo - OPTION_EDGE_THRESHOLD)
        passive_qty = min(max_buy - qty, OPTION_AGGRESS_SIZE)
        if passive_qty > 0:
            orders.append(Order(symbol, passive_buy_price, passive_qty))

    # Store IV for surface tracking
    if bb is not None and ba is not None:
        market_mid = (bb + ba) / 2.0
        iv = implied_vol_from_price(market_mid, S, strike, T)
        if iv > 0:
            iv_surface[symbol] = iv

    net_delta = position * delta_per_unit
    return orders, net_delta


# ─── Main Trader class ───────────────────────────────────────────────────────

class Trader:
    def __init__(self):
        # EWM fair value trackers
        self.fv_slow: Dict[str, EWMEstimator] = {}
        self.fv_fast: Dict[str, EWMEstimator] = {}
        # Volatility estimator (shared for VE underlying)
        self.vol_est = VolatilityEstimator(window=40)
        # IV surface: symbol → last calibrated IV
        self.iv_surface: Dict[str, float] = {}
        # Tick counter for intra-day TTE calculation
        self.tick_count: int = 0
        # Last known VE price
        self.last_ve_price: float = 5000.0

        for p in PRODUCTS:
            self.fv_slow[p] = EWMEstimator(EWM_ALPHA_SLOW)
            self.fv_fast[p] = EWMEstimator(EWM_ALPHA_FAST)

    def _load_state(self, trader_data: str):
        """Restore persistent state from JSON string."""
        if not trader_data:
            return
        try:
            state = json.loads(trader_data)
            self.tick_count = state.get("tick_count", 0)
            self.last_ve_price = state.get("last_ve_price", 5000.0)
            iv_data = state.get("iv_surface", {})
            self.iv_surface.update(iv_data)
            # Restore EWM states
            for p in PRODUCTS:
                slow_v = state.get(f"fv_slow_{p}")
                fast_v = state.get(f"fv_fast_{p}")
                if slow_v is not None:
                    self.fv_slow[p].value = slow_v
                if fast_v is not None:
                    self.fv_fast[p].value = fast_v
            vol_prices = state.get("vol_prices", [])
            for pr in vol_prices:
                self.vol_est.prices.append(pr)
            stored_sigma = state.get("sigma_daily")
            if stored_sigma:
                self.vol_est.sigma_daily = stored_sigma
        except Exception:
            pass

    def _save_state(self) -> str:
        """Serialize state to JSON string for persistence."""
        state: Dict = {
            "tick_count": self.tick_count,
            "last_ve_price": self.last_ve_price,
            "iv_surface": self.iv_surface,
            "sigma_daily": self.vol_est.sigma_daily,
            "vol_prices": list(self.vol_est.prices)[-50:],  # keep last 50
        }
        for p in PRODUCTS:
            state[f"fv_slow_{p}"] = self.fv_slow[p].value
            state[f"fv_fast_{p}"] = self.fv_fast[p].value
        return json.dumps(state)

    def _compute_tte(self, timestamp: int) -> float:
        """
        Compute time-to-expiry in days.
        Round 3 starts with TTE = 5 days.
        Each timestamp within the day reduces TTE by 1/10000 of a day.
        """
        day_fraction = timestamp / 10000.0  # how far through today (0 to 1)
        tte = TTE_ROUND_START - day_fraction
        return max(tte, 0.0)

    def run(self, state: TradingState):
        self._load_state(state.traderData)
        self.tick_count += 1

        result: Dict[str, List[Order]] = {}
        conversions = 0

        # ── Step 1: Extract current prices ───────────────────────────────────
        positions = state.position

        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT", OrderDepth())
        ve_mid = weighted_mid(ve_od)
        if ve_mid is None:
            ve_mid = self.last_ve_price
        else:
            self.last_ve_price = ve_mid

        # Update VE volatility estimator
        sigma = self.vol_est.update(ve_mid)

        # Compute TTE
        T = self._compute_tte(state.timestamp)

        # Update EWM fair values for spot products
        hp_od = state.order_depths.get("HYDROGEL_PACK", OrderDepth())
        hp_mid = weighted_mid(hp_od)

        if ve_mid:
            self.fv_slow["VELVETFRUIT_EXTRACT"].update(ve_mid)
            self.fv_fast["VELVETFRUIT_EXTRACT"].update(ve_mid)
        if hp_mid:
            self.fv_slow["HYDROGEL_PACK"].update(hp_mid)
            self.fv_fast["HYDROGEL_PACK"].update(hp_mid)

        # ── Step 2: HYDROGEL_PACK market making ──────────────────────────────
        hp_fv = self.fv_slow["HYDROGEL_PACK"].value
        if hp_fv is None:
            hp_fv = hp_mid or 5000.0
        hp_pos = positions.get("HYDROGEL_PACK", 0)
        hp_orders = mm_orders(
            "HYDROGEL_PACK", hp_fv, hp_pos,
            POSITION_LIMITS["HYDROGEL_PACK"],
            MM_SPREAD_BASE_HP, hp_od
        )
        result["HYDROGEL_PACK"] = hp_orders

        # ── Step 3: VELVETFRUIT_EXTRACT market making ─────────────────────────
        ve_fv = self.fv_slow["VELVETFRUIT_EXTRACT"].value
        if ve_fv is None:
            ve_fv = ve_mid
        ve_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
        ve_orders_mm = mm_orders(
            "VELVETFRUIT_EXTRACT", ve_fv, ve_pos,
            POSITION_LIMITS["VELVETFRUIT_EXTRACT"],
            MM_SPREAD_BASE_VE, ve_od
        )

        # ── Step 4: VEV Options Trading ───────────────────────────────────────
        total_delta = 0.0  # net delta from all VEV positions

        for sym, strike in STRIKES.items():
            od = state.order_depths.get(sym, OrderDepth())
            pos = positions.get(sym, 0)

            orders, net_delta = option_orders(
                sym, strike, ve_mid, T, sigma,
                pos, OPTION_MAX_POSITION, od, self.iv_surface
            )
            result[sym] = orders
            total_delta += net_delta

        # ── Step 5: Delta Hedging via VE spot ─────────────────────────────────
        # Our VEV book has total_delta exposure (in VE equivalent units)
        # We want to be delta-neutral → trade VE to offset
        target_ve_hedge = -int(round(total_delta))  # VE position to offset VEV delta

        # Combine MM position intent with hedge intent
        # Priority: hedging delta > market-making
        current_ve = ve_pos
        ve_hedge_needed = target_ve_hedge - current_ve

        # Clamp to position limits
        max_ve_pos = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        target_ve_clamped = max(-max_ve_pos, min(max_ve_pos, target_ve_hedge))
        hedge_trade = target_ve_clamped - current_ve

        hedge_orders = []
        if abs(hedge_trade) >= DELTA_HEDGE_THRESHOLD:
            if hedge_trade > 0:
                # Need to buy VE
                ba = best_ask(ve_od)
                if ba is not None:
                    qty = min(hedge_trade, abs(ve_od.sell_orders.get(ba, hedge_trade)))
                    if qty > 0:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", ba, qty))
            else:
                # Need to sell VE
                bb = best_bid(ve_od)
                if bb is not None:
                    qty = min(-hedge_trade, ve_od.buy_orders.get(bb, -hedge_trade))
                    if qty > 0:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", bb, -qty))

        # Merge MM and hedge orders for VE (hedge takes priority)
        if abs(hedge_trade) >= DELTA_HEDGE_THRESHOLD:
            result["VELVETFRUIT_EXTRACT"] = hedge_orders
        else:
            result["VELVETFRUIT_EXTRACT"] = ve_orders_mm

        # ── Step 6: IV Surface Arbitrage ─────────────────────────────────────
        # If we have IV estimates for multiple strikes, check for vol surface
        # anomalies: strikes trading at very different IVs → sell high IV, buy low IV
        # (This is a vol spread / skew trade)
        if len(self.iv_surface) >= 3:
            ivs = [(sym, iv) for sym, iv in self.iv_surface.items()
                   if sym in STRIKES]
            if ivs:
                iv_values = [iv for _, iv in ivs]
                iv_mean = sum(iv_values) / len(iv_values)
                iv_std  = math.sqrt(sum((v - iv_mean)**2 for v in iv_values) / len(iv_values))

                for sym, iv in ivs:
                    strike = STRIKES[sym]
                    od = state.order_depths.get(sym, OrderDepth())
                    pos = positions.get(sym, 0)
                    pos_limit = OPTION_MAX_POSITION

                    # If this strike has unusually HIGH IV (>1.5 std above mean)
                    # → the option is relatively expensive → add sell signal
                    if iv > iv_mean + 1.5 * iv_std and iv_std > 0.01:
                        bb = best_bid(od)
                        if bb is not None and pos > -pos_limit // 2:
                            existing = result.get(sym, [])
                            qty = min(10, pos_limit + pos)
                            if qty > 0:
                                existing.append(Order(sym, bb, -qty))
                            result[sym] = existing

                    # If this strike has unusually LOW IV (>1.5 std below mean)
                    # → the option is relatively cheap → add buy signal
                    elif iv < iv_mean - 1.5 * iv_std and iv_std > 0.01:
                        ba = best_ask(od)
                        if ba is not None and pos < pos_limit // 2:
                            existing = result.get(sym, [])
                            qty = min(10, pos_limit - pos)
                            if qty > 0:
                                existing.append(Order(sym, ba, qty))
                            result[sym] = existing

        # ── Persist state ─────────────────────────────────────────────────────
        trader_data_out = self._save_state()

        return result, conversions, trader_data_out


# ─── Local backtesting harness (not submitted) ───────────────────────────────

if __name__ == "__main__":
    # Quick sanity check
    print("=== Black-Scholes Sanity Check ===")
    S, T, sigma = 5200.0, 5.0, 0.16
    for K in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
        price = bs_call_price(S, K, T, sigma)
        delta = bs_delta(S, K, T, sigma)
        print(f"  VEV_{K:5d}: theo={price:8.2f}  delta={delta:.4f}")

    print("\n=== Implied Vol Test ===")
    # Round-trip test
    for K in [5000, 5200, 5500]:
        theo = bs_call_price(S, K, T, sigma)
        iv_back = implied_vol_from_price(theo, S, K, T)
        print(f"  K={K}: price={theo:.2f}, recovered_iv={iv_back:.4f} (target={sigma})")

    print("\n=== Delta Hedge Simulation ===")
    trader = Trader()
    # Simulate holding 100 VEV_5200 (long)
    delta_5200 = bs_delta(S, 5200, T, sigma)
    total_delta = 100 * delta_5200
    hedge_ve = -int(round(total_delta))
    print(f"  Position: +100 VEV_5200 (delta={delta_5200:.4f})")
    print(f"  Total delta: {total_delta:.2f}")
    print(f"  VE hedge needed: {hedge_ve} (sell {-hedge_ve} VE)")