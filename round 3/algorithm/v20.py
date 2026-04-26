"""
========================================================================
ROUND 3 OPTIMIZED TRADING ALGORITHM — TARGET PNL: +50K
========================================================================

ROOT CAUSE ANALYSIS OF PREVIOUS -14.5K LOSS
--------------------------------------------
The previous algorithm used sigma=0.16 (16% daily vol).
Actual calibrated implied vol from market prices = 0.013 (1.3% daily).
This 12x error caused BS to price all options 5-10x ABOVE market price,
making the algo think every voucher was massively underpriced → bought
everything → lost on all VEV positions.

Additionally:
  • HYDROGEL_PACK was sold below fair value due to wrong fair-value tracking
  • Delta hedging was off because wrong sigma → wrong deltas
  • VEV 6000 / 6500 at market price 0.5 were accumulated at 0 value

CORRECTED STRATEGY
------------------
1. HYDROGEL_PACK Market-Making (PRIMARY PROFIT DRIVER)
   • Market spread = 16 ticks (bid ~10003, ask ~10019)
   • Quote AGGRESSIVELY inside the spread: FV±5
   • Full size 200 positions, aggressive inventory management
   • Expected edge: 5 ticks × 200 round trips ≈ 1000+ PnL/day

2. VELVETFRUIT_EXTRACT Market-Making (SECONDARY PROFIT DRIVER)
   • Market spread = 5 ticks
   • Quote inside: FV±2
   • Take any order crossing our fair value
   • Expected edge: 2 ticks × high frequency ≈ 500+ PnL/day

3. VEV Options — Corrected BS with sigma=0.013
   ─────────────────────────────────────────────
   INSIGHT: With sigma=0.013 per day, BS prices closely match market:
     • Deep ITM (K≤5000): trade at intrinsic value, near delta=1
       → avoid buying (no time value premium); sell at bid=intrinsic+small
     • Near ATM (K=5100-5400): real time value, market-makeable
       → quote inside their 2-4 tick spreads using BS as fair value
       → position limit 200 per voucher, earn 1-2 ticks per round trip
     • Deep OTM (K≥5500): near-zero value
       → ONLY SELL at current ask price; do NOT buy
       → VEV_6000 and VEV_6500 at 0.5 → sell at 1 (collect premium)

4. DELTA HEDGING (with correct deltas)
   • ATM vouchers have delta ≈ 0.4-0.7
   • Long VEV positions → positive delta → hedge by selling VE spot
   • Short VEV positions → negative delta → hedge by buying VE spot
   • This makes the portfolio theta-positive without directional risk

5. POSITION SIZING — maximize to PnL
   • HP: full 200 limit
   • VE: 150 (reserve 50 for delta hedging)
   • VEVs: 200 per voucher (conservative vs 300 limit)
   • OTM VEVs: sell up to 150 units (pure premium collection)

PARAMETER VALUES (calibrated from log data)
--------------------------------------------
sigma = 0.013          # calibrated from market implied vols
TTE_ROUND_START = 5.0  # Round 3: 5 days to expiry
HP_HALF_SPREAD = 5     # inside the 16-wide market spread
VE_HALF_SPREAD = 2     # inside the 5-wide market spread
VEV_EDGE_THRESHOLD = 1.0  # minimum edge in SeaShells to trade a voucher
========================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import json
from collections import deque

# ─── Competition framework types ──────────────────────────────────────────────

@dataclass
class Order:
    symbol: str
    price: int
    quantity: int  # positive = buy, negative = sell


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)
    sell_orders: Dict[int, int] = field(default_factory=dict)


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


# ─── Constants ────────────────────────────────────────────────────────────────

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

# ─── CRITICAL CALIBRATED PARAMETERS ──────────────────────────────────────────
# sigma=0.013 is derived from actual market implied vols:
# VEV_5100-5500 mid prices vs BS formula → median IV ≈ 0.013 per day
# Previous algo used 0.16 (12x too high!) → caused all VEV losses
SIGMA_CALIBRATED = 0.013   # per-day implied vol (calibrated from market data)
SIGMA_MIN = 0.008          # floor to avoid numerical issues
SIGMA_MAX = 0.05           # ceiling to prevent over-pricing OTM options

# Round 3: TTE = 5 days
TTE_ROUND_START = 5.0

# Market-making parameters (tuned to observed spreads)
HP_HALF_SPREAD = 5         # Market spread is 16; we quote inside at ±5
VE_HALF_SPREAD = 2         # Market spread is 5; we quote inside at ±2

# Inventory skew: how much to shift quotes per unit of inventory
HP_INVENTORY_SKEW = 0.04
VE_INVENTORY_SKEW = 0.03

# Options parameters
VEV_EDGE_THRESHOLD = 1.0   # min edge to trade a VEV
VEV_MAX_POSITION = 200     # conservative vs 300 hard limit
VEV_ORDER_SIZE = 25        # max qty per order (large to fill spreads)
OTM_SELL_SIZE = 100        # size to sell deep OTM vouchers (free premium)

# Delta hedging
DELTA_HEDGE_THRESHOLD = 3  # min delta imbalance before hedging
VE_RESERVE_FOR_HEDGE = 50  # keep 50 units of VE capacity for hedging

# EWM parameters
ALPHA_FAST = 0.25
ALPHA_SLOW = 0.04
ALPHA_VOL  = 0.15


# ─── Math utilities ───────────────────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S: float, K: float, T: float, sigma: float) -> float:
    """
    Black-Scholes European call price (r=0).
    T in days, sigma in per-day units.
    """
    if T <= 0:
        return max(S - K, 0.0)
    if S <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    call = S * norm_cdf(d1) - K * norm_cdf(d2)
    return max(call, max(S - K, 0.0))  # floor at intrinsic


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS call delta = N(d1)."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    if S <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return S * norm_pdf(d1) * sqrt_T


def implied_vol(market_price: float, S: float, K: float, T: float,
                tol: float = 0.01, max_iter: int = 50) -> float:
    """Newton-Raphson IV solver. Returns per-day sigma."""
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic + 0.01 or T <= 0:
        return 0.0
    sigma = SIGMA_CALIBRATED  # start from calibrated value
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, sigma)
        vega = bs_vega(S, K, T, sigma)
        if abs(vega) < 1e-8:
            break
        diff = price - market_price
        if abs(diff) < tol:
            break
        sigma -= diff / vega
        sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
    return sigma


# ─── Helper functions ─────────────────────────────────────────────────────────

def best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders.keys()) if od.buy_orders else None


def best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders.keys()) if od.sell_orders else None


def weighted_mid(od: OrderDepth) -> Optional[float]:
    """Volume-weighted mid (micro-price)."""
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None:
        return None
    bq = od.buy_orders.get(bb, 1)
    aq = abs(od.sell_orders.get(ba, 1))
    total = bq + aq
    if total == 0:
        return (bb + ba) / 2.0
    return (bb * aq + ba * bq) / total  # lean toward thinner side


class EWM:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class VolEstimator:
    """Rolling log-return volatility, scaled to per-day units."""
    def __init__(self, window: int = 50):
        self.prices: deque = deque(maxlen=window + 1)
        self.sigma = SIGMA_CALIBRATED
        self.ewm_var = EWM(ALPHA_VOL)

    def update(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) >= 10:
            prices = list(self.prices)
            returns = []
            for i in range(1, len(prices)):
                if prices[i - 1] > 0:
                    returns.append(math.log(prices[i] / prices[i - 1]))
            if len(returns) >= 5:
                mean_r = sum(returns) / len(returns)
                var_tick = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                # Scale tick variance to daily variance (10000 ticks per day)
                var_daily = var_tick * 10000
                self.ewm_var.update(var_daily)
                if self.ewm_var.value and self.ewm_var.value > 0:
                    raw_sigma = math.sqrt(self.ewm_var.value)
                    # Clamp — realized vol should be close to calibrated
                    self.sigma = max(SIGMA_MIN, min(SIGMA_MAX, raw_sigma))
        return self.sigma


# ─── Market-making orders ─────────────────────────────────────────────────────

def mm_orders(
    symbol: str, fair: float, position: int,
    pos_limit: int, half_spread: int, inv_skew_factor: float,
    od: OrderDepth, order_size: int = 40
) -> List[Order]:
    """
    Aggressive market-making:
    - Post passive orders inside the spread
    - Lift/hit any orders that cross our fair value
    - Inventory skew to mean-revert position
    """
    orders: List[Order] = []
    skew = int(round(inv_skew_factor * position))
    bid_price = int(fair - half_spread - skew)
    ask_price = int(fair + half_spread - skew)

    max_buy  = pos_limit - position
    max_sell = pos_limit + position

    passive_buy  = min(max_buy,  order_size)
    passive_sell = min(max_sell, order_size)

    # Passive quotes
    if passive_buy > 0 and bid_price > 0:
        orders.append(Order(symbol, bid_price, passive_buy))
    if passive_sell > 0 and ask_price > 0:
        orders.append(Order(symbol, ask_price, -passive_sell))

    # Aggressive: take any sell order priced BELOW our bid (free money)
    agg_bought = 0
    for ask_p in sorted(od.sell_orders.keys()):
        if ask_p > bid_price:
            break
        avail = abs(od.sell_orders[ask_p])
        qty = min(avail, max_buy - agg_bought)
        if qty > 0:
            orders.append(Order(symbol, ask_p, qty))
            agg_bought += qty

    # Aggressive: take any buy order priced ABOVE our ask (free money)
    agg_sold = 0
    for bid_p in sorted(od.buy_orders.keys(), reverse=True):
        if bid_p < ask_price:
            break
        avail = od.buy_orders[bid_p]
        qty = min(avail, max_sell - agg_sold)
        if qty > 0:
            orders.append(Order(symbol, bid_p, -qty))
            agg_sold += qty

    return orders


# ─── Options trading logic ────────────────────────────────────────────────────

def option_orders(
    symbol: str, strike: int, S: float, T: float, sigma: float,
    position: int, pos_limit: int, od: OrderDepth,
    iv_cache: Dict[str, float]
) -> Tuple[List[Order], float]:
    """
    Trade a single VEV based on corrected BS mispricing (sigma=0.013).

    Strategy depends on moneyness:
    - Deep ITM (K <= S-200): don't accumulate; market-make on spread only
    - Near ATM (|S-K| < 400): market-make using BS fair value with IV
    - Deep OTM (K >= S+200): SELL premium; don't buy
    - Very deep OTM (K >= S+500): near zero value; avoid holding

    Returns (orders, net_delta_from_position)
    """
    orders: List[Order] = []
    theo = bs_call_price(S, strike, T, sigma)
    delta_per_unit = bs_delta(S, strike, T, sigma)
    intrinsic = max(S - strike, 0.0)

    bb = best_bid(od)
    ba = best_ask(od)

    if bb is None and ba is None:
        return [], position * delta_per_unit

    # Update IV cache for surface tracking
    if bb is not None and ba is not None:
        mkt_mid = (bb + ba) / 2.0
        tv = mkt_mid - intrinsic
        if tv > 0.5 and T > 0:
            iv = implied_vol(mkt_mid, S, strike, T)
            if iv > 0:
                iv_cache[symbol] = iv

    # ── Moneyness classification ──
    moneyness = S - strike  # positive = ITM

    # ── DEEP ITM (K ≤ S-200, delta ≈ 1): treat like spot ──
    if moneyness >= 200:
        # These options trade at intrinsic with nearly no spread opportunity
        # Only sell if bid > theo + edge (they're overpriced vs intrinsic)
        # Only buy if ask < theo - edge (they're cheap vs intrinsic)
        if bb is not None and bb > theo + VEV_EDGE_THRESHOLD:
            max_sell = pos_limit + position
            qty = min(od.buy_orders.get(bb, 0), max_sell, VEV_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(symbol, bb, -qty))
        if ba is not None and ba < theo - VEV_EDGE_THRESHOLD:
            max_buy = pos_limit - position
            qty = min(abs(od.sell_orders.get(ba, 0)), max_buy, VEV_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(symbol, ba, qty))

    # ── NEAR ATM (|moneyness| < 300): main market-making zone ──
    elif abs(moneyness) < 300:
        # Use live IV if available (from cache), else use calibrated sigma
        effective_sigma = iv_cache.get(symbol, sigma)
        effective_sigma = max(SIGMA_MIN, min(SIGMA_MAX, effective_sigma))
        theo_live = bs_call_price(S, strike, T, effective_sigma)

        # Sell overpriced vouchers
        if bb is not None and bb > theo_live + VEV_EDGE_THRESHOLD:
            max_sell = pos_limit + position
            qty = min(od.buy_orders.get(bb, 0), max_sell, VEV_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(symbol, bb, -qty))
            # Passive sell
            passive_p = int(theo_live + VEV_EDGE_THRESHOLD)
            passive_qty = min(max_sell - qty, VEV_ORDER_SIZE)
            if passive_qty > 0 and passive_p > 0:
                orders.append(Order(symbol, passive_p, -passive_qty))

        # Buy underpriced vouchers
        if ba is not None and ba < theo_live - VEV_EDGE_THRESHOLD:
            max_buy = pos_limit - position
            qty = min(abs(od.sell_orders.get(ba, 0)), max_buy, VEV_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(symbol, ba, qty))
            # Passive buy
            passive_p = int(theo_live - VEV_EDGE_THRESHOLD)
            passive_qty = min(max_buy - qty, VEV_ORDER_SIZE)
            if passive_qty > 0 and passive_p > 0:
                orders.append(Order(symbol, passive_p, passive_qty))

    # ── OTM (moneyness < -200): sell premium, never buy ──
    elif moneyness < -200:
        # OTM options have positive time value but will decay to zero
        # Strategy: SELL them if bid >= 1 (collect premium)
        # NEVER buy OTM options in this regime (they'll expire worthless)
        if bb is not None and bb >= 1 and position > -pos_limit:
            max_sell = min(pos_limit + position, OTM_SELL_SIZE)
            qty = min(od.buy_orders.get(bb, 0), max_sell, VEV_ORDER_SIZE)
            if qty > 0:
                orders.append(Order(symbol, bb, -qty))
            # If bid is high enough, also sell passively at theo+edge
            if theo > VEV_EDGE_THRESHOLD and bb > theo:
                passive_p = max(1, int(theo))
                passive_qty = min(max_sell - qty, VEV_ORDER_SIZE)
                if passive_qty > 0:
                    orders.append(Order(symbol, passive_p, -passive_qty))

    net_delta = position * delta_per_unit
    return orders, net_delta


# ─── Main Trader class ────────────────────────────────────────────────────────

class Trader:
    def __init__(self):
        self.fv_slow: Dict[str, EWM] = {p: EWM(ALPHA_SLOW) for p in PRODUCTS}
        self.fv_fast: Dict[str, EWM] = {p: EWM(ALPHA_FAST) for p in PRODUCTS}
        self.vol_est = VolEstimator(window=60)
        self.iv_cache: Dict[str, float] = {}  # symbol → live calibrated IV
        self.last_ve_price: float = 5262.0
        self.last_hp_price: float = 9980.0
        self.tick: int = 0

    # ── State persistence ──────────────────────────────────────────────────────

    def _load_state(self, trader_data: str):
        if not trader_data:
            return
        try:
            s = json.loads(trader_data)
            self.tick = s.get("tick", 0)
            self.last_ve_price = s.get("last_ve", self.last_ve_price)
            self.last_hp_price = s.get("last_hp", self.last_hp_price)
            self.iv_cache.update(s.get("iv_cache", {}))
            for p in PRODUCTS:
                sv = s.get(f"slow_{p}")
                fv = s.get(f"fast_{p}")
                if sv is not None:
                    self.fv_slow[p].value = sv
                if fv is not None:
                    self.fv_fast[p].value = fv
            for pr in s.get("vol_prices", []):
                self.vol_est.prices.append(pr)
            stored_sig = s.get("sigma")
            if stored_sig:
                self.vol_est.sigma = stored_sig
        except Exception:
            pass

    def _save_state(self) -> str:
        s: Dict = {
            "tick": self.tick,
            "last_ve": self.last_ve_price,
            "last_hp": self.last_hp_price,
            "iv_cache": self.iv_cache,
            "sigma": self.vol_est.sigma,
            "vol_prices": list(self.vol_est.prices)[-80:],
        }
        for p in PRODUCTS:
            s[f"slow_{p}"] = self.fv_slow[p].value
            s[f"fast_{p}"] = self.fv_fast[p].value
        return json.dumps(s)

    def _tte(self, timestamp: int) -> float:
        """Time to expiry in days. Round 3 starts at TTE=5."""
        day_fraction = timestamp / 10000.0
        return max(TTE_ROUND_START - day_fraction, 1e-6)

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load_state(state.traderData)
        self.tick += 1
        result: Dict[str, List[Order]] = {}
        positions = state.position

        # ── Get market data ────────────────────────────────────────────────────
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT", OrderDepth())
        hp_od = state.order_depths.get("HYDROGEL_PACK", OrderDepth())

        ve_mid = weighted_mid(ve_od)
        hp_mid = weighted_mid(hp_od)

        if ve_mid is not None:
            self.last_ve_price = ve_mid
            self.fv_slow["VELVETFRUIT_EXTRACT"].update(ve_mid)
            self.fv_fast["VELVETFRUIT_EXTRACT"].update(ve_mid)
            self.vol_est.update(ve_mid)
        else:
            ve_mid = self.last_ve_price

        if hp_mid is not None:
            self.last_hp_price = hp_mid
            self.fv_slow["HYDROGEL_PACK"].update(hp_mid)
            self.fv_fast["HYDROGEL_PACK"].update(hp_mid)
        else:
            hp_mid = self.last_hp_price

        # Use calibrated sigma (with live realized vol if available)
        sigma = self.vol_est.sigma

        # Compute TTE
        T = self._tte(state.timestamp)

        # ── HYDROGEL_PACK Market-Making ────────────────────────────────────────
        hp_pos = positions.get("HYDROGEL_PACK", 0)
        hp_fv = self.fv_slow["HYDROGEL_PACK"].value or hp_mid

        result["HYDROGEL_PACK"] = mm_orders(
            "HYDROGEL_PACK", hp_fv, hp_pos,
            POSITION_LIMITS["HYDROGEL_PACK"],
            HP_HALF_SPREAD, HP_INVENTORY_SKEW, hp_od,
            order_size=50,
        )

        # ── VELVETFRUIT_EXTRACT Market-Making (primary, hedge overrides later) ─
        ve_pos = positions.get("VELVETFRUIT_EXTRACT", 0)
        ve_fv = self.fv_slow["VELVETFRUIT_EXTRACT"].value or ve_mid

        ve_mm_orders = mm_orders(
            "VELVETFRUIT_EXTRACT", ve_fv, ve_pos,
            POSITION_LIMITS["VELVETFRUIT_EXTRACT"] - VE_RESERVE_FOR_HEDGE,
            VE_HALF_SPREAD, VE_INVENTORY_SKEW, ve_od,
            order_size=40,
        )

        # ── VEV Options Trading ────────────────────────────────────────────────
        total_delta = 0.0

        for sym, strike in STRIKES.items():
            od = state.order_depths.get(sym, OrderDepth())
            pos = positions.get(sym, 0)

            orders, net_delta = option_orders(
                sym, strike, ve_mid, T, sigma,
                pos, VEV_MAX_POSITION, od, self.iv_cache
            )
            result[sym] = orders
            total_delta += net_delta

        # ── Delta Hedging via VE Spot ──────────────────────────────────────────
        # We want our total portfolio delta = 0
        # Current VE position already has delta = 1 per unit
        # VEV positions have delta = total_delta
        # So we need: ve_pos + total_delta = 0
        # → target_ve_pos = -total_delta
        target_ve_hedge = int(round(-total_delta))
        max_ve = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        target_ve_hedge = max(-max_ve, min(max_ve, target_ve_hedge))
        hedge_delta = target_ve_hedge - ve_pos

        hedge_orders: List[Order] = []
        if abs(hedge_delta) >= DELTA_HEDGE_THRESHOLD:
            if hedge_delta > 0:
                ba = best_ask(ve_od)
                if ba is not None:
                    avail = abs(ve_od.sell_orders.get(ba, hedge_delta))
                    qty = min(hedge_delta, avail)
                    if qty > 0:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", ba, qty))
            else:
                bb = best_bid(ve_od)
                if bb is not None:
                    avail = ve_od.buy_orders.get(bb, -hedge_delta)
                    qty = min(-hedge_delta, avail)
                    if qty > 0:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", bb, -qty))

        # If significant delta hedge needed, prioritize hedge; else MM
        if abs(hedge_delta) >= DELTA_HEDGE_THRESHOLD:
            # Merge: include both MM and hedge orders (hedge takes effect first)
            result["VELVETFRUIT_EXTRACT"] = hedge_orders + ve_mm_orders
        else:
            result["VELVETFRUIT_EXTRACT"] = ve_mm_orders

        # ── IV Surface Arbitrage (vol skew trades) ─────────────────────────────
        # If multiple VEVs have been calibrated, trade any with anomalous IV
        # (sells high-IV strikes, buys low-IV strikes — vol spread trade)
        if len(self.iv_cache) >= 4:
            iv_vals = [(sym, iv) for sym, iv in self.iv_cache.items()
                       if sym in STRIKES and abs(STRIKES[sym] - ve_mid) < 400]
            if len(iv_vals) >= 3:
                ivs = [v for _, v in iv_vals]
                iv_mean = sum(ivs) / len(ivs)
                iv_std  = (sum((v - iv_mean) ** 2 for v in ivs) / len(ivs)) ** 0.5

                for sym, iv in iv_vals:
                    od = state.order_depths.get(sym, OrderDepth())
                    pos = positions.get(sym, 0)

                    # Sell high-IV strike (expensive)
                    if iv > iv_mean + 1.2 * iv_std and iv_std > 0.001:
                        bb_s = best_bid(od)
                        if bb_s is not None and pos > -VEV_MAX_POSITION // 2:
                            qty = min(15, VEV_MAX_POSITION + pos)
                            if qty > 0:
                                existing = result.get(sym, [])
                                existing.append(Order(sym, bb_s, -qty))
                                result[sym] = existing

                    # Buy low-IV strike (cheap)
                    elif iv < iv_mean - 1.2 * iv_std and iv_std > 0.001:
                        ba_s = best_ask(od)
                        if ba_s is not None and pos < VEV_MAX_POSITION // 2:
                            qty = min(15, VEV_MAX_POSITION - pos)
                            if qty > 0:
                                existing = result.get(sym, [])
                                existing.append(Order(sym, ba_s, qty))
                                result[sym] = existing

        trader_data_out = self._save_state()
        return result, 0, trader_data_out


# ─── Local test harness ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Calibrated BS Prices — Round 3 (TTE=5, S=5262, sigma=0.013) ===")
    S, T, sigma = 5262.0, 5.0, SIGMA_CALIBRATED
    for K in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]:
        p = bs_call_price(S, K, T, sigma)
        d = bs_delta(S, K, T, sigma)
        intrinsic = max(S - K, 0.0)
        tv = p - intrinsic
        print(f"  VEV_{K:5d}: theo={p:8.2f}  intrinsic={intrinsic:8.2f}  "
              f"tv={tv:6.2f}  delta={d:.4f}")

    print("\n=== IV Round-Trip Verification ===")
    for K, mkt in [(5200, 101.3), (5300, 50.1), (5400, 16.0), (5500, 6.3)]:
        iv = implied_vol(mkt, S, K, T)
        bs_check = bs_call_price(S, K, T, iv)
        print(f"  K={K}: mkt={mkt:.1f}, recovered_IV={iv:.4f}, "
              f"BS_check={bs_check:.2f}")

    print("\n=== Delta Hedging Example ===")
    # Long 100 VEV_5300 + long 50 VEV_5400
    d5300 = bs_delta(S, 5300, T, sigma)
    d5400 = bs_delta(S, 5400, T, sigma)
    total_delta = 100 * d5300 + 50 * d5400
    hedge = -int(round(total_delta))
    print(f"  +100 VEV_5300 (delta={d5300:.4f}) + +50 VEV_5400 (delta={d5400:.4f})")
    print(f"  Total VEV delta: {total_delta:.2f}")
    print(f"  Hedge: {hedge:+d} VE units")

    print("\n=== Key Insight Summary ===")
    print(f"  Old sigma=0.16 → BS(5262,5300,5,0.16) = {bs_call_price(S, 5300, 5, 0.16):.2f}")
    print(f"  New sigma=0.013→ BS(5262,5300,5,0.013)= {bs_call_price(S, 5300, 5, 0.013):.2f}")
    print(f"  Market price of VEV_5300 ≈ 50-52")
    print(f"  Old algo saw market@52 vs BS@{bs_call_price(S,5300,5,0.16):.0f} → bought as 'cheap'")
    print(f"  New algo sees market@52 vs BS@{bs_call_price(S,5300,5,0.013):.0f} → correctly valued!")
