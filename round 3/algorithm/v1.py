"""
IMC Prosperity Round 3 – Trading Algorithm
==========================================
Products
  HYDROGEL_PACK           delta-1  pos_limit=200
  VELVETFRUIT_EXTRACT     delta-1  pos_limit=200  (underlying for vouchers)
  VEV_4000 … VEV_6500     call options, 10 strikes, pos_limit=300 each
                           TTE=5 days at Round 3 start

Strategies
  1. HYDROGEL_PACK        → market-making around EMA fair value
  2. VELVETFRUIT_EXTRACT  → market-making + delta hedge for options book
  3. Options              → Black-Scholes mispricing: buy cheap / sell dear
                            Hedge aggregate delta via VELVETFRUIT_EXTRACT
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import math, json

# ────────────────────────── constants ─────────────────────────────────────────

POS_LIMIT = {
    "HYDROGEL_PACK":         200,
    "VELVETFRUIT_EXTRACT":   200,
}
OPTION_POS_LIMIT = 300   # per voucher

STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
    "VEV_6500": 6500,
}

# Round 3: TTE=5 days at timestamp 0.
# Each round day ≈ 10 000 steps × 100 ms = 1 000 000 timestamp units.
ROUND_START_TTE    = 5           # days
TIMESTAMPS_PER_DAY = 1_000_000
TRADING_DAYS_YEAR  = 252

# ── vol calibration ──────────────────────────────────────────────────────────
# Historical VEV mid-prices (from prices_round_3_day_*.csv) cluster around
# 5000–5500 with intra-day ranges of ~200–300 ticks.
# Daily σ ≈ 250/5250 ≈ 4.7 %;  annualised ≈ 4.7% × √252 ≈ 75 %.
# Use a conservative 70 % to avoid overpaying for options.
SIGMA = 0.70   # annualised implied vol

# ── market-making parameters ──────────────────────────────────────────────
MM_SPREAD_HALF  = 1       # passive quote offset from fair value
MM_ORDER_SIZE   = 20      # passive qty per side
MEAN_REV_ALPHA  = 0.04    # EMA coefficient (lower = smoother)
TAKE_EDGE       = 0.5     # take from book if (fair - ask) > this

# ── option parameters ─────────────────────────────────────────────────────
OPT_EDGE_BUY   = 3.0   # min edge to buy an option (ask < fair - edge)
OPT_EDGE_SELL  = 3.0   # min edge to sell an option (bid > fair + edge)
OPT_ORDER_SIZE = 15    # max units per option trade

DELTA_HEDGE_THRESHOLD = 8   # hedge when |net_delta| exceeds this


# ────────────────────────── Black-Scholes ─────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price (r=0)."""
    if T <= 1e-8:
        return max(S - K, 0.0)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    return S * _ncdf(d1) - K * _ncdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Delta of European call (r=0)."""
    if T <= 1e-8:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)


# ────────────────────────── helpers ──────────────────────────────────────────

def get_mid(od: OrderDepth):
    if od.buy_orders and od.sell_orders:
        return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
    if od.buy_orders:
        return float(max(od.buy_orders))
    if od.sell_orders:
        return float(min(od.sell_orders))
    return None

def clamp(qty: int, pos: int, limit: int) -> int:
    """Clamp qty so pos+qty stays in [-limit, +limit]."""
    if qty > 0:
        return min(qty, limit - pos)
    return max(qty, -limit - pos)


# ────────────────────────── Trader class ──────────────────────────────────────

class Trader:

    # ── state persistence ──────────────────────────────────────────────────

    def _load(self, td: str):
        if td:
            try:
                d = json.loads(td)
                self._ema = d.get("ema", {})
                return
            except Exception:
                pass
        self._ema: Dict[str, float] = {}

    def _save(self) -> str:
        return json.dumps({"ema": self._ema})

    # ── time to expiry ─────────────────────────────────────────────────────

    def _tte(self, ts: int) -> float:
        """TTE in years, clamped to a small positive value."""
        days = max(ROUND_START_TTE - ts / TIMESTAMPS_PER_DAY, 1e-4)
        return days / TRADING_DAYS_YEAR

    # ── market-making for a delta-1 product ───────────────────────────────

    def _mm(self, sym: str, od: OrderDepth, pos: int,
            limit: int, fair: float) -> List[Order]:
        orders: List[Order] = []
        running_pos = pos

        # Take cheap sell orders
        for ask, vol in sorted(od.sell_orders.items()):   # ascending
            if ask < fair - TAKE_EDGE:
                q = clamp(-vol, running_pos, limit)        # vol is negative
                if q > 0:
                    orders.append(Order(sym, ask, q))
                    running_pos += q
            else:
                break

        # Take expensive buy orders
        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid > fair + TAKE_EDGE:
                q = clamp(-vol, running_pos, limit)        # sell, so negative
                if q < 0:
                    orders.append(Order(sym, bid, q))
                    running_pos += q
            else:
                break

        # Passive market-making quotes
        # Skew quotes toward zero position (inventory management)
        skew = -running_pos / limit   # in [-1, 1]
        buy_px  = int(fair - MM_SPREAD_HALF + skew)
        sell_px = int(fair + MM_SPREAD_HALF + skew)

        bq = clamp(MM_ORDER_SIZE, running_pos, limit)
        sq = clamp(-MM_ORDER_SIZE, running_pos, limit)

        if bq > 0:
            orders.append(Order(sym, buy_px, bq))
        if sq < 0:
            orders.append(Order(sym, sell_px, sq))

        return orders

    # ── option edge trades ─────────────────────────────────────────────────

    def _opt_trade(self, sym: str, strike: int, od: OrderDepth,
                   pos: int, spot: float, T: float
                   ) -> Tuple[List[Order], float]:
        """Return (orders, delta_added)."""
        orders: List[Order] = []
        delta_added = 0.0
        fair  = bs_call(spot, strike, T, SIGMA)
        delta = bs_delta(spot, strike, T, SIGMA)

        # Buy if ask is below fair - edge
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            if best_ask < fair - OPT_EDGE_BUY:
                raw_vol = abs(od.sell_orders[best_ask])
                q = clamp(min(OPT_ORDER_SIZE, raw_vol), pos, OPTION_POS_LIMIT)
                if q > 0:
                    orders.append(Order(sym, best_ask, q))
                    pos += q
                    delta_added += q * delta

        # Sell if bid is above fair + edge
        if od.buy_orders:
            best_bid = max(od.buy_orders)
            if best_bid > fair + OPT_EDGE_SELL:
                raw_vol = od.buy_orders[best_bid]
                q = clamp(-min(OPT_ORDER_SIZE, raw_vol), pos, OPTION_POS_LIMIT)
                if q < 0:
                    orders.append(Order(sym, best_bid, q))
                    pos += q
                    delta_added += q * delta

        return orders, delta_added

    # ── delta hedge leg ────────────────────────────────────────────────────

    def _hedge(self, od: OrderDepth, pos: int, net_delta: float) -> List[Order]:
        """
        net_delta > 0 → long delta overall → sell underlying to flatten.
        net_delta < 0 → short delta → buy underlying.
        """
        orders: List[Order] = []
        if abs(net_delta) < DELTA_HEDGE_THRESHOLD:
            return orders
        limit  = POS_LIMIT["VELVETFRUIT_EXTRACT"]
        target = clamp(-int(round(net_delta)), pos, limit)
        if target == 0:
            return orders
        if target > 0 and od.sell_orders:
            orders.append(Order("VELVETFRUIT_EXTRACT", min(od.sell_orders), target))
        elif target < 0 and od.buy_orders:
            orders.append(Order("VELVETFRUIT_EXTRACT", max(od.buy_orders), target))
        return orders

    # ── main ───────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load(state.traderData)
        result: Dict[str, List[Order]] = {}
        T = self._tte(state.timestamp)

        # ── update EMA for delta-1 underlyings ──────────────────────────
        for sym in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
            od = state.order_depths.get(sym)
            if od is None:
                continue
            mid = get_mid(od)
            if mid is None:
                continue
            prev = self._ema.get(sym, mid)
            self._ema[sym] = MEAN_REV_ALPHA * mid + (1 - MEAN_REV_ALPHA) * prev

        spot = self._ema.get("VELVETFRUIT_EXTRACT")

        # ── HYDROGEL_PACK market-making ──────────────────────────────────
        hp_od = state.order_depths.get("HYDROGEL_PACK")
        if hp_od and "HYDROGEL_PACK" in self._ema:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._mm(
                "HYDROGEL_PACK", hp_od, pos,
                POS_LIMIT["HYDROGEL_PACK"], self._ema["HYDROGEL_PACK"])
        else:
            result["HYDROGEL_PACK"] = []

        # ── Option trades + track aggregate delta ────────────────────────
        net_delta = 0.0  # sum of (position × delta) across all options

        for sym, strike in STRIKES.items():
            od = state.order_depths.get(sym)
            pos = state.position.get(sym, 0)

            # Existing position delta
            if spot and pos != 0:
                net_delta += pos * bs_delta(spot, strike, T, SIGMA)

            if od is None or spot is None:
                result[sym] = []
                continue

            opt_orders, d_added = self._opt_trade(sym, strike, od, pos, spot, T)
            result[sym] = opt_orders
            net_delta += d_added

        # ── VELVETFRUIT_EXTRACT: MM + delta hedge ────────────────────────
        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if vev_od and spot is not None:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            mm_orders = self._mm(
                "VELVETFRUIT_EXTRACT", vev_od, pos,
                POS_LIMIT["VELVETFRUIT_EXTRACT"], spot)

            pos_after = pos + sum(o.quantity for o in mm_orders)
            hedge_orders = self._hedge(vev_od, pos_after, net_delta)
            result["VELVETFRUIT_EXTRACT"] = mm_orders + hedge_orders
        else:
            result["VELVETFRUIT_EXTRACT"] = []

        return result, 0, self._save()