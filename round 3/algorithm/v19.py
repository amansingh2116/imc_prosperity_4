from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math

# =========================
# Core constants
# =========================

PRODUCTS = [
    "HYDROGEL_PACK",
    "VELVETFRUIT_EXTRACT",
    "VEV_4000",
    "VEV_4500",
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
    "VEV_6000",
    "VEV_6500",
]

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

STRIKES: Dict[str, int] = {f"VEV_{k}": k for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]}

# Prosperity timestamps: 0..999900 with 100 ticks per step, 10,000 ticks/day.
TICKS_PER_DAY = 1_000_000
TTE_START_DAYS = 5.0
TRADING_DAYS = 252.0

# Volatility / pricing
SIGMA_INIT = 0.215
SIGMA_MIN = 0.10
SIGMA_MAX = 0.35
SIGMA_SMOOTH = 0.08  # low-pass update on smile-derived vol

# Spot fair value trackers
HP_EMA_ALPHA = 0.004
VE_EMA_ALPHA = 0.03

# Market making / execution
HP_HALF_SPREAD = 4
HP_TAKE_EDGE = 3
HP_MAKE_SIZE = 60
HP_MAX_ACTIVE = 120

VE_HALF_SPREAD = 2
VE_TAKE_EDGE = 2
VE_MAKE_SIZE = 12
VE_MAX_ACTIVE = 40

# Options
OPT_TAKE_EDGE = 0.75
OPT_MAKE_EDGE = 0.25
OPT_MAX_ORDER = 20

# Option-specific conviction thresholds
BUY_STRIKES = {"VEV_5400", "VEV_5000", "VEV_4500"}
SELL_STRIKES = {"VEV_5200", "VEV_5300", "VEV_5500"}
SKIP_STRIKES = {"VEV_6000", "VEV_6500"}

# Far OTM / deep ITM special handling
DEEP_ITM = {"VEV_4000", "VEV_4500"}

# Hedge control
DELTA_HEDGE_THRESHOLD = 8
DELTA_HEDGE_CAP = 70

# =========================
# Math helpers
# =========================

def ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0:
        return 0.0
    if T <= 1e-12:
        return max(S - K, 0.0)
    sigma = max(1e-8, sigma)
    vol = sigma * math.sqrt(T)
    if vol <= 1e-12:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    return S * ncdf(d1) - K * ncdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0:
        return 0.0
    if T <= 1e-12:
        return 1.0 if S > K else 0.0
    sigma = max(1e-8, sigma)
    vol = sigma * math.sqrt(T)
    if vol <= 1e-12:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol
    return ncdf(d1)

def implied_vol_from_price(price: float, S: float, K: float, T: float) -> Optional[float]:
    """Solve for annualized implied vol by bisection."""
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-9 or T <= 1e-12 or S <= 0:
        return None

    lo, hi = 0.01, 1.50
    # Ensure bracket
    for _ in range(15):
        if bs_call(S, K, T, lo) > price:
            lo *= 0.5
        elif bs_call(S, K, T, hi) < price:
            hi *= 1.5
        else:
            break

    if bs_call(S, K, T, lo) > price or bs_call(S, K, T, hi) < price:
        return None

    for _ in range(50):
        mid = (lo + hi) / 2.0
        val = bs_call(S, K, T, mid)
        if abs(val - price) < 1e-4:
            return mid
        if val < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

def best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders) if od.buy_orders else None

def best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders) if od.sell_orders else None

def micro_mid(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None:
        return None
    bq = od.buy_orders.get(bb, 1)
    aq = abs(od.sell_orders.get(ba, -1))
    tot = bq + aq
    if tot <= 0:
        return (bb + ba) / 2.0
    # more weight on thinner side
    return (bb * aq + ba * bq) / tot

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

# =========================
# Simple state trackers
# =========================

class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

class Trader:
    def __init__(self):
        self.hp_fair = EMA(HP_EMA_ALPHA)
        self.ve_fair = EMA(VE_EMA_ALPHA)
        self.sigma = SIGMA_INIT

    # ---------- persistence ----------
    def _load(self, trader_data: str) -> None:
        if not trader_data:
            return
        try:
            state = json.loads(trader_data)
            if state.get("hp_fair") is not None:
                self.hp_fair.value = state["hp_fair"]
            if state.get("ve_fair") is not None:
                self.ve_fair.value = state["ve_fair"]
            if state.get("sigma") is not None:
                self.sigma = float(state["sigma"])
        except Exception:
            pass

    def _save(self) -> str:
        return json.dumps({
            "hp_fair": self.hp_fair.value,
            "ve_fair": self.ve_fair.value,
            "sigma": self.sigma,
        })

    # ---------- time ----------
    def _tte_years(self, timestamp: int) -> float:
        days_left = max(TTE_START_DAYS - timestamp / TICKS_PER_DAY, 0.01)
        return days_left / TRADING_DAYS

    # ---------- execution helpers ----------
    def _take_and_make_spot(self, sym: str, fair: float, pos: int, limit: int, od: OrderDepth,
                            half_spread: int, take_edge: float, make_size: int,
                            max_active: int) -> List[Order]:
        orders: List[Order] = []

        bb = best_bid(od)
        ba = best_ask(od)

        max_buy = limit - pos
        max_sell = limit + pos

        # Aggressive takes
        if ba is not None and ba <= fair - take_edge and max_buy > 0:
            qty = min(max_buy, abs(od.sell_orders[ba]))
            if qty > 0:
                orders.append(Order(sym, ba, qty))
                pos += qty
                max_buy -= qty

        if bb is not None and bb >= fair + take_edge and max_sell > 0:
            qty = min(max_sell, od.buy_orders[bb])
            if qty > 0:
                orders.append(Order(sym, bb, -qty))
                pos -= qty
                max_sell -= qty

        # Passive quotes with inventory skew
        skew = int(round(0.05 * pos))
        bid_px = int(round(fair - half_spread - skew))
        ask_px = int(round(fair + half_spread - skew))

        if max_buy > 0 and bid_px > 0:
            orders.append(Order(sym, bid_px, min(make_size, max_buy)))
        if max_sell > 0:
            orders.append(Order(sym, ask_px, -min(make_size, max_sell)))

        # If inventory gets too large, use the opposite side more aggressively.
        if abs(pos) > max_active:
            if pos > 0 and bb is not None:
                orders.append(Order(sym, bb, -min(pos - max_active, od.buy_orders[bb])))
            elif pos < 0 and ba is not None:
                orders.append(Order(sym, ba, min(-pos - max_active, abs(od.sell_orders[ba]))))

        return orders

    def _option_quotes(self, sym: str, K: int, S: float, T: float, sigma: float,
                       pos: int, limit: int, od: OrderDepth) -> Tuple[List[Order], float]:
        """
        Returns orders and delta exposure of current inventory.
        """
        orders: List[Order] = []

        bb = best_bid(od)
        ba = best_ask(od)
        theo = bs_call(S, K, T, sigma)
        delta = bs_delta(S, K, T, sigma)

        # annualized spread cost of hedging one delta unit in VE (half-spread about 2.5)
        hedge_cost = 2.5 * delta

        max_buy = limit - pos
        max_sell = limit + pos

        # Deep ITM: keep small, symmetric, and only if edge survives hedge cost.
        if sym in DEEP_ITM:
            if ba is not None:
                net_buy = theo - ba - hedge_cost
                if net_buy > OPT_TAKE_EDGE and max_buy > 0:
                    qty = min(abs(od.sell_orders[ba]), max_buy, OPT_MAX_ORDER)
                    if qty > 0:
                        orders.append(Order(sym, ba, qty))
                        pos += qty
                        max_buy -= qty
            if bb is not None:
                net_sell = bb - theo - hedge_cost
                if net_sell > OPT_TAKE_EDGE and max_sell > 0:
                    qty = min(od.buy_orders[bb], max_sell, OPT_MAX_ORDER)
                    if qty > 0:
                        orders.append(Order(sym, bb, -qty))
                        pos -= qty
                        max_sell -= qty

            # mild passive MM only when not too close to limits
            if max_buy > 0 and max_sell > 0 and ba is not None and bb is not None:
                buy_px = max(1, int(round(theo - 1)))
                sell_px = int(round(theo + 1))
                if buy_px < ba:
                    orders.append(Order(sym, buy_px, min(5, max_buy)))
                if sell_px > bb:
                    orders.append(Order(sym, sell_px, -min(5, max_sell)))

            return orders, pos * delta

        # Far OTM: only sell if someone is paying us, otherwise skip.
        if sym in {"VEV_6000", "VEV_6500"}:
            if bb is not None and bb > 0 and max_sell > 0:
                # sell a small amount into any positive bid
                qty = min(od.buy_orders[bb], max_sell, 10)
                if qty > 0 and bb >= 1:
                    orders.append(Order(sym, bb, -qty))
            return orders, pos * delta

        # Core strikes: trade only when the net edge after hedge cost is positive.
        if ba is not None:
            net_buy = theo - ba - hedge_cost
            if net_buy > OPT_TAKE_EDGE and max_buy > 0:
                qty = min(abs(od.sell_orders[ba]), max_buy, OPT_MAX_ORDER)
                if qty > 0:
                    orders.append(Order(sym, ba, qty))
                    pos += qty
                    max_buy -= qty
            elif net_buy > OPT_MAKE_EDGE and max_buy > 0:
                px = min(int(round(theo - 1)), ba - 1)
                if px > 0:
                    orders.append(Order(sym, px, min(10, max_buy)))

        if bb is not None:
            net_sell = bb - theo - hedge_cost
            if net_sell > OPT_TAKE_EDGE and max_sell > 0:
                qty = min(od.buy_orders[bb], max_sell, OPT_MAX_ORDER)
                if qty > 0:
                    orders.append(Order(sym, bb, -qty))
                    pos -= qty
                    max_sell -= qty
            elif net_sell > OPT_MAKE_EDGE and max_sell > 0:
                px = max(int(round(theo + 1)), bb + 1)
                orders.append(Order(sym, px, -min(10, max_sell)))

        return orders, pos * delta

    def run(self, state: TradingState):
        self._load(state.traderData)
        result: Dict[str, List[Order]] = {}

        # -------- spot fair values --------
        hp_od = state.order_depths.get("HYDROGEL_PACK", OrderDepth())
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT", OrderDepth())

        hp_mid = micro_mid(hp_od)
        ve_mid = micro_mid(ve_od)

        if hp_mid is not None:
            hp_fair = self.hp_fair.update(hp_mid)
        else:
            hp_fair = self.hp_fair.value if self.hp_fair.value is not None else 10000.0

        if ve_mid is not None:
            ve_fair = self.ve_fair.update(ve_mid)
        else:
            ve_fair = self.ve_fair.value if self.ve_fair.value is not None else 5250.0

        # -------- volatility / smile calibration --------
        T = self._tte_years(state.timestamp)

        ivs = []
        # use liquid strikes to stabilize sigma; exclude deep ITM and far OTM
        for sym in ["VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]:
            od = state.order_depths.get(sym)
            if not od:
                continue
            bb = best_bid(od)
            ba = best_ask(od)
            if bb is None or ba is None or ve_fair <= 0:
                continue
            mid = 0.5 * (bb + ba)
            K = STRIKES[sym]
            iv = implied_vol_from_price(mid, ve_fair, K, T)
            if iv is not None and 0.05 <= iv <= 0.60:
                ivs.append(iv)

        if ivs:
            # Median is robust against the odd outlier strike.
            ivs.sort()
            med = ivs[len(ivs) // 2]
            self.sigma = max(SIGMA_MIN, min(SIGMA_MAX, (1.0 - SIGMA_SMOOTH) * self.sigma + SIGMA_SMOOTH * med))

        # -------- HYDROGEL_PACK --------
        hp_pos = state.position.get("HYDROGEL_PACK", 0)
        result["HYDROGEL_PACK"] = self._take_and_make_spot(
            "HYDROGEL_PACK", hp_fair, hp_pos, POSITION_LIMITS["HYDROGEL_PACK"],
            hp_od, HP_HALF_SPREAD, HP_TAKE_EDGE, HP_MAKE_SIZE, HP_MAX_ACTIVE
        )

        # -------- VELVETFRUIT_EXTRACT --------
        # Use VE mostly as hedge instrument, but still quote lightly around fair.
        ve_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        ve_orders = self._take_and_make_spot(
            "VELVETFRUIT_EXTRACT", ve_fair, ve_pos, POSITION_LIMITS["VELVETFRUIT_EXTRACT"],
            ve_od, VE_HALF_SPREAD, VE_TAKE_EDGE, VE_MAKE_SIZE, VE_MAX_ACTIVE
        )

        # -------- Options --------
        total_delta = 0.0
        option_orders_all: Dict[str, List[Order]] = {}

        for sym, K in STRIKES.items():
            od = state.order_depths.get(sym, OrderDepth())
            pos = state.position.get(sym, 0)
            orders, delta_exposure = self._option_quotes(sym, K, ve_fair, T, self.sigma, pos, POSITION_LIMITS[sym], od)
            if orders:
                option_orders_all[sym] = orders
            total_delta += delta_exposure

        result.update(option_orders_all)

        # -------- Delta hedge on VE only when meaningful --------
        # Hedge current option inventory delta. Do not overtrade for tiny exposure.
        current_ve_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)

        target_ve_pos = -int(round(total_delta))
        hedge_qty = target_ve_pos - current_ve_pos

        # Don't let hedge logic completely overwrite profitable VE quotes
        if abs(hedge_qty) >= DELTA_HEDGE_THRESHOLD:
            hedge_orders: List[Order] = []
            if hedge_qty > 0:
                ba = best_ask(ve_od)
                if ba is not None:
                    qty = min(hedge_qty, abs(ve_od.sell_orders[ba]), DELTA_HEDGE_CAP)
                    if qty > 0:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", ba, qty))
            else:
                bb = best_bid(ve_od)
                if bb is not None:
                    qty = min(-hedge_qty, ve_od.buy_orders[bb], DELTA_HEDGE_CAP)
                    if qty > 0:
                        hedge_orders.append(Order("VELVETFRUIT_EXTRACT", bb, -qty))

            # hedge only if it is non-trivial
            if hedge_orders:
                result["VELVETFRUIT_EXTRACT"] = hedge_orders
        else:
            result["VELVETFRUIT_EXTRACT"] = ve_orders

        trader_data = self._save()
        conversions = 0
        return result, conversions, trader_data
