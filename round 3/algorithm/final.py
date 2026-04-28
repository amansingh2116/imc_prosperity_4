from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional
import json
import math

# ============================================================
# Round 3 strategy v21, updated with:
# - end-of-simulation flattening
# - zero VE position clearing
# - aggressive close-only inventory reduction
# - adverse-selection filter against large MM-style orders
# ============================================================

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

# Time / option settings
TICKS_PER_DAY = 1_000_000
TTE_START_DAYS = 5.0
TRADING_DAYS = 252.0

# End-game windows (timestamps in round are ~0..100_000)
CLOSE_ONLY_START = 90_000   # stop opening fresh risk
FLATTEN_START = 96_000      # aggressively flatten everything
VE_FLATTEN_START = 94_000   # zero out VE before final close

# Volatility / pricing
SIGMA = 0.205

# Spot market making
HP_EMA_ALPHA_FAST = 0.12
HP_EMA_ALPHA_SLOW = 0.004
VE_EMA_ALPHA_FAST = 0.10
VE_EMA_ALPHA_SLOW = 0.020

HP_TAKE_EDGE = 3.5
HP_PASSIVE_OFFSET = 4
HP_ORDER_SIZE = 70

VE_TAKE_EDGE = 1.8
VE_PASSIVE_OFFSET = 2
VE_ORDER_SIZE = 25

# small-bot / MM separation
SMALL_ORDER_MAX = 15
MM_ORDER_MIN = 20

# OBI fair-value shift (small, to reduce adverse selection)
HP_OBI_WEIGHT = 0.14
VE_OBI_WEIGHT = 0.10

# Options: only keep the strikes that worked best
OPTION_CONFIG = {
    "VEV_5000": {"short_threshold": 1.0, "cover_threshold": 0.5, "base_size": 8,  "soft_cap": 90},
    "VEV_5100": {"short_threshold": 1.0, "cover_threshold": 0.5, "base_size": 10, "soft_cap": 120},
    "VEV_5300": {"short_threshold": 1.0, "cover_threshold": 0.5, "base_size": 10, "soft_cap": 120},
    "VEV_5400": {"short_threshold": 1.1, "cover_threshold": 0.6, "base_size": 6,  "soft_cap": 60},
}

SKIP_STRIKES = {"VEV_4000", "VEV_4500", "VEV_5200", "VEV_5500", "VEV_6000", "VEV_6500"}

# ------------------------------------------------------------
# Math helpers
# ------------------------------------------------------------

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

def best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders.keys()) if od.buy_orders else None

def best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders.keys()) if od.sell_orders else None

def mid(od: OrderDepth) -> Optional[float]:
    bb, ba = best_bid(od), best_ask(od)
    if bb is None or ba is None:
        return None
    return 0.5 * (bb + ba)

def micro_mid(od: OrderDepth) -> Optional[float]:
    bb, ba = best_bid(od), best_ask(od)
    if bb is None or ba is None:
        return None
    bq = max(1, od.buy_orders.get(bb, 1))
    aq = max(1, abs(od.sell_orders.get(ba, -1)))
    return (bb * aq + ba * bq) / (bq + aq)

def spread(od: OrderDepth) -> float:
    bb, ba = best_bid(od), best_ask(od)
    if bb is None or ba is None:
        return 0.0
    return float(ba - bb)

def imbalance(od: OrderDepth) -> float:
    bid_vol = sum(max(v, 0) for v in od.buy_orders.values())
    ask_vol = sum(abs(v) for v in od.sell_orders.values())
    denom = bid_vol + ask_vol
    if denom <= 0:
        return 0.0
    return (bid_vol - ask_vol) / denom

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

class EMA:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, x: Optional[float]) -> Optional[float]:
        if x is None:
            return self.value
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

# ------------------------------------------------------------
# Trader
# ------------------------------------------------------------

class Trader:
    def __init__(self):
        self.hp_fast = EMA(HP_EMA_ALPHA_FAST)
        self.hp_slow = EMA(HP_EMA_ALPHA_SLOW)
        self.ve_fast = EMA(VE_EMA_ALPHA_FAST)
        self.ve_slow = EMA(VE_EMA_ALPHA_SLOW)

        # track persistent option richness
        self.opt_resid_ema: Dict[str, float] = {}

    # persistence
    def _load(self, trader_data: str) -> None:
        if not trader_data:
            return
        try:
            s = json.loads(trader_data)
            self.hp_fast.value = s.get("hp_fast")
            self.hp_slow.value = s.get("hp_slow")
            self.ve_fast.value = s.get("ve_fast")
            self.ve_slow.value = s.get("ve_slow")
            self.opt_resid_ema = {str(k): float(v) for k, v in s.get("opt_resid_ema", {}).items()}
        except Exception:
            pass

    def _save(self) -> str:
        return json.dumps({
            "hp_fast": self.hp_fast.value,
            "hp_slow": self.hp_slow.value,
            "ve_fast": self.ve_fast.value,
            "ve_slow": self.ve_slow.value,
            "opt_resid_ema": self.opt_resid_ema,
        })

    def _tte_years(self, timestamp: int) -> float:
        days_left = max(TTE_START_DAYS - timestamp / TICKS_PER_DAY, 0.01)
        return days_left / TRADING_DAYS

    def _is_close_only(self, timestamp: int) -> bool:
        return timestamp >= CLOSE_ONLY_START

    def _is_flatten(self, timestamp: int) -> bool:
        return timestamp >= FLATTEN_START

    def _is_ve_flatten(self, timestamp: int) -> bool:
        return timestamp >= VE_FLATTEN_START

    # --------------------------------------------------------
    # Helpers for flattening and adverse selection
    # --------------------------------------------------------

    def _flatten_orders(self, symbol: str, position: int, od: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        bb = best_bid(od)
        ba = best_ask(od)

        # Flatten by crossing the book. This is intentionally aggressive.
        if position > 0:
            if bb is not None:
                orders.append(Order(symbol, bb, -position))
            elif ba is not None:
                orders.append(Order(symbol, ba, -position))
        elif position < 0:
            if ba is not None:
                orders.append(Order(symbol, ba, -position))
            elif bb is not None:
                orders.append(Order(symbol, bb, -position))

        return [o for o in orders if o.quantity != 0]

    def _take_small_only(
        self,
        symbol: str,
        fair: float,
        position: int,
        od: OrderDepth,
        take_edge: float,
        base_size: int,
        allow_passive: bool = True,
    ) -> List[Order]:
        """
        Adverse-selection filter:
        - only take against small top-of-book orders
        - do not chase MM-sized orders around fair value
        """
        orders: List[Order] = []
        bb = best_bid(od)
        ba = best_ask(od)

        buy_cap = POSITION_LIMITS[symbol] - position
        sell_cap = POSITION_LIMITS[symbol] + position

        # Take only from small orders, not MM-style liquidity
        if ba is not None and ba <= fair - take_edge and buy_cap > 0:
            ask_qty = abs(od.sell_orders[ba])
            if ask_qty <= SMALL_ORDER_MAX:
                qty = min(ask_qty, buy_cap, base_size)
                if qty > 0:
                    orders.append(Order(symbol, ba, qty))
                    position += qty
                    buy_cap -= qty
                    sell_cap += qty

        if bb is not None and bb >= fair + take_edge and sell_cap > 0:
            bid_qty = od.buy_orders[bb]
            if bid_qty <= SMALL_ORDER_MAX:
                qty = min(bid_qty, sell_cap, base_size)
                if qty > 0:
                    orders.append(Order(symbol, bb, -qty))
                    position -= qty
                    sell_cap -= qty
                    buy_cap += qty

        # Passive quotes only if we are not in a risk-off window
        if allow_passive:
            inv_skew = 2.0 * position / max(1, POSITION_LIMITS[symbol])
            bid_px = int(round(fair - (4 if symbol == "HYDROGEL_PACK" else 2) - inv_skew))
            ask_px = int(round(fair + (4 if symbol == "HYDROGEL_PACK" else 2) - inv_skew))

            if ba is not None:
                bid_px = min(bid_px, ba - 1)
            if bb is not None:
                ask_px = max(ask_px, bb + 1)

            quote_size = max(5, base_size - int(abs(position) / 12))
            if buy_cap > 0 and bid_px > 0:
                orders.append(Order(symbol, bid_px, min(quote_size, buy_cap)))
            if sell_cap > 0:
                orders.append(Order(symbol, ask_px, -min(quote_size, sell_cap)))

        return [o for o in orders if o.quantity != 0]

    # --------------------------------------------------------
    # Spot block
    # --------------------------------------------------------

    def _spot_orders(
        self,
        symbol: str,
        fair: float,
        position: int,
        od: OrderDepth,
        take_edge: float,
        base_size: int,
        timestamp: int,
    ) -> List[Order]:
        # hard flatten very late in the sim
        if self._is_flatten(timestamp):
            return self._flatten_orders(symbol, position, od)

        # close-only window: no fresh risk on spots either; only small-bot taking and reduced passive
        if self._is_close_only(timestamp):
            return self._take_small_only(
                symbol=symbol,
                fair=fair,
                position=position,
                od=od,
                take_edge=take_edge + 0.75,
                base_size=max(4, base_size // 2),
                allow_passive=False,
            )

        return self._take_small_only(
            symbol=symbol,
            fair=fair,
            position=position,
            od=od,
            take_edge=take_edge,
            base_size=base_size,
            allow_passive=True,
        )

    # --------------------------------------------------------
    # Options block
    # --------------------------------------------------------

    def _option_orders(
        self,
        symbol: str,
        strike: int,
        ve_fair: float,
        ve_trend: float,
        timestamp: int,
        position: int,
        od: OrderDepth,
    ) -> List[Order]:
        # hard flatten all option inventory late in the sim
        if self._is_flatten(timestamp):
            return self._flatten_orders(symbol, position, od)

        # from this point on, no new option risk in the close-only window
        if self._is_close_only(timestamp):
            # zero EV position clearing / de-risking
            return self._flatten_orders(symbol, position, od) if position != 0 else []

        if symbol in SKIP_STRIKES:
            return []

        bb = best_bid(od)
        ba = best_ask(od)
        if bb is None or ba is None:
            return []

        T = self._tte_years(timestamp)
        theo = bs_call(ve_fair, float(strike), T, SIGMA)
        mid_px = 0.5 * (bb + ba)

        resid = mid_px - theo
        prev = self.opt_resid_ema.get(symbol, resid)
        self.opt_resid_ema[symbol] = 0.25 * resid + 0.75 * prev
        resid_ema = self.opt_resid_ema[symbol]

        cfg = OPTION_CONFIG.get(symbol)
        if cfg is None:
            return []

        short_threshold = cfg["short_threshold"]
        cover_threshold = cfg["cover_threshold"]
        base_size = cfg["base_size"]
        soft_cap = cfg["soft_cap"]

        # if trend is positive, rich options can remain rich slightly longer
        trend_boost = 1.0 + min(0.35, max(0.0, ve_trend) / 4.0)

        sell_cap = soft_cap + position
        orders: List[Order] = []

        # Only sell to "small quantity bots". Ignore large MM-style orders around fair.
        if bb is not None:
            bid_qty = od.buy_orders[bb]
        else:
            bid_qty = 0
        if ba is not None:
            ask_qty = abs(od.sell_orders[ba])
        else:
            ask_qty = 0

        # Short premium only if the bid is rich and the top-of-book size is small.
        can_sell = (
            sell_cap > 0
            and bid_qty <= SMALL_ORDER_MAX
            and (bb - theo >= short_threshold or resid_ema >= short_threshold * 0.75)
        )
        if can_sell:
            edge = max(bb - theo, resid_ema)
            size = int(round(base_size * trend_boost + max(0.0, edge - short_threshold)))
            size = clamp(size, 1, base_size * 2)
            qty = min(size, bid_qty, sell_cap)
            if qty > 0:
                orders.append(Order(symbol, bb, -qty))
                position -= qty
                sell_cap -= qty

        # Cover shorts only when the ask is cheap and small.
        if position < 0 and ba <= theo - cover_threshold and ask_qty <= SMALL_ORDER_MAX:
            buy_cap = abs(position)
            edge = max(theo - ba, cover_threshold)
            size = int(round(base_size * 0.8 + max(0.0, edge - cover_threshold)))
            size = clamp(size, 1, base_size * 2)
            qty = min(size, ask_qty, buy_cap)
            if qty > 0:
                orders.append(Order(symbol, ba, qty))
                position += qty

        # Passive option quotes only if the book is thin/safe and not near the close
        # (This is intentionally conservative.)
        if not self._is_close_only(timestamp):
            if bb is not None and ba is not None and ba - bb >= 2:
                if bid_qty <= SMALL_ORDER_MAX and position < soft_cap:
                    post_bid = min(int(math.floor(theo - 0.5)), ba - 1)
                    if post_bid > 0 and theo - post_bid >= short_threshold:
                        q = min(max(1, base_size // 2), soft_cap + position)
                        orders.append(Order(symbol, post_bid, -min(q, bid_qty if bid_qty > 0 else q)))

                if ask_qty <= SMALL_ORDER_MAX and position > -soft_cap:
                    post_ask = max(int(math.ceil(theo + 0.5)), bb + 1)
                    if post_ask > 0 and post_ask - theo >= cover_threshold:
                        q = min(max(1, base_size // 2), abs(position) if position < 0 else soft_cap)
                        orders.append(Order(symbol, post_ask, min(q, ask_qty if ask_qty > 0 else q)))

        return [o for o in orders if o.quantity != 0]

    # --------------------------------------------------------
    # Main
    # --------------------------------------------------------

    def run(self, state: TradingState):
        self._load(state.traderData)

        result: Dict[str, List[Order]] = {p: [] for p in PRODUCTS}

        hp_od = state.order_depths.get("HYDROGEL_PACK", OrderDepth())
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT", OrderDepth())

        hp_mid = micro_mid(hp_od)
        ve_mid = micro_mid(ve_od)

        hp_fast = self.hp_fast.update(hp_mid)
        hp_slow = self.hp_slow.update(hp_mid)
        ve_fast = self.ve_fast.update(ve_mid)
        ve_slow = self.ve_slow.update(ve_mid)

        # Fair values: EMA anchor + small order-flow pressure adjustment
        hp_fair = (hp_slow if hp_slow is not None else hp_mid if hp_mid is not None else 10000.0)
        ve_fair = (ve_slow if ve_slow is not None else ve_mid if ve_mid is not None else 5250.0)

        hp_fair += HP_OBI_WEIGHT * spread(hp_od) * imbalance(hp_od)
        ve_fair += VE_OBI_WEIGHT * spread(ve_od) * imbalance(ve_od)

        hp_trend = (hp_fast - hp_slow) if (hp_fast is not None and hp_slow is not None) else 0.0
        ve_trend = (ve_fast - ve_slow) if (ve_fast is not None and ve_slow is not None) else 0.0

        # Spot blocks
        result["HYDROGEL_PACK"] = self._spot_orders(
            symbol="HYDROGEL_PACK",
            fair=hp_fair,
            position=state.position.get("HYDROGEL_PACK", 0),
            od=hp_od,
            take_edge=HP_TAKE_EDGE,
            base_size=HP_ORDER_SIZE,
            timestamp=state.timestamp,
        )

        result["VELVETFRUIT_EXTRACT"] = self._spot_orders(
            symbol="VELVETFRUIT_EXTRACT",
            fair=ve_fair,
            position=state.position.get("VELVETFRUIT_EXTRACT", 0),
            od=ve_od,
            take_edge=VE_TAKE_EDGE,
            base_size=VE_ORDER_SIZE,
            timestamp=state.timestamp,
        )

        # Options blocks
        for symbol, strike in STRIKES.items():
            od = state.order_depths.get(symbol, OrderDepth())
            result[symbol] = self._option_orders(
                symbol=symbol,
                strike=strike,
                ve_fair=ve_fair,
                ve_trend=ve_trend,
                timestamp=state.timestamp,
                position=state.position.get(symbol, 0),
                od=od,
            )

        return result, 0, self._save()