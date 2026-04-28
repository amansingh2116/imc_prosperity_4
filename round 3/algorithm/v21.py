from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Optional, Tuple
import json
import math


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

TICKS_PER_DAY = 1_000_000
TRADING_DAYS = 252.0
TTE_START_DAYS = 5.0
SIGMA = 0.205

# Spot market making
HP_EMA_ALPHA = 0.004
VE_EMA_ALPHA = 0.10

HP_TAKE_EDGE = 4.0
HP_PASSIVE_OFFSET = 4
HP_ORDER_SIZE = 60

VE_TAKE_EDGE = 2.0
VE_PASSIVE_OFFSET = 2
VE_ORDER_SIZE = 20

HP_EOD_START = 980_000
VE_EOD_START = 985_000

# Voucher trading
OPTION_BASE_EDGE = 1.0
OPTION_DEEP_ITM_EDGE = 1.5
OPTION_FAR_OTM_EDGE = 0.5

DEEP_ITM = {"VEV_4000", "VEV_4500"}
FAR_OTM = {"VEV_6000", "VEV_6500"}


def norm_cdf(x: float) -> float:
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
    return S * norm_cdf(d1) - K * norm_cdf(d2)


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
    return norm_cdf(d1)


def best_bid(od: OrderDepth) -> Optional[int]:
    return max(od.buy_orders.keys()) if od.buy_orders else None


def best_ask(od: OrderDepth) -> Optional[int]:
    return min(od.sell_orders.keys()) if od.sell_orders else None


def micro_mid(od: OrderDepth) -> Optional[float]:
    bb = best_bid(od)
    ba = best_ask(od)
    if bb is None or ba is None:
        return None
    bid_qty = od.buy_orders.get(bb, 0)
    ask_qty = abs(od.sell_orders.get(ba, 0))
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.5 * (bb + ba)
    return (bb * ask_qty + ba * bid_qty) / total


def clamp_buy(symbol: str, position: int, qty: int) -> int:
    return max(0, min(qty, POSITION_LIMITS[symbol] - position))


def clamp_sell(symbol: str, position: int, qty: int) -> int:
    return max(0, min(qty, POSITION_LIMITS[symbol] + position))


def conviction_to_size(edge: float, threshold: float, cap: int, power: float = 1.25) -> int:
    if edge <= threshold:
        return 0
    scale = (edge / threshold) - 1.0
    size = int(round(cap * min(1.0, scale ** power)))
    return max(1, min(cap, size))


class Trader:
    def __init__(self) -> None:
        self.hp_ema: Optional[float] = None
        self.ve_ema: Optional[float] = None

    def _load(self, trader_data: str) -> None:
        if not trader_data:
            return
        try:
            state = json.loads(trader_data)
            self.hp_ema = state.get("hp_ema")
            self.ve_ema = state.get("ve_ema")
        except Exception:
            pass

    def _save(self) -> str:
        return json.dumps({
            "hp_ema": self.hp_ema,
            "ve_ema": self.ve_ema,
        })

    def _update_ema(self, old: Optional[float], new: Optional[float], alpha: float) -> Optional[float]:
        if new is None:
            return old
        if old is None:
            return new
        return alpha * new + (1.0 - alpha) * old

    def _tte_years(self, timestamp: int) -> float:
        days_left = max(TTE_START_DAYS - timestamp / TICKS_PER_DAY, 0.001)
        return days_left / TRADING_DAYS

    def _market_make_spot(
        self,
        symbol: str,
        fair: float,
        position: int,
        order_depth: OrderDepth,
        take_edge: float,
        passive_offset: int,
        base_size: int,
        eod_start: int,
        timestamp: int,
    ) -> List[Order]:
        orders: List[Order] = []
        running = position
        bb = best_bid(order_depth)
        ba = best_ask(order_depth)

        buy_cap = POSITION_LIMITS[symbol] - running
        sell_cap = POSITION_LIMITS[symbol] + running

        for ask in sorted(order_depth.sell_orders.keys()):
            if ask > fair - take_edge or buy_cap <= 0:
                break
            qty = clamp_buy(symbol, running, min(abs(order_depth.sell_orders[ask]), base_size, buy_cap))
            if qty <= 0:
                continue
            orders.append(Order(symbol, ask, qty))
            running += qty
            buy_cap -= qty
            sell_cap += qty

        for bid in sorted(order_depth.buy_orders.keys(), reverse=True):
            if bid < fair + take_edge or sell_cap <= 0:
                break
            qty = clamp_sell(symbol, running, min(order_depth.buy_orders[bid], base_size, sell_cap))
            if qty <= 0:
                continue
            orders.append(Order(symbol, bid, -qty))
            running -= qty
            sell_cap -= qty
            buy_cap += qty

        # Late session: quote one-sided to reduce inventory.
        late_mode = timestamp >= eod_start
        inv_skew = 2.0 * running / max(1, POSITION_LIMITS[symbol])
        bid_px = int(round(fair - passive_offset - inv_skew))
        ask_px = int(round(fair + passive_offset - inv_skew))

        if ba is not None:
            bid_px = min(bid_px, ba - 1)
        if bb is not None:
            ask_px = max(ask_px, bb + 1)

        if late_mode:
            if running < 0 and buy_cap > 0:
                qty = clamp_buy(symbol, running, min(base_size, abs(running), buy_cap))
                if qty > 0 and bid_px > 0:
                    orders.append(Order(symbol, bid_px, qty))
            elif running > 0 and sell_cap > 0:
                qty = clamp_sell(symbol, running, min(base_size, running, sell_cap))
                if qty > 0:
                    orders.append(Order(symbol, ask_px, -qty))
        else:
            quote_size = max(5, base_size - int(abs(running) / 10))
            if buy_cap > 0 and bid_px > 0:
                orders.append(Order(symbol, bid_px, clamp_buy(symbol, running, min(quote_size, buy_cap))))
            if sell_cap > 0:
                orders.append(Order(symbol, ask_px, -clamp_sell(symbol, running, min(quote_size, sell_cap))))

        return [o for o in orders if o.quantity != 0]

    def _option_orders(
        self,
        symbol: str,
        strike: int,
        spot_fair: float,
        timestamp: int,
        position: int,
        order_depth: OrderDepth,
    ) -> List[Order]:
        orders: List[Order] = []
        bb = best_bid(order_depth)
        ba = best_ask(order_depth)
        if bb is None and ba is None:
            return orders

        T = self._tte_years(timestamp)
        fair = bs_call(spot_fair, float(strike), T, SIGMA)
        delta = bs_delta(spot_fair, float(strike), T, SIGMA)
        intrinsic = max(spot_fair - strike, 0.0)

        buy_cap = POSITION_LIMITS[symbol] - position
        sell_cap = POSITION_LIMITS[symbol] + position

        if symbol in DEEP_ITM:
            threshold = OPTION_DEEP_ITM_EDGE
            size_cap = 8
        elif symbol in FAR_OTM:
            threshold = OPTION_FAR_OTM_EDGE
            size_cap = 20
        else:
            threshold = OPTION_BASE_EDGE
            size_cap = 14

        # For very far OTM names, avoid buying lottery tickets.
        allow_buy = symbol not in FAR_OTM

        if ba is not None and allow_buy and buy_cap > 0:
            buy_edge = fair - ba
            # Favor names where time value is real and delta is not negligible.
            if symbol not in DEEP_ITM and symbol not in FAR_OTM and delta < 0.05:
                buy_edge = -1.0
            qty = conviction_to_size(buy_edge, threshold, size_cap)
            qty = clamp_buy(symbol, position, min(qty, buy_cap))
            if qty > 0:
                orders.append(Order(symbol, ba, qty))
                position += qty
                buy_cap -= qty
                sell_cap += qty

        if bb is not None and sell_cap > 0:
            sell_edge = bb - fair
            if symbol in FAR_OTM:
                # Willingly sell a little premium if the market pays at least 1.
                sell_edge = max(sell_edge, bb - intrinsic)
            qty = conviction_to_size(sell_edge, threshold, size_cap)
            qty = clamp_sell(symbol, position, min(qty, sell_cap))
            if qty > 0:
                orders.append(Order(symbol, bb, -qty))
                position -= qty
                sell_cap -= qty
                buy_cap += qty

        # Passive quotes only when the spread exists and we still have room.
        if bb is not None and ba is not None and ba - bb >= 2:
            theo_bid = int(math.floor(fair - 0.5))
            theo_ask = int(math.ceil(fair + 0.5))
            post_bid = min(theo_bid, ba - 1)
            post_ask = max(theo_ask, bb + 1)

            if allow_buy and buy_cap > 0 and post_bid > 0 and fair - post_bid >= threshold:
                qty = clamp_buy(symbol, position, min(max(1, size_cap // 2), buy_cap))
                if qty > 0:
                    orders.append(Order(symbol, post_bid, qty))
            if sell_cap > 0 and post_ask > 0 and post_ask - fair >= threshold:
                qty = clamp_sell(symbol, position, min(max(1, size_cap // 2), sell_cap))
                if qty > 0:
                    orders.append(Order(symbol, post_ask, -qty))

        return [o for o in orders if o.quantity != 0]

    def run(self, state: TradingState):
        self._load(state.traderData)
        result: Dict[str, List[Order]] = {p: [] for p in PRODUCTS}

        hp_od = state.order_depths.get("HYDROGEL_PACK", OrderDepth())
        ve_od = state.order_depths.get("VELVETFRUIT_EXTRACT", OrderDepth())

        hp_mid = micro_mid(hp_od)
        ve_mid = micro_mid(ve_od)

        self.hp_ema = self._update_ema(self.hp_ema, hp_mid, HP_EMA_ALPHA)
        self.ve_ema = self._update_ema(self.ve_ema, ve_mid, VE_EMA_ALPHA)

        hp_fair = self.hp_ema if self.hp_ema is not None else (hp_mid if hp_mid is not None else 10_000.0)
        ve_fair = self.ve_ema if self.ve_ema is not None else (ve_mid if ve_mid is not None else 5_250.0)

        hp_pos = state.position.get("HYDROGEL_PACK", 0)
        result["HYDROGEL_PACK"] = self._market_make_spot(
            "HYDROGEL_PACK",
            hp_fair,
            hp_pos,
            hp_od,
            HP_TAKE_EDGE,
            HP_PASSIVE_OFFSET,
            HP_ORDER_SIZE,
            HP_EOD_START,
            state.timestamp,
        )

        ve_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        result["VELVETFRUIT_EXTRACT"] = self._market_make_spot(
            "VELVETFRUIT_EXTRACT",
            ve_fair,
            ve_pos,
            ve_od,
            VE_TAKE_EDGE,
            VE_PASSIVE_OFFSET,
            VE_ORDER_SIZE,
            VE_EOD_START,
            state.timestamp,
        )

        for symbol, strike in STRIKES.items():
            od = state.order_depths.get(symbol)
            if od is None:
                continue
            pos = state.position.get(symbol, 0)
            result[symbol] = self._option_orders(symbol, strike, ve_fair, state.timestamp, pos, od)

        return result, 0, self._save()