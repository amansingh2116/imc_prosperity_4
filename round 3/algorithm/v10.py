"""
IMC Prosperity Round 3 - Trader v10

What changed from v8/v9:
1. HYDROGEL_PACK stays close to the profitable v8 logic, but now has
   a late-session unwind to reduce the end-of-day mark-to-market crash.
2. VEV_5400 is priced from a live IV smile fit using anchor strikes
   5000/5100/5200/5300 instead of a stale flat prior.
3. VEV_5400 execution is conservative:
   - very large edge: cross the ask
   - otherwise: post inside the spread like v9, but with better pricing
4. Far OTM premium selling was removed because historical market trades
   print at the bid, so our ask-side orders never get hit in the official logs.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json
import math


LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

ALL_PRODUCTS = [
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


# HP parameters
HP_EMA_ALPHA = 0.004
HP_TAKE_EDGE = 4
HP_MAKE_OFFSET = 4
HP_SIZE = 70
HP_EOD_START = 96_000
HP_EOD_END = 100_000
HP_EOD_CLEAR_SIZE = 8


# Option pricing
TRADING_DAYS = 252
TTE_START = 5.0
TICKS_PER_DAY = 1_000_000
SIGMA_FALLBACK = 0.215
SMILE_ANCHORS = [5000, 5100, 5200, 5300]


# 5400 outlier trade
OPT_BUY_STRIKE = 5400
OPT_BUY_MAX_POS = 100
OPT_TAKE_EDGE = 2.0
OPT_POST_EDGE = 1.0
OPT_EXIT_EDGE = 1.5
OPT_EOD_START = 96_000
OPT_EOD_CLEAR_SIZE = 8


# Deep ITM wide-spread MM
DEEP_ITM = [4000, 4500]
ITM_MAKE_OFFSET = 2
ITM_SIZE = 5
ITM_MAX_INV = 20


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 1e-8:
        return max(S - K, 0.0)
    vol = sigma * math.sqrt(T)
    if vol <= 1e-12:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol
    d2 = d1 - vol
    return S * _ncdf(d1) - K * _ncdf(d2)


def _implied_vol_bisect(mkt: float, S: float, K: float, T: float) -> float:
    intrinsic = max(S - K, 0.0)
    if T <= 1e-8 or mkt <= intrinsic + 0.01:
        return SIGMA_FALLBACK

    lo, hi = 0.05, 1.5
    try:
        if bs_call(S, K, T, lo) >= mkt or bs_call(S, K, T, hi) <= mkt:
            return SIGMA_FALLBACK
    except Exception:
        return SIGMA_FALLBACK

    for _ in range(50):
        mid = (lo + hi) / 2.0
        price = bs_call(S, K, T, mid)
        if abs(price - mkt) < 0.005:
            return mid
        if price < mkt:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def tte_years(ts: int) -> float:
    days = max(TTE_START - ts / TICKS_PER_DAY, 0.001)
    return days / TRADING_DAYS


def best_bid_ask(od: OrderDepth) -> Tuple[int | None, int | None]:
    bb = max(od.buy_orders.keys()) if od.buy_orders else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    return bb, ba


def book_mid(od: OrderDepth) -> float:
    bid_notional = 0.0
    bid_volume = 0.0
    ask_notional = 0.0
    ask_volume = 0.0

    for price, volume in od.buy_orders.items():
        if volume > 0:
            bid_notional += price * volume
            bid_volume += volume

    for price, volume in od.sell_orders.items():
        qty = abs(volume)
        ask_notional += price * qty
        ask_volume += qty

    if bid_volume > 0 and ask_volume > 0:
        return 0.5 * (bid_notional / bid_volume + ask_notional / ask_volume)

    bb, ba = best_bid_ask(od)
    if bb is not None and ba is not None:
        return 0.5 * (bb + ba)
    return float(bb or ba or 0.0)


def clamp_buy(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] - pos))


def clamp_sell(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] + pos))


def _linear_predict(xs: List[float], ys: List[float], xq: float, fallback: float) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return fallback

    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    denom = sum((x - x_bar) * (x - x_bar) for x in xs)
    if denom <= 1e-12:
        return y_bar

    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom
    pred = y_bar + slope * (xq - x_bar)
    low = min(ys) - 0.03
    high = max(ys) + 0.03
    return max(low, min(high, pred))


def _tiered_size(edge: float, low: int, mid: int, high: int) -> int:
    if edge >= 3.0:
        return high
    if edge >= 2.0:
        return mid
    if edge >= 1.0:
        return low
    return 0


class Trader:
    def _load(self, td: str) -> Dict:
        if td:
            try:
                data = json.loads(td)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def _save(self, data: Dict) -> str:
        try:
            return json.dumps(data)
        except Exception:
            return "{}"

    def _fit_smile_iv(self, state: TradingState, spot: float, T: float, target_strike: int, fallback: float) -> Tuple[float, float]:
        xs: List[float] = []
        ys: List[float] = []
        for strike in SMILE_ANCHORS:
            sym = f"VEV_{strike}"
            od = state.order_depths.get(sym)
            if od is None:
                continue
            mid = book_mid(od)
            if mid <= 0.5 or spot <= 0:
                continue
            iv = _implied_vol_bisect(mid, spot, float(strike), T)
            if 0.05 < iv < 1.0:
                xs.append(math.log(strike / spot))
                ys.append(iv)

        if len(xs) < 2:
            return fallback, fallback

        target_x = math.log(target_strike / spot)
        pred = _linear_predict(xs, ys, target_x, fallback)
        base = sum(ys) / len(ys)
        return pred, base

    def _trade_hp(self, od: OrderDepth, pos: int, fair: float, ts: int) -> List[Order]:
        orders: List[Order] = []
        sym = "HYDROGEL_PACK"
        bb, ba = best_bid_ask(od)
        running = pos
        late_mode = HP_EOD_START <= ts < HP_EOD_END

        # Aggressive taker
        for ask, vol in sorted(od.sell_orders.items()):
            if ask >= fair - HP_TAKE_EDGE:
                break
            max_qty = min(HP_SIZE, abs(vol))
            if late_mode:
                if running >= 0:
                    break
                max_qty = min(max_qty, abs(running))
            q = clamp_buy(sym, running, max_qty)
            if q <= 0:
                break
            orders.append(Order(sym, int(ask), q))
            running += q

        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid <= fair + HP_TAKE_EDGE:
                break
            max_qty = min(HP_SIZE, abs(vol))
            if late_mode:
                if running <= 0:
                    break
                max_qty = min(max_qty, running)
            q = clamp_sell(sym, running, max_qty)
            if q <= 0:
                break
            orders.append(Order(sym, int(bid), -q))
            running -= q

        # Late-session inventory reduction: cross a little, then quote one-sided.
        if late_mode:
            if running < 0 and ba is not None:
                q = clamp_buy(sym, running, min(HP_EOD_CLEAR_SIZE, abs(running)))
                if q > 0:
                    orders.append(Order(sym, int(ba), q))
                    running += q
            elif running > 0 and bb is not None:
                q = clamp_sell(sym, running, min(HP_EOD_CLEAR_SIZE, running))
                if q > 0:
                    orders.append(Order(sym, int(bb), -q))
                    running -= q

        # Passive market maker
        lim = LIMITS[sym]
        skew = int(round(running / lim * HP_MAKE_OFFSET))
        buy_px = int(round(fair)) - HP_MAKE_OFFSET - skew
        sell_px = int(round(fair)) + HP_MAKE_OFFSET - skew
        if ba is not None:
            buy_px = min(buy_px, int(ba) - 1)
        if bb is not None:
            sell_px = max(sell_px, int(bb) + 1)

        if late_mode:
            make_size = min(HP_SIZE, max(0, abs(running)))
            if running < 0 and make_size > 0:
                q = clamp_buy(sym, running, make_size)
                if q > 0:
                    orders.append(Order(sym, buy_px, q))
            elif running > 0 and make_size > 0:
                q = clamp_sell(sym, running, make_size)
                if q > 0:
                    orders.append(Order(sym, sell_px, -q))
        else:
            bsz = clamp_buy(sym, running, HP_SIZE)
            ssz = clamp_sell(sym, running, HP_SIZE)
            if bsz > 0:
                orders.append(Order(sym, buy_px, bsz))
            if ssz > 0:
                orders.append(Order(sym, sell_px, -ssz))

        return orders

    def _trade_5400(self, od: OrderDepth, pos: int, fair: float, ts: int) -> List[Order]:
        sym = f"VEV_{OPT_BUY_STRIKE}"
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return []

        orders: List[Order] = []
        running = pos
        if ts >= OPT_EOD_START:
            frac = (HP_EOD_END - ts) / float(max(1, HP_EOD_END - OPT_EOD_START))
            target_pos = int(round(OPT_BUY_MAX_POS * max(0.0, frac)))
        else:
            target_pos = OPT_BUY_MAX_POS

        if running > target_pos:
            qty = min(running - target_pos, OPT_EOD_CLEAR_SIZE + max(0, (running - target_pos) // 8))
            qty = clamp_sell(sym, running, qty)
            if qty > 0:
                orders.append(Order(sym, int(bb), -qty))
                running -= qty

        # Exit when the market bid is above our model or when we are too late in the day.
        exit_edge = bb - fair
        if running > 0 and exit_edge >= OPT_EXIT_EDGE:
            qty = _tiered_size(exit_edge + 0.5, 4, 6, 8)
            qty = min(qty, running)
            qty = clamp_sell(sym, running, qty)
            if qty > 0:
                orders.append(Order(sym, int(bb), -qty))
                running -= qty
                return orders

        did_take = False

        # Strong signal: cross the ask.
        take_edge = fair - ba
        if running < target_pos and take_edge >= OPT_TAKE_EDGE:
            qty = _tiered_size(take_edge, 4, 6, 8)
            qty = min(qty, target_pos - running)
            qty = clamp_buy(sym, running, qty)
            if qty > 0:
                orders.append(Order(sym, int(ba), qty))
                running += qty
                did_take = True

        # Milder signal: rest just inside the bid.
        if (not did_take) and running < target_pos:
            post_px = int(bb) + 1 if int(bb) + 1 < int(ba) else int(bb)
            post_edge = fair - post_px
            if post_edge >= OPT_POST_EDGE:
                qty = _tiered_size(post_edge, 2, 4, 6)
                qty = min(qty, target_pos - running)
                qty = clamp_buy(sym, running, qty)
                if qty > 0:
                    orders.append(Order(sym, post_px, qty))

        return orders

    def _trade_deep_itm(self, sym: str, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        orders: List[Order] = []
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return orders

        if pos < ITM_MAX_INV:
            buy_px = int(bb) + ITM_MAKE_OFFSET
            if buy_px < fair:
                qty = clamp_buy(sym, pos, min(ITM_SIZE, ITM_MAX_INV - max(pos, 0)))
                if qty > 0:
                    orders.append(Order(sym, buy_px, qty))

        if pos > -ITM_MAX_INV:
            sell_px = int(ba) - ITM_MAKE_OFFSET
            if sell_px > fair:
                qty = clamp_sell(sym, pos, min(ITM_SIZE, ITM_MAX_INV - max(-pos, 0)))
                if qty > 0:
                    orders.append(Order(sym, sell_px, -qty))

        return orders

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        ema_dict = data.get("ema", {})
        smile_iv = float(data.get("smile_iv", SIGMA_FALLBACK))
        result: Dict[str, List[Order]] = {sym: [] for sym in ALL_PRODUCTS}

        ts = state.timestamp
        T = tte_years(ts)

        # Update HP EMA.
        hp_od = state.order_depths.get("HYDROGEL_PACK")
        if hp_od is not None:
            hp_mid = book_mid(hp_od)
            if hp_mid > 0:
                prev = float(ema_dict.get("HYDROGEL_PACK", hp_mid))
                ema_dict["HYDROGEL_PACK"] = HP_EMA_ALPHA * hp_mid + (1.0 - HP_EMA_ALPHA) * prev

        hp_fair = ema_dict.get("HYDROGEL_PACK")
        if hp_od is not None and hp_fair:
            hp_pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._trade_hp(hp_od, hp_pos, float(hp_fair), ts)

        # No direct VEV trading.
        result["VELVETFRUIT_EXTRACT"] = []

        # Smile-based option pricing.
        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_mid = book_mid(vev_od) if vev_od is not None else 0.0
        if vev_mid > 0:
            sigma_5400, sigma_base = self._fit_smile_iv(state, vev_mid, T, OPT_BUY_STRIKE, smile_iv)
            smile_iv = sigma_5400

            od_5400 = state.order_depths.get("VEV_5400")
            if od_5400 is not None:
                pos_5400 = state.position.get("VEV_5400", 0)
                fair_5400 = bs_call(vev_mid, 5400.0, T, sigma_5400)
                result["VEV_5400"] = self._trade_5400(od_5400, pos_5400, fair_5400, ts)

            for strike in DEEP_ITM:
                sym = f"VEV_{strike}"
                od = state.order_depths.get(sym)
                if od is None:
                    continue
                pos = state.position.get(sym, 0)
                fair = bs_call(vev_mid, float(strike), T, sigma_base)
                result[sym] = self._trade_deep_itm(sym, od, pos, fair)

        new_data = {
            "ema": ema_dict,
            "smile_iv": smile_iv,
        }
        return result, 0, self._save(new_data)
