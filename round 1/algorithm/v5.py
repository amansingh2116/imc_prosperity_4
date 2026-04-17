import json
from typing import Dict, List, Optional, Tuple
from datamodel import OrderDepth, TradingState, Order

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

POSITION_LIMITS: Dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 20,
    "ASH_COATED_OSMIUM": 10,
}

# PEPPER: buy early, hold long, only reduce on strong reversal
PEPPER_CFG = {
    "fast_alpha": 0.30,
    "slow_alpha": 0.06,
    "target_long": 20,
    "build_window": 120,          # early phase: accumulate faster
    "aggr_buy_levels": 5,
    "max_buy_per_tick": 20,
    "reversal_threshold": -2.0,   # only reduce if trend becomes clearly negative
    "reduce_size": 4,
}

# OSMIUM: passive market making
OSMIUM_CFG = {
    "ema_alpha": 0.05,
    "base_edge": 6,
    "vol_factor": 0.35,
    "quote_size": 4,
    "skew_factor": 0.80,
    "max_pos_for_bid": 8,
    "max_pos_for_ask": -8,
    "soft_reduce_start": 6,
}

# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    bb = max(od.buy_orders) if od.buy_orders else None
    ba = min(od.sell_orders) if od.sell_orders else None
    return bb, ba

def arith_mid(bb: Optional[int], ba: Optional[int]) -> Optional[float]:
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0

def safe_buy_qty(desired: int, pos: int, limit: int) -> int:
    return max(0, min(desired, limit - pos))

def safe_sell_qty(desired: int, pos: int, limit: int) -> int:
    return max(0, min(desired, limit + pos))

# ══════════════════════════════════════════════════════════════════════════════
# PEPPER: BUY EARLY, HOLD LONG, DO NOT SELL EARLY
# ══════════════════════════════════════════════════════════════════════════════

def trade_pepper(
    od: OrderDepth,
    pos: int,
    limit: int,
    cfg: dict,
    saved: dict,
    timestamp: int
) -> Tuple[List[Order], dict]:
    orders: List[Order] = []

    bb, ba = best_bid_ask(od)
    if bb is None or ba is None:
        return orders, saved

    mid = arith_mid(bb, ba)
    if mid is None:
        return orders, saved

    if "pepper_fast" not in saved:
        saved["pepper_fast"] = mid
        saved["pepper_slow"] = mid
        saved["pepper_prev_fast"] = mid

    prev_fast = saved["pepper_prev_fast"]
    fast = saved["pepper_fast"]
    slow = saved["pepper_slow"]

    fast = cfg["fast_alpha"] * mid + (1.0 - cfg["fast_alpha"]) * fast
    slow = cfg["slow_alpha"] * mid + (1.0 - cfg["slow_alpha"]) * slow

    saved["pepper_prev_fast"] = saved["pepper_fast"]
    saved["pepper_fast"] = fast
    saved["pepper_slow"] = slow

    trend = fast - slow
    slope = fast - prev_fast

    # Early phase: build position fast and keep it.
    if timestamp < cfg["build_window"]:
        target = cfg["target_long"]
    else:
        # After early phase, if trend is still positive, stay long.
        # If trend weakens, still keep a large long bias.
        if trend > 2.0:
            target = cfg["target_long"]
        elif trend > 0.5:
            target = 16
        elif trend > 0.0:
            target = 12
        else:
            target = 8

    # Buy up to target, but do not churn inventory.
    if pos < target:
        qty_needed = min(target - pos, cfg["max_buy_per_tick"])

        # Aggressively lift asks first
        levels_swept = 0
        for ask_px in sorted(od.sell_orders.keys()):
            if levels_swept >= cfg["aggr_buy_levels"]:
                break
            if qty_needed <= 0:
                break

            available = abs(od.sell_orders[ask_px])
            qty = safe_buy_qty(min(available, qty_needed), pos, limit)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", ask_px, qty))
                pos += qty
                qty_needed -= qty
                levels_swept += 1

        # If still not enough, place a small passive bid just inside the best bid.
        if qty_needed > 0 and bb is not None:
            qty = safe_buy_qty(qty_needed, pos, limit)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", bb + 1, qty))

    # Only reduce on a clear reversal. Do not sell on small rises.
    if pos > 0 and trend <= cfg["reversal_threshold"] and slope < 0:
        sell_qty = min(cfg["reduce_size"], pos)
        if sell_qty > 0:
            orders.append(Order("INTARIAN_PEPPER_ROOT", bb, -sell_qty))

    return orders, saved

# ══════════════════════════════════════════════════════════════════════════════
# OSMIUM: PASSIVE MARKET MAKING
# ══════════════════════════════════════════════════════════════════════════════

def trade_osmium(
    od: OrderDepth,
    pos: int,
    limit: int,
    cfg: dict,
    saved: dict
) -> Tuple[List[Order], dict]:
    orders: List[Order] = []

    bb, ba = best_bid_ask(od)
    if bb is None or ba is None:
        return orders, saved

    mid = arith_mid(bb, ba)
    if mid is None:
        return orders, saved

    if "osmium_ema" not in saved:
        saved["osmium_ema"] = mid
        saved["osmium_hist"] = []

    ema = saved["osmium_ema"]
    ema = cfg["ema_alpha"] * mid + (1.0 - cfg["ema_alpha"]) * ema
    saved["osmium_ema"] = ema

    hist = saved["osmium_hist"]
    hist.append(mid)
    if len(hist) > 25:
        hist.pop(0)
    saved["osmium_hist"] = hist

    if len(hist) >= 6:
        diffs = [hist[i] - hist[i - 1] for i in range(1, len(hist))]
        mean_diff = sum(diffs) / len(diffs)
        var_diff = sum((x - mean_diff) ** 2 for x in diffs) / len(diffs)
        vol = var_diff ** 0.5
    else:
        vol = 1.5

    edge = round(cfg["base_edge"] + cfg["vol_factor"] * vol)
    edge = max(5, min(8, edge))

    skew = pos * cfg["skew_factor"]

    bid_px = round(ema - edge - skew)
    ask_px = round(ema + edge - skew)

    if bid_px >= ask_px:
        center = (bid_px + ask_px) // 2
        bid_px = center - 1
        ask_px = center + 1

    abs_pos = abs(pos)
    if abs_pos <= cfg["soft_reduce_start"]:
        size = cfg["quote_size"]
    else:
        frac = (limit - abs_pos) / max(limit - cfg["soft_reduce_start"], 1)
        size = max(1, round(cfg["quote_size"] * frac))

    if pos < cfg["max_pos_for_bid"]:
        buy_qty = safe_buy_qty(size, pos, limit)
        if buy_qty > 0 and bid_px < ba:
            orders.append(Order("ASH_COATED_OSMIUM", bid_px, buy_qty))

    if pos > cfg["max_pos_for_ask"]:
        sell_qty = safe_sell_qty(size, pos, limit)
        if sell_qty > 0 and ask_px > bb:
            orders.append(Order("ASH_COATED_OSMIUM", ask_px, -sell_qty))

    return orders, saved

# ══════════════════════════════════════════════════════════════════════════════
# TRADER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class Trader:
    def run(self, state: TradingState):
        try:
            saved: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if product not in POSITION_LIMITS:
                result[product] = []
                continue

            limit = POSITION_LIMITS[product]
            pos = state.position.get(product, 0)

            if product == "INTARIAN_PEPPER_ROOT":
                orders, saved = trade_pepper(
                    od, pos, limit, PEPPER_CFG, saved, state.timestamp
                )
            elif product == "ASH_COATED_OSMIUM":
                orders, saved = trade_osmium(od, pos, limit, OSMIUM_CFG, saved)
            else:
                orders = []

            result[product] = orders

        traderData = json.dumps(saved)
        return result, 0, traderData