import json
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION v6
# ══════════════════════════════════════════════════════════════════════════════

POSITION_LIMITS: Dict[str, int] = {
    "INTARIAN_PEPPER_ROOT": 80,
    "ASH_COATED_OSMIUM": 80,
}

class Trader:
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
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
            orders: List[Order] = []

            bb = max(od.buy_orders.keys()) if od.buy_orders else None
            ba = min(od.sell_orders.keys()) if od.sell_orders else None
            
            if bb is None or ba is None:
                result[product] = orders
                continue

            mid = (bb + ba) / 2.0

            # ══════════════════════════════════════════════════════════════════
            # PEPPER: BUY-AND-HOLD (SWEEP TO LIMIT)
            # ══════════════════════════════════════════════════════════════════
            if product == "INTARIAN_PEPPER_ROOT":
                if "pepper_intercept" not in saved:
                    saved["pepper_intercept"] = mid
                
                # Formula: Intercept + 0.001 * timestamp
                fair_value = saved["pepper_intercept"] + (0.001 * state.timestamp)
                
                qty_needed = limit - pos
                if qty_needed > 0:
                    # Sweep asks <= fair + 20
                    for ask_px in sorted(od.sell_orders.keys()):
                        if qty_needed <= 0:
                            break
                        if ask_px <= fair_value + 20:
                            available = abs(od.sell_orders[ask_px])
                            buy_qty = min(available, qty_needed)
                            orders.append(Order(product, ask_px, buy_qty))
                            pos += buy_qty
                            qty_needed -= buy_qty
                    
                    # Park remaining order safely at the best bid to soak up fills
                    if qty_needed > 0:
                        orders.append(Order(product, bb, qty_needed))

            # ══════════════════════════════════════════════════════════════════
            # OSMIUM: TAKE AND MAKE MEAN-REVERSION
            # ══════════════════════════════════════════════════════════════════
            elif product == "ASH_COATED_OSMIUM":
                if "osmium_ema" not in saved:
                    saved["osmium_ema"] = mid
                
                # Update EMA track
                alpha = 0.10
                ema = alpha * mid + (1.0 - alpha) * saved["osmium_ema"]
                saved["osmium_ema"] = ema

                # --- PHASE 1: TAKE ---
                # Grab any ask < EMA (Guaranteed +EV)
                for ask_px in sorted(od.sell_orders.keys()):
                    if ask_px < ema and pos < limit:
                        available = abs(od.sell_orders[ask_px])
                        buy_qty = min(available, limit - pos)
                        if buy_qty > 0:
                            orders.append(Order(product, ask_px, buy_qty))
                            pos += buy_qty

                # Grab any bid > EMA (Guaranteed +EV)
                for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                    if bid_px > ema and pos > -limit:
                        available = od.buy_orders[bid_px]
                        sell_qty = min(available, limit + pos)
                        if sell_qty > 0:
                            orders.append(Order(product, bid_px, -sell_qty))
                            pos -= sell_qty

                # --- PHASE 2: MAKE ---
                # Calculate Inventory Skew (-1.0 to 1.0)
                skew_ratio = pos / limit
                skew_ticks = int(skew_ratio * 3) # Shift up to 3 ticks based on inventory
                
                # Post 1 tick inside the book, adjusting for skew
                my_bid = min(bb + 1, ba - 2) - skew_ticks
                my_ask = max(ba - 1, bb + 2) - skew_ticks
                
                # Safety check: avoid crossing own orders
                if my_bid >= my_ask:
                    center = (my_bid + my_ask) // 2
                    my_bid = center - 1
                    my_ask = center + 1

                # Taper size if position > 60
                abs_pos = abs(pos)
                base_quote_size = 12
                
                if abs_pos > 60:
                    taper_frac = (limit - abs_pos) / 20.0
                    quote_size = max(1, int(base_quote_size * taper_frac))
                else:
                    quote_size = base_quote_size

                # Submit MAKE orders
                if pos < limit:
                    buy_qty = min(quote_size, limit - pos)
                    if buy_qty > 0:
                        orders.append(Order(product, my_bid, buy_qty))
                        
                if pos > -limit:
                    sell_qty = min(quote_size, limit + pos)
                    if sell_qty > 0:
                        orders.append(Order(product, my_ask, -sell_qty))

            result[product] = orders

        return result, 0, json.dumps(saved)