from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import math

POSITION_LIMIT = 10
MAX_LOT = 3

class Trader:
    def __init__(self):
        # Dynamic tracking dictionaries
        self.ema_prices: Dict[str, float] = {}

    def get_order_book_imbalance(self, od: OrderDepth) -> float:
        """Calculates normalized pressure on the order book."""
        bid_vol = sum(od.buy_orders.values()) if od.buy_orders else 0
        ask_vol = sum(abs(v) for v in od.sell_orders.values()) if od.sell_orders else 0
        
        total_vol = bid_vol + ask_vol
        if total_vol == 0: return 0.0
        
        # Returns a value between -1.0 (heavy sell pressure) and 1.0 (heavy buy pressure)
        return (bid_vol - ask_vol) / total_vol

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            orders: List[Order] = []
            
            if not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid

            # 1. Dynamic EMA Tracking (No hardcoded starting values)
            if product not in self.ema_prices:
                self.ema_prices[product] = mid_price
            else:
                # Alpha of 0.1 gives a smooth tracking of recent ticks
                self.ema_prices[product] = 0.1 * mid_price + 0.9 * self.ema_prices[product]

            pos = state.position.get(product, 0)
            
            # 2. Dynamic Order Book Imbalance (OBI)
            # Replaces the hardcoded "EARLY_SKEW" lists by letting the book dictate the pressure.
            obi = self.get_order_book_imbalance(od)
            
            # Calculate Fair Value: EMA + Spread adjustment based on OBI
            fair_value = self.ema_prices[product] + (obi * spread * 0.2)

            # 3. Dynamic Inventory Skew (Replaces hardcoded panic unwinds)
            # The closer we get to the limit, the more aggressively we shift our quotes to get flat.
            inventory_skew = (pos / POSITION_LIMIT) * (spread * 0.4)
            
            # 4. Adaptive Quoting Base
            quote_bid = math.floor(fair_value - (spread * 0.3) - inventory_skew)
            quote_ask = math.ceil(fair_value + (spread * 0.3) - inventory_skew)

            # Ensure we don't cross the book accidentally unless intentionally taking edge
            quote_bid = min(quote_bid, best_ask - 1)
            quote_ask = max(quote_ask, best_bid + 1)

            buy_room = POSITION_LIMIT - pos
            sell_room = POSITION_LIMIT + pos

            # 5. Smart Passive Execution
            if buy_room > 0:
                orders.append(Order(product, int(quote_bid), min(MAX_LOT, buy_room)))

            if sell_room > 0:
                orders.append(Order(product, int(quote_ask), -min(MAX_LOT, sell_room)))

            if orders:
                result[product] = orders

        return result, 0, ""