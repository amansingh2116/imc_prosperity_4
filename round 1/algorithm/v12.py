"""
IMC Prosperity Round 1 — trader_v12.py
=======================================
Key data findings (from 3 days of prices_round_1 data):

INTARIAN_PEPPER_ROOT:
  - Perfect linear trend: FV(t) = intercept + 0.001 * timestamp
    intercept = ~10000 (day -2), ~11000 (day -1), ~12000 (day 0)
  - Residual std ≈ 2 units — extremely tight, highly predictable
  - Spread ≈ 14 units; bot asks cluster at FV + 6–8
  Strategy → BUY to max position ASAP, HOLD all day.
             80 units × 1000 rise/day ≈ 80 000 XIRECs/day.

ASH_COATED_OSMIUM:
  - Mean-reverts to 10 000 with std ≈ 5 units
  - Bot bids: 9990–9998; bot asks: 10 005–10 014
  Strategy → Market-make inside the bot spread + take aggressive
             mispricings. ACO supplements IPR.

Expected PnL (full 10 000-iteration simulation, all 3 days):
  IPR trend  ≈ 240 000 XIRECs
  ACO spread ≈  10 000–15 000 XIRECs
  Total      ≈ 250 000+ XIRECs (target is 200 000)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"
IPR_SLOPE = 0.001   # Measured precisely across all 3 sample days
IPR_LIMIT = 80
ACO_LIMIT = 80
ACO_FV    = 10_000


class Trader:

    # ------------------------------------------------------------------ #
    #  Bid method (required for Round 2 — ignored in Round 1)             #
    # ------------------------------------------------------------------ #
    def bid(self):
        return 15

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def run(self, state: TradingState):
        # Deserialise persistent state (survives between iterations)
        td: Dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}

        t = state.timestamp
        result = {}

        for product, order_depth in state.order_depths.items():
            pos = state.position.get(product, 0)

            if product == IPR:
                result[product] = self._trade_ipr(order_depth, pos, t, td)
            elif product == ACO:
                result[product] = self._trade_aco(order_depth, pos, t, td)
            else:
                result[product] = []

        return result, 0, json.dumps(td)

    # ------------------------------------------------------------------ #
    #  IPR — buy-and-hold trend follower                                  #
    # ------------------------------------------------------------------ #
    def _trade_ipr(
        self,
        od: OrderDepth,
        pos: int,
        t: int,
        td: dict,
    ) -> List[Order]:
        """
        Goal: hold +80 (maximum long) throughout the day.

        Fair value rises at exactly 0.001 XIRECs per timestamp.
        We calibrate the intercept once per day (at t == 0 or on first call).
        Every ask at or below FV + 8 is a good buy; we also post a passive
        bid to top-up the position whenever we're below the limit.
        We never sell unless a bot is paying significantly above FV.
        """
        orders: List[Order] = []
        asks = sorted(od.sell_orders.items())        # ascending price
        bids = sorted(od.buy_orders.items(), reverse=True)   # descending price

        if not asks and not bids:
            return orders

        # --- Calibrate intercept (once per day, reset on t == 0) ---
        if t == 0 or "ipr0" not in td:
            if asks and bids:
                mid = (asks[0][0] + bids[0][0]) / 2
            elif asks:
                mid = asks[0][0]
            else:
                mid = bids[0][0]
            # intercept = current_mid - slope * t
            # (at t=0 this is just mid; at a later t it back-adjusts)
            td["ipr0"] = mid - IPR_SLOPE * t

        fv = td["ipr0"] + IPR_SLOPE * t

        buy_cap  = IPR_LIMIT - pos   # how many more units we can buy
        sell_cap = IPR_LIMIT + pos   # how many units we can sell (if pos > 0)

        # ---- 1. Aggressive taking: buy all asks ≤ FV + 8 ----
        # Buffer of 8 ticks ≈ half the typical bot spread.
        # Even buying at FV + 8 is recovered in 8 / 0.001 = 8 000 timestamps.
        if buy_cap > 0 and asks:
            BUY_BUFFER = 8
            remaining = buy_cap
            for ask_price, ask_vol in asks:
                if remaining <= 0:
                    break
                if ask_price <= fv + BUY_BUFFER:
                    qty = min(abs(ask_vol), remaining)
                    orders.append(Order(IPR, ask_price, qty))
                    remaining -= qty
                else:
                    break   # asks are sorted ascending; no point going further

            # ---- 2. Passive bid to top-up position ----
            # Post just above the best bot bid to attract bot sellers.
            if remaining > 0 and bids:
                best_bot_bid = bids[0][0]
                # Stay at most 5 ticks below FV so we only pay a fair price.
                our_bid = min(best_bot_bid + 1, int(fv) - 1)
                # Sanity: don't cross asks we just decided not to take.
                if our_bid < (asks[0][0] if asks else our_bid + 1):
                    orders.append(Order(IPR, our_bid, remaining))

        # ---- 3. Opportunistic sell: only if bot pays well above FV ----
        # This is rare (residual std ≈ 2) but free money if it happens.
        SELL_PREMIUM = 12   # > half typical spread → genuinely above FV
        if sell_cap > 0 and pos > 0 and bids:
            best_bot_bid = bids[0][0]
            if best_bot_bid > fv + SELL_PREMIUM:
                qty = min(sell_cap, abs(bids[0][1]))
                orders.append(Order(IPR, best_bot_bid, -qty))

        return orders

    # ------------------------------------------------------------------ #
    #  ACO — market-maker around 10 000                                   #
    # ------------------------------------------------------------------ #
    def _trade_aco(
        self,
        od: OrderDepth,
        pos: int,
        t: int,
        td: dict,
    ) -> List[Order]:
        """
        ACO mean-reverts tightly around 10 000 (std ≈ 5, spread ≈ 16).

        Two layers:
        1. Aggressive: take any obvious mispricing immediately.
           - Buy if best ask < FV - 1 (bot selling cheap)
           - Sell if best bid > FV + 1 (bot buying expensive)
        2. Passive: post quotes inside the bot spread with inventory skew.
           - Our bid > best bot bid, our ask < best bot ask
           - Skew: if we're long, lower both quotes to encourage selling;
             if short, raise both to encourage buying.
        """
        orders: List[Order] = []
        asks = sorted(od.sell_orders.items())
        bids = sorted(od.buy_orders.items(), reverse=True)

        if not asks or not bids:
            return orders

        best_bot_ask = asks[0][0]
        best_bot_bid = bids[0][0]

        buy_cap  = ACO_LIMIT - pos
        sell_cap = ACO_LIMIT + pos

        # ---- 1. Aggressive taking ----
        # Take the full stack of any ask below FV - 1
        remaining_buy = buy_cap
        for ask_price, ask_vol in asks:
            if ask_price < ACO_FV - 1 and remaining_buy > 0:
                qty = min(abs(ask_vol), remaining_buy)
                orders.append(Order(ACO, ask_price, qty))
                remaining_buy -= qty
            else:
                break   # sorted ascending; done once we pass the threshold

        # Take the full stack of any bid above FV + 1
        remaining_sell = sell_cap
        for bid_price, bid_vol in bids:
            if bid_price > ACO_FV + 1 and remaining_sell > 0:
                qty = min(abs(bid_vol), remaining_sell)
                orders.append(Order(ACO, bid_price, -qty))
                remaining_sell -= qty
            else:
                break   # sorted descending

        # ---- 2. Passive market-making with inventory skew ----
        # Skew: each unit of position shifts our quotes by ~0.05 ticks
        # Capped at ±3 ticks so we never quote the wrong side of FV.
        MAX_SKEW = 3
        skew = int(pos / ACO_LIMIT * MAX_SKEW)   # negative when short

        our_bid = ACO_FV - 1 - skew
        our_ask = ACO_FV + 1 - skew

        # Make sure we're strictly inside the bot spread
        our_bid = min(our_bid, best_bot_bid + 1)
        our_ask = max(our_ask, best_bot_ask - 1)

        # Never let bid ≥ FV or ask ≤ FV (don't subsidise the bots)
        our_bid = min(our_bid, ACO_FV - 1)
        our_ask = max(our_ask, ACO_FV + 1)

        # Don't cross
        if our_bid >= our_ask:
            our_bid = ACO_FV - 1
            our_ask = ACO_FV + 1

        PASSIVE_SIZE = 20   # moderate size; position limit handles the rest

        if remaining_buy > 0 and our_bid < best_bot_ask:
            q = min(PASSIVE_SIZE, remaining_buy)
            orders.append(Order(ACO, our_bid, q))

        if remaining_sell > 0 and our_ask > best_bot_bid:
            q = min(PASSIVE_SIZE, remaining_sell)
            orders.append(Order(ACO, our_ask, -q))

        return orders