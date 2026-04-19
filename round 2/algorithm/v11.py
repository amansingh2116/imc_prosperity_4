"""
IMC Prosperity Round 2 — trader_v8.py
======================================
Diagnosis from v4 and v7 logs:

┌─────────────────────────────────────────────────────────────────────┐
│  IPR — WORKING PERFECTLY IN BOTH v4 AND v7. DO NOT CHANGE.         │
│  v4: 7,243 · v7: 7,291 — both ~7,270 per test (=72,700/full day)   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  ACO — TWO BUGS FOUND:                                               │
│                                                                      │
│  BUG 1 (v4): Hardcoded TAKE threshold (bid > 10001) triggers on    │
│  nearly every tick because best-bid is often 10001-10003.            │
│  → Creates short positions when mid is persistently above 10000.    │
│  → Loss: -511 drawdown at ts=2800.                                  │
│                                                                      │
│  BUG 2 (v4 and v7): Passive MAKE quotes are placed too far from      │
│  the inside spread. v4 posts ask at max(10001, best_ask-1) = 10015, │
│  which bots almost never hit. v7 posts at fair±3 = 9997/10003,      │
│  better but still not maximally competitive.                         │
│                                                                      │
│  Market structure (from log):                                        │
│    Best bid: mean=9996, range 9988-10010                             │
│    Best ask: mean=10012, range 9998-10021                            │
│    Spread: ~16 ticks (61% of ticks), occasionally 5-21              │
│    Bid > 10001: only 4% of ticks → HARDCODED TAKE is nearly blind   │
│    Ask < 9999:  only 0.2% of ticks → BUY TAKE almost never fires   │
│                                                                      │
│  FIX: No TAKE at all (it barely fires and creates the drawdown).    │
│  Instead: pure passive market-making 1 tick inside the spread.       │
│    Our bid  = best_bot_bid  + 1  (shifted down by inventory skew)   │
│    Our ask  = best_bot_ask  - 1  (shifted down by inventory skew)   │
│  This captures ~14/16 of the spread per round trip.                 │
│  Inventory skew ensures we revert toward flat position.              │
└─────────────────────────────────────────────────────────────────────┘

Expected performance vs v4/v7:
  IPR:  same ~7,270 per test                  (×10 = 72,700/full day)
  ACO:  estimate ~800-1,200 per test          (×10 = 8,000-12,000/full day)
  Total test: ~8,000-8,500 (v4=8,207, v7=7,820)
  Full 3-day: ~240,000-250,000+ XIRECs
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json

# ─── Products ────────────────────────────────────────────────────────────────
IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"

# ─── Limits ──────────────────────────────────────────────────────────────────
IPR_LIMIT = 80
ACO_LIMIT = 80

# ─── IPR parameters (validated across all 5 data days) ───────────────────────
IPR_SLOPE      = 0.001   # XR per timestamp, exact across all days
IPR_BUY_TOL    = 10      # pay up to fair+10 to fill fast at open

# ─── ACO parameters ──────────────────────────────────────────────────────────
# NO TAKE phase — it fires too rarely (ask<9999: 0.2%, bid>10001: 4%)
# and when it does fire for sells (bid>10001), it creates harmful short
# positions in days where mid is persistently above 10000.
#
# PASSIVE MAKE ONLY — post 1 tick inside the live spread.
# Inventory skew: shift both quotes down when long (encourages selling),
# shift both up when short (encourages buying).
ACO_QUOTE_INSIDE = 1      # ticks inside best_bid / best_ask
ACO_SKEW_FACTOR  = 0.10   # ticks of skew per unit of position
ACO_MAX_SKEW     = 4      # maximum skew in either direction (ticks)
ACO_QUOTE_SIZE   = 15     # units per passive order
ACO_TAPER_START  = 55     # abs-position above which we start tapering size
MAF_BID          = 1_000  # Round-2 Market Access Fee bid


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _book(od: Optional[OrderDepth]):
    """Return (bids, asks) sorted best-first; safe if od is None."""
    if od is None:
        return [], []
    bids = sorted(od.buy_orders.items(), reverse=True)
    asks = sorted(od.sell_orders.items())
    return bids, asks

def _mid(bids, asks) -> Optional[float]:
    if bids and asks:
        return (bids[0][0] + asks[0][0]) / 2.0
    if bids:
        return float(bids[0][0])
    if asks:
        return float(asks[0][0])
    return None


# ─── Trader ──────────────────────────────────────────────────────────────────

class Trader:

    def bid(self) -> int:
        """Market Access Fee bid (Round 2 only, ignored elsewhere)."""
        return MAF_BID

    def run(self, state: TradingState):
        try:
            saved: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result: Dict[str, List[Order]] = {}
        ts = state.timestamp

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            if product == IPR:
                result[product] = self._ipr(od, pos, ts, saved)
            elif product == ACO:
                result[product] = self._aco(od, pos)
            else:
                result[product] = []

        return result, 0, json.dumps(saved)

    # ─── IPR: buy-and-hold trend follower ────────────────────────────────────

    def _ipr(self, od: Optional[OrderDepth], pos: int,
             ts: int, saved: dict) -> List[Order]:
        """
        Hold +80 all day. Price rises 0.001 XR/ts = 1,000/day exactly.
        Spread cost (~8-9 XR/unit paid at open) recovers by ts ≈ 8,000.
        Log confirms: 80 units filled by ts=300, then +8/100ts thereafter.
        """
        bids, asks = _book(od)
        mid = _mid(bids, asks)
        if mid is None:
            return []

        needed = IPR_LIMIT - pos
        if needed <= 0:
            return []   # already at limit — hold and earn trend

        # Calibrate day intercept on first call
        if "ipr_int" not in saved:
            saved["ipr_int"] = float(mid) - IPR_SLOPE * ts

        fair  = saved["ipr_int"] + IPR_SLOPE * ts
        limit = fair + IPR_BUY_TOL

        orders: List[Order] = []

        # Phase 1: Sweep all asks ≤ fair + tolerance (fills fast at open)
        for ask_px, ask_vol in asks:
            if needed <= 0 or ask_px > limit:
                break
            qty = min(abs(ask_vol), needed)
            orders.append(Order(IPR, ask_px, qty))
            pos    += qty
            needed -= qty

        # Phase 2: Passive top-up — sit 1 tick above best bot bid
        # Gets hit if any bot wants to sell. Capped at fair+tolerance.
        if needed > 0 and bids:
            passive_px = min(bids[0][0] + 1, int(limit))
            # Don't cross any ask we already decided not to take
            lowest_ask = asks[0][0] if asks else passive_px + 1
            if passive_px < lowest_ask:
                orders.append(Order(IPR, passive_px, needed))

        return orders

    # ─── ACO: pure passive market maker ──────────────────────────────────────

    def _aco(self, od: Optional[OrderDepth], pos: int) -> List[Order]:
        """
        Post passive quotes 1 tick inside the live bot spread.
        No aggressive TAKE — data shows it fires too rarely for buys
        (ask < 9999 only 0.2% of ticks) and causes harmful short positions
        for sells (bid > 10001 fires 4% of ticks but mid is often >10000,
        making those short positions immediately underwater).

        Quote placement:
            our_bid = best_bot_bid + 1 − skew   (shift down when long)
            our_ask = best_bot_ask − 1 − skew   (shift down when long)
        Skew naturally mean-reverts inventory:
            long  → quotes shift down → bots buy from us (we sell) → flatten
            short → quotes shift up   → bots sell to us (we buy)  → flatten

        Fill mechanics (how Prosperity exchange works):
            Our passive buy at our_bid fills when a bot's ask falls to our_bid.
            Our passive sell at our_ask fills when a bot's bid rises to our_ask.
            Both can fill in the same tick (round trip = full spread captured).
        """
        bids, asks = _book(od)
        if not bids or not asks:
            return []

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        orders: List[Order] = []

        # ── Inventory skew: shift quotes proportionally to position ──────────
        # Positive pos (long) → negative skew → quotes shift DOWN
        #   → our_ask falls closer to best_bid → easier for bots to hit our ask
        # Negative pos (short) → positive skew → quotes shift UP
        #   → our_bid rises closer to best_ask → easier for bots to fill our bid
        raw_skew = pos * ACO_SKEW_FACTOR
        skew = int(max(-ACO_MAX_SKEW, min(ACO_MAX_SKEW, raw_skew)))

        our_bid = best_bid + ACO_QUOTE_INSIDE - skew
        our_ask = best_ask - ACO_QUOTE_INSIDE - skew

        # Safety: never cross (bid must be strictly below ask)
        if our_bid >= our_ask:
            mid_int = (best_bid + best_ask) // 2
            our_bid = mid_int - 1
            our_ask = mid_int + 1

        # Safety: don't post bid above best_ask or ask below best_bid
        # (that would cause an immediate self-cross and incorrect fill)
        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        # ── Quote size: taper near position limit ─────────────────────────────
        abs_pos = abs(pos)
        if abs_pos <= ACO_TAPER_START:
            qsize = ACO_QUOTE_SIZE
        else:
            taper = (ACO_LIMIT - abs_pos) / max(ACO_LIMIT - ACO_TAPER_START, 1)
            qsize = max(1, round(ACO_QUOTE_SIZE * taper))

        # ── Post passive bid ─────────────────────────────────────────────────
        # Only if we can still buy and the quote is strictly inside the book
        room_buy = ACO_LIMIT - pos
        if room_buy > 0 and our_bid < best_ask:
            qty = min(qsize, room_buy)
            orders.append(Order(ACO, our_bid, qty))

        # ── Post passive ask ─────────────────────────────────────────────────
        # Only if we can still sell and the quote is strictly inside the book
        room_sell = ACO_LIMIT + pos
        if room_sell > 0 and our_ask > best_bid:
            qty = min(qsize, room_sell)
            orders.append(Order(ACO, our_ask, -qty))

        return orders