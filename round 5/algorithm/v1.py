"""
IMC Prosperity Round 5 — trader_round5_v6.py
=============================================

WHAT WENT WRONG IN v5 — ROOT CAUSE ANALYSIS
─────────────────────────────────────────────
v5 introduced an EMA(15)/EMA(60) crossover to switch products into a
"trend-following mode" where we held max directional positions. Result: -2.6k
vs v4's +13.4k. The EMA crossover turned out to be the wrong instrument:

  • Every single product oscillated in/out of "trend mode" 150–200 times per day.
  • Each oscillation caused a position flip (long → short or vice versa), each
    flip burning ~3 × spread in round-trip cost.
  • 200 flips × 2 lots × 10 spread = ~4,000 ticks of pure friction per product.
  • Products that v4 was profitably MM-ing (PEBBLES_M, UV_VISOR_YELLOW, etc.)
    were destroyed: PEBBLES_M went from +631 to -1523 (Δ -2154), UV_VISOR_YELLOW
    from +85 to -1516 (Δ -1601).

WHAT ACTUALLY CAUSED v4's LOSSES
──────────────────────────────────
Detailed analysis of the v4 evaluation log reveals that the 13 products which
lost money in BOTH v4 and v5 (combined -11.9k in v4) all share one trait:
the market price drifted strongly during the evaluation day, and v4's symmetric
MM accumulated inventory on the wrong side faster than the skew could unwind it.

The drift was not predictable from the day-2 sample (the evaluation day ran a
different price path — GALAXY_SOUNDS_PLANETARY_RINGS was +358 drift on day 2
but -927 on eval day). So we cannot hardcode "skip this product." The losers
change by day.

KEY INSIGHT: SIDE SUPPRESSION vs TREND FOLLOWING
──────────────────────────────────────────────────
We do NOT want to trade directionally. The position limit of 10 makes any
directional bet small. We want to KEEP MAKING MARKETS but with ONE SIDE
turned off when the price is persistently drifting in that direction.

Concretely:
  • If EMA(50) > EMA(200) by more than SUPPRESS_THRESH × spread:
    ↳ Price is drifting UPWARD. Stop quoting BIDS (don't buy into an uptrend).
    ↳ Keep quoting ASKS — we'll sell into it if we have inventory.
  • If EMA(200) > EMA(50) by more than SUPPRESS_THRESH × spread:
    ↳ Price is drifting DOWNWARD. Stop quoting ASKS (don't sell into a downtrend).
    ↳ Keep quoting BIDS — we'll buy into it if we have inventory.
  • Otherwise: full symmetric MM as in v4.

This is asymmetric and cautious:
  • We never flip sides (no v5-style whipsaw losses).
  • We never accumulate a large directional position — we just decline to add
    more exposure on the side that's being adversely selected.
  • If we're already stuck with inventory from before suppression kicked in,
    the remaining active side + unwind logic clears it.
  • The signal uses a slow EMA(200) so it's stable and won't oscillate.

PARAMETER CHANGES FROM v4
───────────────────────────
  EMA_FAST   = 50   (was 200 for mid-tracking; now used for suppression signal)
  EMA_SLOW   = 200  (context anchor)
  SUPPRESS_THRESH = 1.5  (turn off one side when drift > 1.5 × spread)
  SKEW_START = 2   (was 3 — start skewing 1 step earlier)
  SKEW_PER_POS = 0.8 (was 0.6 — more aggressive inventory correction)
  UNWIND_THRESH = 6  (was 7 — unwind 1 step sooner)
  MAX_LOT = 2        (unchanged)
  INSIDE_TICKS = 1   (unchanged)

EXPECTED OUTCOME
──────────────────
v4 baseline: +13.4k.
v4 consistent-loser drag: -11.9k (from 13 products that lost in both runs).
Suppression target: prevent most of that -11.9k by not quoting the losing side.
Conservative estimate: +20–30k.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle


# ── Constants ─────────────────────────────────────────────────────────────────

POS_LIMIT = 10

# Only skip confirmed zero-edge products with two independent runs of losses
SKIP_PRODUCTS = frozenset({
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
})

# ── Side-suppression EMAs ─────────────────────────────────────────────────────
# EMA_FAST tracks price over ~50 steps.
# EMA_SLOW tracks price over ~200 steps.
# When they diverge by > SUPPRESS_THRESH × spread, one quoting side is suspended.
# Using slow spans means the signal is stable and won't oscillate intra-day.
EMA_FAST_SPAN = 50
EMA_SLOW_SPAN = 200

# Dimensionless threshold: (ema_fast - ema_slow) / spread must exceed this
# to suppress quoting. 1.5 means: the fast average must be 1.5 spreads above
# the slow average before we stop posting bids. High enough to avoid false
# triggers on flat products; low enough to catch persistent drifters.
SUPPRESS_THRESH = 1.5

# ── MM execution ──────────────────────────────────────────────────────────────
INSIDE_TICKS  = 1    # quote 1 tick inside best price
MAX_LOT       = 2    # max lots per side per step

# ── Inventory management ──────────────────────────────────────────────────────
SKEW_START    = 2    # begin skewing at |pos| >= 2 (earlier than v4's 3)
SKEW_PER_POS  = 0.8  # ticks of quote shift per unit position (more than v4's 0.6)
UNWIND_THRESH = 6    # emergency cross-spread unwind at |pos| >= 6 (sooner than v4's 7)
MIN_SPREAD    = 6    # skip if spread collapses


def _ema(prev, val: float, span: int) -> float:
    if prev is None:
        return float(val)
    a = 2.0 / (span + 1)
    return a * float(val) + (1.0 - a) * float(prev)


class Trader:

    def bid(self):
        return 10

    def run(self, state: TradingState):
        # ── Restore state ─────────────────────────────────────────────────────
        try:
            saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        ema_fast: Dict[str, float] = saved.get("ema_fast", {})
        ema_slow: Dict[str, float] = saved.get("ema_slow", {})

        result: Dict[str, List[Order]] = {}
        conversions = 0

        for product, od in state.order_depths.items():
            od: OrderDepth

            if product in SKIP_PRODUCTS:
                result[product] = []
                continue

            orders: List[Order] = []
            pos       = state.position.get(product, 0)
            buy_room  = POS_LIMIT - pos
            sell_room = POS_LIMIT + pos

            # ── Read order book ───────────────────────────────────────────────
            asks = sorted(od.sell_orders.keys())
            bids = sorted(od.buy_orders.keys(), reverse=True)

            if not asks or not bids:
                result[product] = []
                continue

            best_ask = asks[0]
            best_bid = bids[0]
            spread   = best_ask - best_bid
            mid      = (best_ask + best_bid) / 2.0

            if spread < MIN_SPREAD:
                result[product] = []
                continue

            # ── Update EMAs ───────────────────────────────────────────────────
            ema_fast[product] = _ema(ema_fast.get(product), mid, EMA_FAST_SPAN)
            ema_slow[product] = _ema(ema_slow.get(product), mid, EMA_SLOW_SPAN)

            ef = ema_fast[product]
            es = ema_slow[product]

            # ── Side suppression signal ───────────────────────────────────────
            # Normalise by spread so the threshold is scale-free.
            drift_ratio = (ef - es) / spread

            # Positive: fast > slow → price drifting UP → suppress BID side
            # (We don't want to keep buying into an uptrend and get left holding
            # stock when the market eventually reverses.)
            suppress_bid = drift_ratio > SUPPRESS_THRESH

            # Negative: fast < slow → price drifting DOWN → suppress ASK side
            # (We don't want to keep selling short into a downtrend and end up
            # short when the market eventually reverses.)
            suppress_ask = drift_ratio < -SUPPRESS_THRESH

            # ── Emergency unwind ──────────────────────────────────────────────
            # Cross the spread to de-risk. Overrides all quoting logic.
            if pos >= UNWIND_THRESH and sell_room > 0:
                qty = min(pos - (UNWIND_THRESH - 3), sell_room)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                result[product] = orders
                continue

            if pos <= -UNWIND_THRESH and buy_room > 0:
                qty = min(-pos - (UNWIND_THRESH - 3), buy_room)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                result[product] = orders
                continue

            # ── Inventory skew ────────────────────────────────────────────────
            # Shift both quotes against the direction of our current position.
            # Starts earlier (pos >= SKEW_START=2) and is more aggressive
            # (SKEW_PER_POS=0.8) than v4 to clear inventory faster.
            skew_magnitude = max(0, abs(pos) - SKEW_START + 1) * SKEW_PER_POS
            skew = int(round(skew_magnitude * (1 if pos > 0 else -1)))

            # ── Passive quotes ────────────────────────────────────────────────
            our_bid = best_bid + INSIDE_TICKS - skew
            our_ask = best_ask - INSIDE_TICKS - skew

            # Ensure valid spread
            if our_bid >= our_ask:
                our_bid = int(mid) - 1
                our_ask = int(mid) + 1

            our_bid = min(our_bid, best_ask - 2)
            our_ask = max(our_ask, best_bid + 2)

            if our_bid >= our_ask:
                result[product] = []
                continue

            # ── Post orders (respecting side suppression) ─────────────────────
            # BID side: post unless suppressed (price trending up = adverse for buying)
            # Exception: if we're already short, we WANT to buy to reduce the short.
            # So override suppression when it would prevent natural inventory unwind.
            if buy_room > 0 and (not suppress_bid or pos < 0):
                qty = min(MAX_LOT, buy_room)
                orders.append(Order(product, our_bid, qty))

            # ASK side: post unless suppressed (price trending down = adverse for selling short)
            # Exception: if we're already long, we WANT to sell to reduce the long.
            if sell_room > 0 and (not suppress_ask or pos > 0):
                qty = min(MAX_LOT, sell_room)
                orders.append(Order(product, our_ask, -qty))

            result[product] = orders

        # ── Persist state ─────────────────────────────────────────────────────
        trader_data = jsonpickle.encode({"ema_fast": ema_fast, "ema_slow": ema_slow})
        return result, conversions, trader_data