"""
IMC Prosperity Round 5 — trader_round5_v4.py
=============================================

DATA-DRIVEN ANALYSIS SUMMARY (Day 2 sample, 10,000 timestamps, 50 products):
─────────────────────────────────────────────────────────────────────────────

SPREAD-TO-VOL RATIO (determines MM edge):
  SNACKPACK:     spread=16.7, vol=6.9,  ratio=2.41  ← EXCELLENT
  UV_VISOR:      spread=13.0, vol=10.2, ratio=1.27  ← GOOD
  GALAXY_SOUNDS: spread=13.3, vol=10.4, ratio=1.27  ← GOOD
  OXYGEN_SHAKE:  spread=12.8, vol=10.9, ratio=1.17  ← MODERATE
  TRANSLATOR:    spread=8.7,  vol=9.9,  ratio=0.89  ← MARGINAL
  SLEEP_POD:     spread=9.2,  vol=10.4, ratio=0.88  ← MARGINAL
  PANEL:         spread=9.6,  vol=10.0, ratio=0.95  ← MARGINAL
  PEBBLES:       spread=12.8, vol=18.1, ratio=0.71  ← POOR (pebbles have huge vol)
  ROBOT:         spread=7.2,  vol=10.5, ratio=0.68  ← POOR (avoid)
  MICROCHIP:     spread=8.9,  vol=14.1, ratio=0.63  ← POOR (huge vol)

SIMULATION RESULTS (quoting 1 tick inside, all products, 10k steps):
  Total PnL all 50 products: +122k
  Group breakdown (descending):
    TRANSLATOR:    +30k   MICROCHIP:  +21k  UV_VISOR:  +18k
    SNACKPACK:     +13k   OXYGEN:     +12k  PANEL:     +12k
    GALAXY_SOUNDS:  +9k   PEBBLES:    +8k   SLEEP_POD:  +4k
    ROBOT:          -6k  ← ONLY losing group; SKIP

KEY INSIGHT — WHY v3 UNDERPERFORMED:
  v3 only traded 5 SNACKPACK products (13k/day). The other 45 products
  contribute ~109k additional PnL (with ROBOTS being the only losing group).
  The fix is to trade ALL products EXCEPT ROBOTS.

INSIDE TICKS = 1 vs 3:
  v3 used INSIDE_TICKS=3. This gives round-trip profit of spread-6=10 ticks.
  INSIDE_TICKS=1 gives spread-2=14.7 ticks per round trip on SNACKPACK.
  Fill rate is approximately EQUAL: bots always prefer the best price in market.
  Setting INSIDE_TICKS=1 maximises profit per fill with no fill rate penalty.

FILL MECHANISM (confirmed from trade data analysis):
  All 229 bot-to-bot trades per product land exactly at best_bid or best_ask.
  Our quote at bid+1 / ask-1 becomes the new best price → bots that hit the
  market will prefer our price → we get filled in place of the market-making bot.

POSITION/INVENTORY MANAGEMENT:
  With pos_limit=10 and lot=2 we can fill 5 times before hitting the limit.
  Large losses in simulation came from drift: SLEEP_POD_LAMB_WOOL (-16k), 
  PEBBLES_L (-15k) — both products with slope ~-7 to -9 ticks per 100 steps.
  Fix: unwind aggressively at pos=7, skew quotes strongly against inventory.

PRODUCTS TO TRADE:
  All 50 minus the 5 ROBOTS (which have spread/vol=0.68 → estimated -5.5k/day).
  That leaves 45 products across 9 groups.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle


# ── Constants ─────────────────────────────────────────────────────────────────

POS_LIMIT = 10

# Skip ROBOTS: only group with negative MM edge (spread 7.2, vol 10.5)
SKIP_PRODUCTS = frozenset({
    "ROBOT_VACUUMING",
    "ROBOT_MOPPING",
    "ROBOT_DISHES",
    "ROBOT_LAUNDRY",
    "ROBOT_IRONING",
})

# Quote 1 tick inside the bot spread.
# This makes us the best bid AND best ask simultaneously.
# Bots that want to trade always prefer the best price → they trade with us.
# Round-trip capture: spread - 2 ticks (= 14.7 on SNACKPACK, ~7-11 on others).
# Adverse selection per fill: ~0.3 * vol ≈ 2-3 ticks on average.
# Net per round trip: ~10 ticks for SNACKPACK, ~1-5 for the rest.
INSIDE_TICKS = 1

# Max lots per passive order per side per iteration.
# Small lot = less inventory build-up per fill = less directional risk.
MAX_LOT = 2

# At |pos| >= SKEW_START we start aggressively shifting our quotes.
SKEW_START = 3

# Ticks of quote shift per unit of position.
# At pos=5: skew=2.5 → our_bid is 2.5 ticks worse → bots less likely to sell to us.
# Larger skew → faster mean-reversion of inventory.
SKEW_PER_POS = 0.6

# At |pos| >= UNWIND_THRESH we stop passive quoting and cross the spread.
UNWIND_THRESH = 7

# Minimum spread required to quote. If the spread collapses (unusual regime)
# we stand aside to avoid being caught in a volatile period.
MIN_SPREAD = 6

# EMA span for mid price smoothing (used only for context logging, not entry).
EMA_SPAN = 200


def _ema(prev, val: float, span: int) -> float:
    if prev is None:
        return float(val)
    a = 2.0 / (span + 1)
    return a * float(val) + (1.0 - a) * float(prev)


class Trader:

    def bid(self):
        """Required for Round 2 compatibility; ignored in Round 5."""
        return 10

    def run(self, state: TradingState):
        # ── Restore persisted state ───────────────────────────────────
        try:
            saved = jsonpickle.decode(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        ema_mid: Dict[str, float] = saved.get("ema_mid", {})

        result: Dict[str, List[Order]] = {}
        conversions = 0

        for product, od in state.order_depths.items():
            od: OrderDepth

            # Skip products with negative or zero expected edge.
            if product in SKIP_PRODUCTS:
                result[product] = []
                continue

            orders: List[Order] = []
            pos = state.position.get(product, 0)
            buy_room  = POS_LIMIT - pos   # how much more long we can go
            sell_room = POS_LIMIT + pos   # how much more short we can go

            # ── Read the order book ───────────────────────────────────
            asks = sorted(od.sell_orders.keys())          # ascending
            bids = sorted(od.buy_orders.keys(), reverse=True)  # descending

            if not asks or not bids:
                result[product] = []
                continue

            best_ask = asks[0]
            best_bid = bids[0]
            spread   = best_ask - best_bid
            mid      = (best_ask + best_bid) / 2.0

            # Update EMA (not used for entry; useful for future extensions)
            ema_mid[product] = _ema(ema_mid.get(product), mid, EMA_SPAN)

            # Sanity check: don't quote in a collapsed spread environment.
            if spread < MIN_SPREAD:
                result[product] = []
                continue

            # ── Emergency unwind: inventory too large ─────────────────
            # Cross the spread aggressively to reduce inventory to safe levels.
            # We absorb the spread cost in exchange for risk reduction.
            if pos >= UNWIND_THRESH and sell_room > 0:
                # Hit the best bid (sell at best_bid = guaranteed fill)
                unwind_qty = min(pos - (UNWIND_THRESH - 3), sell_room)
                if unwind_qty > 0:
                    orders.append(Order(product, best_bid, -unwind_qty))
                result[product] = orders
                continue

            if pos <= -UNWIND_THRESH and buy_room > 0:
                # Lift the best ask (buy at best_ask = guaranteed fill)
                unwind_qty = min(-pos - (UNWIND_THRESH - 3), buy_room)
                if unwind_qty > 0:
                    orders.append(Order(product, best_ask, unwind_qty))
                result[product] = orders
                continue

            # ── Compute inventory skew ────────────────────────────────
            # Shifts BOTH quotes against the direction of our current position.
            # Long (+pos): lower our bid (less attractive to sellers) AND lower
            #              our ask (more attractive to buyers) → sell more.
            # Short (-pos): raise our bid AND raise our ask → buy more.
            # The skew is applied as a continuous function starting from SKEW_START.
            skew_magnitude = max(0, abs(pos) - SKEW_START + 1) * SKEW_PER_POS
            skew = int(round(skew_magnitude * (1 if pos > 0 else -1)))

            # ── Compute our passive quotes ────────────────────────────
            # INSIDE_TICKS=1 → we improve the bot's price by 1 tick on each side.
            # This makes us the new best bid AND best ask simultaneously.
            # Any bot that hits the market will prefer our price over the bot's.
            our_bid = best_bid + INSIDE_TICKS - skew
            our_ask = best_ask - INSIDE_TICKS - skew

            # ── Safety guards ─────────────────────────────────────────
            # Ensure we always have a valid two-sided quote (bid < ask).
            if our_bid >= our_ask:
                # Collapse to mid ± 1 as fallback
                our_bid = int(mid) - 1
                our_ask = int(mid) + 1

            # Don't accidentally cross the spread (that would be an immediate
            # taker trade, not a passive market-maker order).
            # We want passive fills: stay strictly inside the bot spread.
            our_bid = min(our_bid, best_ask - 2)
            our_ask = max(our_ask, best_bid + 2)

            # One final check after clamping:
            if our_bid >= our_ask:
                result[product] = []
                continue

            # ── Post passive orders ───────────────────────────────────
            if buy_room > 0:
                qty = min(MAX_LOT, buy_room)
                orders.append(Order(product, our_bid, qty))

            if sell_room > 0:
                qty = min(MAX_LOT, sell_room)
                orders.append(Order(product, our_ask, -qty))

            result[product] = orders

        # ── Persist EMA state across iterations ───────────────────────
        trader_data = jsonpickle.encode({"ema_mid": ema_mid})
        return result, conversions, trader_data