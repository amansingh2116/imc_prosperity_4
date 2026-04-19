"""
IMC Prosperity Round 2  —  trader_final.py
===========================================

WHAT THE DATA ACTUALLY PROVED  (v4/v7/v12 log analysis)
=========================================================

IPR:
  - Perfect linear trend FV(t) = 13000 + 0.001*t on the test day
    (intercept shifts each day; slope is constant at 0.001 exactly)
  - All 80 units filled by ts=400 in v12.  Max overpay above FV: 9.9 XR.
    Recovery time: 9.9 / 0.001 = 9,900 ts.  Fine.
  - Strategy: buy everything at ask levels, stop.  No selling, no passive top-up.
    (The passive top-up in v12 never fired; IPR ask volume is sufficient.)
  - Only "parameter": tolerance above FV.  Observed asks are at FV+7 to FV+10,
    so tolerance=11 captures all levels without overpaying more than 10 ticks.
    This is data-derived, not hardcoded arbitrarily.

ACO:
  - 100% of all 42 fills in v12 were PASSIVE MAKE fills.  Take never fired.
  - Average edge per fill: 7.18 XR from mid (std=1.14, always positive).
  - Fill quantity per event: 2–10 units, avg=4.8.  Our quote of 15 was never
    fully consumed, so LARGER quotes do not improve fill rate.
  - Fill rate: 42 fills / 1000 ticks = 4.2%.  This is driven by BOT FLOW,
    not by our quote size.  Quoting 15 or 80 gives the same fills.
  - Bot spread: median=16, std=2.6, range [5,21].  We quote 1 tick inside
    each side, capturing 14/16 = 87.5% of the spread per round trip.
  - Position range in v12: [-22, +10].  Skew works correctly — no blowup.
  - v7 regressed (530 vs v12's 1358) because it removed take.  But take also
    never fired in v12!  The real issue: v7's make placement logic was worse.
    v12's "best_bid+1 / best_ask-1" is the right quote placement.

WHAT CHANGES vs v12:
  1. Remove every hardcoded constant that could overfit:
     - No ACO_FV=10000 (used only for take; take never fires anyway)
     - No EMA at all (only needed if take is meaningful)
     - No take phase (100% of real PnL came from make; take creates position risk)
     - No "anchor strength" magic number
  2. Derive the ONE real parameter (quote size) from observed fill data:
     - Max fill observed: 10 units.  Quote size=20 ensures we never miss volume
       while giving headroom.  Above 20 provides no benefit.
  3. Skew formula kept from v12 but capped at half-spread:
     - skew in [-half_spread+1, half_spread-1] so quotes stay inside book
     - derived from spread observed at each tick, not hardcoded
  4. IPR tolerance=11 (observed max ask offset above FV was 10 in logs)

MAF BID (Round 2):
  - Extra ACO flow from MAF ≈ 25% × 2-day ACO PnL ≈ 25% × 27,000 ≈ 6,750 XR
  - My R1 rank: 908 / 5700 = top 16%.  Sophisticated competitor pool.
  - Top-16% players will compute similarly → median bid ≈ 2,500–4,000
  - Optimal bid: 4,500 (safely above likely peer median, cost well below MAF value)
  - If median turns out to be 6,000 (aggressive field), we lose access but pay
    nothing.  If median is 2,000 (conservative), we overpay by 2,500 but gain
    6,750 — net +4,250.  Expected value positive across all reasonable scenarios.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json

# ─── Products ────────────────────────────────────────────────────────────────
IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"

# ─── Position limits ─────────────────────────────────────────────────────────
LIMIT: Dict[str, int] = {IPR: 80, ACO: 80}

# ─── IPR constants ───────────────────────────────────────────────────────────
# Slope measured across all 5 sample days: exactly 0.001 XR/timestamp.
IPR_SLOPE = 0.001

# Maximum price we pay above our calibrated fair value.
# From logs: actual ask levels are at most FV+10.  Set 11 to sweep all of them
# without leaving free money behind.  Derived from observed data, not guessed.
IPR_MAX_ABOVE_FV = 11

# ─── ACO constants ───────────────────────────────────────────────────────────
# Quote 1 tick inside the live bot spread on each side.  From log analysis:
# median spread=16; quoting 1 inside captures 14 ticks per round trip (87.5%).
# Quoting 2 inside would capture 12 (75%) but get filled more often — tested
# in v7 equivalent and gave fewer fills because bots rarely reach that far.
ACO_INSIDE = 1

# Passive quote size.  Log shows max fill = 10 units per event; bot flow is
# the binding constraint, not our size.  20 gives headroom without wasting
# position capacity on one side.
ACO_QUOTE_SIZE = 20

# Inventory skew cap: maximum ticks we shift our quotes due to position.
# At limit (pos=±80): skew = ±ACO_MAX_SKEW.  4 ticks keeps us inside the
# bot spread even at extreme inventory while still incentivising reversion.
ACO_MAX_SKEW = 4

# ─── MAF bid (Round 2 only, ignored in all other rounds) ─────────────────────
MAF_BID = 4_500


# ─── Trader ──────────────────────────────────────────────────────────────────

class Trader:

    def bid(self) -> int:
        """Market Access Fee bid for Round 2."""
        return MAF_BID

    def run(self, state: TradingState):
        try:
            saved: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)
            lim = LIMIT.get(product, 20)

            if product == IPR:
                result[product] = self._ipr(od, pos, state.timestamp, lim, saved)
            elif product == ACO:
                result[product] = self._aco(od, pos, lim)
            else:
                result[product] = []

        return result, 0, json.dumps(saved)

    # ─── IPR ─────────────────────────────────────────────────────────────────

    def _ipr(self, od: OrderDepth, pos: int, ts: int,
             lim: int, saved: dict) -> List[Order]:
        """
        Buy-and-hold.  Fill as much as possible at open; hold the rest of the day.

        Fair value = intercept + 0.001 × timestamp.
        Intercept is calibrated from the first mid-price we observe.  Because
        the intercept shifts each day (~1000 XR lower per day), we re-calibrate
        whenever traderData is empty (i.e., at the start of each day/run).

        We sweep every ask level at or below FV + IPR_MAX_ABOVE_FV.
        No selling — the trend earns ~1,000 XR per unit per day, so holding
        dominates any marginal spread we could capture by selling.
        """
        bids = sorted(od.buy_orders.items(), reverse=True)
        asks = sorted(od.sell_orders.items())          # ascending price

        # Need at least one side to calibrate
        if not asks and not bids:
            return []

        mid: Optional[float] = None
        if bids and asks:
            mid = (bids[0][0] + asks[0][0]) / 2.0
        elif asks:
            mid = float(asks[0][0])
        else:
            mid = float(bids[0][0])

        # Calibrate intercept once per simulation run (day).
        # Reset happens naturally because traderData starts empty each submission.
        if "ipr_int" not in saved:
            saved["ipr_int"] = mid - IPR_SLOPE * ts

        fv = saved["ipr_int"] + IPR_SLOPE * ts
        ceiling = fv + IPR_MAX_ABOVE_FV

        needed = lim - pos
        if needed <= 0:
            return []

        orders: List[Order] = []
        for ask_px, ask_vol in asks:
            if needed <= 0:
                break
            if ask_px > ceiling:
                break                              # asks are ascending; stop here
            qty = min(abs(ask_vol), needed)
            orders.append(Order(IPR, ask_px, qty))
            needed -= qty

        return orders

    # ─── ACO ─────────────────────────────────────────────────────────────────

    def _aco(self, od: OrderDepth, pos: int, lim: int) -> List[Order]:
        """
        Pure passive market-making.  Post one bid and one ask each tick,
        1 tick inside the live bot spread, skewed by current inventory.

        Why no EMA / take phase?
          Log analysis shows 100% of real fills came from passive makes.
          Take fires ~0% of ticks in practice because our EMA-based fair value
          stays close to mid, and the bot book rarely crosses it.  Adding take
          only creates position risk (v4's -511 drawdown came from a misfire).

        Skew logic (from v12, validated):
          Long position → shift both quotes DOWN by skew ticks:
            our_ask moves closer to bot_bid → bots hit our ask more → we sell → flatten
            our_bid moves lower → bots less likely to fill our bid → we buy less
          Short position → shift UP → encourages buying, discourages selling.

        Quote size:
          Fixed at ACO_QUOTE_SIZE=20.  Max observed bot fill per event: 10 units.
          Bot flow is the binding constraint, not our size.  20 gives headroom.
          We still cap at remaining room on each side.
        """
        bids = sorted(od.buy_orders.items(), reverse=True)
        asks = sorted(od.sell_orders.items())

        if not bids or not asks:
            return []

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = best_ask - best_bid

        # Don't quote into a crossed or 0-spread book (data anomaly)
        if spread <= 1:
            return []

        # Inventory skew: proportional to position, capped at ACO_MAX_SKEW.
        # Also capped so that neither quote crosses the mid of the bot spread,
        # preventing us from accidentally posting an aggressive (crossing) order.
        raw_skew = round((pos / lim) * ACO_MAX_SKEW)
        max_safe  = spread // 2 - 1          # stay inside the spread with headroom
        skew = max(-max_safe, min(max_safe, raw_skew))

        our_bid = best_bid + ACO_INSIDE - skew
        our_ask = best_ask - ACO_INSIDE - skew

        # Final safety: bid must be strictly below ask, both strictly inside book
        if our_bid >= our_ask:
            mid_int  = (best_bid + best_ask) // 2
            our_bid  = mid_int - 1
            our_ask  = mid_int + 1

        our_bid = min(our_bid, best_ask - 1)
        our_ask = max(our_ask, best_bid + 1)

        orders: List[Order] = []

        room_buy  = lim - pos
        room_sell = lim + pos

        if room_buy > 0 and our_bid < best_ask:
            orders.append(Order(ACO, our_bid, min(ACO_QUOTE_SIZE, room_buy)))

        if room_sell > 0 and our_ask > best_bid:
            orders.append(Order(ACO, our_ask, -min(ACO_QUOTE_SIZE, room_sell)))

        return orders