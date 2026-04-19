"""
IMC Prosperity Round 2 — Upgraded Trading Strategy
=====================================================

FINDINGS FROM DATA ANALYSIS
-----------------------------

INTARIAN_PEPPER_ROOT (IPR):
  • Perfect linear trend: fair_value = anchor + 0.001 * timestamp
    (Confirmed exact slope=0.001000 across ALL 5 days / both rounds — never deviates)
  • Residual noise: std ≈ 2.4 ticks, max deviation ≈ 11 ticks → very predictable
  • Bot spread: ~14 ticks (bid1 ~7 below mid, ask1 ~7 above mid)
  • OPTIMAL STRATEGY: buy max position (80) ASAP at day-start, hold all day,
    unwind at day-end.  Expected gain: 80 × ~986 ≈ 78,880 XIRECS/day
  • Current code bug: IndexError on empty bids crashes iterator → fixed

ASH_COATED_OSMIUM (ACO):
  • Mean-reverting around 10,000 with half-life ≈ 2–3 iterations
  • Return autocorrelation lag-1 = −0.50, lag-2 ≈ 0 → pure AR(1)
  • 91 % of mid prices within [9990, 10010]; 99 % within [9985, 10015]
  • Bot spread: ~16 ticks (bid1 ~7-8 below mid, ask1 ~7-8 above mid)
  • OPTIMAL STRATEGY: three-layer market-maker + stat-arb taking
      Layer 1  Aggressive take   — hit mispricings beyond ±5 from fair
      Layer 2  Passive make      — post inside bot spread at fair ± 3
      Layer 3  Second level       — deeper passive quotes as backup
  • End-of-day flatten added for both products

WHAT WAS WRONG IN v1
---------------------
  1. IndexError crash at iteration 0 (accessing bids[0] when bids is empty)
     → All 999 successful iterations were AFTER the position was already maxed,
       so IPR just drifted profitably by accident.  ACO made minimal trades.
  2. IPR strategy tried to market-make rather than trend-follow.
     Missed ~78,880 − 13,100 = 65,780 XIRECS per test run.
  3. IPR sell-back logic triggered too early and unnecessarily.
  4. ACO EMA alpha=0.06 was far too slow for a half-life-2 process; alpha
     should be ~0.5.
  5. ACO position skewing was ad-hoc; now uses clean inventory-penalty formula.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple, Optional
import json

# ── Product names ──────────────────────────────────────────────────────────────
IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"

# ── Position limits ────────────────────────────────────────────────────────────
IPR_LIMIT = 80
ACO_LIMIT = 80

# ── IPR constants ──────────────────────────────────────────────────────────────
# Slope confirmed at exactly 0.001 across every day of both rounds.
IPR_SLOPE: float = 0.001

# How far above fair value we're still willing to buy early in the day.
# Since price rises 1 tick / 1000-ts, paying +20 extra = broken even after 20k-ts.
# With a 1,000,000-ts day we have enormous headroom.
IPR_BUY_BUFFER_EARLY: int = 25   # t < 100_000
IPR_BUY_BUFFER_LATE:  int = 12   # t ≥ 100_000

# Unwind phase start (timestamp)
IPR_UNWIND_START: int = 990_000

# ── ACO constants ──────────────────────────────────────────────────────────────
ACO_LONG_TERM_FAIR: float = 10_000.0   # True long-run mean

# EMA for short-term fair value; alpha≈0.4 → half-life ≈ 2 steps (matches data)
ACO_EMA_FAST_ALPHA: float = 0.40
# Slow EMA for regime detection / anchor
ACO_EMA_SLOW_ALPHA: float = 0.02

# Aggressive-take thresholds (in ticks from fair)
ACO_TAKE_THRESHOLD:  int = 5

# Passive-quote offset from fair (Layer 2)
ACO_PASSIVE_OFFSET:  int = 3

# Base order size for passive quotes
ACO_BASE_SIZE:  int = 18
ACO_MAX_SIZE:   int = 28
ACO_MIN_SIZE:   int  = 5

# Inventory at which we start skewing aggressively
ACO_SKEW_START: int = 20

# End-of-day flatten timestamp
ACO_EOD_START: int = 995_000

# ── Round-2 bid auction ────────────────────────────────────────────────────────
MAF_BID: int = 1800


# ══════════════════════════════════════════════════════════════════════════════
def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _sorted_book(od: Optional[OrderDepth]):
    """Return (bids, asks) sorted best-first; safe even if od is None."""
    if od is None:
        return [], []
    bids = sorted(od.buy_orders.items(),  reverse=True)  # highest bid first
    asks = sorted(od.sell_orders.items())                 # lowest ask first
    return bids, asks


def _mid_price(bids, asks) -> Optional[float]:
    if bids and asks:
        return 0.5 * (bids[0][0] + asks[0][0])
    if bids:
        return float(bids[0][0])
    if asks:
        return float(asks[0][0])
    return None


# ══════════════════════════════════════════════════════════════════════════════
class Trader:

    # Required for Round 2 auction mechanism
    def bid(self) -> int:
        return MAF_BID

    # ── Main entry point ───────────────────────────────────────────────────────
    def run(self, state: TradingState):
        # Restore persistent state (survives across iterations via traderData)
        td: Dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}

        result: Dict[str, List[Order]] = {}

        result[IPR] = self._trade_ipr(
            state.order_depths.get(IPR),
            state.position.get(IPR, 0),
            state.timestamp,
            td,
        )

        result[ACO] = self._trade_aco(
            state.order_depths.get(ACO),
            state.position.get(ACO, 0),
            state.timestamp,
            td,
        )

        conversions = 0
        return result, conversions, json.dumps(td)

    # ══════════════════════════════════════════════════════════════════════════
    # IPR — Trend-Following Buy-and-Hold
    # ══════════════════════════════════════════════════════════════════════════
    def _trade_ipr(
        self,
        od: Optional[OrderDepth],
        pos: int,
        t: int,
        td: Dict,
    ) -> List[Order]:

        bids, asks = _sorted_book(od)
        mid = _mid_price(bids, asks)
        if mid is None:
            return []

        # ── Calibrate anchor once per day ────────────────────────────────────
        # anchor = mid_0 − slope × t_0  →  fair(t) = anchor + slope × t
        if "ipr_anchor" not in td:
            td["ipr_anchor"] = float(mid) - IPR_SLOPE * t

        fair: float = float(td["ipr_anchor"]) + IPR_SLOPE * t

        orders: List[Order] = []
        buy_cap  = IPR_LIMIT - pos   # units we can still buy
        sell_cap = IPR_LIMIT + pos   # units we can still sell

        # ── End-of-day unwind: sell everything aggressively ──────────────────
        if t >= IPR_UNWIND_START:
            if pos > 0:
                remaining = pos
                for bid_px, bid_vol in bids:
                    if remaining <= 0:
                        break
                    qty = min(abs(bid_vol), remaining)
                    if qty > 0:
                        orders.append(Order(IPR, int(bid_px), -qty))
                        remaining -= qty
                # If still long after hitting all bids, post at best bid - 1
                if remaining > 0 and bids:
                    orders.append(Order(IPR, int(bids[-1][0]) - 1, -remaining))
            elif pos < 0:
                # Cover any accidental short
                remaining = -pos
                for ask_px, ask_vol in asks:
                    if remaining <= 0:
                        break
                    qty = min(abs(ask_vol), remaining)
                    if qty > 0:
                        orders.append(Order(IPR, int(ask_px), qty))
                        remaining -= qty
            return orders

        # ── Active buy phase: accumulate maximum position ─────────────────────
        if buy_cap <= 0:
            # Already at limit — nothing to do until unwind
            return []

        buffer = IPR_BUY_BUFFER_EARLY if t < 100_000 else IPR_BUY_BUFFER_LATE
        remaining = buy_cap

        # Layer A: Immediately hit any ask price ≤ fair + buffer
        for ask_px, ask_vol in asks:
            if remaining <= 0:
                break
            if ask_px <= fair + buffer:
                qty = min(abs(ask_vol), remaining)
                if qty > 0:
                    orders.append(Order(IPR, int(ask_px), qty))
                    remaining -= qty
            else:
                break  # asks are sorted ascending; no point checking further

        # Layer B: Post passive bid just inside current spread to catch more
        if remaining > 0:
            if asks:
                # Offer 1 tick below best ask to attract fills
                passive_bid = int(asks[0][0]) - 1
            elif bids:
                passive_bid = int(bids[0][0]) + 1
            else:
                passive_bid = int(fair)

            # Cap at fair + buffer so we don't chase irrationally
            passive_bid = min(passive_bid, int(fair) + buffer)
            passive_bid = max(passive_bid, 1)

            orders.append(Order(IPR, passive_bid, remaining))

        return orders

    # ══════════════════════════════════════════════════════════════════════════
    # ACO — Multi-Layer Market Maker + Statistical Arbitrage
    # ══════════════════════════════════════════════════════════════════════════
    def _trade_aco(
        self,
        od: Optional[OrderDepth],
        pos: int,
        t: int,
        td: Dict,
    ) -> List[Order]:

        bids, asks = _sorted_book(od)
        mid = _mid_price(bids, asks)
        if mid is None:
            return []

        # ── Dual-speed EMA (fast for fair value, slow for regime anchor) ──────
        if "aco_fast" not in td:
            td["aco_fast"] = float(mid)
            td["aco_slow"] = float(mid)
        else:
            td["aco_fast"] = (
                (1 - ACO_EMA_FAST_ALPHA) * float(td["aco_fast"])
                + ACO_EMA_FAST_ALPHA * float(mid)
            )
            td["aco_slow"] = (
                (1 - ACO_EMA_SLOW_ALPHA) * float(td["aco_slow"])
                + ACO_EMA_SLOW_ALPHA * float(mid)
            )

        # Blend fast EMA with long-run anchor.
        # When fast EMA strays, anchor pulls it back.
        fair: float = 0.7 * float(td["aco_fast"]) + 0.3 * ACO_LONG_TERM_FAIR
        fair_i: int = int(round(fair))

        best_bid: int = int(bids[0][0]) if bids else fair_i - 8
        best_ask: int = int(asks[0][0]) if asks else fair_i + 8

        buy_cap  = ACO_LIMIT - pos
        sell_cap = ACO_LIMIT + pos
        orders: List[Order] = []
        rem_buy  = buy_cap
        rem_sell = sell_cap

        # ── End-of-day flatten (flatten position by day close) ────────────────
        if t >= ACO_EOD_START:
            if pos > 0 and bids:
                remaining = pos
                for bid_px, bid_vol in bids:
                    if remaining <= 0:
                        break
                    qty = min(abs(bid_vol), remaining)
                    if qty > 0:
                        orders.append(Order(ACO, int(bid_px), -qty))
                        remaining -= qty
            elif pos < 0 and asks:
                remaining = -pos
                for ask_px, ask_vol in asks:
                    if remaining <= 0:
                        break
                    qty = min(abs(ask_vol), remaining)
                    if qty > 0:
                        orders.append(Order(ACO, int(ask_px), qty))
                        remaining -= qty
            return orders

        # ── Inventory penalty: shifts take-thresholds to mean-revert position ─
        # When long, we lower the sell threshold (easier to sell).
        # When short, we lower the buy threshold (easier to buy).
        inv_penalty: float = pos / ACO_LIMIT  # ∈ [−1, +1]

        take_buy_th  = fair - ACO_TAKE_THRESHOLD - max(0.0,  inv_penalty * 6)
        take_sell_th = fair + ACO_TAKE_THRESHOLD + max(0.0, -inv_penalty * 6)

        # ── LAYER 1: Aggressive takes (statistical-arbitrage / mean-reversion) ─
        # Buy when bot's ask is significantly below fair (price will revert up).
        for ask_px, ask_vol in asks:
            if rem_buy <= 0:
                break
            if ask_px <= take_buy_th:
                qty = min(abs(ask_vol), rem_buy)
                if qty > 0:
                    orders.append(Order(ACO, int(ask_px), qty))
                    rem_buy -= qty
            else:
                break

        # Sell when bot's bid is significantly above fair (price will revert down).
        for bid_px, bid_vol in bids:
            if rem_sell <= 0:
                break
            if bid_px >= take_sell_th:
                qty = min(abs(bid_vol), rem_sell)
                if qty > 0:
                    orders.append(Order(ACO, int(bid_px), -qty))
                    rem_sell -= qty
            else:
                break

        # ── LAYER 2: Passive market-making (spread capture) ──────────────────
        # Post quotes inside the bot spread (bots trade ~fair±8; we post at ±3–5).
        # Inventory skew: if long, lower both quotes to sell more easily.
        pos_skew: int = _clamp(int(round(pos / ACO_SKEW_START)), -4, 4)

        l2_bid = fair_i - ACO_PASSIVE_OFFSET - pos_skew
        l2_ask = fair_i + ACO_PASSIVE_OFFSET - pos_skew

        # Enforce: bid < best_ask, ask > best_bid, and bid < ask
        l2_bid = min(l2_bid, best_ask - 1)
        l2_ask = max(l2_ask, best_bid + 1)
        if l2_bid >= l2_ask:
            l2_bid = fair_i - 2
            l2_ask = fair_i + 2

        # Size: shrink as we approach position limit
        fill_ratio = abs(pos) / ACO_LIMIT            # 0 → free, 1 → at limit
        base = int(round(ACO_BASE_SIZE * (1 - fill_ratio * 0.5)))
        base = _clamp(base, ACO_MIN_SIZE, ACO_MAX_SIZE)

        # Asymmetric sizing to lean against inventory
        if pos > ACO_SKEW_START:
            bid_sz = _clamp(base - int((pos - ACO_SKEW_START) // 6), ACO_MIN_SIZE, base)
            ask_sz = _clamp(base + int((pos - ACO_SKEW_START) // 4), base, ACO_MAX_SIZE)
        elif pos < -ACO_SKEW_START:
            bid_sz = _clamp(base + int((-pos - ACO_SKEW_START) // 4), base, ACO_MAX_SIZE)
            ask_sz = _clamp(base - int((-pos - ACO_SKEW_START) // 6), ACO_MIN_SIZE, base)
        else:
            bid_sz = base
            ask_sz = base

        # Hard gates at position limit
        if pos >= ACO_LIMIT:
            bid_sz = 0
        if pos <= -ACO_LIMIT:
            ask_sz = 0

        if rem_buy > 0 and bid_sz > 0 and l2_bid < best_ask:
            q = min(rem_buy, bid_sz)
            orders.append(Order(ACO, int(l2_bid), q))
            rem_buy -= q

        if rem_sell > 0 and ask_sz > 0 and l2_ask > best_bid:
            q = min(rem_sell, ask_sz)
            orders.append(Order(ACO, int(l2_ask), -q))
            rem_sell -= q

        # ── LAYER 3: Deeper passive quotes (additional fill probability) ──────
        # Post a second level 4–5 ticks wider as a backup / inventory buffer.
        if pos < ACO_LIMIT - 10:
            l3_bid = l2_bid - 5
            l3_bid_sz = _clamp(ACO_BASE_SIZE // 2, ACO_MIN_SIZE, ACO_MAX_SIZE)
            if rem_buy > 0 and l3_bid >= 1:
                q = min(rem_buy, l3_bid_sz)
                orders.append(Order(ACO, int(l3_bid), q))
                rem_buy -= q

        if pos > -ACO_LIMIT + 10:
            l3_ask = l2_ask + 5
            l3_ask_sz = _clamp(ACO_BASE_SIZE // 2, ACO_MIN_SIZE, ACO_MAX_SIZE)
            if rem_sell > 0:
                q = min(rem_sell, l3_ask_sz)
                orders.append(Order(ACO, int(l3_ask), -q))
                rem_sell -= q

        return orders