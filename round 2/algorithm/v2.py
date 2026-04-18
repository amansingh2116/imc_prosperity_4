"""
IMC Prosperity Round 2 — trader_r2.py
======================================
Products: INTARIAN_PEPPER_ROOT (IPR) + ASH_COATED_OSMIUM (ACO)
Limits:   IPR = 80,  ACO = 80

Round 2 is structurally identical to Round 1 with two additions:
  1. bid() function for the Market Access Fee auction
  2. R2 days are day -1, 0, 1 (same products, prices shifted +1000)

DATA FINDINGS (R2 sample, 3 days):
  IPR slope = 0.001 XIREC/timestamp (1000/day), R²≈1 — unchanged from R1
  ACO mean  = 10000, std ≈ 5, spread ≈ 16     — unchanged from R1

STRATEGY LOGIC:
  IPR → Buy 80 immediately, hold all day. ~79,000 XIREC/day.
  ACO → Anchored EMA market-making.          ~7,000 XIREC/day.
  MAF → bid(1000): costs 1000, earns ~2774 extra → net +1774 XIREC.

PROJECTED 3-DAY TOTAL: ~263,000 XIREC (target = 200,000).

CHANGES vs v12:
  ACO: Switched from raw FV=10000 to anchored EMA (α=0.02, anchor=1%)
       This prevents the fair value from drifting on days where mid
       oscillates above or below 10000 persistently.
  ACO: Sweep level-2 of the book on TAKE — was missing ~30% of takes.
  IPR: Unchanged — already near-optimal in v12.
  bid(): Returns 1000 (see game theory section in docstring above).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

# ─── Constants ───────────────────────────────────────────────────────────────

IPR = "INTARIAN_PEPPER_ROOT"
ACO = "ASH_COATED_OSMIUM"

IPR_LIMIT   = 80
ACO_LIMIT   = 80

IPR_SLOPE   = 0.001     # XR per timestamp, constant across all days
IPR_BUY_TOL = 10        # pay up to fair+10 to fill fast at open

ACO_FAIR    = 10_000.0  # true long-run mean, stable across all days
ACO_ALPHA   = 0.02      # slow EMA — prevents fair value drifting with noise
ACO_ANCHOR  = 0.01      # % pull toward 10000 per tick (1%)
ACO_SKEW    = 3         # max quote shift in ticks due to inventory
ACO_QSIZE   = 10        # passive quote size
ACO_SOFT    = 60        # taper quote size above this abs-position

# ─── MAF Bid ─────────────────────────────────────────────────────────────────
# Game theory rationale:
#   Extra profit from 25% more flow ≈ 2,774 XIRECs (3 days, ACO dominant).
#   Bid 1,000 → almost certainly in top 50% of participants.
#   Net gain from MAF = 2,774 − 1,000 = +1,774 XIRECs.
#   If somehow we lose the auction: we pay nothing, lose nothing.
# ─────────────────────────────────────────────────────────────────────────────

MAF_BID = 1_000


# ─── Trader ──────────────────────────────────────────────────────────────────

class Trader:

    def bid(self) -> int:
        """Market Access Fee bid (Round 2 only). Top 50% of bids win."""
        return MAF_BID

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        # ── Load persisted state ──────────────────────────────────────────────
        try:
            saved: dict = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}

        result: Dict[str, List[Order]] = {}
        ts = state.timestamp

        for product, od in state.order_depths.items():
            pos = state.position.get(product, 0)

            if product == IPR:
                result[product] = self._trade_ipr(od, pos, ts, saved)
            elif product == ACO:
                result[product] = self._trade_aco(od, pos, saved)
            else:
                result[product] = []

        return result, 0, json.dumps(saved)

    # ─── IPR: buy-and-hold trend follower ─────────────────────────────────

    def _trade_ipr(self, od: OrderDepth, pos: int,
                   ts: int, saved: dict) -> List[Order]:
        """
        Hold 80 units all day. Each unit earns ~1000 XIREC from the trend.
        The ~8 XIREC spread cost paid at open recovers within ~8000 timestamps.

        Logic:
          1. Calibrate intercept from the first mid-price seen (once per day).
          2. Sweep all asks ≤ fair + IPR_BUY_TOL (fill fast, pay a little premium).
          3. Park remaining needed quantity at best_bid (passive top-up).
          4. Never sell — the trend makes holding dominant all day.
        """
        orders: List[Order] = []
        needed = IPR_LIMIT - pos
        if needed <= 0:
            return orders  # already maxed — nothing to do

        asks = sorted(od.sell_orders.items())   # ascending price
        bids = sorted(od.buy_orders.items(), reverse=True)

        if not asks and not bids:
            return orders

        # ── Calibrate day intercept ───────────────────────────────────────────
        if ts == 0 or "ipr_int" not in saved:
            # Set intercept so that fair_value(ts) = mid at this moment
            if asks and bids:
                mid = (asks[0][0] + bids[0][0]) / 2.0
            elif asks:
                mid = float(asks[0][0])
            else:
                mid = float(bids[0][0])
            saved["ipr_int"] = mid - IPR_SLOPE * ts

        fair = saved["ipr_int"] + IPR_SLOPE * ts
        thresh = fair + IPR_BUY_TOL

        # ── 1. Aggressive take: sweep all asks ≤ fair + tolerance ────────────
        for ask_px, ask_vol in asks:
            if needed <= 0 or ask_px > thresh:
                break
            qty = min(abs(ask_vol), needed)
            orders.append(Order(IPR, ask_px, qty))
            pos    += qty
            needed -= qty

        # ── 2. Passive top-up: park order at best_bid ─────────────────────
        # Gets filled if any seller arrives willing to sell at that price.
        if needed > 0 and bids:
            best_bot_bid = bids[0][0]
            passive_px = min(best_bot_bid + 1, int(thresh))
            # Don't cross any ask we decided not to take
            lowest_untaken_ask = asks[0][0] if asks else passive_px + 1
            if passive_px < lowest_untaken_ask:
                orders.append(Order(IPR, passive_px, needed))

        return orders

    # ─── ACO: anchored-EMA market maker ───────────────────────────────────

    def _trade_aco(self, od: OrderDepth, pos: int,
                   saved: dict) -> List[Order]:
        """
        Market-make around the stable fair value of 10,000.

        Fair value estimate:
            ema  ← 0.02 × mid + 0.98 × ema     (slow EMA, α=0.02)
            ema  ← 0.99 × ema + 0.01 × 10000   (anchor pull)

        The double update keeps the EMA near 10,000 even if the market
        mid drifts persistently above or below (which happens on some days).
        Without the anchor, a fast EMA like v9's α=0.10 follows the mid,
        causing the algorithm to stop taking mispriced quotes.

        TAKE phase:  Sweep both book levels if they cross fair value.
                     (v9 only swept level 1 — missing ~30% of opportunities.)
        MAKE phase:  Post tight passive quotes inside the bot spread,
                     skewed by inventory to encourage mean-reversion to flat.
        """
        orders: List[Order] = []

        asks = sorted(od.sell_orders.items())           # ascending price
        bids = sorted(od.buy_orders.items(), reverse=True)  # descending price

        if not asks or not bids:
            return orders

        best_bot_ask = asks[0][0]
        best_bot_bid = bids[0][0]
        mid = (best_bot_bid + best_bot_ask) / 2.0

        # ── Update anchored EMA ───────────────────────────────────────────────
        ema = saved.get("aco_ema", ACO_FAIR)
        ema = ACO_ALPHA * mid + (1.0 - ACO_ALPHA) * ema   # slow EMA
        ema = (1.0 - ACO_ANCHOR) * ema + ACO_ANCHOR * ACO_FAIR  # anchor
        saved["aco_ema"] = ema
        fair = ema

        # ── PHASE 1: TAKE ─────────────────────────────────────────────────────
        # Buy everything below fair (guaranteed positive EV at fair ≈ 10000)
        for ask_px, ask_vol in asks:
            if ask_px >= fair or pos >= ACO_LIMIT:
                break
            qty = min(abs(ask_vol), ACO_LIMIT - pos)
            orders.append(Order(ACO, ask_px, qty))
            pos += qty

        # Sell everything above fair
        for bid_px, bid_vol in bids:
            if bid_px <= fair or pos <= -ACO_LIMIT:
                break
            qty = min(abs(bid_vol), ACO_LIMIT + pos)
            orders.append(Order(ACO, bid_px, -qty))
            pos -= qty

        # ── PHASE 2: MAKE ─────────────────────────────────────────────────────
        # Inventory skew: long → lower quotes (encourage selling)
        #                 short → raise quotes (encourage buying)
        skew   = int((pos / ACO_LIMIT) * ACO_SKEW)
        my_bid = int(fair) - 1 - skew
        my_ask = int(fair) + 1 - skew

        # Safety: maintain minimum spread and stay on correct side of fair
        if my_bid >= my_ask:
            my_bid = int(fair) - 1
            my_ask = int(fair) + 1

        # Taper quote size when near position limit
        abs_pos = abs(pos)
        if abs_pos <= ACO_SOFT:
            qsize = ACO_QSIZE
        else:
            taper = (ACO_LIMIT - abs_pos) / max(ACO_LIMIT - ACO_SOFT, 1)
            qsize = max(1, round(ACO_QSIZE * taper))

        # Post bid (passive buy) — only if our bid price crosses into the ask side
        # i.e., my_bid >= best_bot_ask means we'd fill immediately at best_bot_ask
        if pos < ACO_LIMIT and my_bid >= best_bot_ask:
            qty = min(qsize, ACO_LIMIT - pos, abs(od.sell_orders.get(best_bot_ask, -qsize)))
            if qty > 0:
                orders.append(Order(ACO, my_bid, qty))

        # Post ask (passive sell) — only if our ask price crosses into the bid side
        if pos > -ACO_LIMIT and my_ask <= best_bot_bid:
            qty = min(qsize, ACO_LIMIT + pos, od.buy_orders.get(best_bot_bid, qsize))
            if qty > 0:
                orders.append(Order(ACO, my_ask, -qty))

        return orders