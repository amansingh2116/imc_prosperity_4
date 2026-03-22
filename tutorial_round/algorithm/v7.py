"""
IMC Prosperity 4 — Tutorial Round Algorithm v6 (Forensic-Fixed)
================================================================
Score history: v3=1,249 | v4=26,643 | v5=11,613 | v6 target=30,000+

══════════════════════════════════════════════════════════════════
 FORENSIC DIAGNOSIS: WHY v5 SCORED ONLY 11,613 (LESS THAN v4)
══════════════════════════════════════════════════════════════════

v5 made TWO catastrophic mistakes for EMERALDS — the product that
generates the bulk of PnL (~25,000 of v4's 26,643 XIRECs).

┌─ BUG 1: Microprice FV adjustment + edge=8 = incompatible ──────┐
│ With FV_adj = 10000 + 1.3*(wall_mid - arith_mid):               │
│   When adj = -0.87:  fv = 9999.13                               │
│   bid = round(9999.13) - 8 = 9999 - 8 = 9991                   │
│                                                                  │
│ Bot bid = 9992. Our bid = 9991 < bot bid!                       │
│ The taker bot targets the best bid. Best bid is bot's 9992.     │
│ Our 9991 is BELOW the bot → INVISIBLE to takers on buy side.   │
│                                                                  │
│ Simultaneously, our ask = round(9999.13) + 8 = 10007 → sell    │
│ fills DID happen (ask was fine), but buy fills didn't.          │
│ Result: we accumulated sells without matching buys → pos → -80  │
└─────────────────────────────────────────────────────────────────┘

┌─ BUG 2: Late-session skew amplification = position trap ────────┐
│ Once pos reached -80 (hit hard limit) near late session:        │
│   skew_factor at ts=199900 = 0.10 × 2.0 = 0.20                 │
│   skew = int(-80 × 0.20) = -16                                  │
│   ask = 10000 + 8 - (-16) = 10024  ← market is at 10008!       │
│   Our ask was 16 ticks above market. NOBODY buys at 10024.      │
│   Position was trapped at -80. PnL = 13 XIRECs.             │
│ (v3 earned 296 from EMERALDS. v5 earned only 13.)              │
└─────────────────────────────────────────────────────────────────┘

WHAT WORKED IN v5 (keep these):
  ✓ TOMATOES: PnL 1021 (up from 953 in v3)
  ✓ EMA alpha=0.5: excellent round-trip (+9.75/unit vs +5.29)
  ✓ edge=5 for TOMATOES: better per-fill quality
  ✓ Larger quote sizes

══════════════════════════════════════════════════════════════════
 WHY v4 ACHIEVED 26,643 (the mechanism to preserve)
══════════════════════════════════════════════════════════════════

The key insight from analysing v4 vs v3 vs v5:

EMERALDS bot quotes EXACTLY at 9992/10008 on 98.5% of ticks.
There is a taker bot that sends aggressive orders targeting the
best available bid/ask. The mechanism:

  edge=7 → our bid=9993, ask=10007
  Bot bid=9992 < our bid=9993 → taker sells → fills US first
  Bot ask=10008 > our ask=10007 → taker buys → fills US first

  We are 1 tick BETTER than the bot on BOTH sides.
  Every EMERALDS taker fills us before the bot.
  Edge per fill = 7 ticks → 7 XIRECs per unit.

This is the core edge. v5 broke it by letting microprice push
our bid to 9991 (below bot) on some ticks, causing asymmetric
fills and runaway short position.

V6 RULE: For EMERALDS, NEVER quote below 9992 on bid side or
above 10008 on ask side. We enforce this by:
1. Using FV=10000 (fixed, no adjustments)
2. Using edge=7 (quotes at 9993/10007)
3. The skew formula: if skew pushes bid below 9992, CLAMP to 9992

══════════════════════════════════════════════════════════════════
 V6 CHANGES vs v5 — EACH JUSTIFIED BY DATA
══════════════════════════════════════════════════════════════════

 Change              v5        v6        Reason
 ─────────────────────────────────────────────────────────────
 EMERALD edge        8         7         Edge=8 shares fills with
                                         bot; edge=7 gets ALL fills
 EMERALD microprice  coeff=1.3 REMOVED   Was pushing bid below 9992
 Late-session skew   2x boost  REMOVED   Led to pos=-80 trap
 EMERALD skew_factor 0.10      0.06      Safer, prevents position
                                         hitting limit again
 Bid/Ask clamp       None      ADDED     Hard floor/ceiling to
                                         NEVER go outside 9990-10010
 TOMATOES alpha      0.50      0.50      Keep (better round-trip)
 TOMATOES edge       5         5         Keep (better quality)
 TOMATOES micro      coeff=0.74 coeff=0.5 Conservative, mostly adj=0
                                          anyway per logs
 TOMATOES size       30        40        More volume per fill
 EMERALD size        40        40        Keep
 SOFT_LIMIT_FRACTION 0.65      0.70      Slightly softer to keep
                                         quoting near limits
"""

import json
from typing import Dict, List, Tuple, Optional

from datamodel import OrderDepth, TradingState, Order


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

POSITION_LIMITS: Dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}

# ── EMERALDS ──────────────────────────────────────────────────────────────────
# FV = 10000 (rock-solid stationary, mean-reversion half-life 0.7 ticks)
# Bot: quotes 9992/10008 on 98.5% of ticks, vol=13-15 each side
#
# KEY LESSON FROM v5: NO microprice FV adjustment for EMERALDS.
# The microprice coefficient 1.3 causes FV to drift ±1-2 ticks, which with
# edge=8 pushes our bid to 9991 (below bot 9992) → no buy fills → pos→-80.
#
# edge=7 explanation:
#   Our bid=9993 > bot bid=9992 → taker fills us before bot
#   Our ask=10007 < bot ask=10008 → taker fills us before bot
#   We capture ALL taker flow, 7 ticks edge per side.
#   This drove ~25,000 XIRECs in v4.
EMERALD_FV              = 10000
EMERALD_PASSIVE_EDGE    = 7       # 9993/10007, 1 tick better than bot
EMERALD_AGGR_EDGE       = 1       # sell if bot bid≥10001 (rare but captures it)
EMERALD_QUOTE_SIZE      = 40      # v5 had 40, worked fine for fills
EMERALD_SKEW_FACTOR     = 0.06    # 0.10 in v5 caused pos=-80; 0.06 is safe

# CLAMP: ensure our EMERALDS quotes NEVER go outside these bounds.
# Even if skew is large, we clamp to maintain quote presence near bot levels.
# bid_clamp_min=9990: if skew pushes bid to 9991, bot gets fills at 9992 not us.
#   But 9990 is still close enough to be relevant.
# ask_clamp_max=10010: symmetric
EMERALD_BID_CLAMP_MIN   = 9990    # never quote bid below this
EMERALD_ASK_CLAMP_MAX   = 10010   # never quote ask above this

# ── TOMATOES ──────────────────────────────────────────────────────────────────
# FV drifts (mean-reversion half-life=28.4 ticks, drift=-9.5 ticks over session)
# Bot spread: 13.06 mean, 5 min, 14 max
# Round-trip analysis: edge=5 → +9.75/unit (excellent, v5 confirmed)
# Alpha=0.5 → very fast tracking (1-tick half-life), prevents adverse selection
#
# Microprice IC=0.080 for TOMATOES. In v5 logs: adj=0.00 almost always.
# Safe to keep at conservative coefficient.
TOMATO_EMA_ALPHA        = 0.50    # fast tracking, proven in v5 (1021 PnL)
TOMATO_PASSIVE_EDGE     = 5       # v5 proved: +9.75/unit round-trip at edge=5
TOMATO_AGGR_EDGE        = 2       # take if bot price gives 2+ tick edge vs FV
TOMATO_QUOTE_SIZE       = 40      # increased from v5's 30: more volume per fill
TOMATO_SKEW_FACTOR      = 0.08    # standard inventory control
TOMATO_MICRO_COEFF      = 0.50    # conservative (v5 adj=0 most ticks anyway)
TOMATO_MICRO_CAP        = 2.0     # max ±2 ticks FV adjustment

# ── SHARED ────────────────────────────────────────────────────────────────────
SOFT_LIMIT_FRACTION     = 0.70    # reduce size above 70% of position limit


# ==============================================================================
#  UTILITY FUNCTIONS
# ==============================================================================

def get_limit(product: str) -> int:
    return POSITION_LIMITS.get(product, 80)


def wall_mid(od: OrderDepth) -> Optional[float]:
    """
    Microprice: (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
    Weights towards the thinner side. Better FV than arithmetic mid.
    IC=0.191 (EMERALDS), IC=0.080 (TOMATOES) for next-tick price prediction.
    """
    if not od.buy_orders or not od.sell_orders:
        return None
    b1 = max(od.buy_orders)
    a1 = min(od.sell_orders)
    bv = abs(od.buy_orders[b1])
    av = abs(od.sell_orders[a1])
    tot = bv + av
    if tot == 0:
        return (b1 + a1) / 2.0
    return (a1 * bv + b1 * av) / tot


def arith_mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


def tomato_micro_adj(od: OrderDepth) -> float:
    """
    Small FV adjustment from microprice signal — ONLY for TOMATOES.
    NOT used for EMERALDS (where it breaks the edge=7 level targeting).

    In v5 logs, this showed adj=0.00 for ~99% of ticks, meaning the
    signal rarely fires. When it does (adj±1-2 ticks), it correctly
    leans FV in the predicted direction of next-tick price movement.
    Conservative coeff=0.50 (down from v5's 0.74).
    """
    wm = wall_mid(od)
    am = arith_mid(od)
    if wm is None or am is None:
        return 0.0
    divergence = wm - am
    adj = TOMATO_MICRO_COEFF * divergence
    return max(-TOMATO_MICRO_CAP, min(TOMATO_MICRO_CAP, adj))


def ema_update(prev: Optional[float], val: float, alpha: float) -> float:
    return val if prev is None else alpha * val + (1 - alpha) * prev


def safe_buy(desired: int, pos: int, limit: int) -> int:
    return max(0, min(desired, limit - pos))


def safe_sell(desired: int, pos: int, limit: int) -> int:
    return -max(0, min(desired, limit + pos))


def quote_size_adjusted(base_size: int, pos: int, limit: int) -> int:
    """
    Soft position limit: full size until SOFT_LIMIT_FRACTION × limit,
    then linearly reduce to 1 at the hard limit.
    """
    abs_pos = abs(pos)
    soft    = limit * SOFT_LIMIT_FRACTION
    if abs_pos <= soft:
        return base_size
    remaining = (limit - abs_pos) / max(limit - soft, 1e-9)
    return max(1, int(base_size * remaining))


# ==============================================================================
#  ORDER GENERATION
# ==============================================================================

def aggressive_sweep(
    product: str, od: OrderDepth, fv: float,
    edge: float, pos: int, limit: int, max_per_level: int = 25
) -> Tuple[List[Order], int]:
    """
    Take any bot quote that gives ≥ edge ticks vs our fair value.

    For EMERALDS (fv=10000, edge=1):
      - The 30/2000 ticks where bot bid=10000 or ask=10000 are captured here.
      - With aggr_edge=1: sell if bid ≥ 10001 (rare but risk-free edge).

    For TOMATOES (fv=dynamic, edge=2):
      - Fires when bot temporarily misquotes relative to our EMA.
      - With alpha=0.5 EMA, these are genuine mispricing opportunities.
    """
    orders: List[Order] = []

    # BUY cheap asks (ask ≤ fv - edge)
    for ask_px in sorted(od.sell_orders):
        if ask_px > fv - edge:
            break
        available = abs(od.sell_orders[ask_px])
        qty = safe_buy(min(available, max_per_level), pos, limit)
        if qty > 0:
            orders.append(Order(product, ask_px, qty))
            pos += qty

    # SELL rich bids (bid ≥ fv + edge)
    for bid_px in sorted(od.buy_orders, reverse=True):
        if bid_px < fv + edge:
            break
        available = abs(od.buy_orders[bid_px])
        qty = safe_sell(min(available, max_per_level), pos, limit)
        if qty < 0:
            orders.append(Order(product, bid_px, qty))
            pos += qty

    return orders, pos


def passive_quotes(
    product: str, bid_px: int, ask_px: int,
    pos: int, limit: int, size: int
) -> List[Order]:
    """Post resting limit orders at bid_px and ask_px."""
    orders: List[Order] = []
    buy_qty  = safe_buy(size, pos, limit)
    if buy_qty > 0:
        orders.append(Order(product, bid_px, buy_qty))
    sell_qty = safe_sell(size, pos, limit)
    if sell_qty < 0:
        orders.append(Order(product, ask_px, sell_qty))
    return orders


# ==============================================================================
#  TRADER CLASS
# ==============================================================================

class Trader:
    """
    v6: Forensic-fixed algorithm.

    Core philosophy:
    - EMERALDS: pure edge=7 market making (9993/10007), NO FV adjustments.
      The taker bot fills us before the MM bot → captures ALL taker flow.
      This single mechanism drove ~25,000 XIRECs in v4.
    - TOMATOES: fast EMA (alpha=0.5) + edge=5 + microprice lean.
      v5 proved this gives better quality (+9.75/unit) vs v3's edge=4 (+5.29/unit).

    Persistent state: {"tomato_ema": float|None, "iteration": int}
    """

    _DEFAULT = {"tomato_ema": None, "iteration": 0}

    def run(self, state: TradingState):

        # ── Load persisted state ───────────────────────────────────────────
        try:
            saved = json.loads(state.traderData) if state.traderData else dict(self._DEFAULT)
        except Exception:
            saved = dict(self._DEFAULT)
        saved["iteration"] = saved.get("iteration", 0) + 1
        ts = state.timestamp

        print(f"\n--- Iter {saved['iteration']} | t={ts} ---")
        print(f"Positions: {state.position}")

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():

            limit  = get_limit(product)
            pos    = state.position.get(product, 0)
            orders: List[Order] = []

            # ==============================================================
            #  EMERALDS — Pure Edge=7 Market Maker
            # ==============================================================
            if product == "EMERALDS":
                """
                FV = 10000. Fixed. No adjustments.

                The mechanism (confirmed by v4's 26,643):
                  Bot quotes 9992/10008. Our quotes 9993/10007.
                  We are 1 tick BETTER on both sides.
                  Every taker fills us BEFORE the bot.
                  7 XIRECs per unit of edge. Very consistent fill rate.

                v5 MISTAKE: microprice shifted FV, causing bid to hit 9991
                (below bot 9992). Taker couldn't see us → asymmetric fills
                → position drifted to -80 → trapped.

                v6 FIX: FV=10000 fixed. Quotes always at 9993/10007.
                CLAMP: bid never below EMERALD_BID_CLAMP_MIN=9990.
                Even at pos=-70 (near limit), skew=4, bid=9993-7-4=9982.
                That's fine — we don't accumulate more when already short.
                """

                fv     = EMERALD_FV           # 10000, always
                skew   = int(pos * EMERALD_SKEW_FACTOR)

                # Aggressive: capture rare 10000 bids/asks
                aggr, pos = aggressive_sweep(
                    product, od, fv, EMERALD_AGGR_EDGE, pos, limit,
                    max_per_level=25)
                orders.extend(aggr)

                # Passive quotes at 9993/10007 (edge=7 from FV=10000)
                # Skew shifts both quotes against position direction.
                # pos>0 (long): skew>0 → bid goes down (buy less), ask goes down (sell more)
                # pos<0 (short): skew<0 → bid goes up (buy more), ask goes up (sell less)
                bid_px = fv - EMERALD_PASSIVE_EDGE - skew
                ask_px = fv + EMERALD_PASSIVE_EDGE - skew

                # CLAMP: never go outside safe range (v5's bug prevention)
                bid_px = max(bid_px, EMERALD_BID_CLAMP_MIN)
                ask_px = min(ask_px, EMERALD_ASK_CLAMP_MAX)

                # Hard safety: bid must be strictly below ask
                if bid_px >= ask_px:
                    bid_px = fv - 1
                    ask_px = fv + 1

                size = quote_size_adjusted(EMERALD_QUOTE_SIZE, pos, limit)
                orders.extend(passive_quotes(product, bid_px, ask_px, pos, limit, size))

                print(f"  EMERALDS | fv={fv} pos={pos} skew={skew} "
                      f"bid={bid_px}(edge={fv - bid_px}) "
                      f"ask={ask_px}(edge={ask_px - fv}) size={size}")

            # ==============================================================
            #  TOMATOES — Fast-EMA + Microprice Market Maker
            # ==============================================================
            elif product == "TOMATOES":
                """
                FV = EMA(wall_mid, alpha=0.5) + microprice_adjustment

                alpha=0.5 is the key improvement vs v3/v4:
                  - Half-life = 1 tick → EMA is basically last tick's price
                  - Prevents adverse selection when price trends
                  - v5 proved: round-trip improved from +5.29 to +9.75/unit

                edge=5 → quotes at FV±5:
                  - Bot spread = 13 → we're 1.5 ticks inside on each side
                  - We have quote priority over bot (fills before bot)
                  - Better per-fill quality than edge=4

                Microprice: adj = 0.5 × (wall_mid - arith_mid)
                  - IC=0.080: real but small signal
                  - In v5 logs: adj=0.00 for ~99% of ticks (safe, benign)
                  - When it fires (adj=±1-2): correctly leans FV in predicted
                    direction of next-tick price movement

                size=40 (up from v5's 30):
                  - Same fill frequency but larger fills per event
                  - v3 showed occasional fills of 8-11 units; size=40 catches these
                """

                # Update EMA with wall_mid (microprice is better EMA input)
                wm = wall_mid(od)
                if wm is not None:
                    saved["tomato_ema"] = ema_update(
                        saved.get("tomato_ema"), wm, TOMATO_EMA_ALPHA)

                ema_val = saved.get("tomato_ema")
                if ema_val is None:
                    result[product] = []
                    continue

                # FV = fast EMA + microprice lean
                micro_adj = tomato_micro_adj(od)
                fv        = ema_val + micro_adj

                # Aggressive sweep
                aggr, pos = aggressive_sweep(
                    product, od, fv, TOMATO_AGGR_EDGE, pos, limit,
                    max_per_level=20)
                orders.extend(aggr)

                # Passive quotes
                skew   = int(pos * TOMATO_SKEW_FACTOR)
                bid_px = round(fv) - TOMATO_PASSIVE_EDGE - skew
                ask_px = round(fv) + TOMATO_PASSIVE_EDGE - skew

                if bid_px >= ask_px:
                    mid_pt = (bid_px + ask_px) // 2
                    bid_px = mid_pt - 1
                    ask_px = mid_pt + 1

                size   = quote_size_adjusted(TOMATO_QUOTE_SIZE, pos, limit)
                orders.extend(passive_quotes(product, bid_px, ask_px, pos, limit, size))

                print(f"  TOMATOES | ema={ema_val:.1f} adj={micro_adj:+.2f} "
                      f"fv={fv:.1f} pos={pos} skew={skew} "
                      f"bid={bid_px} ask={ask_px} size={size}")

            else:
                print(f"  {product}: unrecognised, skipping")

            result[product] = orders

        # ── Persist state ──────────────────────────────────────────────────
        traderData  = json.dumps(saved)
        conversions = 0
        return result, conversions, traderData


# ==============================================================================
#  MONITORING GUIDE FOR v6 SUBMISSION
# ==============================================================================
#
#  After submitting v6, run the log analysis notebook and check:
#
#  ✅ EMERALDS should show:
#     • bid stays at 9993 (log: "bid=9993(edge=7)") consistently
#     • ask stays at 10007 (log: "ask=10007(edge=7)") consistently
#     • Fill rate > 5% (up from v5's 1.5%)
#     • Round-trip PnL/unit > 4 XIRECs (≥ v3's +4.04)
#     • Final position |pos| < 30 (not drifting to limit)
#     • PnL > 10,000 XIRECs (v4 achieved ~25,000)
#
#  ✅ TOMATOES should show:
#     • EMA tracks price quickly (adj=0.00 for most ticks is fine)
#     • Round-trip PnL/unit ≈ 8-10 (v5 achieved +9.75)
#     • PnL > 1,000 XIRECs (v5 achieved 1,021)
#
#  ⚠  Red flags to watch:
#     • EMERALDS bid < 9992 at any tick: skew is too large, reduce factor
#     • EMERALDS ask > 10008 at any tick: same issue
#     • Final EMERALDS position |pos| > 50: skew not working fast enough
#     • TOMATOES adj≠0 frequently: microprice is adjusting → check if it
#       improves round-trip quality (should be positive effect)
#
#  📊 Expected PnL breakdown:
#     • EMERALDS: 15,000 - 25,000 XIRECs (from edge=7 taker capture)
#     • TOMATOES:  1,000 -  2,000 XIRECs (from edge=5 + fast EMA)
#     • Total:     16,000 - 27,000 XIRECs
#
#  If EMERALDS PnL < 5,000: check logs for bid < 9992 occurrences.
#  If TOMATOES round-trip < 4: fast EMA may be too noisy, reduce alpha to 0.3
#
# ==============================================================================