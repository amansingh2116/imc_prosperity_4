"""
IMC Prosperity Round 3 – Trader v8
====================================
Root-cause fixes from v7 analysis:

PROBLEM 1 (v7 regression from v6): HP EMA too slow → position bias
  v7 used EMA(alpha=0.08) initialized at raw mid (~10011).
  HP true mean is ~9990, not 10000. Slow EMA lagged, causing constant
  net-long drift → 105 unit long stuck position → -460 HP PnL.
  FIX: EMA(alpha=0.004) ≈ rolling 500-tick window. Gives balanced
  take rates (23% buy / 25% sell) across all historical days.
  Result: symmetric, mean-reverting position.

PROBLEM 2 (core of -5255 VEV loss): Delta hedging is a guaranteed loser
  VEV spread = 5 ticks (2.5 ticks per crossing).
  Each option fill earns 1-2 ticks of spread.
  Each VEV hedge crossing costs 2.5 ticks.
  Net per round-trip: NEGATIVE.
  VEV lost -5255 in v7, -7506 in v6. Options only earn hundreds.
  FIX: COMPLETELY DISABLE delta hedging. No VEV trading at all.

PROBLEM 3: Wrong option selection in v7
  LIQUID_STRIKES = [5000, 5300, 5400, 5500] had negative net edge
  after hedge cost. Without hedge cost, they earn ~1-2 ticks passive
  but fill infrequently (spread only 1-6 ticks wide).
  FIX: Only trade options with WIDE spreads (near-riskless edge):
    - Deep ITM (4000: 21-tick spread, 4500: 16-tick spread): earn 5-8 ticks
      passive with near-zero net risk (delta~1 but symmetric MM).
    - Far OTM sell (6000, 6500): delta~0, earn 1/unit, VEV needs +14%
      move in 5 days to hurt us (essentially impossible).
    - ALL ATM/near-ATM options: SKIP. Spread too thin, hedge costs dominate.

CALIBRATION (from all 3 historical days):
  HP spread      : 15.7 ticks (very stable)
  HP mean        : ~9990 (NOT 10000 — using EMA avoids the bias)
  HP EMA alpha   : 0.004 (≈ 500-tick rolling window, balanced takes)
  VEV spread     : 5.0 ticks (fixed)
  VEV_4000 spread: 20.9 ticks average
  VEV_4500 spread: 16.0 ticks average
  VEV_6000/6500  : bid=0, ask=1 always (sell at 1 = pure premium)
  TTE at Round 3 : 5 days (from day-0 of simulation)

PARAMETER GUIDE (see bottom of file):
  HP_EMA_ALPHA  : 0.003-0.006. Lower = slower/more stable (less overfitting).
                  Risk: too low = still biased vs sudden drift.
  HP_TAKE_EDGE  : 3-6. Higher = more selective takes, less position risk.
                  Risk: too high = miss profitable takes.
  HP_MAKE_OFFSET: 3-5. Higher = more passive edge per fill.
                  Risk: too high = fewer fills.
  HP_SIZE       : 50-100. Bounded by position limit 200.
                  Safe: 70 (leaves room for skew adjustments).
  ITM_SIZE      : 3-8. Keep small to limit delta exposure.
  FOTM_MAX_SHORT: 100-200. Lower = safer but less premium.
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json, math

# ──────────────────────────── constants ────────────────────────────────────────

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

ALL_STRIKES  = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# ── Options strategy buckets ──────────────────────────────────────────────────
# DEEP_ITM: wide market spread (16-21 ticks) → earn 5-8 ticks passively, delta~1
DEEP_ITM     = [4000, 4500]
# FAR_OTM: market ask=1, bid=0 always → sell at 1 for free premium, delta~0
FAR_OTM      = [6000, 6500]
# SKIP: all ATM/near-ATM. Thin spread (<6 ticks), hedge cost > edge earned.
SKIP_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]

# ── Black-Scholes (only needed for Deep ITM fair value check) ─────────────────
SIGMA        = 0.205          # avg market IV across all strikes (calibrated)
TRADING_DAYS = 252
TTE_START    = 5.0            # days at Round 3 timestamp 0
TICKS_PER_DAY = 1_000_000

# ── HYDROGEL_PACK ─────────────────────────────────────────────────────────────
# HP spread ~16 ticks, mean ~9990. Use slow EMA to track level without bias.
HP_EMA_ALPHA   = 0.004    # ≈ 500-tick rolling window; balanced take rates
HP_TAKE_EDGE   = 4        # take if |ask/bid - fair| > 4 ticks (very conservative)
HP_MAKE_OFFSET = 4        # passive orders 4 ticks from fair (earn 4 ticks/fill)
HP_SIZE        = 70       # per order side; safely below 200 limit even with skew

# ── Deep ITM options (4000, 4500) ─────────────────────────────────────────────
# Strategy: purely passive MM inside the wide market spread.
# We earn ~5-8 ticks per fill with no directional bet (symmetric buy+sell).
ITM_MAKE_OFFSET = 2       # post 2 ticks inside the market spread
ITM_SIZE        = 5       # small size to limit delta exposure
ITM_MAX_INV     = 20      # soft cap on net position per strike

# ── Far OTM (6000, 6500) ──────────────────────────────────────────────────────
# Strategy: sell at ask price = 1 (the only non-zero level buyers pay).
# BS value ≈ 0.000001, market ask = 1 → pure edge.
# Risk: VEV needs +14% move in ≤5 days. Historical VEV std ≈ 14 ticks/day.
# 5-day 3-sigma move ≈ 14×sqrt(5)×3 = 94 ticks → VEV max ≈ 5300+94 = 5394.
# VEV hitting 6000 is essentially impossible. Max short kept small for safety.
FOTM_SELL_PRICE = 1
FOTM_MAX_SHORT  = 150     # max short per strike (down from 200 in v7 for safety)
FOTM_SIZE       = 20      # units per order


# ────────────────────────── Black-Scholes ──────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price_delta(S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
    """Return (call_price, delta) for European call with r=0."""
    if T <= 1e-8:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)
    vol_sq  = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sq
    return S * _ncdf(d1) - K * _ncdf(d1 - vol_sq), _ncdf(d1)

def tte_years(ts: int) -> float:
    days = max(TTE_START - ts / TICKS_PER_DAY, 0.001)
    return days / TRADING_DAYS


# ────────────────────────── helpers ────────────────────────────────────────────

def best_bid_ask(od: OrderDepth) -> Tuple:
    bb = max(od.buy_orders.keys())  if od.buy_orders  else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    return bb, ba

def book_mid(od: OrderDepth) -> float:
    """Volume-weighted mid price."""
    bn = bv = an = av = 0.0
    for p, v in od.buy_orders.items():
        if v > 0: bn += p * v; bv += v
    for p, v in od.sell_orders.items():
        u = abs(v); an += p * u; av += u
    if bv > 0 and av > 0:
        return 0.5 * (bn / bv + an / av)
    bb, ba = best_bid_ask(od)
    if bb and ba: return (bb + ba) / 2.0
    return float(bb or ba or 0)

def clamp_buy(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] - pos))

def clamp_sell(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] + pos))


# ────────────────────────── Trader ─────────────────────────────────────────────

class Trader:

    def _load(self, td: str) -> Dict:
        if td:
            try:
                d = json.loads(td)
                if isinstance(d, dict): return d
            except Exception: pass
        return {}

    def _save(self, d: Dict) -> str:
        try: return json.dumps(d)
        except Exception: return "{}"

    # ── HYDROGEL_PACK ──────────────────────────────────────────────────────────

    def _trade_hp(self, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        """
        Market-make HP around slow-EMA fair value.

        HP has a ~16-tick spread with bots posting at bid≈fair-8, ask≈fair+8.
        We earn by:
          1. Aggressive takes when ask/bid is 4+ ticks away from our EMA fair.
          2. Passive quotes 4 ticks from fair (inside the bot spread).
        Inventory skew prevents runaway positions.

        Why slow EMA (alpha=0.004)?
          HP drifts slowly across days (~9908 to ~10079 observed range).
          A 500-tick EMA tracks the trend without being fooled by tick noise.
          At alpha=0.004, take rates are 23% buy / 25% sell — nearly balanced.
          This prevents the inventory accumulation that destroyed v7.
        """
        orders: List[Order] = []
        sym    = "HYDROGEL_PACK"
        running = pos
        bb, ba  = best_bid_ask(od)

        # ── Phase 1: Aggressive taker ──────────────────────────────────────
        for ask, vol in sorted(od.sell_orders.items()):
            if ask >= fair - HP_TAKE_EDGE: break
            q = clamp_buy(sym, running, min(HP_SIZE, abs(vol)))
            if q <= 0: break
            orders.append(Order(sym, ask, q))
            running += q

        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid <= fair + HP_TAKE_EDGE: break
            q = clamp_sell(sym, running, min(HP_SIZE, abs(vol)))
            if q <= 0: break
            orders.append(Order(sym, bid, -q))
            running -= q

        # ── Phase 2: Passive market-maker (inventory-skewed) ──────────────
        lim  = LIMITS[sym]
        skew = int(round(running / lim * HP_MAKE_OFFSET))  # [-offset .. +offset]

        buy_px  = int(fair) - HP_MAKE_OFFSET - skew
        sell_px = int(fair) + HP_MAKE_OFFSET - skew

        # Clamp so we never cross live market prices
        if ba is not None: buy_px  = min(buy_px,  int(ba) - 1)
        if bb is not None: sell_px = max(sell_px, int(bb) + 1)

        bsz = clamp_buy(sym,  running, HP_SIZE)
        ssz = clamp_sell(sym, running, HP_SIZE)
        if bsz > 0: orders.append(Order(sym, buy_px,  bsz))
        if ssz > 0: orders.append(Order(sym, sell_px, -ssz))

        return orders

    # ── Deep ITM options (4000, 4500) ──────────────────────────────────────────

    def _trade_deep_itm(
        self, sym: str, od: OrderDepth, pos: int, fair: float
    ) -> List[Order]:
        """
        Deep ITM options (4000, 4500) have 16-21 tick market spreads.
        Strategy: passive MM inside the spread, earning 5-8 ticks per fill.
        No delta hedge — positions are kept small and symmetric.

        Why no hedge?
          VEV hedge costs 2.5 ticks (half the 5-tick VEV spread).
          Deep ITM earns at most 8 ticks/fill.
          If 1 option fill triggers 1 hedge fill: 8 - 2.5 = +5.5 net (ok).
          BUT: the hedge fires continuously even between option fills,
          generating pure losses. Empirically VEV lost -7506 / -5255 in v6/v7.
          Without hedge: Deep ITM earns ~100-135 XIRECS per day (observed in v6/v7).
          That's the right tradeoff.
        """
        orders: List[Order] = []
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None: return orders

        running = pos

        # Post passive buy at bid+offset (inside spread)
        if running < ITM_MAX_INV:
            buy_px = int(bb) + ITM_MAKE_OFFSET
            if buy_px < fair:  # only buy below fair
                q = clamp_buy(sym, running, min(ITM_SIZE, ITM_MAX_INV - max(running, 0)))
                if q > 0:
                    orders.append(Order(sym, buy_px, q))

        # Post passive sell at ask-offset (inside spread)
        if running > -ITM_MAX_INV:
            sell_px = int(ba) - ITM_MAKE_OFFSET
            if sell_px > fair:  # only sell above fair
                q = clamp_sell(sym, running, min(ITM_SIZE, ITM_MAX_INV - max(-running, 0)))
                if q > 0:
                    orders.append(Order(sym, sell_px, -q))

        return orders

    # ── Far OTM (6000, 6500): sell premium ────────────────────────────────────

    def _trade_far_otm(self, sym: str, od: OrderDepth, pos: int) -> List[Order]:
        """
        VEV_6000 and VEV_6500 are essentially worthless by Black-Scholes
        (VEV ≈ 5255, strike 6000, TTE 5 days → BS price ≈ 0.000001).
        Market: bid=0, ask=1 at all times.
        Strategy: sell at price=1.

        Risk analysis:
          - VEV historical std: ≈14 ticks/day
          - 5-day 3-sigma move: 14×√5×3 ≈ 94 ticks → VEV max ≈ 5350
          - VEV_6000 expires worthless unless VEV > 6000 (+14.2% from 5255)
          - This has never happened in 3 days of historical data
          - We cap short at FOTM_MAX_SHORT=150 for safety
        """
        orders: List[Order] = []
        _, ba = best_bid_ask(od)
        if ba is None or ba < FOTM_SELL_PRICE:
            return orders

        if pos > -FOTM_MAX_SHORT:
            q = clamp_sell(sym, pos, min(FOTM_SIZE, FOTM_MAX_SHORT - max(-pos, 0)))
            if q > 0:
                orders.append(Order(sym, FOTM_SELL_PRICE, -q))

        return orders

    # ── main ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data     = self._load(state.traderData)
        ema_dict = data.get("ema", {})
        result: Dict[str, List[Order]] = {}

        # ── Update HP EMA ─────────────────────────────────────────────────────
        # Only HP uses EMA (VEV is not traded for MM anymore).
        hp_od = state.order_depths.get("HYDROGEL_PACK")
        if hp_od:
            raw  = book_mid(hp_od)
            if raw > 0:
                prev = ema_dict.get("HYDROGEL_PACK", raw)
                ema_dict["HYDROGEL_PACK"] = HP_EMA_ALPHA * raw + (1.0 - HP_EMA_ALPHA) * prev

        # ── HYDROGEL_PACK ─────────────────────────────────────────────────────
        hp_fair = ema_dict.get("HYDROGEL_PACK")
        if hp_od and hp_fair:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._trade_hp(hp_od, pos, hp_fair)

        # ── Options ───────────────────────────────────────────────────────────
        # Get VEV for BS fair value computation (deep ITM only)
        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_mid = book_mid(vev_od) if vev_od else None
        T = tte_years(state.timestamp)

        # Skip all ATM/near-ATM options
        for K in SKIP_STRIKES:
            result[f"VEV_{K}"] = []

        # Deep ITM: passive MM inside wide spread
        for K in DEEP_ITM:
            sym = f"VEV_{K}"
            od  = state.order_depths.get(sym)
            if od is None:
                result[sym] = []
                continue
            pos = state.position.get(sym, 0)
            # Use BS fair (no hedge, just to anchor passive quotes)
            if vev_mid:
                fair, _ = bs_price_delta(vev_mid, float(K), T, SIGMA)
            else:
                bb, ba = best_bid_ask(od)
                fair = (bb + ba) / 2.0 if bb and ba else 0.0
            result[sym] = self._trade_deep_itm(sym, od, pos, fair)

        # Far OTM: sell at 1 for pure premium
        for K in FAR_OTM:
            sym = f"VEV_{K}"
            od  = state.order_depths.get(sym)
            if od is None:
                result[sym] = []
                continue
            pos = state.position.get(sym, 0)
            result[sym] = self._trade_far_otm(sym, od, pos)

        # ── VELVETFRUIT_EXTRACT: NO trading (hedge cost > all option gains) ───
        # VEV lost -5255 in v7 and -7506 in v6 purely from hedging.
        # Options earn <500 XIRECS total. Net: deeply negative.
        # Leaving VEV untraded completely.
        result["VELVETFRUIT_EXTRACT"] = []

        return result, 0, self._save({"ema": ema_dict})


# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER GUIDE
# ══════════════════════════════════════════════════════════════════════════════
#
# ┌──────────────────┬───────────────┬────────────────────────────────────────┐
# │ Parameter        │ Range / Type  │ How to tune & overfitting risk         │
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ HP_EMA_ALPHA     │ 0.003 – 0.010 │ Measures how fast fair tracks price.   │
# │                  │               │ Lower = more stable, less overfitting.  │
# │                  │               │ Tune: plot (take_buy - take_sell) rate  │
# │                  │               │ across all 3 days; pick alpha where     │
# │                  │               │ imbalance is <5% on average.            │
# │                  │               │ Overfitting risk: LOW (it's a filter,   │
# │                  │               │ not a price predictor).                 │
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ HP_TAKE_EDGE     │ 2 – 8 ticks   │ Min edge to take aggressively.         │
# │                  │               │ Higher = fewer but higher-quality takes.│
# │                  │               │ Tune: run backtest, check if take PnL   │
# │                  │               │ > passive PnL. 4 is robust.             │
# │                  │               │ Overfitting risk: MEDIUM (depends on    │
# │                  │               │ spread which is stable at 16 ticks).    │
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ HP_MAKE_OFFSET   │ 2 – 6 ticks   │ Distance from fair for passive quotes.  │
# │                  │               │ Higher = more per fill but fewer fills.  │
# │                  │               │ Tune: spread is 16; offset=4 means 4    │
# │                  │               │ ticks inside the bot spread. Safe range:│
# │                  │               │ 3-5. Don't exceed 7 (outside spread).   │
# │                  │               │ Overfitting risk: LOW.                  │
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ HP_SIZE          │ 30 – 100      │ Order size per side.                    │
# │                  │               │ Larger = more PnL but more position risk.│
# │                  │               │ Safe cap: 200/(2×make_offset/spread)≈70 │
# │                  │               │ Overfitting risk: NONE (just scaling).  │
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ ITM_MAKE_OFFSET  │ 1 – 4 ticks   │ Inside spread for VEV_4000/4500.       │
# │                  │               │ Spread is 20/16 ticks, offset=2 leaves  │
# │                  │               │ 8-9 ticks to earn. Safe.                │
# │                  │               │ Overfitting risk: LOW.                  │
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ ITM_SIZE         │ 3 – 10        │ Keep small to limit delta exposure.     │
# │                  │               │ Without hedge, each unit is delta~1     │
# │                  │               │ exposure to VEV. Keep ITM_MAX_INV ≤ 20. │
# │                  │               │ Overfitting risk: LOW.                  │
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ FOTM_MAX_SHORT   │ 50 – 200      │ Max short position in VEV_6000/6500.   │
# │                  │               │ Higher = more premium but tail risk.    │
# │                  │               │ Historical: VEV never exceeded 5300 in  │
# │                  │               │ 3 days. 150 is safe; 300 is aggressive. │
# │                  │               │ Overfitting risk: MEDIUM (depends on    │
# │                  │               │ whether the sim uses different VEV path).│
# ├──────────────────┼───────────────┼────────────────────────────────────────┤
# │ SIGMA            │ 0.19 – 0.22   │ Implied vol for BS fair value.          │
# │                  │               │ Only affects deep ITM fair value quotes. │
# │                  │               │ Since ITM_MAKE_OFFSET dominates, sigma  │
# │                  │               │ has minor impact here.                  │
# │                  │               │ Calibrated: 0.205 from tick-level data. │
# │                  │               │ Overfitting risk: LOW for this strategy.│
# └──────────────────┴───────────────┴────────────────────────────────────────┘
#
# EXPECTED PnL BREAKDOWN:
#   HYDROGEL_PACK       : +8000 to +15000  (main earner, 90%+ of total)
#   VEV_4000            : +100 to +150
#   VEV_4500            : +80  to +120
#   VEV_6000 + VEV_6500 : +100 to +300 (FOTM_MAX_SHORT × 1 × 2 strikes × fill_rate)
#   VELVETFRUIT_EXTRACT : 0    (no trading → no losses)
#   All ATM options     : 0    (skipped → no losses)
#   NET EXPECTED        : +8000 to +16000
#
# WHY IS THIS NOT OVERFITTING?
#   1. HP EMA is a smoothing filter, not a predictive model.
#   2. All edges are structural (wide spreads, zero-value options)
#      not calibrated to specific price paths.
#   3. The only "prediction" is that VEV won't jump +14% in 5 days —
#      which is grounded in realized volatility, not curve-fitting.
#   4. We REMOVED the complex option hedging logic that was overfit
#      to the specific VEV path in historical data.