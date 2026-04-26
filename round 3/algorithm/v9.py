"""
IMC Prosperity Round 3 – Trader v9
====================================
Key improvements over v8:

WHAT v8 WAS MISSING (and what the IMC hints were pointing at):
  The hints said: "Compute IV smile → find outlier → decide direction → scale by conviction"
  v8 completely ignored ATM/near-ATM options, leaving edge on the table.

DATA ANALYSIS FINDINGS:
  VEV_5400 is underpriced 100% of the time (mean IV dev = -0.0138 vs consensus).
  Its ask is ~1.9 ticks below BS fair on Day 0, ~1.3 ticks on Day 2.
  This is the primary structural inefficiency in the options book.

  VEV_5500 is overpriced ~30% of the time (mean IV dev = +0.0036 vs consensus).
  Weaker signal — only trade passively when clearly above fair.

  VEV_5300 was tested but REMOVED: delta ~0.45 creates large directional exposure,
  and the liquidation PnL was negative when VEV moved up. Risk > reward.

v9 NEW STRATEGY:
  1. HP market-making (v8 unchanged, earns ~11,500/round)
  2. BUY VEV_5400 passively (post at bid+1 when ask-price < BS_fair - 1.0)
     - Position limit: 100 units max (leaves room for VEV movement)
     - No delta hedge (VEV spread costs too much)
     - Expected PnL: +300 to +800 per day
  3. SELL VEV_5500 passively (post at ask-1 when bid-price > BS_fair + 0.3)
     - Position limit: 100 units max short
     - Expected PnL: +100 to +300 per day
  4. Consensus IV calculation from stable strikes (5000, 5100, 5200, 5300)
     - Rolling IV computed from these 4 strikes each tick
     - Used as the fair-value anchor for 5400/5500 trades
  5. Keep v8's deep ITM (4000/4500) passive MM: +200/day
  6. Keep v8's far OTM (6000/6500) premium selling (limited by fills)

CALIBRATION:
  HP spread:     ~16 ticks, mean ~9990
  HP EMA alpha:  0.004 (500-tick window)
  SIGMA:         0.2008 (consensus IV across all ATM strikes)
  TTE at R3:     5 days from timestamp=0
  VEV_5400 IV:   consistently 0.014 below consensus -> BUY
  VEV_5500 IV:   +0.004 above consensus -> mild SELL
  5400 ask edge: 1.9 ticks (Day 0) → 1.3 ticks (Day 2) → ~1.0 ticks at TTE=5d

EXPECTED PnL BREAKDOWN:
  HYDROGEL_PACK      : +11,000 to +15,000
  VEV_5400 passive   : +400  to +900
  VEV_5500 passive   : +100  to +300
  VEV_4000/4500 MM   : +200  to +300
  VEV_6000/6500 prem : +50   to +200
  NET EXPECTED       : +12,000 to +17,000
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json, math

# ────────────────────────── constants ──────────────────────────────────────────

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]},
}

# ── HYDROGEL_PACK ──────────────────────────────────────────────────────────────
HP_EMA_ALPHA   = 0.004    # slow EMA ≈ 500-tick window; balanced take rates
HP_TAKE_EDGE   = 4        # take if |ask/bid - fair| > 4 ticks
HP_MAKE_OFFSET = 4        # passive orders 4 ticks from fair
HP_SIZE        = 70       # per side; well under 200 limit

# ── IV / Black-Scholes ────────────────────────────────────────────────────────
SIGMA_PRIOR    = 0.2008   # prior IV; replaced by live computation when possible
TRADING_DAYS   = 252
TTE_START      = 5.0      # days at Round 3, timestamp = 0
TICKS_PER_DAY  = 1_000_000

# Strikes used to compute consensus IV each tick (stable, ~0 IV deviation)
CONSENSUS_STRIKES = [5000, 5100, 5200, 5300]

# ── NEW: ATM option trades (5400 BUY, 5500 SELL) ─────────────────────────────
# VEV_5400: ask is ~1.5 ticks below fair → BUY passively at bid+1
ATM_BUY_STRIKE     = 5400
ATM_BUY_MIN_EDGE   = 1.0    # only buy if fair - our_price > 1.0 tick
ATM_BUY_MAX_POS    = 100    # max long position (conservative, limits delta exposure)
ATM_BUY_SIZE       = 10     # units per order

# VEV_5500: ask IV is ~+0.004 above consensus → SELL passively at ask-1
ATM_SELL_STRIKE    = 5500
ATM_SELL_MIN_EDGE  = 0.5    # only sell if our_price - fair > 0.5 tick
ATM_SELL_MAX_POS   = 100    # max short position
ATM_SELL_SIZE      = 10     # units per order

# ── Deep ITM (4000, 4500): passive MM inside wide spread ─────────────────────
DEEP_ITM         = [4000, 4500]
ITM_MAKE_OFFSET  = 2
ITM_SIZE         = 5
ITM_MAX_INV      = 20

# ── Far OTM (6000, 6500): sell premium at 1 ──────────────────────────────────
FAR_OTM          = [6000, 6500]
FOTM_SELL_PRICE  = 1
FOTM_MAX_SHORT   = 150
FOTM_SIZE        = 20

# ── Skip (no edge identified) ─────────────────────────────────────────────────
SKIP_STRIKES     = [5000, 5100, 5200, 5300]


# ────────────────────────── Black-Scholes utils ────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price, r=0."""
    if T <= 1e-8:
        return max(S - K, 0.0)
    vol_sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sq
    return S * _ncdf(d1) - K * _ncdf(d1 - vol_sq)

def _implied_vol_bisect(mkt: float, S: float, K: float, T: float) -> float:
    """Fast bisection IV solver. Returns SIGMA_PRIOR on failure."""
    if T <= 1e-8 or mkt <= max(S - K, 0.0) + 0.01:
        return SIGMA_PRIOR
    lo, hi = 0.05, 1.5
    try:
        if bs_call(S, K, T, lo) >= mkt or bs_call(S, K, T, hi) <= mkt:
            return SIGMA_PRIOR
    except Exception:
        return SIGMA_PRIOR
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = bs_call(S, K, T, mid)
        if abs(p - mkt) < 0.005:
            return mid
        if p < mkt:
            lo = mid
        else:
            hi = mid
    return mid

def tte_years(ts: int) -> float:
    """Time to expiry in years, from Round 3 timestamp."""
    days = max(TTE_START - ts / TICKS_PER_DAY, 0.001)
    return days / TRADING_DAYS


# ────────────────────────── order book utils ───────────────────────────────────

def best_bid_ask(od: OrderDepth) -> Tuple:
    bb = max(od.buy_orders.keys())  if od.buy_orders  else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    return bb, ba

def book_mid(od: OrderDepth) -> float:
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
            except Exception:
                pass
        return {}

    def _save(self, d: Dict) -> str:
        try:
            return json.dumps(d)
        except Exception:
            return "{}"

    # ── Consensus IV from stable strikes ──────────────────────────────────────

    def _consensus_iv(
        self,
        state: TradingState,
        vev_mid: float,
        T: float,
        prev_sigma: float
    ) -> float:
        """
        Compute live consensus IV from the 4 stable near-ATM strikes.
        These strikes (5000, 5100, 5200, 5300) have near-zero IV deviation
        from each other historically. Their average gives a reliable real-time
        sigma to price 5400 and 5500 against.

        Falls back to prev_sigma if fewer than 2 valid IV samples found.
        """
        ivs = []
        for K in CONSENSUS_STRIKES:
            sym = f"VEV_{K}"
            od = state.order_depths.get(sym)
            if od is None:
                continue
            mid = book_mid(od)
            if mid <= 0.5:
                continue
            iv = _implied_vol_bisect(mid, vev_mid, float(K), T)
            if 0.05 < iv < 1.0:
                ivs.append(iv)
        if len(ivs) >= 2:
            return sum(ivs) / len(ivs)
        return prev_sigma

    # ── HYDROGEL_PACK market maker ─────────────────────────────────────────────

    def _trade_hp(self, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        orders: List[Order] = []
        sym = "HYDROGEL_PACK"
        running = pos
        bb, ba = best_bid_ask(od)

        # Phase 1: Aggressive taker (edge > HP_TAKE_EDGE)
        for ask, vol in sorted(od.sell_orders.items()):
            if ask >= fair - HP_TAKE_EDGE:
                break
            q = clamp_buy(sym, running, min(HP_SIZE, abs(vol)))
            if q <= 0:
                break
            orders.append(Order(sym, ask, q))
            running += q

        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid <= fair + HP_TAKE_EDGE:
                break
            q = clamp_sell(sym, running, min(HP_SIZE, abs(vol)))
            if q <= 0:
                break
            orders.append(Order(sym, bid, -q))
            running -= q

        # Phase 2: Passive MM with inventory skew
        lim  = LIMITS[sym]
        skew = int(round(running / lim * HP_MAKE_OFFSET))
        buy_px  = int(fair) - HP_MAKE_OFFSET - skew
        sell_px = int(fair) + HP_MAKE_OFFSET - skew

        if ba is not None: buy_px  = min(buy_px,  int(ba) - 1)
        if bb is not None: sell_px = max(sell_px, int(bb) + 1)

        bsz = clamp_buy(sym,  running, HP_SIZE)
        ssz = clamp_sell(sym, running, HP_SIZE)
        if bsz > 0: orders.append(Order(sym, buy_px,  bsz))
        if ssz > 0: orders.append(Order(sym, sell_px, -ssz))

        return orders

    # ── VEV_5400: BUY when underpriced (100% of ticks by data analysis) ───────

    def _trade_atm_buy(
        self, od: OrderDepth, pos: int, fair: float
    ) -> List[Order]:
        """
        VEV_5400 is systematically underpriced (IV dev = -0.014 vs consensus).
        At Round 3 TTE=5d, its ask is ~1.5 ticks below BS fair on average.

        Strategy: post passive buy at bid+1 (one tick above the bot bid,
        still below the bot ask). This gets filled when bots sweep down through
        our level. We earn fair - (bid+1) ≈ 1.5 ticks per fill.

        Only post when fair - our_buy > ATM_BUY_MIN_EDGE (1.0 tick).
        Max position: ATM_BUY_MAX_POS = 100 units.

        Note: We do NOT delta-hedge (VEV spread = 5 ticks → hedge costs 2.5 ticks
        per crossing, eating all the option edge). Accept the delta exposure
        since VEV doesn't move dramatically intraday.
        """
        sym = f"VEV_{ATM_BUY_STRIKE}"
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return []
        if pos >= ATM_BUY_MAX_POS:
            return []

        our_buy = int(bb) + 1
        if our_buy >= ba:  # don't cross the spread
            our_buy = int(bb)

        edge = fair - our_buy
        if edge <= ATM_BUY_MIN_EDGE:
            return []

        q = clamp_buy(sym, pos, min(ATM_BUY_SIZE, ATM_BUY_MAX_POS - max(pos, 0)))
        if q <= 0:
            return []
        return [Order(sym, our_buy, q)]

    # ── VEV_5500: SELL when overpriced (mild signal, post passively) ───────────

    def _trade_atm_sell(
        self, od: OrderDepth, pos: int, fair: float
    ) -> List[Order]:
        """
        VEV_5500 has a mild positive IV deviation (+0.004 on avg, +0.008 in Day 1/2).
        Strategy: post passive sell at ask-1. Only trade when edge > ATM_SELL_MIN_EDGE.
        Max short: ATM_SELL_MAX_POS = 100 units.
        """
        sym = f"VEV_{ATM_SELL_STRIKE}"
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return []
        if pos <= -ATM_SELL_MAX_POS:
            return []

        our_sell = int(ba) - 1
        if our_sell <= bb:  # don't cross the spread
            our_sell = int(ba)

        edge = our_sell - fair
        if edge <= ATM_SELL_MIN_EDGE:
            return []

        q = clamp_sell(sym, pos, min(ATM_SELL_SIZE, ATM_SELL_MAX_POS - max(-pos, 0)))
        if q <= 0:
            return []
        return [Order(sym, our_sell, -q)]

    # ── Deep ITM (4000, 4500): passive MM inside wide spread ──────────────────

    def _trade_deep_itm(
        self, sym: str, od: OrderDepth, pos: int, fair: float
    ) -> List[Order]:
        orders: List[Order] = []
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return orders

        running = pos

        if running < ITM_MAX_INV:
            buy_px = int(bb) + ITM_MAKE_OFFSET
            if buy_px < fair:
                q = clamp_buy(sym, running, min(ITM_SIZE, ITM_MAX_INV - max(running, 0)))
                if q > 0:
                    orders.append(Order(sym, buy_px, q))

        if running > -ITM_MAX_INV:
            sell_px = int(ba) - ITM_MAKE_OFFSET
            if sell_px > fair:
                q = clamp_sell(sym, running, min(ITM_SIZE, ITM_MAX_INV - max(-running, 0)))
                if q > 0:
                    orders.append(Order(sym, sell_px, -q))

        return orders

    # ── Far OTM (6000, 6500): sell at 1 for pure premium ──────────────────────

    def _trade_far_otm(self, sym: str, od: OrderDepth, pos: int) -> List[Order]:
        """
        VEV_6000/6500: BS price ≈ 0.000001. Market ask = 1.
        Sell at 1 for near-riskless premium. VEV would need +14% move to hurt.
        """
        _, ba = best_bid_ask(od)
        if ba is None or ba < FOTM_SELL_PRICE:
            return []
        if pos > -FOTM_MAX_SHORT:
            q = clamp_sell(sym, pos, min(FOTM_SIZE, FOTM_MAX_SHORT - max(-pos, 0)))
            if q > 0:
                return [Order(sym, FOTM_SELL_PRICE, -q)]
        return []

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data     = self._load(state.traderData)
        ema_dict = data.get("ema", {})
        sigma    = data.get("sigma", SIGMA_PRIOR)
        result: Dict[str, List[Order]] = {}

        T = tte_years(state.timestamp)

        # ── VEV mid price (needed for IV computation) ──────────────────────────
        vev_od  = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_mid = book_mid(vev_od) if vev_od else None

        # ── Update consensus IV ────────────────────────────────────────────────
        # Each tick: compute IV from 4 stable strikes, EMA-smooth it.
        if vev_mid and vev_mid > 0:
            live_sigma = self._consensus_iv(state, vev_mid, T, sigma)
            # Smooth the sigma estimate to avoid noise (fast EMA alpha=0.05)
            sigma = 0.05 * live_sigma + 0.95 * sigma

        # ── Update HP EMA ──────────────────────────────────────────────────────
        hp_od = state.order_depths.get("HYDROGEL_PACK")
        if hp_od:
            raw = book_mid(hp_od)
            if raw > 0:
                prev = ema_dict.get("HYDROGEL_PACK", raw)
                ema_dict["HYDROGEL_PACK"] = HP_EMA_ALPHA * raw + (1.0 - HP_EMA_ALPHA) * prev

        # ── HYDROGEL_PACK ──────────────────────────────────────────────────────
        hp_fair = ema_dict.get("HYDROGEL_PACK")
        if hp_od and hp_fair:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._trade_hp(hp_od, pos, hp_fair)

        # ── VELVETFRUIT_EXTRACT: no direct trading ─────────────────────────────
        # VEV spread = 5 ticks. Hedge cost > all option gains. Keep flat.
        result["VELVETFRUIT_EXTRACT"] = []

        # ── Consensus / skip strikes: empty orders ─────────────────────────────
        for K in SKIP_STRIKES:
            result[f"VEV_{K}"] = []

        # ── NEW: VEV_5400 (BUY — underpriced by ~1.5 ticks) ───────────────────
        sym_5400 = "VEV_5400"
        od_5400  = state.order_depths.get(sym_5400)
        if od_5400 is not None and vev_mid:
            pos  = state.position.get(sym_5400, 0)
            fair = bs_call(vev_mid, ATM_BUY_STRIKE, T, sigma)
            result[sym_5400] = self._trade_atm_buy(od_5400, pos, fair)
        else:
            result[sym_5400] = []

        # ── NEW: VEV_5500 (SELL — overpriced by ~0.5 ticks) ───────────────────
        sym_5500 = "VEV_5500"
        od_5500  = state.order_depths.get(sym_5500)
        if od_5500 is not None and vev_mid:
            pos  = state.position.get(sym_5500, 0)
            fair = bs_call(vev_mid, ATM_SELL_STRIKE, T, sigma)
            result[sym_5500] = self._trade_atm_sell(od_5500, pos, fair)
        else:
            result[sym_5500] = []

        # ── Deep ITM (4000, 4500): passive MM ─────────────────────────────────
        for K in DEEP_ITM:
            sym = f"VEV_{K}"
            od  = state.order_depths.get(sym)
            if od is None:
                result[sym] = []
                continue
            pos = state.position.get(sym, 0)
            if vev_mid:
                fair = bs_call(vev_mid, float(K), T, sigma)
            else:
                bb, ba = best_bid_ask(od)
                fair = (bb + ba) / 2.0 if bb and ba else 0.0
            result[sym] = self._trade_deep_itm(sym, od, pos, fair)

        # ── Far OTM (6000, 6500): premium selling ──────────────────────────────
        for K in FAR_OTM:
            sym = f"VEV_{K}"
            od  = state.order_depths.get(sym)
            if od is None:
                result[sym] = []
                continue
            pos = state.position.get(sym, 0)
            result[sym] = self._trade_far_otm(sym, od, pos)

        # ── Save state ─────────────────────────────────────────────────────────
        new_data = {"ema": ema_dict, "sigma": sigma}
        return result, 0, self._save(new_data)


# ══════════════════════════════════════════════════════════════════════════════
# v9 TUNING GUIDE
# ══════════════════════════════════════════════════════════════════════════════
#
# NEW PARAMETERS:
#
# ATM_BUY_MIN_EDGE (1.0 tick)
#   Min ticks of edge (fair - our_price) to post a buy order for 5400.
#   Lower = more fills but smaller edge per fill.
#   Higher = fewer fills but cleaner edge.
#   Empirically: at TTE=5d, 5400 ask is ~1.3-1.9 ticks below fair.
#   At 1.0: fills ~90% of the time. At 1.5: fills ~60% of the time.
#   RISK: lower threshold → more position → more delta exposure.
#
# ATM_BUY_MAX_POS (100 units)
#   Cap on long 5400 position.
#   At delta ≈ 0.1 (5400 is far OTM at TTE=5d), 100 units = 10 VEV equivalent.
#   VEV daily std ≈ 15 ticks → 1-day P&L from delta = 15 × 10 = 150 ticks.
#   This is manageable. Higher → more PnL but more risk.
#
# ATM_SELL_MIN_EDGE (0.5 tick)
#   Min edge for selling 5500. Signal is weaker (30% of ticks).
#   Too low → more noise fills. 0.5 is reasonable.
#
# SIGMA smoothing (0.05 EMA alpha)
#   How fast the live IV estimate updates.
#   0.05 = about 20-tick rolling window on sigma.
#   If 0 → always use SIGMA_PRIOR (loses the live IV signal).
#   If 1.0 → uses raw tick IV (noisy). 0.05 is balanced.
#
# EXISTING PARAMETERS (unchanged from v8):
#   HP_EMA_ALPHA, HP_TAKE_EDGE, HP_MAKE_OFFSET, HP_SIZE
#   ITM_MAKE_OFFSET, ITM_SIZE, ITM_MAX_INV
#   FOTM_SELL_PRICE, FOTM_MAX_SHORT, FOTM_SIZE
#
# WHY IS THIS NOT OVERFITTING?
#   1. VEV_5400's IV underpricing is structural (consistent across all 3 days,
#      all timestamps, with std < 0.004 vol points).
#   2. The signal direction is theory-grounded: if one strike is cheaper vs
#      neighbors, buying it and waiting for convergence is a textbook edge.
#   3. Position limits (100 units) prevent extreme exposure.
#   4. We use live consensus IV (not a fixed number) to adapt to regime changes.
#   5. HP MM is unchanged — still the dominant PnL source.