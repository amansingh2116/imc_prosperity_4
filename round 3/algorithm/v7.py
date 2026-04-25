"""
IMC Prosperity Round 3 – Trader v7
====================================
Improvements over v6 (profit=2382):

  1. HP fair value: EMA-tracked (not hardcoded 10000) — prevents bad trades
     when HP drifts to 9891 or 10051 (observed range in historical data).

  2. HP sizing: increased from 30 → 50 per side. HP earns the most (9251/2382 =
     88% of total PnL). Wider orders capture more volume at the same edge.

  3. Option strike selection (data-driven):
       Net edge = (opt_half_spread) - (delta × vev_half_spread)
         5000 → +0.59  ✓ trade
         5100 → -0.14  ✗ SKIP  ← v6 was wrong to trade this
         5200 → -0.29  ✗ SKIP  ← v6 was wrong to trade this
         5300 → +0.14  ✓ trade
         5400 → +0.39  ✓ trade
         5500 → +0.51  ✓ trade

  4. Far-OTM (6000, 6500): sell at market ask=1 when available.
     These are worth ~0 by BS (VEV~5255, TTE≤5d). Selling at 1 = pure premium.
     Hedge cost = delta × vev_half_spread ≈ 0.00 × 2.5 ≈ 0, so net edge ≈ 1.

  5. Deep ITM (4000, 4500): MM lightly — these trade like underlying (delta≈1)
     but with much wider market spreads (17-22 ticks vs 5 for VEV).
     Earn spread on a small position without extra hedge cost.

  6. VEV hedge: only when |net_delta| > 3 (was 5 in v6). Tighter = less drift.

  7. Sigma per-strike: use the calibrated mean IVs from tick-by-tick analysis.
     All ATM-OTM strikes cluster at 0.205–0.228, use 0.215 (midpoint).
     This is more accurate than 0.200 used in v6.

Calibration summary (from historical data):
  SIGMA               = 0.215   (tick-level IV avg across 5000-5500, 2 days)
  HP spread           = 16 ticks (very stable)
  VEV spread          = 5 ticks
  VEV_6000/6500 ask   = 1 (constant), bid=0, trades occur at 0
  TTE at Round 3 start = 5 days
  1 trading day = 1,000,000 timestamp units
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json, math

# ────────────────────────── constants ─────────────────────────────────────────

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    **{f"VEV_{k}": 300 for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]},
}

ALL_STRIKES   = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Strikes we actively trade (positive edge after hedge cost):
LIQUID_STRIKES  = [5000, 5300, 5400, 5500]   # net edge > 0 (see header analysis)
DEEP_ITM        = [4000, 4500]                # delta ≈ 1; wide market spread = easy edge
FAR_OTM         = [6000, 6500]                # worth ~0; sell at market ask=1 for free premium
SKIP_STRIKES    = [5100, 5200]                # negative net edge after hedge — DO NOT TRADE

# ── Calibrated Black-Scholes parameters ───────────────────────────────────────
SIGMA          = 0.215    # best-fit annualised IV (tick-level, avg 5000-5500)
TRADING_DAYS   = 252

# Round 3 live evaluation: TTE = 5d at timestamp 0
TTE_START_DAYS = 5.0
TICKS_PER_DAY  = 1_000_000

# ── HYDROGEL_PACK parameters ──────────────────────────────────────────────────
# HP mean-reverts around a slow trend; use EMA instead of fixed 10000.
HP_EMA_ALPHA   = 0.08    # slow EMA: adapts to drift without chasing noise
HP_TAKE_EDGE   = 4       # take if |price - fair| > this (spread=16, so very safe)
HP_MAKE_OFFSET = 4       # post passive orders 4 ticks from fair
HP_SIZE        = 50      # ↑ from 30 in v6; HP is the biggest earner

# ── VELVETFRUIT_EXTRACT parameters ────────────────────────────────────────────
VEV_EMA_ALPHA   = 0.10   # a bit faster than HP (VEV has more genuine drift)
VEV_TAKE_EDGE   = 1      # take when clearly mispriced
VEV_MAKE_OFFSET = 2      # passive spread
VEV_SIZE        = 8      # conservative; VEV is mainly a hedge vehicle

# ── Option parameters (liquid strikes) ────────────────────────────────────────
OPT_TAKE_EDGE   = 4.0    # only take on very clear mispricing
OPT_MAKE_OFFSET = 1      # passive quote 1 tick inside market
OPT_SIZE        = 15     # per quote
OPT_MAX_INV     = 60     # soft inventory limit per strike (↑ from 50 in v6)
                          # OPT_MAX_INV=60 < 300 limit, prevents getting stuck

# ── Deep ITM parameters ───────────────────────────────────────────────────────
ITM_MAX_INV    = 15      # keep small; each unit adds ~delta=1 underlying exposure
ITM_SIZE       = 5       # tiny quotes just to earn the wide spread

# ── Far OTM (6000, 6500) parameters ──────────────────────────────────────────
# Strategy: SELL at the market ask=1 (or best_bid+1 if higher).
# These options are worth ~0.0001 by BS → edge ≈ 1 per unit.
# Risk: if VEV spikes above 6000 (very unlikely in 5 days from ~5255), we lose.
FOTM_SELL_PRICE = 1      # sell at 1 (the constant market ask price)
FOTM_MAX_SHORT  = 200    # max short position per far OTM strike (conservative)
FOTM_SIZE       = 20     # units per order

# ── Delta hedging ─────────────────────────────────────────────────────────────
HEDGE_THRESHOLD = 3      # ↓ from 5 in v6; tighter delta management
HEDGE_SIZE_CAP  = 50     # max hedge units per tick


# ────────────────────────── Black-Scholes ─────────────────────────────────────

def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price_delta(S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
    """Return (call_price, delta) for a European call with r=0."""
    if T <= 1e-8:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)
    sq = math.sqrt(T)
    vol_sq = sigma * sq
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sq
    d2 = d1 - vol_sq
    price = S * _ncdf(d1) - K * _ncdf(d2)
    return price, _ncdf(d1)


# ────────────────────────── helpers ──────────────────────────────────────────

def best_bid_ask(od: OrderDepth) -> Tuple:
    bb = max(od.buy_orders.keys())  if od.buy_orders  else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    return bb, ba

def wall_mid(od: OrderDepth) -> float:
    """Volume-weighted mid — more stable than simple mid."""
    bn = bv = an = av = 0.0
    for p, v in od.buy_orders.items():
        if v > 0: bn += p*v; bv += v
    for p, v in od.sell_orders.items():
        u = abs(v); an += p*u; av += u
    if bv > 0 and av > 0:
        return 0.5*(bn/bv + an/av)
    bb, ba = best_bid_ask(od)
    if bb and ba: return (bb+ba)/2.0
    return float(bb or ba or 0)

def clamp_buy(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] - pos))

def clamp_sell(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] + pos))

def tte_years(ts: int) -> float:
    """Current time-to-expiry in years (decreases each tick)."""
    days = max(TTE_START_DAYS - ts / TICKS_PER_DAY, 0.001)
    return days / TRADING_DAYS


# ────────────────────────── Trader ────────────────────────────────────────────

class Trader:

    # ── persistence ────────────────────────────────────────────────────────

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

    # ── HYDROGEL_PACK ───────────────────────────────────────────────────────

    def _trade_hp(self, od: OrderDepth, pos: int, fair: float) -> List[Order]:
        """
        Aggressive take + passive make around EMA fair value.
        HP spread is always ~16 ticks → 8 ticks edge per side.
        Using EMA fair prevents mis-quoting when HP drifts (range 9891-10051).
        """
        orders: List[Order] = []
        sym = "HYDROGEL_PACK"
        running = pos
        bb, ba = best_bid_ask(od)

        # ── Phase 1: Aggressive taker ──────────────────────────────────────
        # Buy any ask that is clearly below fair
        for ask, vol in sorted(od.sell_orders.items()):
            if ask >= fair - HP_TAKE_EDGE:
                break
            q = clamp_buy(sym, running, min(HP_SIZE, abs(vol)))
            if q <= 0: break
            orders.append(Order(sym, ask, q))
            running += q

        # Sell any bid that is clearly above fair
        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid <= fair + HP_TAKE_EDGE:
                break
            q = clamp_sell(sym, running, min(HP_SIZE, abs(vol)))
            if q <= 0: break
            orders.append(Order(sym, bid, -q))
            running -= q

        # ── Phase 2: Passive market-maker ──────────────────────────────────
        # Skew quotes based on inventory to mean-revert position
        limit = LIMITS[sym]
        skew = int(round(running / limit * HP_MAKE_OFFSET))  # [-offset, +offset]

        buy_px  = int(fair) - HP_MAKE_OFFSET - skew
        sell_px = int(fair) + HP_MAKE_OFFSET - skew

        # Ensure we don't cross the live market
        if ba is not None: buy_px  = min(buy_px,  int(ba) - 1)
        if bb is not None: sell_px = max(sell_px, int(bb) + 1)

        bsz = clamp_buy(sym,  running, HP_SIZE)
        ssz = clamp_sell(sym, running, HP_SIZE)

        if bsz > 0: orders.append(Order(sym, buy_px,  bsz))
        if ssz > 0: orders.append(Order(sym, sell_px, -ssz))

        return orders

    # ── VELVETFRUIT_EXTRACT ─────────────────────────────────────────────────

    def _trade_vev(self, od: OrderDepth, pos: int, fair: float,
                   hedge_qty: int) -> List[Order]:
        """
        Light market-making + delta hedge layer.
        VEV is used mainly for hedging; keep MM size small.
        """
        orders: List[Order] = []
        sym = "VELVETFRUIT_EXTRACT"
        running = pos
        bb, ba = best_bid_ask(od)

        # Aggressive take (clear mispricings)
        for ask, vol in sorted(od.sell_orders.items()):
            if ask >= fair - VEV_TAKE_EDGE: break
            q = clamp_buy(sym, running, min(VEV_SIZE, abs(vol)))
            if q <= 0: break
            orders.append(Order(sym, ask, q))
            running += q

        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid <= fair + VEV_TAKE_EDGE: break
            q = clamp_sell(sym, running, min(VEV_SIZE, abs(vol)))
            if q <= 0: break
            orders.append(Order(sym, bid, -q))
            running -= q

        # Passive MM quotes (inventory-skewed)
        limit = LIMITS[sym]
        skew = int(round(running / limit * VEV_MAKE_OFFSET))
        buy_px  = int(fair) - VEV_MAKE_OFFSET - skew
        sell_px = int(fair) + VEV_MAKE_OFFSET - skew
        if ba is not None: buy_px  = min(buy_px,  int(ba) - 1)
        if bb is not None: sell_px = max(sell_px, int(bb) + 1)

        bsz = clamp_buy(sym,  running, VEV_SIZE)
        ssz = clamp_sell(sym, running, VEV_SIZE)
        if bsz > 0: orders.append(Order(sym, buy_px,  bsz))
        if ssz > 0: orders.append(Order(sym, sell_px, -ssz))

        # ── Delta hedge ─────────────────────────────────────────────────────
        # hedge_qty = how much VEV we need to buy/sell to flatten net delta
        if abs(hedge_qty) >= HEDGE_THRESHOLD:
            h = max(-HEDGE_SIZE_CAP, min(HEDGE_SIZE_CAP, hedge_qty))
            h = int(round(h))
            if h > 0 and ba is not None:
                q = clamp_buy(sym, running, h)
                if q > 0: orders.append(Order(sym, int(ba), q))
            elif h < 0 and bb is not None:
                q = clamp_sell(sym, running, -h)
                if q > 0: orders.append(Order(sym, int(bb), -q))

        return orders

    # ── Liquid options (5000, 5300, 5400, 5500) ───────────────────────────

    def _trade_liquid_option(
        self, sym: str, od: OrderDepth, pos: int,
        fair: float, delta: float
    ) -> Tuple[List[Order], float]:
        """
        Earn the bid-ask spread on liquid strikes.
        Only passive quotes — avoid paying spread via aggressive takes.
        Take aggressively only when edge is very large (OPT_TAKE_EDGE=4).
        Returns (orders, delta_added).
        """
        orders: List[Order] = []
        d_added = 0.0
        running = pos
        bb, ba = best_bid_ask(od)

        # Rare aggressive take (large mispricing only)
        if ba is not None and ba < fair - OPT_TAKE_EDGE:
            if abs(running) < OPT_MAX_INV:
                vol = abs(od.sell_orders[ba])
                q = clamp_buy(sym, running, min(OPT_SIZE, vol))
                q = min(q, OPT_MAX_INV - max(running, 0))
                if q > 0:
                    orders.append(Order(sym, ba, q))
                    running += q
                    d_added += q * delta

        if bb is not None and bb > fair + OPT_TAKE_EDGE:
            if abs(running) < OPT_MAX_INV:
                vol = od.buy_orders[bb]
                q = clamp_sell(sym, running, min(OPT_SIZE, vol))
                q = min(q, OPT_MAX_INV - max(-running, 0))
                if q > 0:
                    orders.append(Order(sym, bb, -q))
                    running -= q
                    d_added -= q * delta

        # Passive quotes (main income)
        if bb is not None and ba is not None and ba - bb >= 2:
            # Buy side
            if running < OPT_MAX_INV:
                buy_px = int(bb) + OPT_MAKE_OFFSET
                if buy_px < fair:
                    bsz = clamp_buy(sym, running, min(OPT_SIZE, OPT_MAX_INV - max(running, 0)))
                    if bsz > 0:
                        orders.append(Order(sym, buy_px, bsz))
                        # Don't add to d_added for passive (unfilled yet)

            # Sell side
            if running > -OPT_MAX_INV:
                sell_px = int(ba) - OPT_MAKE_OFFSET
                if sell_px > fair:
                    ssz = clamp_sell(sym, running, min(OPT_SIZE, OPT_MAX_INV - max(-running, 0)))
                    if ssz > 0:
                        orders.append(Order(sym, sell_px, -ssz))

        return orders, d_added

    # ── Deep ITM options (4000, 4500) ─────────────────────────────────────

    def _trade_deep_itm(
        self, sym: str, od: OrderDepth, pos: int,
        fair: float, delta: float
    ) -> Tuple[List[Order], float]:
        """
        Deep ITM options (delta≈1) have wide market spreads (17-22 ticks).
        Market-make lightly to earn spread without large directional exposure.
        """
        orders: List[Order] = []
        d_added = 0.0
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return orders, d_added

        running = pos
        max_pos = ITM_MAX_INV

        if running < max_pos:
            buy_px = int(bb) + 1
            if buy_px < fair:
                q = clamp_buy(sym, running, min(ITM_SIZE, max_pos - max(running, 0)))
                if q > 0:
                    orders.append(Order(sym, buy_px, q))
                    d_added += q * delta

        if running > -max_pos:
            sell_px = int(ba) - 1
            if sell_px > fair:
                q = clamp_sell(sym, running, min(ITM_SIZE, max_pos - max(-running, 0)))
                if q > 0:
                    orders.append(Order(sym, sell_px, -q))
                    d_added -= q * delta

        return orders, d_added

    # ── Far OTM (6000, 6500): sell for free premium ───────────────────────

    def _trade_far_otm(
        self, sym: str, od: OrderDepth, pos: int, delta: float
    ) -> Tuple[List[Order], float]:
        """
        VEV_6000 and VEV_6500 are essentially worthless by BS.
        Market: bid=0, ask=1 (constant).
        Strategy: sell at ask price = 1. Buyer pays us 1 per unit.
        On expiry, option value ≈ 0 → we keep the 1 unit premium.
        Delta ≈ 0 → minimal hedge cost.
        Risk: if VEV spikes >6000, we owe (VEV-6000) per unit.
              With VEV~5255 and only 5d TTE, this requires a +14% move → very unlikely.
        """
        orders: List[Order] = []
        d_added = 0.0
        bb, ba = best_bid_ask(od)

        # Only sell if there's a buyer (bid > 0) — trades at price=0 suggest
        # no buyers willing to pay 1, so we post and wait.
        # We can post at ask=1 as a maker; if nobody buys, no harm.
        if pos > -FOTM_MAX_SHORT:
            sell_px = FOTM_SELL_PRICE
            # Only sell above 0 (otherwise no edge)
            if ba is not None and ba >= sell_px:
                q = clamp_sell(sym, pos, min(FOTM_SIZE, FOTM_MAX_SHORT - max(-pos, 0)))
                if q > 0:
                    orders.append(Order(sym, sell_px, -q))
                    d_added -= q * delta  # delta≈0 for far OTM

        return orders, d_added

    # ── main ──────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data      = self._load(state.traderData)
        ema_dict  = data.get("ema", {})
        result: Dict[str, List[Order]] = {}

        # ── Update EMAs ─────────────────────────────────────────────────────
        for sym in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
            od = state.order_depths.get(sym)
            if od is None: continue
            raw = wall_mid(od)
            if raw == 0: continue
            alpha = HP_EMA_ALPHA if sym == "HYDROGEL_PACK" else VEV_EMA_ALPHA
            prev = ema_dict.get(sym, raw)
            ema_dict[sym] = alpha * raw + (1.0 - alpha) * prev

        # ── HYDROGEL_PACK ────────────────────────────────────────────────────
        hp_od  = state.order_depths.get("HYDROGEL_PACK")
        hp_ema = ema_dict.get("HYDROGEL_PACK")
        if hp_od and hp_ema:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._trade_hp(hp_od, pos, hp_ema)

        # ── Options + VEV ────────────────────────────────────────────────────
        vev_od  = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_ema = ema_dict.get("VELVETFRUIT_EXTRACT")

        net_delta    = 0.0   # existing option inventory delta
        new_od_delta = 0.0   # expected delta from new orders this tick
        T = tte_years(state.timestamp)

        if vev_ema is not None:

            # ── Account for existing option positions' delta ─────────────────
            for K in ALL_STRIKES:
                sym = f"VEV_{K}"
                pos = state.position.get(sym, 0)
                if pos != 0:
                    _, d = bs_price_delta(vev_ema, float(K), T, SIGMA)
                    net_delta += pos * d

            # ── Trade liquid strikes (5000, 5300, 5400, 5500) ────────────────
            for K in LIQUID_STRIKES:
                sym = f"VEV_{K}"
                od = state.order_depths.get(sym)
                if od is None:
                    result[sym] = []
                    continue
                pos = state.position.get(sym, 0)
                fair, delta = bs_price_delta(vev_ema, float(K), T, SIGMA)
                orders, d = self._trade_liquid_option(sym, od, pos, fair, delta)
                result[sym] = orders
                new_od_delta += d

            # ── Skip 5100 and 5200 (negative net edge) ───────────────────────
            for K in SKIP_STRIKES:
                result[f"VEV_{K}"] = []

            # ── Deep ITM (4000, 4500) ─────────────────────────────────────────
            for K in DEEP_ITM:
                sym = f"VEV_{K}"
                od = state.order_depths.get(sym)
                if od is None:
                    result[sym] = []
                    continue
                pos = state.position.get(sym, 0)
                fair, delta = bs_price_delta(vev_ema, float(K), T, SIGMA)
                orders, d = self._trade_deep_itm(sym, od, pos, fair, delta)
                result[sym] = orders
                new_od_delta += d

            # ── Far OTM (6000, 6500) ─────────────────────────────────────────
            for K in FAR_OTM:
                sym = f"VEV_{K}"
                od = state.order_depths.get(sym)
                if od is None:
                    result[sym] = []
                    continue
                pos = state.position.get(sym, 0)
                _, delta = bs_price_delta(vev_ema, float(K), T, SIGMA)
                orders, d = self._trade_far_otm(sym, od, pos, delta)
                result[sym] = orders
                new_od_delta += d

        # ── VELVETFRUIT_EXTRACT: MM + hedge ───────────────────────────────────
        if vev_od and vev_ema:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            total_delta = net_delta + new_od_delta
            # To flatten net delta: buy/sell underlying by -total_delta, then offset pos
            raw_hedge = -total_delta - pos
            hedge_qty = 0 if abs(raw_hedge) < HEDGE_THRESHOLD else int(round(raw_hedge))
            result["VELVETFRUIT_EXTRACT"] = self._trade_vev(
                vev_od, pos, vev_ema, hedge_qty
            )

        return result, 0, self._save({"ema": ema_dict})