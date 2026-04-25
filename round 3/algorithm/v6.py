"""
IMC Prosperity 4 – Round 3 Optimised Trader
============================================
Products
  HYDROGEL_PACK           delta-1, pos_limit=200  (stable ~10000)
  VELVETFRUIT_EXTRACT     delta-1, pos_limit=200  (VEV underlying, ~5200-5280)
  VEV_4000 … VEV_6500     European call options (10 strikes), pos_limit=300 each
                           TTE = 5d at Round 3 start, decreases to ~2d by end

Key calibration from data analysis
  • True implied vol (sigma) = 0.200 (stable across all days, all strikes 5000-5500)
  • HYDROGEL_PACK: fair value = 10000, market spread ~16 ticks → strong MM edge
  • VELVETFRUIT_EXTRACT: random-walks in 5200-5280, spread ~6 ticks
  • Options market: spread 1-7 ticks; market prices options at sigma≈0.20 (well calibrated)
  • VEV_6000, VEV_6500: always priced at 0-1, essentially worthless → skip
  • VEV_4000, VEV_4500: deep ITM (delta≈1.0), always worth (S - K) → skip passive quoting

Strategy overview
  1. HYDROGEL_PACK:         Aggressive taker + passive market-maker around 10000
  2. VELVETFRUIT_EXTRACT:   EMA-based fair-value market-maker (conservative size)
  3. Options liquid (5000-5500):
       - Compute BS fair with sigma=0.20 and current TTE
       - Passive market-make inside the existing spread (earn half bid-ask)
       - Tight inventory limit (max ±40 per strike, to prevent stuck positions)
       - Continuously hedge aggregate net delta via VELVETFRUIT_EXTRACT
  4. Deep ITM (4000, 4500): MM lightly as near-underlying products
  5. Far OTM (6000, 6500):  Skip entirely

Why this beats v1/v3
  • v1: took options aggressively → paid spread, hit pos limit (300), stuck with
        large directional positions, theta decay → lost on 7 of 10 strikes
  • v3: used sigma=0.2525 (wrong, 25% too high) → systematic trading errors → -771
  • v4: sigma=0.200 (correct), only makes not takes on options, strict inventory
        control prevents stuck positions, better HP market-making
"""

from datamodel import Order, OrderDepth, TradingState
from typing import Dict, List, Tuple
import json
import math


# ─────────────────────────── constants ───────────────────────────────────────

LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300, "VEV_4500": 300,
    "VEV_5000": 300, "VEV_5100": 300, "VEV_5200": 300,
    "VEV_5300": 300, "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}

# All option strikes (VEV_<K>)
ALL_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Strikes we actively market-make
LIQUID_STRIKES  = [5000, 5100, 5200, 5300, 5400, 5500]  # ATM/OTM range
DEEP_ITM_STRIKES = [4000, 4500]                          # delta ≈ 1
FAR_OTM_STRIKES  = [6000, 6500]                          # worthless, skip

# ── Calibrated parameters ─────────────────────────────────────────────────────
SIGMA           = 0.200   # annualised IV (from historical data analysis)
TRADING_DAYS    = 252     # trading days per year

# Round 3 live evaluation: TTE = 5d at timestamp 0, decreasing each tick
TTE_START_DAYS  = 5.0
TICKS_PER_DAY   = 1_000_000   # 1M timestamps = 1 trading day

# ── Hydrogel market-making ────────────────────────────────────────────────────
HP_FAIR         = 10_000   # known stable fair value
HP_TAKE_EDGE    = 2        # take if (fair - ask) > this  (HP spread ~16, so easy)
HP_MAKE_OFFSET  = 3        # passive quote offset from fair value
HP_SIZE         = 30       # order size per side

# ── VEV underlying market-making ─────────────────────────────────────────────
VEV_EMA_ALPHA   = 0.10     # slow EMA (underlying trends slowly)
VEV_TAKE_EDGE   = 1        # take if outside fair by this
VEV_MAKE_OFFSET = 2        # passive quote offset
VEV_SIZE        = 10       # conservative size (underlying used for hedging)

# ── Option market-making (liquid strikes 5000-5500) ──────────────────────────
OPT_TAKE_EDGE   = 3.0      # take only if clear edge (rare – usually just make)
OPT_MAKE_OFFSET = 1        # passive quote 1 tick inside market spread
OPT_SIZE        = 15       # max units per passive quote
OPT_MAX_POS     = 50       # inventory limit per strike (prevent stuck positions)

# ── Delta hedging ─────────────────────────────────────────────────────────────
HEDGE_THRESHOLD = 5        # only hedge when |net_delta| exceeds this
HEDGE_SIZE_CAP  = 40       # max hedge size per tick


# ─────────────────────────── Black-Scholes ────────────────────────────────────

def _ncdf(x: float) -> float:
    """Standard normal CDF via math.erf approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price_delta(S: float, K: float, T: float, sigma: float) -> Tuple[float, float]:
    """Return (call_price, delta) for a European call with r=0."""
    if T <= 1e-8:
        intrinsic = max(S - K, 0.0)
        return intrinsic, (1.0 if S > K else 0.0)
    sq_t = math.sqrt(T)
    vol_sqrt = sigma * sq_t
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    price = S * _ncdf(d1) - K * _ncdf(d2)
    delta = _ncdf(d1)
    return price, delta


# ─────────────────────────── helper utilities ─────────────────────────────────

def best_bid_ask(od: OrderDepth):
    """Return (best_bid, best_ask) or (None, None)."""
    bb = max(od.buy_orders.keys()) if od.buy_orders else None
    ba = min(od.sell_orders.keys()) if od.sell_orders else None
    return bb, ba


def mid_price(od: OrderDepth):
    bb, ba = best_bid_ask(od)
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return float(bb or ba or 0)


def wall_mid(od: OrderDepth) -> float:
    """Volume-weighted average mid (more stable than simple mid)."""
    bid_v = bid_n = ask_v = ask_n = 0.0
    for p, v in od.buy_orders.items():
        if v > 0:
            bid_n += p * v
            bid_v += v
    for p, v in od.sell_orders.items():
        av = abs(v)
        ask_n += p * av
        ask_v += av
    if bid_v > 0 and ask_v > 0:
        return 0.5 * (bid_n / bid_v + ask_n / ask_v)
    return mid_price(od)


def clamp_buy(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] - pos))


def clamp_sell(sym: str, pos: int, qty: int) -> int:
    return max(0, min(qty, LIMITS[sym] + pos))


def tte_years(timestamp: int) -> float:
    """Time-to-expiry in years, decreasing each tick during Round 3."""
    days_elapsed = timestamp / TICKS_PER_DAY
    days_remaining = max(TTE_START_DAYS - days_elapsed, 0.001)
    return days_remaining / TRADING_DAYS


# ─────────────────────────── Trader ──────────────────────────────────────────

class Trader:

    # ── state I/O ─────────────────────────────────────────────────────────────

    def _load(self, td: str) -> Dict:
        if td:
            try:
                d = json.loads(td)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        return {}

    def _save(self, d: Dict) -> str:
        try:
            return json.dumps(d)
        except Exception:
            return "{}"

    # ── HYDROGEL_PACK ─────────────────────────────────────────────────────────

    def _trade_hp(self, od: OrderDepth, pos: int) -> List[Order]:
        """Market-make HYDROGEL_PACK around its known fair value of 10000."""
        orders: List[Order] = []
        fair = HP_FAIR
        running = pos

        # Phase 1 – Aggressive take: buy any ask below (fair - HP_TAKE_EDGE)
        for ask, vol in sorted(od.sell_orders.items()):
            if ask >= fair - HP_TAKE_EDGE:
                break
            q = clamp_buy("HYDROGEL_PACK", running, min(HP_SIZE, abs(vol)))
            if q <= 0:
                break
            orders.append(Order("HYDROGEL_PACK", ask, q))
            running += q

        # Phase 1b – Aggressive take: sell any bid above (fair + HP_TAKE_EDGE)
        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid <= fair + HP_TAKE_EDGE:
                break
            q = clamp_sell("HYDROGEL_PACK", running, min(HP_SIZE, abs(vol)))
            if q <= 0:
                break
            orders.append(Order("HYDROGEL_PACK", bid, -q))
            running -= q

        # Phase 2 – Passive quotes: tight inside the market spread
        bb, ba = best_bid_ask(od)
        limit = LIMITS["HYDROGEL_PACK"]

        # Inventory skew: if long, lower buy price and raise sell price slightly
        skew = int(round(running / limit * HP_MAKE_OFFSET))

        buy_px  = fair - HP_MAKE_OFFSET - skew
        sell_px = fair + HP_MAKE_OFFSET - skew

        # Ensure we are inside the real spread if it exists
        if ba is not None:
            buy_px = min(buy_px, ba - 1)
        if bb is not None:
            sell_px = max(sell_px, bb + 1)

        buy_sz  = clamp_buy("HYDROGEL_PACK", running, HP_SIZE)
        sell_sz = clamp_sell("HYDROGEL_PACK", running, HP_SIZE)

        if buy_sz > 0:
            orders.append(Order("HYDROGEL_PACK", int(buy_px), buy_sz))
        if sell_sz > 0:
            orders.append(Order("HYDROGEL_PACK", int(sell_px), -sell_sz))

        return orders

    # ── VELVETFRUIT_EXTRACT ───────────────────────────────────────────────────

    def _trade_vev(self, od: OrderDepth, pos: int, ema: float,
                   hedge_qty: int) -> List[Order]:
        """Market-make VEV around EMA fair, then add delta-hedge orders."""
        orders: List[Order] = []
        fair = ema
        running = pos

        # Take aggressive mispricings
        for ask, vol in sorted(od.sell_orders.items()):
            if ask >= fair - VEV_TAKE_EDGE:
                break
            q = clamp_buy("VELVETFRUIT_EXTRACT", running, min(VEV_SIZE, abs(vol)))
            if q <= 0:
                break
            orders.append(Order("VELVETFRUIT_EXTRACT", ask, q))
            running += q

        for bid, vol in sorted(od.buy_orders.items(), reverse=True):
            if bid <= fair + VEV_TAKE_EDGE:
                break
            q = clamp_sell("VELVETFRUIT_EXTRACT", running, min(VEV_SIZE, abs(vol)))
            if q <= 0:
                break
            orders.append(Order("VELVETFRUIT_EXTRACT", bid, -q))
            running -= q

        # Passive market-making quotes
        bb, ba = best_bid_ask(od)
        limit = LIMITS["VELVETFRUIT_EXTRACT"]
        skew = int(round(running / limit * VEV_MAKE_OFFSET))

        buy_px  = int(fair) - VEV_MAKE_OFFSET - skew
        sell_px = int(fair) + VEV_MAKE_OFFSET - skew

        if ba is not None:
            buy_px = min(buy_px, ba - 1)
        if bb is not None:
            sell_px = max(sell_px, bb + 1)

        buy_sz  = clamp_buy("VELVETFRUIT_EXTRACT", running, VEV_SIZE)
        sell_sz = clamp_sell("VELVETFRUIT_EXTRACT", running, VEV_SIZE)

        if buy_sz > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", buy_px, buy_sz))
        if sell_sz > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", sell_px, -sell_sz))

        # Delta hedge layer: add hedge on top of MM if needed
        if abs(hedge_qty) >= HEDGE_THRESHOLD:
            capped = max(-HEDGE_SIZE_CAP, min(HEDGE_SIZE_CAP, hedge_qty))
            h = int(round(capped))
            if h > 0:
                q = clamp_buy("VELVETFRUIT_EXTRACT", running, h)
                if q > 0 and ba is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", ba, q))
            elif h < 0:
                q = clamp_sell("VELVETFRUIT_EXTRACT", running, -h)
                if q > 0 and bb is not None:
                    orders.append(Order("VELVETFRUIT_EXTRACT", bb, -q))

        return orders

    # ── Option market-making ──────────────────────────────────────────────────

    def _trade_option(
        self,
        sym: str,
        strike: int,
        od: OrderDepth,
        pos: int,
        fair: float,
        delta: float,
    ) -> Tuple[List[Order], float]:
        """
        Market-make one option strike.
        Returns (orders, net_delta_added_by_new_orders).
        """
        orders: List[Order] = []
        delta_added = 0.0
        running = pos

        bb, ba = best_bid_ask(od)

        # ── Aggressive taking (only with large clear edge) ────────────────────
        if ba is not None and ba < fair - OPT_TAKE_EDGE:
            # Only take if not already at inventory limit
            if abs(running) < OPT_MAX_POS:
                vol = abs(od.sell_orders[ba])
                q = clamp_buy(sym, running, min(OPT_SIZE, vol))
                # Extra guard: don't exceed soft inventory limit
                q = min(q, OPT_MAX_POS - max(running, 0))
                if q > 0:
                    orders.append(Order(sym, ba, q))
                    running += q
                    delta_added += q * delta

        if bb is not None and bb > fair + OPT_TAKE_EDGE:
            if abs(running) < OPT_MAX_POS:
                vol = od.buy_orders[bb]
                q = clamp_sell(sym, running, min(OPT_SIZE, vol))
                q = min(q, OPT_MAX_POS - max(-running, 0))
                if q > 0:
                    orders.append(Order(sym, bb, -q))
                    running -= q
                    delta_added -= q * delta

        # ── Passive market-making ─────────────────────────────────────────────
        # Only quote if inventory is within soft limits
        if bb is not None and ba is not None:
            spread = ba - bb
            if spread >= 2:  # only quote if market has a gap to fill
                # Buy side: inside market bid
                if running < OPT_MAX_POS:
                    buy_px = int(bb) + OPT_MAKE_OFFSET
                    if buy_px < fair:           # still profitable
                        buy_sz = min(OPT_SIZE, OPT_MAX_POS - max(running, 0))
                        buy_sz = clamp_buy(sym, running, buy_sz)
                        if buy_sz > 0:
                            orders.append(Order(sym, buy_px, buy_sz))

                # Sell side: inside market ask
                if running > -OPT_MAX_POS:
                    sell_px = int(ba) - OPT_MAKE_OFFSET
                    if sell_px > fair:          # still profitable
                        sell_sz = min(OPT_SIZE, OPT_MAX_POS - max(-running, 0))
                        sell_sz = clamp_sell(sym, running, sell_sz)
                        if sell_sz > 0:
                            orders.append(Order(sym, sell_px, -sell_sz))

        return orders, delta_added

    # ── Deep ITM options (4000, 4500) ─────────────────────────────────────────

    def _trade_deep_itm(
        self, sym: str, strike: int, od: OrderDepth, pos: int,
        fair: float, delta: float
    ) -> Tuple[List[Order], float]:
        """
        Deep ITM options behave like the underlying (delta ≈ 1).
        Simple light market-making; avoid large positions.
        """
        orders: List[Order] = []
        delta_added = 0.0
        bb, ba = best_bid_ask(od)
        if bb is None or ba is None:
            return orders, delta_added

        # Only quote inside market with tiny size to earn spread
        max_pos = 20  # keep small, delta ≈ 1 so each unit = full underlying exposure
        running = pos

        # Buy side
        if running < max_pos:
            buy_px = bb + 1
            if buy_px < fair:
                q = clamp_buy(sym, running, min(5, max_pos - max(running, 0)))
                if q > 0:
                    orders.append(Order(sym, int(buy_px), q))
                    delta_added += q * delta

        # Sell side
        if running > -max_pos:
            sell_px = ba - 1
            if sell_px > fair:
                q = clamp_sell(sym, running, min(5, max_pos - max(-running, 0)))
                if q > 0:
                    orders.append(Order(sym, int(sell_px), -q))
                    delta_added -= q * delta

        return orders, delta_added

    # ── main entry point ──────────────────────────────────────────────────────

    def run(self, state: TradingState):
        data = self._load(state.traderData)
        ema_dict: Dict[str, float] = data.get("ema", {})

        result: Dict[str, List[Order]] = {}

        # ── Update EMA for underlying products ───────────────────────────────
        for sym in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"):
            od = state.order_depths.get(sym)
            if od is None:
                continue
            raw = wall_mid(od)
            if raw == 0:
                continue
            prev = ema_dict.get(sym, raw)
            alpha = VEV_EMA_ALPHA if sym == "VELVETFRUIT_EXTRACT" else 0.05
            ema_dict[sym] = alpha * raw + (1.0 - alpha) * prev

        # ── HYDROGEL_PACK ────────────────────────────────────────────────────
        hp_od = state.order_depths.get("HYDROGEL_PACK")
        if hp_od:
            pos = state.position.get("HYDROGEL_PACK", 0)
            result["HYDROGEL_PACK"] = self._trade_hp(hp_od, pos)

        # ── Option book: compute total net delta from existing positions ─────
        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_ema = ema_dict.get("VELVETFRUIT_EXTRACT")

        net_delta = 0.0          # delta from option inventory (existing)
        new_order_delta = 0.0    # expected delta from new option orders

        if vev_ema is not None:
            T = tte_years(state.timestamp)

            # Track existing positions' delta
            for K in ALL_STRIKES:
                sym = f"VEV_{K}"
                pos = state.position.get(sym, 0)
                if pos != 0:
                    _, d = bs_price_delta(vev_ema, float(K), T, SIGMA)
                    net_delta += pos * d

            # Trade each strike
            for K in LIQUID_STRIKES:
                sym = f"VEV_{K}"
                od = state.order_depths.get(sym)
                if od is None:
                    result[sym] = []
                    continue
                pos = state.position.get(sym, 0)
                fair, delta = bs_price_delta(vev_ema, float(K), T, SIGMA)
                orders, dAdded = self._trade_option(sym, K, od, pos, fair, delta)
                result[sym] = orders
                new_order_delta += dAdded

            for K in DEEP_ITM_STRIKES:
                sym = f"VEV_{K}"
                od = state.order_depths.get(sym)
                if od is None:
                    result[sym] = []
                    continue
                pos = state.position.get(sym, 0)
                fair, delta = bs_price_delta(vev_ema, float(K), T, SIGMA)
                orders, dAdded = self._trade_deep_itm(sym, K, od, pos, fair, delta)
                result[sym] = orders
                new_order_delta += dAdded

            # Far OTM: skip trading, set empty
            for K in FAR_OTM_STRIKES:
                result[f"VEV_{K}"] = []

        # ── VELVETFRUIT_EXTRACT: MM + hedge ──────────────────────────────────
        if vev_od and vev_ema:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            # Total net delta that needs hedging
            total_delta = net_delta + new_order_delta
            # To flatten, we need to change underlying position by -total_delta
            hedge_needed = -int(round(total_delta)) - pos
            # Clamp hedge: only act if meaningful
            if abs(hedge_needed) < HEDGE_THRESHOLD:
                hedge_needed = 0
            result["VELVETFRUIT_EXTRACT"] = self._trade_vev(
                vev_od, pos, vev_ema, hedge_needed
            )

        return result, 0, self._save({"ema": ema_dict})