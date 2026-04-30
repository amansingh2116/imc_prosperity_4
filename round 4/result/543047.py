from datamodel import Order, OrderDepth, TradingState, Trade
from typing import Dict, List, Optional
import json
import math

# --- Configuration & Constants ---
HP = "HYDROGEL_PACK"
VEV = "VELVETFRUIT_EXTRACT"
ACTIVE_STRIKES = [5000, 5100, 5200, 5300, 5400, 5500]
STRIKES = {f"VEV_{k}": k for k in ACTIVE_STRIKES}

POSITION_LIMITS = {
    HP: 200, 
    VEV: 200,
    **{f"VEV_{k}": 300 for k in ACTIVE_STRIKES}
}

TICKS_PER_DAY = 1_000_000
CLOSE_ONLY_START = 900_000
FLATTEN_START = 960_000  

# 🔑 FIXED TTE: We hold this constant and let the dynamic IV map absorb time-decay.
# This prevents out-of-sample failures when day numbers vary.
CONSTANT_TTE = 4.0 / 252.0 

# --- Edges & Parameters ---
HP_TAKE_EDGE = 1.5
HP_PASSIVE_OFFSET = 1.5 
HP_ORDER_SIZE = 60      

VE_TAKE_EDGE = 1.0
VE_PASSIVE_OFFSET = 1.0
VE_ORDER_SIZE = 40      

OPT_TAKE_EDGE = 1.0
OPT_MAKE_EDGE = 0.5
OPT_ORDER_SIZE = 30     

HARD_HEDGE_THRESHOLD = 40

# --- Black-Scholes Engine ---
def ncdf(x: float) -> float: 
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0 or T <= 1e-12: return max(S - K, 0.0)
    vol = max(1e-8, sigma) * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / vol
    return S * ncdf(d1) - K * ncdf(d1 - vol)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if S <= 0 or T <= 1e-12: return 1.0 if S > K else 0.0
    vol = max(1e-8, sigma) * math.sqrt(T)
    return ncdf((math.log(S / K) + 0.5 * sigma * sigma * T) / vol)

def implied_volatility(price: float, S: float, K: float, T: float) -> Optional[float]:
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-9 or T <= 1e-12 or S <= 0: return None
    lo, hi = 0.01, 1.50
    for _ in range(15):
        mid = (lo + hi) / 2.0
        if bs_call(S, K, T, mid) < price: lo = mid
        else: hi = mid
    return (lo + hi) / 2.0

# --- Microstructure Helpers ---
def best_bid(od: OrderDepth) -> Optional[int]: return max(od.buy_orders.keys()) if od.buy_orders else None
def best_ask(od: OrderDepth) -> Optional[int]: return min(od.sell_orders.keys()) if od.sell_orders else None

def micro_mid(od: OrderDepth) -> Optional[float]:
    bb, ba = best_bid(od), best_ask(od)
    if not bb or not ba: return None
    bq, aq = od.buy_orders[bb], abs(od.sell_orders[ba])
    return (bb * aq + ba * bq) / (bq + aq) if (bq + aq) > 0 else (bb + ba) / 2.0

class Trader:
    def __init__(self):
        self.hp_ema = None
        self.ve_ema = None
        self.iv_map: Dict[str, float] = {}

    def _load(self, state_str: str):
        if state_str:
            try:
                data = json.loads(state_str)
                self.hp_ema = data.get("hp")
                self.ve_ema = data.get("ve")
                self.iv_map = data.get("iv_map", {})
            except: pass

    def _save(self) -> str:
        return json.dumps({"hp": self.hp_ema, "ve": self.ve_ema, "iv_map": self.iv_map})

    def _update_ema(self, current: Optional[float], new_val: Optional[float], alpha: float) -> Optional[float]:
        if new_val is None: return current
        if current is None: return new_val
        return alpha * new_val + (1.0 - alpha) * current

    def _get_omni_signal(self, trades: List[Trade], od: OrderDepth, noise_threshold: int = 15) -> float:
        """
        OMNI-SIGNAL: Blends Order Book Imbalance (resting liquidity) with 
        Tick-Tested Flow (aggressive execution) for the ultimate adverse selection filter.
        """
        bb, ba = best_bid(od), best_ask(od)
        if bb is None or ba is None: return 0.0
        
        # 1. Resting OBI
        bq, aq = od.buy_orders.get(bb, 0), abs(od.sell_orders.get(ba, 0))
        obi = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0
        
        # 2. Executed Flow
        if not trades: return obi * 0.5  # Rely purely on OBI if no trades
        
        buy_aggressor_vol = 0
        sell_aggressor_vol = 0
        
        for t in trades:
            if t.quantity < noise_threshold or t.buyer == "SUBMISSION" or t.seller == "SUBMISSION": continue
            if t.price >= ba: buy_aggressor_vol += t.quantity
            elif t.price <= bb: sell_aggressor_vol += t.quantity
                
        tot = buy_aggressor_vol + sell_aggressor_vol
        trade_signal = (buy_aggressor_vol - sell_aggressor_vol) / tot if tot > 0 else 0.0
        
        # Blend: 60% Resting Pressure, 40% Executed Pressure
        return (0.6 * obi) + (0.4 * trade_signal)

    def _trade_spot(self, sym: str, fair: float, pos: int, od: OrderDepth, take_edge: float, 
                    passive_offset: float, size: int, tick: int, effective_pos: float = None) -> List[Order]:
        orders: List[Order] = []
        limit = POSITION_LIMITS[sym]
        bb, ba = best_bid(od), best_ask(od)

        flattening = tick >= FLATTEN_START
        closing = tick >= CLOSE_ONLY_START

        # 1. Hard Flattening Mode
        if flattening:
            if pos > 0 and bb is not None: orders.append(Order(sym, bb, -pos))
            elif pos < 0 and ba is not None: orders.append(Order(sym, ba, -pos))
            return orders

        max_buy = 0 if (closing and pos >= 0) else min(limit - pos, limit if closing else limit * 2)
        max_sell = 0 if (closing and pos <= 0) else min(limit + pos, limit if closing else limit * 2)

        # 2. Aggressive Takes
        if ba and ba <= fair - take_edge and max_buy > 0:
            qty = min(max_buy, abs(od.sell_orders[ba]))
            if qty > 0: orders.append(Order(sym, ba, qty)); pos += qty; max_buy -= qty

        if bb and bb >= fair + take_edge and max_sell > 0:
            qty = min(max_sell, od.buy_orders[bb])
            if qty > 0: orders.append(Order(sym, bb, -qty)); pos -= qty; max_sell -= qty

        # 3. Passive Quotes (Smart Closing)
        skew_pos = effective_pos if effective_pos is not None else pos
        
        # 🔑 SMART CLOSE: If in close-only mode, violently skew quotes to attract takers
        if closing:
            if pos > 0: ask_px = int(math.floor(fair)) # Offer extremely cheap
            else: ask_px = int(math.ceil(fair + passive_offset))
            
            if pos < 0: bid_px = int(math.ceil(fair)) # Bid extremely high
            else: bid_px = int(math.floor(fair - passive_offset))
        else:
            skew = (skew_pos / limit) * (passive_offset * 1.5)  
            bid_px = int(round(fair - passive_offset - skew))
            ask_px = int(round(fair + passive_offset - skew))

        if max_buy > 0 and bid_px > 0: orders.append(Order(sym, bid_px, min(size, max_buy)))
        if max_sell > 0: orders.append(Order(sym, ask_px, -min(size, max_sell)))

        return orders

    def run(self, state: TradingState):
        self._load(state.traderData)
        result: Dict[str, List[Order]] = {}
        
        current_tick = state.timestamp % TICKS_PER_DAY 

        hp_od = state.order_depths.get(HP, OrderDepth())
        ve_od = state.order_depths.get(VEV, OrderDepth())

        hp_micro = micro_mid(hp_od)
        ve_micro = micro_mid(ve_od)

        self.hp_ema = self._update_ema(self.hp_ema, hp_micro, 0.05)
        self.ve_ema = self._update_ema(self.ve_ema, ve_micro, 0.15)

        # Apply Omni-Signal Toxicity Filter
        hp_omni = self._get_omni_signal(state.market_trades.get(HP, []), hp_od, 15)
        ve_omni = self._get_omni_signal(state.market_trades.get(VEV, []), ve_od, 15)

        # Shift Fair Value proportionally to the spread pressure
        hp_fair = (self.hp_ema or 10000.0) + (hp_omni * 2.0) 
        ve_fair = (self.ve_ema or 5250.0) + (ve_omni * 1.5)

        # --- 1. Trade HYDROGEL_PACK ---
        result[HP] = self._trade_spot(HP, hp_fair, state.position.get(HP, 0), hp_od, HP_TAKE_EDGE, HP_PASSIVE_OFFSET, HP_ORDER_SIZE, current_tick)

        # --- 2. Dynamic STRIKE-SPECIFIC IV Calibration ---
        current_ve_spot = ve_micro if ve_micro else 5250.0

        for sym, K in STRIKES.items():
            od = state.order_depths.get(sym)
            if od:
                bb, ba = best_bid(od), best_ask(od)
                if bb and ba:
                    # Let IV absorb the TTE time decay automatically!
                    iv = implied_volatility(0.5*(bb+ba), current_ve_spot, K, CONSTANT_TTE)
                    if iv and 0.05 < iv < 0.80:
                        prev_iv = self.iv_map.get(sym, 0.16)
                        self.iv_map[sym] = 0.1 * iv + 0.9 * prev_iv

        # --- 3. Trade VOUCHERS (Options Arbitrage) ---
        net_delta = 0.0
        for sym, K in STRIKES.items():
            od = state.order_depths.get(sym, OrderDepth())
            pos = state.position.get(sym, 0)
            
            flattening = current_tick >= FLATTEN_START
            closing = current_tick >= CLOSE_ONLY_START

            if flattening:
                orders = []
                bb, ba = best_bid(od), best_ask(od)
                if pos > 0 and bb is not None: orders.append(Order(sym, bb, -pos))
                elif pos < 0 and ba is not None: orders.append(Order(sym, ba, -pos))
                if orders: result[sym] = orders
                continue

            strike_iv = self.iv_map.get(sym, 0.16)
            theo = bs_call(current_ve_spot, K, CONSTANT_TTE, strike_iv)
            delta = bs_delta(current_ve_spot, K, CONSTANT_TTE, strike_iv)
            net_delta += pos * delta 

            max_buy = 0 if (closing and pos >= 0) else min(POSITION_LIMITS[sym] - pos, POSITION_LIMITS[sym])
            max_sell = 0 if (closing and pos <= 0) else min(POSITION_LIMITS[sym] + pos, POSITION_LIMITS[sym])
            
            orders: List[Order] = []
            bb, ba = best_bid(od), best_ask(od)

            if ba and theo - ba > OPT_TAKE_EDGE and max_buy > 0:
                q = min(abs(od.sell_orders[ba]), max_buy, OPT_ORDER_SIZE)
                orders.append(Order(sym, ba, q)); pos += q; max_buy -= q
            if bb and bb - theo > OPT_TAKE_EDGE and max_sell > 0:
                q = min(od.buy_orders[bb], max_sell, OPT_ORDER_SIZE)
                orders.append(Order(sym, bb, -q)); pos -= q; max_sell -= q

            if closing:
                # Smart option unwind via passive quotes
                if pos > 0 and ba: orders.append(Order(sym, ba - 1, -min(pos, OPT_ORDER_SIZE)))
                elif pos < 0 and bb: orders.append(Order(sym, bb + 1, min(abs(pos), OPT_ORDER_SIZE)))
                if orders: result[sym] = orders
                continue

            if max_buy > 0:
                post_bid = min(int(math.floor(theo - OPT_MAKE_EDGE)), ba - 1 if ba else 9999)
                if post_bid > 0: orders.append(Order(sym, post_bid, min(OPT_ORDER_SIZE, max_buy)))
            if max_sell > 0:
                post_ask = max(int(math.ceil(theo + OPT_MAKE_EDGE)), bb + 1 if bb else 0)
                orders.append(Order(sym, post_ask, -min(OPT_ORDER_SIZE, max_sell))) 

            if orders: result[sym] = orders

        # --- 4. Trade VELVETFRUIT_EXTRACT (Hybrid Active/Passive Delta Hedging) ---
        ve_pos = state.position.get(VEV, 0)
        effective_ve_pos = ve_pos + net_delta 
        ve_orders: List[Order] = []
        
        ve_bb, ve_ba = best_bid(ve_od), best_ask(ve_od)

        if not (current_tick >= CLOSE_ONLY_START):
            if effective_ve_pos > HARD_HEDGE_THRESHOLD and ve_bb is not None:
                hedge_qty = min(int(effective_ve_pos - HARD_HEDGE_THRESHOLD), ve_od.buy_orders[ve_bb])
                if hedge_qty > 0:
                    ve_orders.append(Order(VEV, ve_bb, -hedge_qty))
                    ve_pos -= hedge_qty
                    effective_ve_pos -= hedge_qty
            elif effective_ve_pos < -HARD_HEDGE_THRESHOLD and ve_ba is not None:
                hedge_qty = min(int(abs(effective_ve_pos + HARD_HEDGE_THRESHOLD)), abs(ve_od.sell_orders[ve_ba]))
                if hedge_qty > 0:
                    ve_orders.append(Order(VEV, ve_ba, hedge_qty))
                    ve_pos += hedge_qty
                    effective_ve_pos += hedge_qty

        # Skew FV for remaining passive hedging
        delta_skew = (net_delta / POSITION_LIMITS[VEV]) * 2.5
        ve_fair_hedged = ve_fair - delta_skew

        spot_orders = self._trade_spot(VEV, ve_fair_hedged, ve_pos, ve_od, VE_TAKE_EDGE, VE_PASSIVE_OFFSET, VE_ORDER_SIZE, current_tick, effective_pos=effective_ve_pos)
        ve_orders.extend(spot_orders)
        
        if ve_orders: result[VEV] = ve_orders

        return result, 0, self._save()