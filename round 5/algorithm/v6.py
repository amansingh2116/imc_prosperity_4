from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List


POSITION_LIMIT = 10

SKIP_PRODUCTS = frozenset(
    {
        "ROBOT_VACUUMING",
        "ROBOT_MOPPING",
        "ROBOT_DISHES",
        "ROBOT_LAUNDRY",
        "ROBOT_IRONING",
        # PEBBLES_XL has a consistently poor spread/volatility ratio in days
        # 2-4 and lost money in every broad market-making run.
        "PEBBLES_XL",
    }
)

# Most upside comes from v2's broad passive market making. These names get
# earlier skew because the logs consistently show adverse inventory there.
EARLY_SKEW_PRODUCTS = frozenset(
    {
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "MICROCHIP_RECTANGLE",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
        "PANEL_1X2",
        "PANEL_1X4",
        "PANEL_2X2",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_SUEDE",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_VANILLA",
        "TRANSLATOR_SPACE_GRAY",
        "UV_VISOR_AMBER",
    }
)

INSIDE_TICKS = 1
MAX_LOT = 2
MIN_SPREAD = 6

DEFAULT_SKEW_START = 3
EARLY_SKEW_START = 1
SKEW_PER_POS = 0.6

UNWIND_THRESH = 7
UNWIND_TO = 4


def clamp_quote_pair(bid: int, ask: int, best_bid: int, best_ask: int):
    bid = min(bid, best_ask - 2)
    ask = max(ask, best_bid + 2)
    if bid >= ask:
        return None
    return bid, ask


class Trader:
    def bid(self):
        return 10

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            od: OrderDepth
            orders: List[Order] = []
            result[product] = orders

            if product in SKIP_PRODUCTS:
                continue

            if not od.buy_orders or not od.sell_orders:
                continue

            best_bid = max(od.buy_orders)
            best_ask = min(od.sell_orders)
            spread = best_ask - best_bid
            if spread < MIN_SPREAD:
                continue

            pos = state.position.get(product, 0)
            buy_room = POSITION_LIMIT - pos
            sell_room = POSITION_LIMIT + pos

            if pos >= UNWIND_THRESH and sell_room > 0:
                qty = min(pos - UNWIND_TO, sell_room)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                continue

            if pos <= -UNWIND_THRESH and buy_room > 0:
                qty = min(-pos - UNWIND_TO, buy_room)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                continue

            skew_start = EARLY_SKEW_START if product in EARLY_SKEW_PRODUCTS else DEFAULT_SKEW_START
            skew_mag = max(0, abs(pos) - skew_start + 1) * SKEW_PER_POS
            skew = int(round(skew_mag * (1 if pos > 0 else -1)))

            quoted = clamp_quote_pair(
                best_bid + INSIDE_TICKS - skew,
                best_ask - INSIDE_TICKS - skew,
                best_bid,
                best_ask,
            )
            if quoted is None:
                continue

            our_bid, our_ask = quoted

            if buy_room > 0:
                orders.append(Order(product, our_bid, min(MAX_LOT, buy_room)))

            if sell_room > 0:
                orders.append(Order(product, our_ask, -min(MAX_LOT, sell_room)))

        return result, 0, ""
