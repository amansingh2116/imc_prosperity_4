
# IMC Prosperity Round 5 Analysis and Trading Approach

## What the challenge requires
The trader must implement a `Trader` class with a `run()` method that receives a `TradingState` and returns a dictionary of orders, a conversion count, and a serializable `traderData` string. The platform enforces per-product position limits, and the Lambda environment is stateless, so any learning has to be carried through `traderData`. The challenge docs also state that the new round uses only the 50 newly listed products, with a position limit of 10 per product. fileciteturn0file0

## What the data shows
The provided price files for days 2, 3, and 4 have 50 products and 10,000 timestamps per day. A few patterns stand out:

- Several groups behave like a shared basket with strong cross-sectional structure.
- The cleanest baskets are `Protein Snack Packs`, `Purification Pebbles`, `Construction Panels`, `Instant Translators`, and `UV-Visors`.
- Across the sample days, the products in those groups have stable relative offsets versus their group mean.
- The residuals around the basket are mildly mean-reverting, which is exactly the kind of edge a small-limit strategy can exploit.
- The best groups are the ones with the strongest and most stable residual structure, especially `Protein Snack Packs` and `Purification Pebbles`.

## How the trader works
The submitted trader uses three ideas together:

- **Basket fair value**: for each group, it computes the current group mean and adds a product-specific offset estimated from the historical sample data.
- **Stateful smoothing**: it keeps EMAs for each product and each group so the fair value adapts instead of staying fixed.
- **Inventory control**: it skews fair value against the current position so it naturally reduces risk when inventory becomes too large.

## Execution logic
- It buys when the best ask is meaningfully below fair value.
- It sells when the best bid is meaningfully above fair value.
- It uses larger size only when the edge is stronger.
- It keeps orders small and conservative because the position limit is only 10.

## Why this is a reasonable final-round base
This is not a brittle one-product script. It is a structured relative-value trader that:
- uses the category relationships in the data,
- adapts to the live stream through EMAs,
- avoids overfitting to one day,
- and stays within the exchange constraints.

## Files
- Trader code: `trader_round5.py`
