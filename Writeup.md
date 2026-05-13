# IMC Prosperity 4 Writeup - Team Noisy_room

This writeup documents our IMC Prosperity 4 submission from the repository evidence: the problem statements, data folders, notebooks, manual-round notes, final algorithm files, logs, JSON result files, and platform screenshots. The goal is not to make the run look cleaner than it was. The interesting part of this competition was the full process: learning market microstructure, building analysis tooling, finding real edges in early rounds, overreaching in some later strategies, and still extracting strong manual-round results through structured reasoning.

## 1. Executive Summary

IMC Prosperity 4 combined two different styles of trading:

- **Algorithmic trading**, where we submitted a Python `Trader` class that ran repeatedly on order-book states, placed limit orders, respected position limits, and used only the information available at each timestamp.
- **Manual trading**, where each round contained a puzzle-style decision problem: auctions, allocation, bidding games, option portfolios, or news-based portfolio construction.

Our strongest algorithmic work came in the first two rounds. We identified that `INTARIAN_PEPPER_ROOT` had a clean deterministic upward drift and that `ASH_COATED_OSMIUM` rewarded passive spread capture. This produced about **96.8k** in Round 1 and about **92.0k raw** in Round 2 before the market-access fee.

The later algorithmic rounds were more difficult. Round 3 and Round 4 introduced `HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`, and a full strip of VEV vouchers/options. We built Black-Scholes style fair values, residual signals, implied-volatility logic, delta hedging, flattening, adverse-selection filters, and counterparty-aware signals. Those ideas were directionally reasonable but not robust enough live; the final algorithmic results were negative. Round 5 expanded the universe to 50 low-limit products. We researched basket and HMM-style approaches, but the final submitted code was a broad EMA/order-book-imbalance market maker, and it lost money across a few concentrated products.

Manual rounds were the steadier side of the run. We had a **rank 1 manual result in Round 1**, a strong **rank 11 manual result in Round 2**, and positive manual PnL in every later round shown in the screenshots.

## 2. Result Summary

The table below uses the exact repository artifacts:

- Algorithmic JSON values come from `round */result/*.json`.
- Platform values and manual PnL come from the screenshots in each `result/` directory.
- Round 2 has both a raw algorithmic value and a net platform value because the final code bid a `MAF_BID` of 4,500.

| Round | Algorithmic result | Manual result | Round total shown | Notes |
|---|---:|---:|---:|---|
| Round 1 | +96,791.39 | +87,995 | +184,786 | Manual rank 1; algorithm rank 1267th; overall position 908 |
| Round 2 | +92,007.03 raw, +87,507 net | +215,912 | +303,419 | Net algorithm subtracts 4,500 market-access fee; manual rank 11; overall position 218 |
| Round 3 | -29,787.00 | +55,248 | +25,461 | Options round began; manual stayed positive |
| Round 4 | -103,772.00 | +23,566 | -80,206 | Loss dominated by `HYDROGEL_PACK` |
| Round 5 | -95,056.68 | +91,195 | -3,862 | Manual almost offset the algorithmic drawdown |

Product-level algorithmic PnL from the final JSONs:

| Round | Main winners | Main losers |
|---|---|---|
| Round 1 | `INTARIAN_PEPPER_ROOT` +79,255; `ASH_COATED_OSMIUM` +17,536 | None |
| Round 2 | `INTARIAN_PEPPER_ROOT` +79,199; `ASH_COATED_OSMIUM` +12,808 | None before MAF |
| Round 3 | Most skipped vouchers flat; `VEV_5400` only -24 | `VEV_5100` -10,397; `VEV_5000` -7,182; `VELVETFRUIT_EXTRACT` -4,752; `HYDROGEL_PACK` -4,354 |
| Round 4 | `VEV_5200` +2,041; `VEV_5300` +1,273 | `HYDROGEL_PACK` -98,613; `VELVETFRUIT_EXTRACT` -4,433 |
| Round 5 | `PEBBLES_XL` +31,061; `UV_VISOR_YELLOW` +9,876; `OXYGEN_SHAKE_MORNING_BREATH` +9,709 | `PEBBLES_S` -29,553; `SLEEP_POD_NYLON` -17,852; `PEBBLES_M` -13,638; `UV_VISOR_RED` -12,827 |

## 3. Preparation and Prerequisites

Before attacking the live rounds, we built a basic trading foundation from the `theory/` folder:

- `Trading glossary.md` covered the order book, bids, asks, spreads, limit orders, market orders, price-time priority, liquidity taking, and market making.
- `Writing an Algorithm in Python.md` clarified the IMC bot interface: `Trader.run`, `TradingState`, `OrderDepth`, position limits, `traderData`, and the 900ms execution constraint.
- The attached order-book image and theory notes helped translate abstract concepts into the exact data structures we would receive.

This mattered because most early mistakes in Prosperity are not advanced math errors. They are interface and microstructure errors: quoting through the wrong side, forgetting position limits, assuming orders always fill, carrying unsafe inventory, or using future data in a live-style strategy.

### Tooling We Used

The repository shows a clear analysis loop:

1. **Read the round problem statement** from `R*_Problem.md`.
2. **Inspect official CSV data** from each `data/` folder.
3. **Prototype in notebooks**, especially the round-specific product analysis notebooks and the universal log-analysis dashboard.
4. **Submit versions to the platform** and download logs/JSON results.
5. **Use the log analysis notebook** to inspect PnL, inventory, fills, quote placement, adverse selection, and parameter sensitivity.
6. **Iterate final code** into the `result/*.py` file for each round.

The most important reusable tools were:

- `imc_prosperity_log_analysis.ipynb`, our universal log dashboard.
- Round-specific `product_analysis.ipynb` notebooks for fair value, spread, returns, trade flow, ACF/PACF, pair analysis, and sensitivity experiments.
- The community visualizer/backtester workflow noted in `other/optimizations.md`.
- Manual-round markdown files where we wrote the math before submitting.

We also used LLM prompts as an analysis accelerator, not as an auto-trader. The useful parts were asking for sanity checks, root-cause analysis, and alternate explanations after we already had data-driven hypotheses.

## 4. Tutorial Round

### Problem

The tutorial round introduced two products:

- `EMERALDS`
- `TOMATOES`

Both had position limits of 80. The point of the round was to learn the platform, understand the bot interface, and build intuition for spread capture versus fair-value estimation.

### Analysis

From the README and optimization notes, our main observations were:

- `EMERALDS` was highly stationary around a fair value of **10,000**.
- Bots quoted `EMERALDS` around **9992/10008** for most of the session, giving a stable spread to step inside.
- `TOMATOES` had a less stable mean near **4991**, visible short-term drift, and slower mean reversion.
- Microprice and order-book imbalance had some predictive value, but the signal was much cleaner in `EMERALDS` than in `TOMATOES`.

The most important tutorial lesson was that more sophistication was not automatically better. One version added microprice adjustment and late-session skew to `EMERALDS`; this broke a previously good strategy by moving quotes away from the profitable passive levels and trapping inventory. The forensic notes in `tutorial_round/algorithm/v7.py` explicitly document that `EMERALDS` generated most of the profit when treated as a simple fixed-fair-value market-making product.

### Final Tutorial Logic

For `EMERALDS`, the final idea was:

- Fixed fair value: `10000`.
- Quote one tick inside the common bot spread, around `9993/10007`.
- Avoid unnecessary microprice adjustment.
- Use inventory skew only as a small risk-control mechanism.
- Aggressively take only rare mispriced quotes.

For `TOMATOES`, the final idea was:

- Use a fast EMA because the product drifted more.
- Include a small microprice adjustment.
- Quote with a wider edge than `EMERALDS`.
- Skew quotes based on inventory to avoid getting stuck at the limit.

The tutorial round became the template for the rest of the repository: identify the product behavior first, then choose the simplest strategy that matches that behavior.

## 5. Round 1 - Trading Groundwork

### Problem

Round 1 introduced:

- `ASH_COATED_OSMIUM`
- `INTARIAN_PEPPER_ROOT`

Both had position limits of 80. The manual challenge was an exchange auction for `DRYLAND_FLAX` and `EMBER_MUSHROOM`.

### Algorithmic Analysis

The two products behaved very differently.

`INTARIAN_PEPPER_ROOT` had a very clean upward trend. The final submission modeled fair value as:

```text
fair_value = intercept + 0.001 * timestamp
```

The intercept was calibrated from the first observed midprice. Once this was understood, the strategy was not really a market maker; it was closer to a trend-following/buy-and-hold strategy:

- Sweep asks that were below the drifting fair value plus a tolerance.
- Build to the +80 position limit.
- Avoid selling because the deterministic drift made early inventory valuable.

`ASH_COATED_OSMIUM` was more suitable for market making:

- Track an EMA fair value.
- Take clearly cheap asks or rich bids around the EMA.
- Place passive quotes around fair value.
- Skew quotes based on inventory.
- Taper size when inventory got close to the limit.

### Final Algorithm

The final Round 1 file was `round 1/result/269313.py`.

Important behavior:

- `INTARIAN_PEPPER_ROOT`: linear-trend fair value, aggressive buying, hold inventory.
- `ASH_COATED_OSMIUM`: EMA mean reversion, take-and-make logic, passive spread capture.

### Manual Round

The manual puzzle was a clearing-price auction. We analyzed how bid price and quantity interacted with clearing price and profit.

Final orders from `R1_manual.md`:

| Product | Bid quantity | Bid price | Clearing price | Manual PnL |
|---|---:|---:|---:|---:|
| `DRYLAND_FLAX` | 9,999 | 30 | 29 | +9,999 |
| `EMBER_MUSHROOM` | 19,999 | 17 | 16 | +77,996.10 |
| Total | | | | +87,995.10 |

The key insight was that for the final bid price, using more quantity increased absolute profit even when margin per unit was not maximized. This produced a platform manual rank of **1st**.

### Performance

Round 1 was our best complete round:

- Algorithmic PnL: **+96,791.39**
- Manual PnL: **+87,995**
- Round total: **+184,786**

The product breakdown confirms that the round was driven by the trend edge:

- `INTARIAN_PEPPER_ROOT`: **+79,255**
- `ASH_COATED_OSMIUM`: **+17,536.39**

## 6. Round 2 - Market Access Fee and Allocation

### Problem

Round 2 kept the same algorithmic products:

- `ASH_COATED_OSMIUM`
- `INTARIAN_PEPPER_ROOT`

The new twist was a **Market Access Fee**. Teams could bid for 25% more counterparty quotes, but only the top 50% of bids received the extra access. The manual challenge was an allocation problem with a 50,000 budget split across research, scale, and speed.

### Algorithmic Analysis

Round 2 was mostly a refinement round.

For `INTARIAN_PEPPER_ROOT`, the linear trend remained the dominant edge. The final code simplified the idea:

- Use `fair = intercept + 0.001 * timestamp`.
- Buy to the position limit early.
- Use a tolerance around fair value.
- Do not waste edge trying to mean-revert a product that was structurally trending.

For `ASH_COATED_OSMIUM`, the Round 2 result file contains an important data-driven conclusion: profitable fills were almost entirely passive. The strategy moved away from overactive taking and focused on:

- Quoting one tick inside the common spread.
- Using size 20.
- Keeping a simple inventory skew.
- Avoiding unnecessary EMA/take logic when passive spread capture was the cleaner source of PnL.

### Market Access Fee Decision

The final code used:

```python
MAF_BID = 4500
```

The reasoning in `round 2/result/359796.py` estimated the extra-access value as roughly:

```text
25% additional ACO flow * about 27,000 expected ACO edge ~= 6,750
```

A 4,500 bid was intended to be high enough to qualify while still leaving positive expected value. In the final platform result, this explains the difference between:

- Raw JSON algorithmic PnL: **+92,007.03**
- Net screenshot algorithmic PnL: **+87,507**

### Manual Round

The manual challenge, "Invest & Expand", asked us to allocate budget across:

- Research
- Scale
- Speed

The objective was:

```text
Research * Scale * Speed - Budget_Used
```

Our final allocation was:

| Category | Allocation |
|---|---:|
| Research | 15% |
| Scale | 45% |
| Speed | 40% |

The idea was to use the full budget, keep research from becoming overfunded under diminishing returns, and put meaningful weight into speed because speed was likely rank-sensitive.

### Performance

Round 2 was also strong:

- Raw algorithmic PnL: **+92,007.03**
- Net algorithmic platform PnL after MAF: **+87,507**
- Manual PnL: **+215,912**
- Round total shown: **+303,419**
- Platform position after Round 2 screenshot: **218**

Product-level raw PnL:

- `INTARIAN_PEPPER_ROOT`: **+79,199**
- `ASH_COATED_OSMIUM`: **+12,808.03**

Compared with Round 1, `ASH_COATED_OSMIUM` made less, but the strategy was cleaner and more intentionally tied to passive fill edge.

## 7. Round 3 - Gloves Off

### Problem

Round 3 introduced a much harder market:

- Spot-like products:
  - `HYDROGEL_PACK`
  - `VELVETFRUIT_EXTRACT`
- Vouchers/options:
  - `VEV_4000`
  - `VEV_4500`
  - `VEV_5000`
  - `VEV_5100`
  - `VEV_5200`
  - `VEV_5300`
  - `VEV_5400`
  - `VEV_5500`
  - `VEV_6000`
  - `VEV_6500`

The position limits were 200 for `HYDROGEL_PACK` and `VELVETFRUIT_EXTRACT`, and 300 for each voucher. The manual challenge was the Celestial Gardeners' Guild bidding game with reserve prices from 670 to 920 and two submitted bids.

### Algorithmic Analysis

This was the first round where option modeling became central.

For the underlyings, we used:

- Fast and slow EMAs.
- Order-book imbalance.
- Microprice-style fair-value shifts.
- Passive market making around estimated fair value.
- Inventory-aware quote skew.

For the vouchers, we modeled them as call options on `VELVETFRUIT_EXTRACT`:

- Estimate a theoretical price using Black-Scholes style logic.
- Compare market mid to model fair value.
- Track residuals because raw model error can persist by strike.
- Trade only selected strikes where the signal looked meaningful.

The final Round 3 algorithm focused on:

- `VEV_5000`
- `VEV_5100`
- `VEV_5300`
- `VEV_5400`

It skipped the far strikes and several mid strikes where liquidity/model confidence looked worse.

### Final Algorithm

The final file was `round 3/result/485097.py`.

Important features:

- Spot market making in `HYDROGEL_PACK` and `VELVETFRUIT_EXTRACT`.
- OBI-based fair-value shift.
- Voucher valuation using theoretical option price plus residual adjustment.
- Small-order/adverse-selection filters.
- End-of-simulation flattening.
- Zero VE clearing and close-only behavior late in the session.

The strategy was more sophisticated than R1/R2, but it also had more model risk. We had to estimate both the underlying fair value and the correct option mispricing. When either layer was wrong, the option trade could lose even if the code looked mathematically coherent.

### Manual Round

The Round 3 manual challenge asked us to submit two bids. The second bid could be penalized depending on the global average, so the puzzle was not simply "bid high".

Our final recommendation in `R3_manual.md` was:

```text
b1 = 766
b2 = 866
```

This came from:

- Equal-thirds style reserve-price reasoning.
- Behavioral assumptions about where other teams would bid.
- Monte Carlo and Dirichlet-style modeling of crowd averages.
- A risk preference for high expected value without making the second bid too fragile.

### Performance

Round 3 result:

- Algorithmic PnL: **-29,787**
- Manual PnL: **+55,248**
- Round total shown: **+25,461**

Algorithmic loss breakdown:

- `VEV_5100`: **-10,397**
- `VEV_5000`: **-7,182**
- `VELVETFRUIT_EXTRACT`: **-4,752**
- `HYDROGEL_PACK`: **-4,354**
- `VEV_5300`: **-3,078**
- `VEV_5400`: **-24**

The loss was spread across both spot and option components. That is a useful diagnostic: the problem was not one single bad strike; the overall model and execution assumptions were not robust enough.

## 8. Round 4 - Counterparty Information and Exotic Manual Options

### Problem

Round 4 kept the Round 3 product universe, but added counterparty identities to market trades. The idea was that some bots or counterparties might have predictable behavior, allowing us to infer informed flow or exploitable patterns.

The manual challenge was an Aether Crystal option-pricing exercise with vanilla and exotic derivatives.

### Algorithmic Analysis

The final Round 4 file was `round 4/result/543047.py`.

Compared with Round 3, we added:

- Counterparty-aware trade signals.
- An "omni" signal combining order-book imbalance and aggressor trade flow.
- Dynamic strike-specific implied-volatility calibration.
- Black-Scholes pricing for voucher fair values.
- Delta hedging through `VELVETFRUIT_EXTRACT`.
- Hard hedge thresholds.
- A more active selected strike set: 5000, 5100, 5200, 5300, 5400, 5500.

The intended logic was:

1. Use `VELVETFRUIT_EXTRACT` as the option underlying.
2. Price each voucher with a theoretical model.
3. Adjust fair value using dynamic IV and recent residual behavior.
4. Trade options only when edge exceeded transaction/noise thresholds.
5. Hedge delta exposure through the underlying.
6. Use disclosed counterparty flow to bias fair value.

In hindsight, this was a lot of moving parts for a live one-round model. Counterparty information was useful in principle, but we did not turn it into a robust enough "follow/avoid this exact trader" framework. Instead, it became another signal layered onto an already sensitive options model.

### Manual Round

The manual round was one of the most mathematically detailed parts of the repository. In `R4_manual.md`, we priced Aether Crystal derivatives using:

- Black-Scholes-Merton.
- Cox-Ross-Rubinstein binomial trees.
- Monte Carlo checks.
- Careful time conversion.

The important correction was time-to-expiry:

```text
2 weeks = 10 trading days = 40 platform steps
3 weeks = 15 trading days = 60 platform steps
```

This prevented the common mistake of using calendar-style time instead of the challenge's trading-time scale.

The final manual portfolio was:

| Action | Instrument | Quantity |
|---|---|---:|
| Sell | `AC_60_C` | 50 |
| Buy | `AC_50_P_2` | 50 |
| Buy | `AC_50_C_2` | 50 |
| Sell | `AC_50_CO` | 50 |
| Sell | `AC_40_BP` | 50 |
| Buy | `AC_45_KO` | 500 |

Our model estimated roughly +171k to +173k expected PnL, but the platform screenshot showed **+23,566**. That still made the manual side positive, but it also showed that the model expectation was much more optimistic than the realized challenge scoring.

### Performance

Round 4 result:

- Algorithmic PnL: **-103,772**
- Manual PnL: **+23,566**
- Round total shown: **-80,206**

Product-level result:

- `HYDROGEL_PACK`: **-98,613**
- `VELVETFRUIT_EXTRACT`: **-4,433**
- `VEV_5000`: **-2,444**
- `VEV_5100`: **-1,621**
- `VEV_5200`: **+2,041**
- `VEV_5300`: **+1,273**

The key finding is that Round 4 was not mainly an options-theory failure. The final loss was dominated by `HYDROGEL_PACK`. That suggests the spot fair-value or execution logic was too aggressive, or that hedging/market-making inventory risk in the spot product was not controlled well enough.

## 9. Round 5 - Fifty Products and Ashflow Alpha

### Problem

Round 5 expanded the algorithmic universe to 50 products in 10 groups, each with a position limit of 10:

- Galaxy Sounds Recorders
- Vertical Sleeping Pods
- Organic Microchips
- Purification Pebbles
- Domestic Robots
- UV-Visors
- Instant Translators
- Construction Panels
- Oxygen Shakes
- Protein Snack Packs

The manual challenge was "Extra! Extra! Read All About It!", a one-day news-based portfolio construction problem using the Ashflow Alpha newsletter. The budget was 1,000,000, and the fee for a position was quadratic in the percentage allocated.

### Algorithmic Research

The Round 5 repository contains more research than the final submission used.

In `round 5/algorithm/round5_analysis.md`, we identified likely basket/group structure:

- Protein Snack Packs
- Purification Pebbles
- Construction Panels
- Instant Translators
- UV-Visors

The intended approach was a relative-value basket model:

```text
product_fair = group_mean + product_offset
```

Then trade deviations from the group fair value while controlling inventory.

The HMM notebook explored:

- Basket maps.
- Pair/spread candidates.
- Three-state Gaussian HMMs.
- Baum-Welch fitting.
- Viterbi state decoding.
- Regime classification.
- Spread diagnostics.

This was good research, but it did not fully land in the final submitted code.

### Final Algorithm

The final submitted file was `round 5/result/582082.py`.

The actual final code was a broad generic market maker:

- Maintain an EMA per product with alpha 0.1.
- Compute order-book imbalance:

```text
OBI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
```

- Estimate fair value as:

```text
fair = EMA + OBI * spread * 0.2
```

- Apply inventory skew:

```text
inventory_skew = position / 10 * spread * 0.4
```

- Quote around fair value using spread-based offsets.
- Keep max lot small, with `MAX_LOT = 3`.
- Respect the universal position limit of 10.

Earlier versions such as `v7.py` and `v8_robust.py` were more conservative. They skipped some poor products and used passive inside quoting, unwind logic, and product filters. The final result file was simpler and broader.

In hindsight, this was the central Round 5 mistake. A generic EMA/OBI market maker looked safe because every product had a small limit, but with 50 products the small risks added up. Without a strong basket or product-specific edge, the strategy traded too many instruments and let adverse selection accumulate.

### Manual Round

The manual problem had six submitted trades in the platform screenshot:

| Product | Action | Allocation | Investment | Fee | Realized PnL |
|---|---|---:|---:|---:|---:|
| Lava cake | Sell | 13% | 130,000 | 16,900 | +65,459 |
| Obsidian cutlery | Buy | 9% | 90,000 | 8,100 | +824 |
| Thermalite core | Buy | 11% | 110,000 | 12,100 | +12,276 |
| Pyroflex cells | Sell | 8% | 80,000 | 6,400 | +9,228 |
| Sulfur reactor | Buy | 14% | 140,000 | 19,600 | +4,794 |
| Magma ink | Buy | 5% | 50,000 | 2,500 | -1,386 |

Total:

- Budget spent: **60%**
- Investment: **600,000**
- Fees: **65,600**
- Manual PnL: **+91,195**

The reasoning in `imc_prosperity_r5_analysis.docx` was based on a Lagrangian allocation framework:

```text
PnL = sum(mu_i * V_i * 10,000) - sum(100 * V_i^2)
```

This gives the first-order condition:

```text
V_i* = 50 * mu_i
```

where `V_i` is the allocation in percentage points and `mu_i` is the expected fractional return. The key practical insight was that using 100% of the budget was not automatically optimal because the fee grew quadratically. If the marginal alpha did not clear the marginal fee, leaving budget unused was correct.

The strongest news reads were:

- Short Lava Cakes because actual lava was found in the product, triggering health review and legal risk.
- Buy Sulfur Reactor because of index inclusion and forced-flow logic.
- Buy Thermalite Cores because the forecast showed a large increase in projected usage.
- Sell Pyroflex Cells because the tax cut was ending.
- Buy Obsidian Cutlery as a supply-shock trade, not a product-contamination trade.
- Small buy in Magma Ink as the tradable proxy for the Lava Fountain Pen demand story.

### Performance

Round 5 result:

- Algorithmic PnL: **-95,056.68**
- Manual PnL: **+91,195**
- Round total shown: **-3,862**

The algorithmic PnL chart in the screenshot drifted downward through most of the session, reached a large drawdown near the late timestamps, then recovered slightly into the close. Product-level PnL shows why the final total stayed negative despite several winners:

Main winners:

- `PEBBLES_XL`: **+31,061.12**
- `UV_VISOR_YELLOW`: **+9,875.77**
- `OXYGEN_SHAKE_MORNING_BREATH`: **+9,708.80**
- `TRANSLATOR_SPACE_GRAY`: **+8,698.85**

Main losers:

- `PEBBLES_S`: **-29,552.50**
- `SLEEP_POD_NYLON`: **-17,852.19**
- `PEBBLES_M`: **-13,637.84**
- `UV_VISOR_RED`: **-12,827.37**
- `TRANSLATOR_VOID_BLUE`: **-12,127.00**

The lesson is that low position limits do not make a broad strategy low-risk when the product count is high.

## 10. What Worked

### Clean Product-Specific Hypotheses

The best example was `INTARIAN_PEPPER_ROOT`. Once the linear drift was identified, the right strategy was simple: buy early and hold. We did not need a complex market-making model. That product alone generated about **79k** in both Round 1 and Round 2.

### Passive Spread Capture When the Data Supported It

`ASH_COATED_OSMIUM` worked when treated as a passive quoting product. The Round 2 analysis explicitly recognized that most profitable fills came from making, not taking.

### Manual Round Discipline

Manual rounds were handled with written math instead of gut feel:

- Round 1: clearing-price auction optimization.
- Round 2: budget allocation across research, scale, and speed.
- Round 3: two-bid reserve-price/crowd model.
- Round 4: option pricing with time conversion, BSM, CRR, and Monte Carlo checks.
- Round 5: quadratic-fee Lagrangian allocation from news signals.

The results reflect that discipline: every manual screenshot was positive, and Round 1 manual was rank 1.

### Forensic Log Analysis

The repository's most valuable engineering asset was the log-analysis workflow. The notebooks helped inspect:

- PnL by product.
- Inventory over time.
- Fill quality.
- Spread capture.
- Quote placement.
- Directional exposure.
- Adverse selection.
- Parameter sensitivity.

This was especially useful in the tutorial round, where forensic analysis caught why a more complicated `EMERALDS` strategy regressed.

## 11. What Did Not Work

### Too Many Signals in Options Rounds

Round 3 and Round 4 layered many signals:

- EMA fair value.
- OBI.
- Microprice.
- Black-Scholes theoretical value.
- Residual EMA.
- Dynamic IV.
- Counterparty flow.
- Delta hedging.
- Flattening.
- Strike selection.

Each individual idea was defensible, but the combined system was fragile. When live behavior differed from the sample data, it was hard to know which component was wrong quickly enough.

### Underestimating Spot Inventory Risk

Round 4 showed this most clearly. The final loss was dominated by `HYDROGEL_PACK`, not by the option strip. That means the spot strategy carried more risk than intended.

### Research Not Fully Reaching Final Code

Round 5 had basket research and HMM exploration, but the final submitted algorithm was a generic per-product EMA/OBI market maker. The final code did not use the strongest structural hypothesis in the research notes: group-relative fair value.

### Treating Small Limits as Automatic Safety

In Round 5, every product had a limit of only 10. That felt safe. But 50 products meant the total risk surface was large. A few persistent losers overwhelmed a set of smaller winners.

## 12. Final Takeaways

The biggest lesson from our Prosperity 4 run is that simple edges with strong evidence beat complex strategies with weak calibration.

In R1/R2, we had direct, interpretable product behavior:

- One product trended almost deterministically.
- One product rewarded passive spread capture.

The final algorithms matched those behaviors and made money.

In R3/R4, we built more sophisticated models, but the edge was less directly observable. Options pricing required correct underlying fair value, correct volatility assumptions, correct execution, and correct hedging. The more links in that chain, the easier it was for live PnL to break.

In R5, we did good research but submitted a strategy that was not specific enough to the discovered structure. The writeup and notebooks point toward a better final approach: group-relative value with explicit product filters, not generic quoting across all 50 products.

Overall, the repository shows a strong progression:

- We learned the platform interface.
- We built reusable analysis tools.
- We solved the early market-structure products well.
- We produced strong manual-round reasoning.
- We documented failures honestly enough to see what should change next time.

If we were rerunning the competition, the main changes would be:

1. Keep the algorithmic strategy as simple as the evidence allows.
2. Require product-level PnL attribution before adding new signals.
3. Separate fair-value modeling from execution/risk-control experiments.
4. In options rounds, cap spot inventory risk more aggressively.
5. In large-universe rounds, trade only products with a confirmed structural edge.
6. Make sure the final submitted code actually implements the strongest research hypothesis.
