# IMC Prosperity 4 - Team Noisy_room

All the data, algorithms, analysis tools, and learning resources for our team "Noisy_room" participating in IMC Prosperity 4 2026.

## 📋 Quick Overview

**IMC Prosperity 4** is an algorithmic trading competition where teams develop Python algorithms to trade against bots on a simulated exchange. The goal is to maximize profit and loss (PnL) measured in XIRECs (currency of the game) across multiple rounds of trading with different products.

**Our Current Focus:** Tutorial Round (Round 0) with EMERALDS and TOMATOES products
- **Best Score:** v7 algorithm
- **Key Success Factor:** Understanding bot behavior patterns and exploiting edge opportunities

---

## 📁 Repository Structure

### Root Level Files

```bash
- imc_prosperity_log_analysis.ipynb    # Logger: Comprehensive log files analysis notebook for reviewing algorithm performance
- README.md                             # This file, serving as the central documentation hub for our team’s strategy development and resources
```

### `/theory/` — Learning Materials & Documentation
Educational resources for understanding IMC Prosperity and algorithmic trading:

| File | Purpose |
|------|---------|
| `IMC_Prosperity_Complete_Study_Guide.pdf` | Complete IMC Prosperity previous rounds based study guide |
| `IMC_Prosperity_Revised_Guide.pdf` | Revised guide (Part 1) with key concepts |
| `IMC_Prosperity_Revised Guide_Part2.pdf` | Revised guide (Part 2) with advanced concepts |
| `Order_Book.png` | Visual reference for order book structure |
| `Trading glossary.md` | Definitions of trading terms (orders, positions, exchanges, etc.) |
| `Writing an Algorithm in Python.md` | Tutorial on implementing the Trader class and run() method |

**When to use these:**
- Start with `Trading glossary.md` if unfamiliar with trading terminology
- Use `Writing an Algorithm in Python.md` to understand the implementation framework and guide for official rules and constraints
- Refer to the study guides (pdfs) for deeper understanding of market dynamics, strategy development, and lessons from past rounds

### `/other/` — Tools & Resources
Supplementary tools and optimizations discussion:

| File | Purpose |
|------|---------|
| `optimizations.md` | **Key file!** Resources, lessons learned, and strategy optimization notes including: <br> • Useful tools (backtester, visualizer links) <br> • LLM chat conversations for algorithm development <br> • Prompt template for using LLMs to generate/improve algorithms <br> • Link to IMC Discord community |
| `Visualizer template.py` | Python template for logging output in visualizer-compatible format |

**How to use:**
- Before developing new features, check `optimizations.md` for known tools and lessons
- Use `Visualizer template.py` as a reference for implementing logging in your algorithm

### `/tutorial_round/` — Tutorial Round Development (Round 0)
This is our main working directory for algorithm development and testing:

#### `/tutorial_round/algorithm/` — Core Algorithm Files

| File | Purpose |
|------|---------|
| `v7.py` | **Current best algorithm** - Latest version with fixes from forensic analysis |
| `imc_prosperity_analysis.ipynb` | Detailed analysis notebook for tutorial round data (price dynamics, trading patterns) |

#### `/tutorial_round/log output/` — Algorithm Test Results

| File | Purpose |
|------|---------|
| `v7.json` | Machine-readable log from algorithm execution (trades, positions, PnL over time) |
| `v7.log` | Human-readable log output with strategy decisions |

**How to use:**
- Run `imc_prosperity_log_analysis.ipynb` with these logs to visualize performance
- Analyze trade patterns, position movements, and PnL components

#### `/tutorial_round/data/` — Historical Market Data
Historical data for the tutorial round used for backtesting and analysis:

| File | Purpose |
|------|---------|
| `prices_round_0_day_-1.csv` | Order book snapshots for day -1 (bid/ask prices and volumes at each timestamp) |
| `prices_round_0_day_-2.csv` | Order book snapshots for day -2 |
| `trades_round_0_day_-1.csv` | Executed trades log for day -1 (trades between other market participants) |
| `trades_round_0_day_-2.csv` | Executed trades log for day -2 |
| `__MACOSX/` | macOS metadata (can be ignored) |

**CSV Format:**
- **prices CSV:** Columns include timestamp, product, bid/ask prices at multiple levels (1-3), volumes, mid price, PnL
- **trades CSV:** Columns include timestamp, product, buyer, seller, price, quantity

**How to use:**
- Load in analysis notebooks to understand market dynamics
- Use for backtesting new algorithm versions

---

## 🚀 Development Workflow

### For Current Round (Tutorial Round - Ongoing)

1. **Make Changes to Algorithm:**
   ```
   Edit: /tutorial_round/algorithm/v7.py
   ```
   - Modify configuration (POSITION_LIMITS, edge values, EMA alpha, etc.)
   - Update strategy logic in the `run()` method
   - Use comments to explain forensic analysis (see v6→v7 example)

2. **Run Backtests:**
   - Upload `v7.py` to IMC Prosperity website for local backtesting
   - Or use the backtester tool: `https://github.com/kevin-fu1/imc-prosperity-4-backtester.git`
   ```powershell
   cd C:\Users\amans\OneDrive\Documents\imc backtest\imc-prosperity-4-backtester
   $env:PYTHONPATH="C:\Users\amans\OneDrive\Documents\imc backtest\imc-prosperity-4-backtester\prosperity4bt"
   python -m prosperity4bt "C:\Users\amans\Downloads\[ALGORITHM].py" 0
   ```

3. **Analyze Results:**
   - Download `v7.json` and `v7.log` from IMC test results
   - Save to `/tutorial_round/log output/`
   - Run cells in `imc_prosperity_log_analysis.ipynb` to generate visualizations

4. **Document Findings:**
   - Add forensic analysis comments to algorithm file (see v6 example in `v7.py` header)
   - Update `/other/optimizations.md` with new insights

### For Future Rounds

**Structure for Round 1, Round 2, etc.:**

```
/round_1/
  /algorithm/
    v1.py          (first version for round 1)
    analysis.ipynb
    /log output/
      v1.json
      v1.log
  /data/
    prices_round_1_day_0.csv
    prices_round_1_day_1.csv
    trades_round_1_day_0.csv
    trades_round_1_day_1.csv
```

**When Starting a New Round:**

1. Create new directory: `/round_X/` at root level
2. Create subdirectories: `algorithm/`, `data/`, `log output/`
3. Obtain new product data from IMC and place in `data/`
4. Start with `v1.py` as base:
   - Copy best algorithm from previous round
   - Research new products (check Discord announcements, official guides)
   - Adapt configuration for new products:
     ```python
     # v1.py for Round X
     # New Products: [PRODUCT1], [PRODUCT2], ...
     # Historical analysis from /round_X/data/ shows:
     #   - PRODUCT1: behavior characteristics here
     #   - PRODUCT2: behavior characteristics here
     
     POSITION_LIMITS = {
         "[PRODUCT1]": [LIMIT],
         "[PRODUCT2]": [LIMIT],
     }
     ```
5. Create analysis notebook by copying from `tutorial_round/algorithm/imc_prosperity_analysis.ipynb`
6. Test and iterate (v1 → v2 → v3...)

---

## 📊 Key Files Explained

### `imc_prosperity_log_analysis.ipynb` (Root)
**Purpose:** Comprehensive performance analysis of algorithm executions

**Sections:**
1. **Data Loading:** Reads JSON/log files from test results
2. **Trade Analysis:** Examines all trades (buys, sells, fills)
3. **Position Tracking:** Charts position over time for each product
4. **PnL Breakdown:** Calculates PnL by product and analyzes:
   - Round-trip profit (trade entry → exit)
   - Spread capture (bid-ask edge profit)
   - Inventory impact (position-dependent slippage)
5. **Bot Behavior Analysis:** Identifies patterns in market maker quotes
6. **Fair Value Estimation:** Compares algorithm FV to market mid-price
7. **Performance Metrics:** Sharpe ratio, fill rates, efficiency ratios
8. **Visualization:** Plots for price paths, positions, PnL curves

**How to use:**
- After each backtest, update `JSON_PATH` and `LOG_PATH` to point to new logs
- Run all cells to generate comprehensive report
- Export visualizations for documentation

### `/tutorial_round/algorithm/imc_prosperity_analysis.ipynb`
**Purpose:** Analyze market data characteristics specific to tutorial round

**Sections:**
1. **Price Dynamics:** 
   - Mean-reversion vs trending behavior
   - Volatility patterns
   
2. **Bot Behavior:**
   - Quote frequency and patterns
   - Spread analysis (5% min, 14 max for TOMATOES; ~15 ticks for EMERALDS)
   
3. **Edge Identification:**
   - Round-trip PnL at different edge levels
   - Optimal quoting strategy

4. **Product-Specific Insights:**
   - EMERALDS: Rock-solid 10K fair value, bot quotes 9992/10008 98.5% of time
   - TOMATOES: Drifting price, -9.5 tick drift over session, needs EMA tracking

**Key Finding:** v4 achieved 26,643 XIRECs by understanding EMERALDS bot was quoting at 9992/10008 consistently, so quoting at 9993/10007 (edge=7) captured ALL taker flow.

---

## 🔑 Key Lessons & Insights

From our forensic analysis (documented in v7.py header):

### EMERALDS Strategy Success (v4 = 26,643)
- **Core Insight:** Bot quotes EXACTLY at 9992/10008 on 98.5% of ticks
- **Our Edge:** Quote 1 tick better (9993/10007) to intercept all taker orders
- **Key Metric:** Every fill nets 7 ticks of edge

### Common Pitfalls (v5 Regression = 11,613)
1. **Microprice Adjustment Gone Wrong:**
   - FV adjustment of ±1-2 ticks pushed our bid below bot's
   - Order below best market bid = invisible to takers
   - Lost buy-side fills, accumulated short position (-80), PnL trapped

2. **Late-Session Skew Amplification:**
   - Skew = position × skew_factor
   - Late session (high time pressure): 0.10 × 2.0 = 0.20 multiplier
   - At position -80: skew = -16, ask = 10024 (16 ticks above market)
   - Nobody buys → position trapped at hard limit

### Solution (v6→v7)
- **Fixed FV:** For EMERALDS, use static FV=10000 (no adjustments)
- **Conservative Skew:** Reduced factor from 0.10 to 0.06
- **Hard Clamps:** Quotes never go below 9990 bid / above 10010 ask (stay relevant)
- **Result:** Recovers to v4-level performance

---

## 🛠️ Tools & Resources

### Backtesting
- **IMC Official Backtester:** https://prosperity.equirag.com/
- **Community Backtester:** https://github.com/kevin-fu1/imc-prosperity-4-backtester.git
- **Visualizer:** https://jmerle.github.io/imc-prosperity-3-visualizer/ (works with v4 logs)

### Community
- **Discord Server:** https://discord.com/channels/1001852729725046804/
  - #algo-trading: Strategy discussions and quick questions
  - #announcements: Official updates and round info

### LLM Assistance
See `/other/optimizations.md` for curated LLM conversation links:
- IMC writeup with basics and trading glossary
- Orderbook analysis template
- Logger/Visualizer template
- All-rounds analysis (rounds 1, 2, 3)

---

## 📝 Notes

- **Time Constraint:** Algorithm must execute in < 900ms per iteration
- **Position Limits:** Hard limits enforced by exchange (violations cancel all orders for that product)
- **Order Validity:** Unfilled orders automatically cancelled at end of iteration
- **Logging:** Use logger.print() and logger.flush() for visualizer compatibility (not raw print())
- **Data Persistence:** Use traderData dict to maintain state across iterations (algorithm is stateless between runs)

---

## 👥 Team: Noisy_room

All documentation maintained for transparent strategy development and knowledge sharing.
