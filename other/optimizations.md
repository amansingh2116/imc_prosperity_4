# Optimizations and Resources for IMC Prosperity 4

## Useful Resources and Tools for IMC Prosperity 4 so far

1. Utilized IMC Prosperity 4 (https://prosperity.equirag.com/) for local tutorial round leaderboard and log output analysis. Additionally, leverAGED the backtester repository at https://github.com/kevin-fu1/imc-prosperity-4-backtester.git (from terminal/cli: 
`cd C:\Users\amans\OneDrive\Documents\imc backtest\imc-prosperity-4-backtester>`
`$env:PYTHONPATH="C:\\Users\\amans\\OneDrive\Documents\imc backtest\\imc-prosperity-4-backtester\\prosperity4bt"`
`python -m prosperity4bt "C:\\Users\\amans\\Downloads\\trader_v4_optimised.py" 0`) and the visualizer at https://jmerle.github.io/imc-prosperity-3-visualizer/ to locally backtest and visualize log outputs.
2. Useful LLM chats/conversations for IMC Prosperity 4:

   * IMC writeup explanation (basics, trading glossary, writing algorithm, tutorial round sample): https://gemini.google.com/share/798b5d8ba38f
   * Orderbook analysis template and tutorial round sample: https://claude.ai/share/f3548894-0186-4ca6-b48b-79932520efc9
   * Logger/Visualiser template and tutorial round sample: https://claude.ai/share/4dd0ca9b-4bc3-4816-a376-d56ce3da5fe3
   * IMC Prosperity 1,2,3 all rounds analysis: https://claude.ai/share/7f01c0ea-3af9-448d-88ac-db74a3b84fd9

3. Joined the IMC Prosperity 4 Discord server (https://discord.com/channels/1001852729725046804/) for discussion. The server has channels focused on algo trading, general discussion, and programming. The announcement channel is useful for official updates, and the manual trading channel may be useful later. The algo trading channel has some useful discussions and advice, and is a good place to ask quick questions and connect with other competitors. Note that the server also has some noise and trolling, so be cautious when engaging with others.

## Useful LLM Prompt for IMC Prosperity 4 Algorithmic Trading Strategy Development

### Prompt for generating algorithmic trading strategy Python code based on Visualizer and IMC Prosperity template

**File to upload:** Your current python algorithmic strategy, Visualizer python template (logger version), IMC official guide (writing an algorithm notion)

```md
You are an expert quantitative trading engineer specializing in IMC Prosperity-style competitions.

I am building an algorithmic trading strategy and need help converting, optimizing, debugging, and improving my Python trading algorithm.

I have attached the following:

* My current algorithm file
* Visualizer logging template (required format)
* Backtester template (if applicable)
* Official IMC Prosperity documentation and constraints

Your tasks:

1. **Template Compliance**

   * Convert my algorithm into FULLY compliant submission format
   * Must strictly follow:

     * Trader class structure
     * run() return format → (orders, conversions, traderData)
     * logger.print() and logger.flush() format (visualizer compatible)
   * Ensure NO raw print() is used
   * Ensure compatibility with both:

     * backtester
     * IMC submission environment

2. **Correctness & Constraints**

   * Ensure:

     * Position limits are NEVER violated
     * Orders are valid under IMC rules
     * No undefined behavior or runtime risks
   * Respect stateless nature (only use traderData for memory)

3. **Optimization**
   Improve the strategy while keeping logic human-readable:

   * Better execution (spread capture, adverse selection)
   * Inventory management (skewing, soft limits)
   * Smarter fair value estimation
   * Reduce overtrading / unnecessary fills
   * Add risk controls

4. **Performance Constraints**

   * Code must run < 900ms per iteration
   * Avoid heavy computations
   * Avoid unnecessary loops / repeated work

5. **Code Quality**

   * Clean, modular, readable
   * Well-structured helper functions
   * No redundant logic
   * Maintainable for future rounds

6. **Logging**

   * Keep logs informative but concise
   * Avoid excessive logging (visualizer limit)
   * Ensure logs help debugging strategy decisions

7. **Output Requirements**
   Return:

   * Final improved Python code ONLY (ready to submit)
   * No explanation unless I explicitly ask

---

If I later ask for modifications:

* Only change relevant parts
* Do NOT rewrite entire code unless necessary
* Keep style consistent

---

If I provide backtest results:

* Diagnose weaknesses (PnL, inventory, execution)
* Suggest improvements AND implement them

---

If something in my strategy is fundamentally flawed:

* Point it out clearly and fix it

---

Goal:
Produce a high-performance, competition-ready trading algorithm that is robust, efficient, and easy to iterate on.
```

### Prompt for learning quantitative finance and algorithmic trading, specifically for IMC Prosperity 4

**Files to attach:** IMC_Prosperity_Complete_Study_Guide.docx, IMC_Prosperity_Writeup.md, trader_v0.py, imc_prosperity_analysis.ipynb

```md
I am participating in IMC Prosperity 4 (an algorithmic trading competition). 
I am new to trading and quantitative finance. I have been building market-making 
algorithms with the help of LLMs, but I now need to properly understand the 
concepts behind what I have been doing so I can make better decisions myself.

CONTEXT — what I have built so far:
- A market-making algorithm (attached: trader_v6_forensic_fixed.py) for two 
  products: EMERALDS (stationary FV=10000) and TOMATOES (trending/drifting price)
- Score history: v3=1,249 | v4=26,643 | v5=11,613 (broke due to bugs) | v6=TBD
- An orderbook analysis notebook (attached: imc_prosperity_analysis.ipynb) 
  and a log file analysis notebook built from actual submission data
- Study materials (attached: IMC_Prosperity_Complete_Study_Guide.docx and 
  IMC_Prosperity_Writeup.md)

KEY FINDINGS FROM DATA ANALYSIS SO FAR (treat these as ground truth):
- EMERALDS: FV=10000 perfectly stationary, mean-reversion half-life=0.7 ticks,
  bot quotes at exactly 9992/10008 on 98.5% of ticks (spread=16 ticks, vol=13-15)
- TOMATOES: mean=4991, std=6, range=34.5 ticks, drift=-9.5 over session,
  mean-reversion half-life=28.4 ticks, bot spread=13 ticks (dynamic, moves with price)
- Microprice (wall_mid) IC=0.191 for EMERALDS, IC=0.080 for TOMATOES vs next tick
- OLS R²=0.465 EMERALDS, R²=0.412 TOMATOES from: next_ret ~ imbalance + spread_z + 
  velocity + wall_mid_divergence
- Round-trip PnL/unit: EMERALDS=+4.04, TOMATOES=+9.75 (latest working version)

WHAT I NEED TO LEARN (go through each concept in detail with examples):

1. MARKET MICROSTRUCTURE BASICS
   - What is a limit order book? How does it work in Prosperity specifically?
   - What is a maker vs taker? How does fill priority work?
   - What is bid-ask spread and why does it exist?
   - What is price impact and adverse selection?
   - What is market making vs market taking?

2. FAIR VALUE (FV) CONCEPTS
   - What is fair value and how do market makers estimate it?
   - Arithmetic mid vs microprice (wall mid) — derive the formula, explain 
     intuitively why microprice is better, with numerical examples from my data
   - EMA as FV tracker: what is alpha, half-life, how to choose it?
   - What does "mean-reversion half-life of 0.7 ticks" actually mean mathematically?
   - What does "trending vs stationary" mean for FV estimation strategy?

3. MARKET MAKING STRATEGY
   - How does passive quoting work? What determines if we get filled?
   - What is "quote priority" and why does edge=7 beat the bot with edge=8?
   - What is inventory skew and why does it help? Derive the math.
   - What is a soft position limit and why use it instead of a hard stop?
   - What is the difference between aggressive sweep and passive quoting?
   - What is a round-trip PnL and why must it be positive?

4. SIGNALS AND ALPHA
   - What is Information Coefficient (IC) and how to interpret IC=0.191?
   - What does R²=0.465 mean in the OLS regression context?
   - Why does microprice divergence predict next-tick price movement?
   - What is order flow imbalance and how does it predict price?
   - What is momentum vs mean-reversion? How do I tell from autocorrelation?

5. RISK MANAGEMENT
   - What is a drawdown and recovery ratio? How are top strategies at 7-13x?
   - What is adverse selection? How to detect it from log data?
   - Why did position=-80 (hitting hard limit) destroy PnL in my v5?
   - How does inventory skew prevent position buildup mathematically?
   - What is vol-gating and when should it trigger?

6. BACKTESTING AND OVERFITTING
   - What is the difference between the official Prosperity backtester and 
     Jasper's open-source backtester? When to use each?
   - What does overfitting mean in the context of algorithm tuning?
   - Which parameters are safe to tune (robust) vs risky to tune (overfit)?
   - Why did changing EMA alpha from 0.08→0.50 work well for TOMATOES but 
     not necessarily generalise?

For each concept:
- Explain from first principles (assume I know basic math but no trading)
- Connect it explicitly to my data (EMERALDS/TOMATOES numbers from above)
- Show the formula, then explain it in plain English
- If there is a common mistake or misunderstanding, highlight it (especially 
  mistakes I have already made, like v5's microprice bug)
- After explaining, give me a "test question" so I can verify I understood it

Start with Section 1 and proceed in order. After each section ask if I want 
to continue or if I have questions before moving on.
```

### Algorithmic Trading Strategy Refactoring Prompt

**Files to attach:** trader_v0.py, imc_prosperity_log_analysis.ipynb, log_output.log (whichever is the latest log file from your most recent submission), imc_prosperity_analysis.ipynb

```md
I am participating in IMC Prosperity 4. This is a continuation of an ongoing 
algorithm development process. Please read everything below carefully before 
responding — do not start giving advice until you have the full context.

════════════════════════════════════════════════════════
 COMPETITION CONTEXT
════════════════════════════════════════════════════════
- Tutorial round: two products, EMERALDS and TOMATOES
- Position limit: 80 units each
- Timestamps: 0 → 199,900 (2000 ticks at 100-tick intervals)
- Scoring: final mark-to-market PnL in XIRECs
- No live generalisation needed for tutorial round (fixed dataset)
- Interface: submit Python file → website runs it → download .log and .json

════════════════════════════════════════════════════════
 COMPLETE HISTORY OF ITERATIONS
════════════════════════════════════════════════════════

v3 (1,249 XIRECs) — BROKEN, two critical bugs:
  Bug 1: EMERALDS passive_edge=2 → quotes at 9998/10002, INSIDE bot spread
         (9992/10008). Bots trade at their own prices, not ours. Near-zero fills.
  Bug 2: TOMATOES EMA alpha=0.08 too slow → EMA lagged price by 10+ ticks during
         downtrend → bought at prices above market → adverse selection.

v4 (26,643 XIRECs) — WORKING, key fixes:
  Fix 1: EMERALDS edge=7 → quotes at 9993/10007, 1 tick better than bot.
         Taker bot fills us BEFORE the MM bot. Captured all EMERALDS taker flow.
  Fix 2: TOMATOES alpha=0.20, added momentum-bias correction.
  Result: EMERALDS ~25,000, TOMATOES ~1,600 XIRECs

v5 (11,613 XIRECs) — BROKE EMERALDS, two new bugs:
  Bug 1: Added microprice FV adjustment (coeff=1.3). When adj=-0.87: 
         fv=9999.13, bid=round(9999.13)-8=9991 < bot bid 9992 → invisible to takers.
         Sell fills kept happening at 10007 but no buy fills → pos drifted to -80.
  Bug 2: Late-session skew amplification (2x at ts=199900). With pos=-80 and
         skew=0.20: ask=10000+8+16=10024. Market at 10008. Nobody buys at 10024.
         Position trapped at -80. EMERALDS PnL = 13 (vs ~25,000 in v4).
  TOMATOES worked: PnL=1,021, round-trip=+9.75/unit (up from v3's +5.29)

v6 (current, attached as trader_v6_forensic_fixed.py) — forensic fixes:
  EMERALDS: edge=7 restored, NO microprice, NO late-session skew,
            skew_factor=0.06, hard bid_clamp_min=9990
  TOMATOES: alpha=0.5 kept, edge=5, microprice coeff=0.50 (conservative),
            size=40 (up from 30)

════════════════════════════════════════════════════════
 KEY DATA FINDINGS (from analysis notebooks, treat as ground truth)
════════════════════════════════════════════════════════

EMERALDS:
  FV=10000 stationary (mean-reversion half-life=0.7 ticks)
  Bot quotes 9992/10008 on 98.5% of ticks, vol=13-15 each side
  Spread=16 ticks (very stable, std=1.35)
  Microprice IC=0.191 ⭐ (strongest signal found, OLS coeff=1.325)
  OLS R²=0.465: next_ret ≈ 1.325*(wall_mid-arith_mid) + other terms
  Sim sweep: best passive_edge=8 (but edge=7 avoids bid-below-bot risk)
  Round-trip v3: +4.04/unit | v4: estimated ~6-7/unit (from 25k PnL)
  Mid-price vol mean=0.726 pts/tick

TOMATOES:
  Mean=4991, std=6, range=34.5 ticks over session
  Drift=-9.5 ticks (trends down over session)
  Mean-reversion half-life=28.4 ticks
  Bot spread: mean=13.06, std=1.69, min=5, max=14 (dynamic, follows price)
  Microprice IC=0.080, OLS coeff=0.740, R²=0.412
  Best EMA alpha=0.5 (sim), best edge=6 (sim) but edge=5 proved in v5: +9.75/unit
  Mid-price vol mean=1.237 pts/tick

════════════════════════════════════════════════════════
 ANALYSIS TOOLS I HAVE
════════════════════════════════════════════════════════
1. Log analysis notebook (attached: imc_prosperity_log_analysis.ipynb)
   Contains 11+ sections: dashboard, bot fingerprinting, FV analysis, PnL
   attribution, adverse selection, quote placement, alpha signals, parameter
   sweep, leaderboard metrics, microprice, market dynamics, round-trip analysis.
   Run this on every new .log file after each submission.

2. Orderbook analysis notebook (attached: imc_prosperity_analysis.ipynb)
   Contains historical orderbook data analysis that informed all parameters.

3. Files attached to THIS CHAT:
   - trader_v6_forensic_fixed.py (current algorithm)
   - imc_prosperity_log_analysis.ipynb (analysis notebook)
   - [ATTACH LATEST .log FILE FROM YOUR NEWEST SUBMISSION HERE]
   - imc_prosperity_analysis.ipynb (historical data notebook)

════════════════════════════════════════════════════════
 WORKFLOW FOR THIS CHAT
════════════════════════════════════════════════════════

STEP 1: I will attach the log file from my latest submission. 
        Run the analysis mentally (or I will paste key outputs) and diagnose.

STEP 2: We discuss specific issues found and potential fixes.

STEP 3: You produce an updated algorithm with:
        - Every parameter change explained with the specific data that justifies it
        - Clear reasoning connecting notebook outputs → algorithm change
        - Warning if any change risks overfitting vs is a structural improvement

STEP 4: I submit and come back with the new log file. Repeat.

════════════════════════════════════════════════════════
 CURRENT OPEN QUESTIONS (to discuss in this chat)
════════════════════════════════════════════════════════
[ADD YOUR CURRENT QUESTIONS OR OBSERVATIONS HERE BEFORE SENDING]
Example:
- After v6, EMERALDS PnL is [X]. Logs show bid=[Y] at most ticks. What next?
- Is there any way to exploit the 30 ticks where bot bid=10000?
- Can we improve TOMATOES PnL beyond 1,000 without hurting EMERALDS?
- The microprice IC=0.191 for EMERALDS is strong — is there a safe way to use it?

════════════════════════════════════════════════════════
 RULES FOR THIS CHAT
════════════════════════════════════════════════════════
1. Always explain WHY before HOW. Connect every change to a specific metric.
2. Before writing new code, summarise the diagnosis in plain English first.
3. After each algorithm version, give a monitoring checklist: what to look for
   in the next log file to confirm the change worked.
4. Never remove a working feature without stating what it achieves that 
   the replacement does not (e.g., do not remove the working TOMATOES logic).
5. When in doubt between two approaches, implement the more conservative one
   and note the more aggressive option as a follow-up to test separately.

Please confirm you have understood all of the above context, then wait for 
me to either paste log analysis outputs or ask a specific question.
```
