# Invest & Expand: Manual Round 2 Strategy Guide

## 1. What the challenge is really asking

You are given **50,000 XIRECs** and must split that budget across three pillars:

- **Research**: improves your trading edge through quant research, data, machine learning, backtesting, and similar infrastructure.
- **Scale**: expands how broadly you can deploy your strategy, such as more traders, engineers, office capacity, exchange access, and operational reach.
- **Speed**: improves your hit rate through latency-reducing infrastructure such as FPGA boards, colocation, RF towers, microwave links, and similar tools.

You choose percentages from **0 to 100** for each pillar, with total allocation **at most 100%**. Each 1% represents **500 XIRECs** of budget.

Your score is:

\[
\text{PnL} = (\text{Research} \times \text{Scale} \times \text{Speed Multiplier}) - \text{Budget Used}
\]

The important part is that this is **not** a simple “maximize each pillar independently” problem. It is a **trade-off problem**:

- Research has **diminishing returns**.
- Scale is **linear**.
- Speed is **competitive and rank-based**, so it depends on what everyone else does.

That makes this round a blend of:

1. optimization,
2. game theory,
3. simulation,
4. behavioral modeling.

---

## 2. The exact mechanics of each pillar

### Research
Research grows logarithmically:

\[
R(x) = 200{,}000 \cdot \frac{\ln(1+x)}{\ln(101)}
\]

where `x` is the percentage invested in Research.

This means:

- the first few percent matter a lot,
- later percent points matter less and less,
- you should never over-allocate to Research just because it “feels safe.”

### Scale
Scale grows linearly:

\[
S(y) = 7 \cdot \frac{y}{100}
\]

where `y` is the percentage invested in Scale.

This means every extra percent of Scale gives a constant increase. There are no diminishing returns here.

### Speed
Speed is the special part. It is not directly valued by your own chosen percentage; instead, it is converted into a **rank-based multiplier** relative to everyone else.

- Highest speed investment gets multiplier **0.9**.
- Lowest speed investment gets multiplier **0.1**.
- Values in between are interpolated linearly by rank.
- Equal investments share the same rank.

So the speed choice is not just “how much do I want?” It is “what rank will this get me compared to the field?”

That is why Speed is the hardest part of the problem.

---

## 3. The first key simplification: spend the full budget

A central realization is that **leaving budget unused is almost always bad**.

If you do not spend a percent, you are giving up a unit of Scale, and Scale is linear. Since the whole score is multiplicative, unused budget typically reduces your gross output more than any possible “safety margin” can justify.

So the natural baseline is:

\[
\text{Research} + \text{Scale} + \text{Speed} = 100
\]

That does **not** mean every possible allocation is equally good. It only means that once you choose Speed, the rest of the budget should usually be fully assigned to Research and Scale.

---

## 4. Solving the Research/Scale subproblem exactly

Once Speed is fixed, let the leftover budget be:

\[
B = 100 - z
\]

where `z` is your chosen Speed percentage.

Then the remaining optimization problem is:

\[
\max_{x+y=B} R(x) \cdot S(y)
\]

Since `S(y)` is linear and `R(x)` is logarithmic, the optimum is not 50/50. The optimal Research share of the leftover budget is much smaller than many people guess.

A strong rule of thumb is:

- allocate roughly **one quarter** of the leftover budget to Research,
- allocate roughly **three quarters** of the leftover budget to Scale.

This ratio is stable across a wide range of Speed values.

### Why Research should not dominate
Research starts strong but flattens quickly. That creates a trap:

- a player sees Research as the “smart” option,
- over-allocates to it,
- and loses more from underfunding Scale than they gain from the extra Research.

This is why “Research first” or “Research around 23% no matter what” is too rigid.

The correct Research amount depends on how much budget is left after Speed.

---

## 5. Why Speed is a game-theory problem, not a pure math problem

Speed does not have a single universal optimum because it depends on the **distribution of everyone else’s choices**.

If a large fraction of the field invests heavily in Speed, then your rank multiplier changes differently than if the field is moderate or conservative.

This means the correct way to think about Speed is:

1. assume a plausible distribution for the lobby,
2. compute what rank your chosen Speed likely gives,
3. translate rank into a multiplier,
4. then compare total expected PnL across candidate Speed values.

That is why simulation matters.

---

## 6. What a rank-based speed model means in practice

Suppose there are many players and their Speed allocations are spread across the range. Then your Speed multiplier is approximately determined by the fraction of players at or below your chosen Speed.

A useful approximation is:

\[
\text{Multiplier} \approx 0.1 + 0.8F(z)
\]

where `F(z)` is the fraction of the lobby at or below your Speed percentage.

Interpretation:

- If many players are below your Speed, you receive a better multiplier.
- If many players are above you, the multiplier is poor.
- Ties matter because equal values share the same rank.

This is the main reason Speed is not a smooth “more is always better” feature.

---

## 7. Why the 15–35% Speed range became our first instinct

Our early reasoning was that many participants would likely try to “buy” Speed aggressively in order to improve their rank. That led to an initial guess that the useful Speed region might lie in the **15–35%** range.

That was a reasonable first-pass hypothesis because:

- it acknowledges competition,
- it avoids overcommitting to Speed,
- it leaves enough budget for Research and Scale.

But the range was still too vague.

### The main issue with 15–35
If the lobby is genuinely speed-heavy, then 15–20 may be too low.
If the lobby is more balanced than expected, 35 may be too high.

So the range was a starting point, not a final answer.

---

## 8. How we refined the analysis

We explored several layers of reasoning.

### A. Smooth optimization
First, we treated the problem as if Speed were a fixed multiplier and optimized Research/Scale exactly.

That gave us the strong baseline:

- **do not waste budget**, and
- **split the non-speed budget roughly 1/4 Research, 3/4 Scale**.

### B. Rank-based sensitivity
Then we recognized Speed is discrete and relative.
A tiny change in Speed can matter a lot if it changes your rank band, but almost nothing if it does not.

So the best Speed value is often a threshold crossing, not a smooth interior point.

### C. Lobby distribution assumptions
We then considered several possible field shapes:

- conservative fields,
- speed-heavy fields,
- balanced fields,
- clustered/focal-number fields,
- near-uniform distributions.

Each one implies a different best Speed.

### D. Simulation intuition
Instead of pretending we know the true lobby perfectly, we used simulation-style reasoning:

- sample plausible competitor distributions,
- compute your likely rank for each candidate Speed,
- evaluate total PnL,
- compare the results.

This is much more robust than trying to guess a single “magic number” by instinct.

---

## 9. What we learned from the stronger mathematical analyses

A few key lessons emerged from the more advanced analyses that were discussed earlier.

### 1) A single universal optimum is not guaranteed
A Speed value like “40%” can be optimal under one lobby model and only mediocre under another. So the answer is conditional on what the field looks like.

### 2) The Research/Scale split is much more stable than the Speed choice
Once Speed is chosen, the leftover-budget split is fairly consistent. That is the easy part.

### 3) The true bottleneck is the opponent distribution
The most important unknown is how the rest of the field allocates Speed.

### 4) Psychological assumptions can help, but they are not proof
It is useful to think about what humans tend to do:

- round numbers,
- middle numbers,
- overconfidence about “maxing” one pillar,
- anchoring on visible ranges.

But psychology should be treated as a prior, not a theorem.

### 5) Overconfidence is dangerous
Some analyses sounded very certain about a point answer like “40% speed is exactly optimal.” In reality, that is only true under some assumptions, not all assumptions.

The more honest conclusion is: **40% was a strong candidate, not a guaranteed universal truth.**

---

## 10. Practical candidate strategies we compared

Here are the major strategy families we explored.

### Strategy A: Balanced conservative
- Speed around 20–25%
- Research around 20–23%
- Scale uses the rest

This is safe, but may lose rank if the field is speed-heavy.

### Strategy B: Moderate competitive
- Speed around 30–35%
- Research about one quarter of the leftover
- Scale gets most of the remainder

This is a good middle ground and often a strong blind strategy.

### Strategy C: Aggressive competitive
- Speed around 40–41%
- Research and Scale optimized on the remaining budget

This is the strategy we eventually converged toward after considering lobby competition and rank pressure.

### Strategy D: Overcommitted Speed
- Speed 50% or above
- Too little left for Research and Scale

This can only win if Speed rank is extraordinarily valuable and the field is very crowded. Usually this is too expensive.

---

## 11. What the “Claude-style” analysis added

A separate analysis proposed a very specific conclusion:

- **Research = 15%**
- **Scale = 45%**
- **Speed = 40%**

It supported this with claims about:

- multiple hypothetical distributions,
- a PnL landscape peaking near 40%,
- level-k reasoning,
- a psychographic model of the field,
- a Nash-style argument.

### What was useful
That analysis was useful because it reinforced the key point that:

- **Speed is the critical decision variable**, and
- **the optimal point is somewhere in the moderate-to-high range, not at the extremes**.

### What was less convincing
The analysis was too confident about exact numbers. It treated a hypothetical model as if it were conclusive. But without exact lobby data, that level of certainty is stronger than the evidence supports.

So we retained the useful insight — moderate/high Speed is plausible — while not treating the exact 40% claim as mathematically absolute.

---

## 12. Our final reasoning path

After exploring all of the above, our final reasoning was:

1. Use the full 100% budget.
2. Keep Research in a modest range because of logarithmic diminishing returns.
3. Put the majority of the non-Speed budget into Scale.
4. Choose a Speed value that is high enough to matter, but not so high that it destroys the multiplicative engine.
5. Prefer a candidate that remains strong across a wide range of plausible lobby distributions.

That led us away from the earlier 15–35 idea as a “hard cap” and toward a more specific competitive allocation.

---

## 13. Final submitted strategy

Our final submitted style of allocation was:

- **Research: 15%**
- **Scale: 45%**
- **Speed: 40%**

### Why this was chosen

#### Research = 15%
This is enough to capture a large portion of the logarithmic benefit without wasting too much budget on diminishing returns.

#### Scale = 45%
Scale is linear, so every unused percent hurts. With 40% in Speed and 15% in Research, the leftover naturally becomes 45%.

#### Speed = 40%
This was chosen as a strong competitive point: high enough to be meaningful in rank-based competition, but not so high that it starves the Research/Scale engine.

---

## 14. Expected behavior of the final allocation

Under a reasonable mid-field assumption, this strategy has the following qualities:

- It uses the full budget.
- It respects diminishing returns in Research.
- It avoids overcommitting to Speed.
- It stays competitive if the lobby is moderately speed-heavy.
- It is simple enough to defend and consistent enough to submit with confidence.

### Best-case scenario
If the field is more conservative than expected, 40% Speed may place you above many competitors, giving you a strong multiplier while still keeping enough Research/Scale to generate excellent gross output.

### Worst-case scenario
If the field is extremely speed-heavy, 40% may not be enough to secure a top rank band, and your multiplier could be lower than hoped.

### Why we still accepted that risk
Because overcommitting to Speed is even more dangerous if the field does not behave that way. The chosen point is a compromise between rank pressure and preserving the core profit engine.

---

## 15. Key lessons from the round

1. **Budget must be fully used.**
2. **Research has diminishing returns.**
3. **Scale is the stable linear lever.**
4. **Speed is the strategic lever because it is competitive.**
5. **The real challenge is predicting the field, not solving a single formula.**
6. **Simulation and scenario testing are more useful than rigid intuition.**
7. **The best answer is robust, not just theoretically elegant.**

---

## 16. Final takeaway

The manual round is best understood as a two-stage optimization game:

1. choose a Speed level that is likely to put you in a strong rank band,
2. then optimize the leftover budget between Research and Scale.

Our final choice — **15% Research, 45% Scale, 40% Speed** — was the result of combining mathematical optimization, competitive reasoning, and repeated scenario analysis.

It is not a proof that 40% Speed is universally optimal. It is the strongest robust allocation we arrived at after considering the structure of the game and the likely behavior of the field.
