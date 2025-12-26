# Ad Tech & Reinforcement Learning Glossary

A comprehensive guide to understanding the math and terminology used in real-time bidding systems. Written to be accessible while maintaining technical accuracy.

---

## Table of Contents
1. [Advertising Fundamentals](#advertising-fundamentals)
2. [Real-Time Bidding (RTB)](#real-time-bidding-rtb)
3. [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
4. [Reinforcement Learning Basics](#reinforcement-learning-basics)
5. [Advanced RL Concepts](#advanced-rl-concepts)
6. [Control Systems (PID)](#control-systems-pid)
7. [Statistics & Probability](#statistics--probability)
8. [Mathematical Notation](#mathematical-notation)

---

## Advertising Fundamentals

### Impression
**What it is:** A single display of an ad to a user.
**Motivation:** This is the basic unit of measurement in digital advertising. Every time someone's browser loads a page with your ad on it, that counts as one impression.
**Example:** If your TikTok ad appears in 1,000 people's feeds, you have 1,000 impressions.

### Click
**What it is:** When a user actively taps/clicks on your ad.
**Motivation:** Clicks indicate interest—someone saw your ad and wanted to learn more.
**Why it matters:** Clicks cost money and ideally lead to purchases.

### Conversion
**What it is:** When a user completes your desired action (usually a purchase).
**Motivation:** This is the end goal of advertising. You're paying for ads to get people to buy stuff.
**Example:** Someone clicks your Shopify store ad and buys a t-shirt—that's a conversion.

### Attribution
**What it is:** Figuring out which ad deserves credit for a conversion.
**Motivation:** If someone saw your TikTok ad Monday, your Instagram ad Tuesday, and bought Wednesday, which ad "caused" the sale? Attribution answers this.
**Why it's tricky:** Users see many ads before buying, making it hard to assign credit fairly.

### Attribution Window
**What it is:** The time period during which a conversion can be credited to an ad.
**Motivation:** If someone clicked your ad 6 months ago and buys today, did the ad really cause it? Attribution windows set reasonable limits.
**Example:** A 24-hour window means only purchases within 24 hours of clicking count.

---

## Real-Time Bidding (RTB)

### Real-Time Bidding (RTB)
**What it is:** An automated auction that happens in ~100 milliseconds every time a webpage loads.
**Motivation:** Instead of buying ad space in bulk, advertisers compete in real-time for each individual impression. This allows precise targeting and efficiency.
**How it works:**
1. User visits a website
2. The site announces "we have an ad slot to sell"
3. Multiple advertisers bid simultaneously
4. Highest bidder wins (usually pays second-highest price)
5. Their ad appears

All of this happens faster than you can blink.

### Demand-Side Platform (DSP)
**What it is:** Software that advertisers use to automatically bid on ad inventory.
**Motivation:** No human can evaluate and bid on millions of impressions per second. DSPs use algorithms to make these decisions automatically.
**Our system:** The `ad-optimizer-rl` codebase is essentially a smart DSP powered by reinforcement learning.

### Bid
**What it is:** The amount of money you're willing to pay for an impression.
**Motivation:** Higher bids win more auctions but cost more money. The skill is bidding just enough to win without overpaying.

### Win Rate
**What it is:** The percentage of auctions you win.
**Formula:** `wins ÷ total_bids × 100%`
**Motivation:**
- Too low (< 1%): You're bidding too little, missing opportunities
- Too high (> 50%): You're probably overpaying
- Sweet spot: Usually 5-20% depending on your goals

### Clearing Price
**What it is:** The actual price paid in an auction (usually second-highest bid).
**Motivation:** Even if you bid $5, you might only pay $2.50 if that was the second-highest bid. Understanding clearing prices helps optimize bids.

### Bid Shading
**What it is:** Strategically reducing your bid based on expected clearing price.
**Motivation:** Why bid $5 if you can win at $3? Bid shading algorithms predict the minimum bid needed to win.
**Example:** You estimate the next-highest bidder will bid $2. Instead of bidding $5, you bid $2.50—still winning but saving $2.50.

---

## Key Performance Indicators (KPIs)

### CTR (Click-Through Rate)
**What it is:** Percentage of people who click after seeing your ad.
**Formula:** `clicks ÷ impressions × 100%`
**Motivation:** Measures how compelling your ad is. If 1,000 people see your ad and 20 click, CTR = 2%.
**Good CTR:** Varies by platform (0.5-2% is typical for display ads, higher for social).

### CVR (Conversion Rate)
**What it is:** Percentage of clicks that result in purchases.
**Formula:** `conversions ÷ clicks × 100%`
**Motivation:** Measures how well your landing page/store converts interest into sales.
**Example:** 100 clicks → 3 purchases = 3% CVR.

### CPA (Cost Per Acquisition/Action)
**What it is:** How much you spend to get one conversion.
**Formula:** `total_ad_spend ÷ conversions`
**Motivation:** Directly measures efficiency. If you spend $100 on ads and get 10 sales, CPA = $10.
**Why it matters:** If your product profit is $15, a $10 CPA is great. A $20 CPA means you're losing money.

### ROAS (Return on Ad Spend)
**What it is:** Revenue generated per dollar spent on ads.
**Formula:** `revenue ÷ ad_spend`
**Motivation:** The ultimate measure of ad efficiency.
**Example:** Spend $100, generate $300 revenue → ROAS = 3.0 (or "3x")
**Interpretation:**
- ROAS < 1: Losing money
- ROAS = 1: Breaking even (on revenue, not profit)
- ROAS > 2: Generally healthy

### Margin ROAS
**What it is:** Profit generated per dollar spent on ads.
**Formula:** `(revenue - cost_of_goods) ÷ ad_spend`
**Motivation:** Regular ROAS ignores product costs. If you sell a $30 shirt that costs $15 to make, margin ROAS accounts for that.
**Why it matters:** You can have ROAS = 2 but still lose money if your margins are low.

### CPM (Cost Per Mille/Thousand)
**What it is:** Cost to show your ad 1,000 times.
**Formula:** `(ad_spend ÷ impressions) × 1000`
**Motivation:** Standard unit for comparing impression costs across platforms.
**Example:** $50 for 10,000 impressions = $5 CPM.

### CPC (Cost Per Click)
**What it is:** Cost for each click on your ad.
**Formula:** `ad_spend ÷ clicks`
**Motivation:** Alternative pricing model where you only pay when someone clicks.

---

## Reinforcement Learning Basics

### Agent
**What it is:** The decision-maker (our bidding algorithm).
**Motivation:** Just like a human learns from experience, an RL agent learns by trying actions and seeing what happens.
**In our system:** The agent decides how much to bid, which demographics to target, etc.

### Environment
**What it is:** The world the agent interacts with (the ad marketplace).
**Motivation:** The agent can't just think—it has to take actions in the real world and observe the results.
**In our system:** Includes user behavior, competitor bids, platform dynamics.

### State (s)
**What it is:** All the information available to the agent at a decision point.
**Motivation:** To make good decisions, you need context. The state captures everything relevant about the current situation.
**Example in RTB:** Hour of day, remaining budget, current CTR, user demographics, etc.

### Action (a)
**What it is:** What the agent chooses to do.
**Motivation:** The whole point is to make good decisions. Actions are those decisions.
**Example in RTB:** Bid amount, target age group, creative type, platform choice.

### Reward (r)
**What it is:** A numerical score telling the agent how good its action was.
**Motivation:** This is how the agent learns. Good actions lead to positive rewards, bad ones to negative (or zero).
**Example in RTB:** Profit from a conversion minus ad cost.

### Policy (π)
**What it is:** The agent's strategy for choosing actions in each state.
**Notation:** `π(a|s)` = probability of taking action `a` in state `s`
**Motivation:** The policy is what we're trying to optimize. A good policy = profitable bidding strategy.

### Q-Value / Q-Function
**What it is:** The expected total future reward for taking action `a` in state `s`.
**Notation:** `Q(s, a)`
**Motivation:** If you knew the Q-values for all actions, you'd just pick the highest one. Q-learning estimates these values from experience.
**Intuition:** "If I bid $2 on this user, what's my expected profit?"

### Discount Factor (γ)
**What it is:** How much to value future rewards vs immediate rewards (usually 0.9-0.99).
**Notation:** `γ` (gamma)
**Motivation:** Money today is worth more than money tomorrow. A discount factor of 0.95 means a reward tomorrow is worth 95% of the same reward today.
**Formula:** Total value = r₀ + γr₁ + γ²r₂ + γ³r₃ + ...

### Exploration vs Exploitation
**What it is:** The tradeoff between trying new things and sticking with what works.
**Motivation:**
- **Exploitation:** Keep doing what's worked before
- **Exploration:** Try something new to maybe find something better

Too much exploitation = miss opportunities. Too much exploration = waste money on bad actions.

### Epsilon-Greedy (ε-greedy)
**What it is:** A simple exploration strategy.
**How it works:** With probability ε, take a random action. Otherwise, take the best known action.
**Example:** ε = 0.1 means explore 10% of the time, exploit 90%.
**Typical schedule:** Start with high ε (lots of exploration), decrease over time as you learn.

### Experience Replay
**What it is:** Storing past experiences and learning from them multiple times.
**Motivation:** Why throw away old experiences? By replaying past actions and outcomes, the agent learns more efficiently.
**Analogy:** Reviewing game tape instead of only learning during live games.

---

## Advanced RL Concepts

### DQN (Deep Q-Network)
**What it is:** Using a neural network to estimate Q-values.
**Motivation:** When there are millions of possible states (like in RTB), you can't store Q-values in a table. Neural networks can generalize across similar states.

### Double DQN
**What it is:** A fix for DQN's tendency to overestimate Q-values.
**Problem it solves:** Regular DQN uses max(Q) which picks up noise. This leads to thinking bad actions are good.
**Solution:** Use two networks—one to pick the action, another to evaluate it. This reduces overconfidence.

### Target Network
**What it is:** A slowly-updated copy of the Q-network used for computing targets.
**Motivation:** If you update the network and immediately use it to set targets, training becomes unstable. The target network provides a stable reference.

### Conservative Q-Learning (CQL)
**What it is:** A method that learns pessimistic Q-value estimates.
**Motivation:** In offline RL (learning from historical data), the agent might overestimate unseen actions. CQL penalizes Q-values for actions not in the dataset, making the agent cautious.
**Why for RTB:** We can't afford to deploy an overconfident agent that wastes budget on bad bids.

### CMDP (Constrained MDP)
**What it is:** Reinforcement learning with hard constraints.
**Motivation:** Standard RL just maximizes reward. In RTB, you can't just maximize—you have a budget limit. CMDPs formally handle these constraints.
**Example constraint:** "Maximize conversions subject to: total spend ≤ $100"

### Offline RL
**What it is:** Learning entirely from historical data, without live interaction.
**Motivation:** In advertising, exploration is expensive. Offline RL lets you train agents on past campaign data before deploying them live.
**Challenge:** The agent might hallucinate that untested actions are great.

---

## Control Systems (PID)

### PID Controller
**What it is:** A classic control algorithm using three terms: Proportional, Integral, Derivative.
**Motivation:** Perfect for budget pacing—reacting to errors and smoothly correcting them.
**Analogy:** Like cruise control in a car. It sees you're going 55 in a 60 zone and adjusts accordingly.

### Error (e)
**What it is:** The difference between target and actual.
**Formula:** `e = target - actual`
**Example:** Target spend at noon: $15. Actual: $10. Error = $5 (underspending).

### Proportional Term (P)
**What it is:** Response proportional to current error.
**Formula:** `P = Kp × error`
**Motivation:** Bigger error → bigger correction.
**Example:** Underspent by $5? Increase bids proportionally.
**Risk:** Too aggressive → oscillation (overshoots then undershoots).

### Integral Term (I)
**What it is:** Response to accumulated historical error.
**Formula:** `I = Ki × sum(all past errors)`
**Motivation:** Fixes persistent errors that P alone can't solve.
**Example:** Consistently underspending? The integral builds up and forces stronger correction.
**Risk:** Integral "windup" if error persists too long.

### Derivative Term (D)
**What it is:** Response to the rate of change of error.
**Formula:** `D = Kd × (current error - previous error)`
**Motivation:** Anticipates future error and dampens oscillation.
**Example:** Spend is accelerating too fast—D slows things down before overshooting.

### Anti-Windup
**What it is:** Limiting the integral term to prevent it from growing too large.
**Motivation:** If you're stuck unable to spend (e.g., no inventory), the integral would grow forever, then overreact when inventory appears.
**Solution:** Cap the integral at reasonable bounds.

### Pacing
**What it is:** Spreading budget evenly over time.
**Motivation:** Without pacing, an aggressive algorithm might spend the whole daily budget by noon, missing afternoon opportunities.
**Goal:** Smooth, even spending throughout the day/campaign.

---

## Statistics & Probability

### Importance Sampling
**What it is:** A technique to estimate properties of one distribution using samples from another.
**Motivation:** If you have data from old policy π_old but want to evaluate new policy π_new, you can reweight the samples.
**Formula:** `weight = π_new(action) / π_old(action)`
**Use in OPE:** Evaluate new bidding strategy using historical data.

### IPS (Inverse Propensity Scoring)
**What it is:** The basic importance sampling estimator for policy evaluation.
**Formula:** `V̂ = (1/n) Σ [π_new(a|s) / π_old(a|s)] × reward`
**Pros:** Unbiased (correct on average)
**Cons:** High variance (individual estimates can be wildly off)

### SNIPS (Self-Normalized IPS)
**What it is:** IPS but normalized by sum of weights instead of sample count.
**Formula:** `V̂ = Σ(weight × reward) / Σ(weight)`
**Motivation:** Much lower variance than IPS, slightly biased but acceptable.
**Recommendation:** Usually the default choice for RTB evaluation.

### Doubly Robust (DR)
**What it is:** Combines importance sampling with a reward prediction model.
**Motivation:** If either your weights OR your model is correct, DR gives the right answer. Two chances to be right = more robust.
**When to use:** High-stakes decisions where you want extra reliability.

### Confidence Interval
**What it is:** A range likely to contain the true value.
**Example:** "ROAS is 2.5 with 95% CI [2.1, 2.9]" means we're 95% confident true ROAS is between 2.1 and 2.9.
**Motivation:** Point estimates aren't the whole story—uncertainty matters.

### Variance
**What it is:** How spread out values are from the average.
**Motivation:** High variance = unreliable estimates. You want low variance for confidence.
**In OPE:** High weight variance means your policy is very different from logging policy—estimates may be unreliable.

### Effective Sample Size (ESS)
**What it is:** The "true" sample size after accounting for weighting.
**Motivation:** If your importance weights are extreme (some very high, some very low), you're not really using all your data effectively.
**Rule of thumb:** ESS < 100 → estimates are unreliable.

### MDA (Mean Directional Accuracy)
**What it is:** How often you correctly predict which option is better.
**Motivation:** In RTB, you don't need exact values—you need to know "is Strategy A better than Strategy B?"
**Target:** MDA > 80% before trusting deployment decisions.

---

## Mathematical Notation

### Σ (Sigma)
**What it is:** Summation—add up all the things.
**Example:** `Σᵢ xᵢ` = x₁ + x₂ + x₃ + ... + xₙ

### π (Pi, in RL context)
**What it is:** Policy—the agent's decision-making strategy.
**Not to confuse with:** π ≈ 3.14159 (that's a different thing entirely)

### γ (Gamma)
**What it is:** Discount factor for future rewards.
**Range:** 0 to 1, typically 0.95-0.99

### ε (Epsilon)
**What it is:** Exploration rate in ε-greedy.
**Range:** 0 to 1, typically starts high and decays

### α (Alpha)
**What it is:** Learning rate or regularization strength.
**In CQL:** Controls how conservative the algorithm is

### λ (Lambda)
**What it is:** Lagrange multiplier for constraints.
**In CMDP:** Balances reward maximization vs constraint satisfaction

### E[X] or 𝔼[X]
**What it is:** Expected value—the average outcome.
**Example:** E[dice roll] = 3.5

### P(X) or Pr(X)
**What it is:** Probability of X happening.
**Range:** 0 (impossible) to 1 (certain)

### argmax
**What it is:** "The argument that maximizes"—which input gives the highest output.
**Example:** argmax Q(s, a) over all a = "which action has highest Q-value?"

### ∈ (Element of)
**What it is:** "Is a member of"
**Example:** a ∈ A means "action a is in action space A"

### |A| (Cardinality)
**What it is:** The size of a set—how many elements.
**Example:** |A| = 768 means there are 768 possible actions

### ∀ (For all)
**What it is:** "For every..."
**Example:** ∀s ∈ S means "for every state s in the state space"

### ∂ (Partial derivative)
**What it is:** Rate of change with respect to one variable.
**In neural networks:** Used to compute gradients for learning

### log / ln
**What it is:** Logarithm—the inverse of exponentiation.
**Motivation:** Makes very large or small numbers manageable.
**In ML:** Usually natural log (ln), sometimes log₁₀

### exp(x) / eˣ
**What it is:** Exponential function, e ≈ 2.718 raised to power x.
**Use:** Softmax, probability calculations, growth modeling

---

## Platform-Specific Terms

### TikTok Marketing API
**What it is:** TikTok's interface for programmatic ad management.
**Use:** Our system uses this to get metrics and (eventually) place bids.

### Shopify Admin API
**What it is:** Shopify's interface for store management.
**Use:** Get sales data, check inventory, track conversions.

### CAPI (Conversions API)
**What it is:** Server-to-server conversion tracking.
**Motivation:** Browser-based tracking is unreliable (ad blockers, iOS privacy). CAPI sends conversion data directly from your server.

### ttclid / gclid / fbclid
**What it is:** Click tracking identifiers from TikTok / Google / Facebook.
**Motivation:** When someone clicks an ad, this ID follows them. If they convert, we match the ID to attribute the sale.

---

## System Architecture Terms

### Circuit Breaker
**What it is:** A safety mechanism that stops operations when too many failures occur.
**Motivation:** If something is broken, stop trying (and failing) repeatedly. Give the system time to recover.
**States:** Closed (normal), Open (blocked), Half-Open (testing recovery)

### Safe Mode
**What it is:** A conservative fallback strategy when something goes wrong.
**Motivation:** Better to bid conservatively than to blow the budget on a malfunctioning algorithm.

### Shadow Mode
**What it is:** Running the algorithm on real data without actually placing bids.
**Motivation:** Validate that the system works before risking real money.

### Warmup Period
**What it is:** Initial time where the agent collects experiences before learning.
**Motivation:** Learning from one data point is useless. Need enough samples to see patterns.

---

## Quick Reference Formulas

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| CTR | clicks ÷ impressions | Higher = more engaging ad |
| CVR | conversions ÷ clicks | Higher = better landing page |
| CPA | spend ÷ conversions | Lower = more efficient |
| ROAS | revenue ÷ spend | Higher = more profitable |
| CPM | (spend ÷ impressions) × 1000 | Cost per 1k views |
| CPC | spend ÷ clicks | Cost per click |
| Win Rate | wins ÷ bids | Auction success rate |

---

## Why These Concepts Matter

Understanding these terms isn't just academic—they directly impact business outcomes:

1. **ROAS & CPA** → Are you making money?
2. **Pacing & PID** → Will you run out of budget too early?
3. **CQL & OPE** → Can you safely deploy new strategies?
4. **Attribution** → Are you measuring success correctly?
5. **Exploration** → Are you learning or just repeating?

The ad-optimizer-rl system combines all these concepts to make intelligent, profitable bidding decisions automatically. Each component addresses a real-world challenge that human advertisers face daily.

---

*Last updated: December 2024*
*For questions about specific implementations, see the source code in `/src/`*
