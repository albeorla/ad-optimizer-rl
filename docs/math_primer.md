# Mathematical Primer for RL-Powered Ad Optimization

This primer connects the project’s design to core reinforcement learning (RL) and constrained optimization concepts.

## 1) Markov Decision Process (MDP)

- State S: campaign context (time, budget, audience, creative, platform, CTR/CVR, seasonality, competition).
- Action A: budget multiplier, platform allocation, creative swap (optionally targeting/bid).
- Transition P(s'|s,a): how state evolves; simulated here, data-driven in production.
- Reward R(s,a,s'): profit = revenue − spend (plus shaping).
- Objective: maximize E[Σ γ^t r_t], with discount γ ∈ (0,1].

## 2) Tabular Q‑Learning (used here)

- Update: Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ].
- Exploration: ε‑greedy (ε small for live canary; decays in sim).
- Replay: sample past transitions to stabilize and reuse data.
- Convergence: tabular Q converges with sufficient visitation and decaying α (function approximation has no guarantee).

Notes for this repo

- State is discretized (JSON key of day/hour/budget bucket/age/creative/platform).
- Experience replay improves sample efficiency.

## 3) DQN (optional upgrade)

- Approximate Q(s,a; θ) with a neural net.
- Stabilization: target network, experience replay, gradient clipping, Huber loss.
- Target: y = r + γ max_a' Q_target(s', a'). Loss: (y − Q(s,a))^2.

## 4) Reward Design (Profit + Shaping)

- Base: r = (revenue − spend) / C (normalize by C≈1000).
- Shaping (small, consistent):
  - ROAS bonus: +b if ROAS > threshold.
  - Overspend penalty: down-weight aggressive budget multipliers.
  - Conversion bonus: +κ per purchase.

## 5) Lagrangian Budget Constraint (Cost‑Sensitive)

Goal: maximize profit subject to daily spend ≤ B (e.g., $30/day).

- Lagrangian reward: r ← r − λ · adSpend, with λ ≥ 0.
- Daily λ update (dual ascent):
  - dev = (spendToDate − B)/B
  - λ ← clip[0,λ_max](λ + η · dev), with step η (0.01–0.05), EMA smoothing.
- Intuition: Overspend → λ increases (spend “costs” more); underspend → λ decreases.
- Always pair with hard guardrails (daily cap, per‑hour delta clamp, peak hours).

## 6) Bandits vs RL (Budget Pre‑Phase)

- Use UCB/Thompson to allocate tiny hourly budget to best arms (platform×hour×creative) by ROAS.
- Feed findings to RL to reduce expensive exploration.

## 7) Attribution Lag

- Purchases arrive late; naive hourly profit can over‑penalize exploration.
- Mitigate with backward credit over k hours, multi‑window attribution (1/7/28‑day), and EMA smoothing.

## 8) Practical Defaults

- γ=0.95, α=0.01 (tabular), ε∈[0.05,0.10] live.
- Normalize rewards so |r| is O(10).
- Diminishing returns: saturate multipliers for large budgets.

## 9) Diagnostics

- ε, λ trends; Q‑table coverage; TD‑error stats.
- Spend vs target; ROAS/CPA distribution; profit/day.
- Action entropy; guardrail trigger counts.

## 10) Safety & Caveats

- DQN has no convergence guarantees; rely on target nets, small steps, replay, and guardrails.
- Enforce: daily cap, delta clamp, peak hours, no‑sales freeze, kill switch.

## 11) Code Map

| Component | File |
|-----------|------|
| Tabular Q-learning | `src/agent/dqnAgent.ts` |
| Neural DQN | `src/agent/dqnAgentNN.ts` |
| Conservative Q-Learning (CQL) | `src/agent/cqlAgent.ts` |
| State/Action encoding | `src/agent/encoding.ts` |
| Experience replay | `src/agent/replay.ts` |
| PID controllers | `src/control/PidController.ts` |
| Delayed feedback (GDFM) | `src/data/AttributionBuffer.ts` |
| Offline Policy Evaluation | `src/evaluation/OPE.ts` |
| Simulator shaping | `src/platforms/mockTikTok.ts`, `src/platforms/mockInstagram.ts` |
| Environment reward | `src/environment/simulator.ts` |
| Guardrails | `src/execution/guardrails.ts` |
| Safety layer | `src/execution/SafetyLayer.ts` |
| State enrichment | `src/types/EnrichedState.ts` |
| Real runner | `src/run/real.ts` |
| Shadow training | `src/run/shadowTraining.ts` |

---

*Last updated: December 2025*
