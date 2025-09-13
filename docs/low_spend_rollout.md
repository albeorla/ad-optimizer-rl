# Low-Spend RL Rollout Guide

This guide details how to operate the RL system with very small budgets while still accumulating episodes and improving performance over time.

Use this alongside the Production Integration Guide (docs/real_integration.md) when connecting to Shopify + TikTok + Instagram.

## Objectives

- Minimize upfront ad spend while learning.
- Accumulate many episodes over days/weeks, not hours.
- Keep strict safety guardrails and clear rollback paths.

## Cost-Sensitive Objective (Minimize Spend, Maximize Profit)

Two complementary ways to encode “spend as little as possible while maximizing profit”:

- Profit-minus-cost reward: optimize r = profit − λ_spend · adSpend. Start with λ_spend ≈ 0.1–0.3 and tune.
- Constrained RL: maximize profit subject to a daily spend budget. Use a Lagrangian multiplier λ updated online:
  - λ ← max(0, λ + η · (spendToDate − budgetTarget)/budgetTarget)
  - This increases the penalty when you overshoot the budget, reducing future spend.

Practical defaults:
- Begin with small λ_spend (e.g., 0.1) and a tight daily budget target.
- Increase λ_spend or the Lagrangian step η when trailing ROAS is poor or spend tracks above plan.

Implementation notes:
- Add `LAMBDA_SPEND` and optional `LAMBDA_STEP` (η) config. Apply in the reward service/environment.
- Keep the ROAS bonus and overspend penalties; the λ_spend term provides smooth cost pressure across all states.

## Core Strategy

- Shadow Mode (read-only): run for 1–2 weeks to validate state, reward, and proposed actions without spend.
- Canary Spend: manage 1–2 adsets per platform with $5–$20/day total.
- Hourly Episodes: treat each hour as a step; form one 24-step episode per day per adset.
- Offline + Online: nightly offline updates from logs; small online exploration in canary.
- Dayparting First: focus tiny budget on peak hours to maximize learning per dollar.

## Budget Guardrails

- Daily Cap: strict per-account and per-adset caps (e.g., $10/day total during pilot).
- Per-Step Delta: clamp budget changes to ±10%/hour (or less, e.g., 5%).
- Min/Max Floors: enforce minimum (e.g., $0.50/hour) and modest maximum (e.g., $2/hour) during pilot.
- Freeze Conditions: stop changes if trailing 6–12h ROAS < 1.5 or CPA > target.
- Canary Scope: keep 90–95% of spend unmanaged until stable.
- Kill Switch: instant rollback to last known budgets; pause runner on anomalies.

Cost-aware guardrails:
- Daily Spend Target: per-account and per-adset (e.g., $10/day); reduce per-hour caps as you approach target.
- Hourly Knapsack: allocate limited hourly budget only to adsets/hours with best expected ROAS (greedy or UCB/Thompson sampling).

## Learning Design

- Episodes From Logs: reconstruct hourly transitions from TikTok/Meta insights and Shopify revenue; credit conversions backward if needed (attribution lag).
- Replay Buffer in DB: persist experiences (state, action, reward, next_state) to accumulate samples over time.
- Exploration Policy: low online epsilon (0.05–0.15) for canary; do most exploration in shadow/offline phases.
- Bandit Pre-Phase: start with budget-only actions (0.5x, 0.75x, 1.0x) before enabling more complex changes.

Budget-allocation bandit (optional before full RL):
- Treat hours×platforms as arms with reward = profit/adSpend (ROAS proxy).
- Allocate a fixed tiny hourly budget to top arms via UCB/Thompson; clamp spend to the cap.
- Feed the resulting episodes into the RL agent for richer policy learning later.

## Rollout Phases

- Phase 0 – Shadow (1–2 weeks)
  - Read insights, compute actions, log “would-apply” decisions.
  - Validate reward fidelity (Shopify revenue − spend), ensure state features are correct.
- Phase 1 – Canary Budgets (1–2 weeks)
  - Apply only budget deltas to a small set of adsets; tight caps and delta limits.
  - Focus on 18–22 (local time) to maximize learning.
- Phase 2 – Creative Rotation (optional, +1–2 weeks)
  - Allow one vetted creative swap per day from a predefined pool.
  - Keep targeting fixed; measure fatigue and rotation impact.
- Phase 3 – Gradual Expansion
  - Increase managed adsets and budgets slowly based on trailing 7–14 day ROAS/profit.

## Scheduling

- Hourly Loop: ingest → featurize → propose (shadow/canary) → apply if safe → log.
- Nightly Train: retrain/update policy on previous day’s episodes; checkpoint model; reduce epsilon slightly.
- Weekly Review: widen caps and scope if KPIs are healthy; otherwise hold or roll back.

Adaptive λ schedule:
- Recompute λ_spend daily from the ratio of actual spend vs budget target.
- If conversions are sparse, increase λ_spend to prioritize minimal spend until signal returns.

## What To Configure Now

- Hard Caps: strict per-day and per-hour ceilings in guardrails.
- Canary List: 1–2 adsets per platform with small budgets.
- Epsilon/Decay: `epsilonStart ~ 0.1`, slow decay; raise only after positive, stable profits.
- Peak Hours: allow actions only in 18–22 initially (configurable window).
- Shadow Logs: persist “suggested vs applied” for audit and backtesting.

Cost parameters:
- `LAMBDA_SPEND` (e.g., 0.1–0.3)
- `DAILY_BUDGET_TARGET` per platform/account
- `LAGRANGE_STEP` (η) for budget constraint tuning (e.g., 0.05)

## Repository Plug-In Points

- Real Adapters: `src/platforms/meta.ts`, `src/platforms/tiktok.ts` (extend `AdPlatformAPI`).
- Shopify Revenue: `src/datasources/shopify.ts` and `src/services/reward.ts` to compute hourly profit.
- Guardrails: `src/execution/guardrails.ts` for caps, deltas, freeze rules, allowlist of adsets/hours.
- Real Environment: `src/environment/realEnvironment.ts` to orchestrate state, actions, and safety checks.
- Runner: `src/run/real.ts` (shadow/pilot flags) on an hourly schedule.

See docs/real_integration.md for detailed data and API setup.

## Suggested Config Defaults (Pilot)

- `DAILY_CAP_TOTAL`: $10 (per platform during pilot)
- `ADSET_MIN_HOURLY`: $0.50
- `ADSET_MAX_HOURLY`: $2.00
- `BUDGET_DELTA_MAX`: 0.10 (10% per hour)
- `PEAK_HOURS`: 18–22 local time
- `EPSILON_START`: 0.10, `EPSILON_DECAY`: 0.9995, `MIN_EPSILON`: 0.05
- `LR_DECAY`: 0.99, checkpoint every 50–100 episodes
- `FREEZE_IF_ROAS_LT`: 1.5 for 6–12h window

## Episode Accounting & Replay Buffer

Persist experiences in a DB to grow samples daily:

- Table `experiences(adset_external_id, platform, ts, state_json, action_json, reward, next_state_json)`
- Maintain a rolling window (e.g., last 60–90 days) to keep training relevant.
- During nightly training, sample batches from this table for updates.

Also persist budget constraint state:
- Table `budget_state(date, platform, target_spend, actual_spend, lambda_spend)` for auditability and reproducibility.

## Shadow-Mode Runner Spec

- Flags: `--mode=shadow|pilot`, `--peak-hours=18-22`, `--daily-cap=10`, `--delta-max=0.1`, `--canary-list=...`.
- Behavior (shadow): only log proposed actions to `actions_log` with status `shadow`.
- Behavior (pilot): apply budget deltas up to delta cap; respect daily/hourly caps and freeze conditions.

Cost-aware flags:
- `--lambda-spend=0.2` penalty weight on spend in reward computation
- `--daily-budget-target=10` (USD)
- `--lagrange-step=0.05` dual ascent step size

## KPIs & Alerts

- KPIs: ROAS, CPA, profit, spend by platform/creative/time; budget change counts.
- Alerts: spend spikes, ROAS below threshold, API failures, queue lag, missing data.
- Dashboards: trailing 7/14/30-day KPIs; per-hour heatmaps; suggested vs applied changes.

Cost-aware KPIs:
- Spend vs budget target (daily/weekly), λ_spend trend, marginal profit per $.

## Checklist

1. Identify canary adsets, set tiny budgets, and configure caps.
2. Run shadow mode 1–2 weeks; validate reward/state mapping.
3. Enable pilot: budget-only changes in peak hours; monitor daily.
4. Persist experiences and retrain nightly; keep exploration low.
5. Expand scope slowly after sustained positive ROAS/profit.

## No-Sales Periods (Prolonged Zero Conversions)

Expected behavior:
- Rewards turn negative; the live policy contracts spend toward minimums.
- ROAS bonuses vanish; overspend penalties and λ_spend dominate decisions.

Safety actions:
- Freeze increases when zero conversions for N hours (e.g., 12) or trailing ROAS < threshold.
- Restrict to peak hours and minimum floors only; optionally switch to shadow mode until signal returns.
- Use attribution lag handling (credit a fraction of conversions backward k hours) to avoid over-penalizing recent spend.

Learning hygiene:
- Keep exploration low live (ε ~ 0.05–0.10); rely on replay buffer and prior positive episodes to avoid catastrophic forgetting.
- Maintain canary scope; do not expand managed spend until sales resume.

---

If you want, we can scaffold a shadow-mode runner, guardrails module, and a simple Postgres schema for the replay buffer so you can start accumulating episodes immediately with minimal spend.
