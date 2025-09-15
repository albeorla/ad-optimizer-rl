# PoC Launch Checklist (Shadow → Pilot, $30/day Cap)

Use this checklist to launch a low‑spend RL proof‑of‑concept that controls ad budgets across TikTok + Instagram (Meta) with a strict $30/day cap.

## 0) Goals & Constraints

- [ ] Primary KPI: daily profit (Shopify revenue − spend − COGS/fees)
- [ ] Secondary KPIs: ROAS, CPA, conversions, spend vs target
- [ ] Daily budget target: $30 (combined)
- [ ] Pilot scope: 1 adset per platform, peak hours only (18–22 local)

## 1) Accounts, Access, and Tokens

- [ ] Shopify custom app (Admin API access token) secured
  - Store domain: \***\*\_\_\*\***
  - Admin token: (in `.env` only)
- [ ] Meta Marketing API (Instagram) access
  - Ad account ID: `act_____________`
  - Long‑lived token: (in `.env` only)
  - App scopes: `ads_management`, `ads_read`, `business_management`
- [ ] TikTok Business Ads access
  - Ad account ID: \***\*\_\_\_\_\*\***
  - Access token: (in `.env` only)
- [ ] Timezone confirmed (e.g., `TZ=America/New_York`)

## 2) Canary Selection (Pilot Targets)

- [ ] Choose 1 adset per platform to manage
  - TikTok adset ID: \***\*\_\_\_\_\*\*** (baseline daily budget: $\_\_)
  - Instagram adset ID: \***\*\_\_\_\_\*\*** (baseline daily budget: $\_\_)
- [ ] Creative mapping available (tag creatives: `ugc` | `lifestyle` | `product` | `discount`)
- [ ] Target audiences documented (age buckets, interests)

## 3) Repository & Configuration

- [ ] Install deps (`npm ci`) and build once (`npm run build`)
- [ ] Create `.env` with placeholders
  - [ ] `SHOPIFY_STORE_DOMAIN=`
  - [ ] `SHOPIFY_ADMIN_TOKEN=`
  - [ ] `META_LONG_LIVED_TOKEN=`
  - [ ] `META_AD_ACCOUNT_ID=`
  - [ ] `TIKTOK_ACCESS_TOKEN=`
  - [ ] `TIKTOK_AD_ACCOUNT_ID=`
  - [ ] `DAILY_BUDGET_TARGET=30`
  - [ ] `TZ=Your/Timezone`
  - Optional tuning:
    - [ ] `LAMBDA_SPEND=0.25`
    - [ ] `LAGRANGE_STEP=0.05`
    - [ ] `EPS_DECAY=0.9995`, `LR_DECAY=0.99`

## 4) Data & Storage (Lightweight to Start)

- [ ] Pick storage (SQLite/Postgres). For team/shared ops, prefer Postgres.
- [ ] Create tables (or run migrations) for:
  - [ ] `insights_hourly`
  - [ ] `orders`
  - [ ] `rewards`
  - [ ] `experiences`
  - [ ] `actions_log`
  - [ ] `budget_state`
- [ ] Verify connectivity from the app to DB

## 5) Adapters (Read‑Only First)

- [ ] Scaffold Meta adapter (`src/platforms/meta.ts`) with:
  - [ ] Hourly adset insights (spend, impressions, clicks, purchases)
  - [ ] Rate limiting / retry / backoff
- [ ] Scaffold TikTok adapter (`src/platforms/tiktok.ts`) with:
  - [ ] Hourly adgroup/adset reports
  - [ ] Rate limiting / retry / backoff
- [ ] Wire to a `RealEnvironment` (shadow mode uses reads only)

## 6) Reward & Featurization

- [ ] Implement Shopify datasource (`src/datasources/shopify.ts`):
  - [ ] Hourly order fetch (or webhooks) with totals and attribution
- [ ] Join ads → revenue to compute hourly profit (lag‑aware)
- [ ] Implement reward service: `profit − λ_spend · spend + bonuses(ROAS, conversions)`
- [ ] Implement featurizer to produce `AdEnvironmentState` per adset‑hour

## 7) Guardrails (Cost & Safety)

- [ ] Daily cap: $30 combined (stop increases when projected spend ≈ target)
- [ ] Per‑hour delta clamp: ±10%
- [ ] Min/max hourly floors: $0.50 – $3.00 during pilot
- [ ] Peak hours only: 18–22 local time
- [ ] No‑sales freeze: zero conversions for 12 hours (lag‑aware) or ROAS < 1.5
- [ ] Kill switch + rollback to last known budgets

## 8) Runner (Shadow → Pilot)

- [ ] Add hourly runner `src/run/real.ts` with flags:
  - [ ] `--mode=shadow|pilot`
  - [ ] `--daily-budget-target=30`
  - [ ] `--peak-hours=18-22`
  - [ ] `--delta-max=0.10`
  - [ ] `--lambda-spend=0.25`
  - [ ] `--lagrange-step=0.05`
  - [ ] `--canary-list=platform:adset_id,...`
- [ ] Shadow mode: ingest → featurize → decide → log “would‑apply” actions (no writes)
- [ ] Pilot mode (after shadow): apply budget deltas within guardrails

## 9) Monitoring & Alerts

- [ ] Dashboards: spend vs target, profit/day, ROAS by hour/platform, suggested vs applied
- [ ] Alerts: zero‑conversion streaks, low ROAS, API failures, queue lag
- [ ] Structured logs to `actions_log`

## 10) Training & Model

- [ ] Nightly retrain from `experiences` (replay buffer)
- [ ] Save checkpoints every 50–100 episodes
- [ ] Keep live epsilon low (0.05–0.10); larger exploration offline
- [ ] Adjust λ_spend daily to track $30 target

## 11) Shadow Readiness Review (Go/No‑Go for Pilot)

- [ ] State correctness validated (features align with platform configs)
- [ ] Reward fidelity validated (Shopify revenue vs platform conversions reconciled)
- [ ] Suggested actions look sane (no wild oscillations)
- [ ] Guardrails exercised in dry runs
- [ ] Rollback plan tested

## 12) Pilot Activation (Budget‑Only)

- [ ] Enable `--mode=pilot` for the two canary adsets
- [ ] Verify per‑hour budget stays within $3/adset (combined ≈ $6/hr, peak hours)
- [ ] Monitor daily; freeze on no‑sales or low ROAS
- [ ] Retrain nightly; checkpoint saved

## 13) Weekly Review & Expansion Criteria

- [ ] Trailing 7–14 day ROAS ≥ target; daily profit positive
- [ ] Suggested vs applied alignment; low override rate
- [ ] API reliability acceptable (low error/429 rates)
- [ ] If green: add one more adset per platform or widen hours slightly

## 14) Sign‑Offs

- [ ] Business owner approval for pilot
- [ ] Engineering sign‑off on safety/rollback
- [ ] Data/analytics sign‑off on reward and attribution

---

References:

- docs/api_spec.md — Endpoints, scopes, flags
- docs/real_integration.md — Architecture, adapters, reward/guardrails
- docs/low_spend_rollout.md — Cost‑sensitive strategy & no‑sales handling
