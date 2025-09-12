# Production Integration Guide: Shopify + TikTok + Instagram

This guide explains how to convert the simulator into a production RL system that operates a real Shopify store’s ad spend across TikTok and Instagram (Meta), with strong safety guardrails and a staged rollout.

## 1) Scope & Prerequisites

- Shopify store with permission to create a private/custom app.
- TikTok Business Ads account and app; Meta Business Manager + app for Marketing API.
- Data store (Postgres/BigQuery) for ads metrics, orders, actions, rewards.
- Scheduler/worker (e.g., cron + Node worker, or a queue runner).
- Secure secrets management (dotenv for local, secret manager in prod).

## 2) Architecture Overview

Data ingress → Feature store → Policy (RL) → Safety checks → Actuator → Feedback loop.

```
[Shopify]───┐         ┌────────────┐         ┌───────────┐         ┌───────────┐
            │         │  Featurize │         │   Policy  │         │  Actuator │
[TikTok Ads]├─► ETL ─►│   State    ├─► Decide│  (DQN/Bandit)
            │         │ + Reward   │         │           │         │ (APIs)    │
[Meta/IG]───┘         └────┬───────┘         └────┬──────┘         └────┬──────┘
                            │                     │                    │
                           DB ◄───────────────────┴───── Feedback ◄────┘
```

Key repo components you’ll extend:
- `src/platforms/base.ts` (AdPlatformAPI)
- `src/platforms/factory.ts`
- `src/environment/simulator.ts` (replace with a real environment service)
- `src/agent/*` (policy stays the same initially)
- `src/training/pipeline.ts` (hourly loop for real runs)

## 3) Data Model (Suggested Tables)

Create tables (Postgres style, adapt as needed):
- `ad_accounts(platform, external_id, name)`
- `campaigns(id, account_id, platform, external_id, name)`
- `adsets(id, campaign_id, platform, external_id, name, target_age_group, interests[], creative_group)`
- `ads(id, adset_id, platform, external_id, name, creative_type)`
- `insights_hourly(platform, entity_type, entity_id, ts_hour, impressions, clicks, spend, purchases, revenue)`
- `orders(order_id, ts, total, currency, attribution_source, adset_external_id, platform)`
- `actions_log(ts, platform, adset_external_id, action, payload, status, error)`
- `rewards(ts_hour, adset_external_id, platform, profit, roas, conversions)`
- `checkpoints(ts, model_blob_location, meta)`

## 4) Shopify Integration

- App & OAuth: create a Custom App; store Admin API token securely.
- Pull orders via GraphQL (recommended) or REST, hourly:
  - fields: id, createdAt, currentTotalPrice, lineItems, discounts, canceledAt.
  - preserve UTM params if available (for attribution join).
- Webhooks: subscribe to `orders/create`, `orders/updated`, `refunds/create`.
- Persist orders into `orders` and aggregate per hour for rewards.

Code placeholder (create later): `src/datasources/shopify.ts`

## 5) Meta (Instagram) Marketing API

- App with permissions: `ads_management`, `ads_read`, `business_management`.
- OAuth and long‑lived tokens. Respect rate limits and app review requirements.
- Read hourly insights per adset: spend, impressions, clicks, purchases.
- Update budgets and bid strategies on adsets.

Code placeholder: `src/platforms/meta.ts`

Skeleton:
```ts
import { AdPlatformAPI } from "./base";
import { AdEnvironmentState, AdAction, RewardMetrics } from "../types";

export class MetaAdsAPI extends AdPlatformAPI {
  constructor(private token: string) { super(); }

  async updateCampaign(adsetId: string, params: any) {
    // PATCH adset budget/bid via Marketing API
  }

  async getCampaignMetrics(adsetId: string): Promise<RewardMetrics> {
    // Map insights to RewardMetrics (revenue via Shopify join preferred)
    return { revenue: 0, adSpend: 0, profit: 0, roas: 0, conversions: 0 };
  }

  simulatePerformance(_s: AdEnvironmentState, _a: AdAction): RewardMetrics {
    throw new Error("simulatePerformance not used in production environment");
  }
}
```

## 6) TikTok Business Ads API

- Create app, OAuth; implement HMAC signing when required.
- Endpoints: get insights per adgroup/ad; update budgets/bids.
- Respect rate limits and retries; batch pulls.

Code placeholder: `src/platforms/tiktok.ts`

## 7) Feature Engineering → `AdEnvironmentState`

For each adset-hour, produce:
- `dayOfWeek`, `hourOfDay`: store timezone-aware.
- `currentBudget`: adset daily budget from API.
- `targetAgeGroup`, `targetInterests`: from adset config (normalize across platforms).
- `creativeType`: tag creatives by group (ugc/lifestyle/product/discount).
- `platform`: `tiktok` | `instagram`.
- `historicalCTR`: moving average of clicks/impressions.
- `historicalCVR`: moving average of purchases/clicks (use Shopify attribution if possible).
- `competitorActivity`: proxy via CPM deltas.
- `seasonality`: weekly/hourly seasonal factors (from orders history).

Implement a featurizer: `src/features/featurizer.ts` (new) that queries DB and produces state rows.

## 8) Reward Computation (Production)

Recommended canonical reward source: Shopify revenue (joined to adsets), not platform‑reported conversions.
- Profit = revenue − spend per adset/hour.
- Bonuses: ROAS tiers (>2, >3, >4), small per‑conversion bonus.
- Penalties: high spend with poor ROAS, budget spikes.
- Handle attribution lag: optionally distribute a fraction of a purchase backward across previous k hours.

Implement: `src/services/reward.ts` to compute and persist into `rewards` table hourly.

## 9) Real Environment (replace simulator for live runs)

Create `src/environment/realEnvironment.ts` that implements the same step API used by the pipeline:
- Build state from feature store for the current hour.
- Use policy to propose actions.
- Apply guardrails (budget caps, delta limits, freeze windows).
- If in shadow mode: log actions and exit.
- If in pilot mode: call platform APIs to update budgets.
- Compute reward next hour using Shopify revenue/spend; feed back into agent.

## 10) Actions Mapping

- `budgetAdjustment`: map to adset daily budget changes; clamp deltas (e.g., ±25% per hour) and global min/max.
- `bidStrategy`: map to platform equivalents (Meta lowest_cost/cost_cap, TikTok similar).
- `creativeType`: rotate among predefined ad IDs tagged with the type; do this carefully (one rotation per day in early pilots).
- `targetAgeGroup/Interests`: prefer A/B clones; avoid changing an adset’s targeting mid‑flight.

## 11) Safety Guardrails

- Hard caps: max daily spend per adset/campaign/account.
- Per‑change limits: max ±X% per hour; cool‑down windows.
- Pause conditions: if ROAS below threshold for N hours, stop changes and alert.
- Canary: start with 5–10% of budget managed by RL; expand on success.
- Kill switch: immediate rollback to last known budgets.

Implement a guardrail module: `src/execution/guardrails.ts`.

## 12) Execution Pipeline (Shadow → Pilot → Expand)

Add a runner (hourly):
- Shadow mode: compute actions, write to `actions_log` with status `shadow`.
- Pilot mode: apply only budgets to canary adsets; write status `applied` with API response.
- Expand: add creative rotations; later, add targeting changes via A/B adsets.

Runner entrypoint: `src/run/real.ts`.

## 13) Configuration & Secrets

Create `.env` values (example):

```
SHOPIFY_STORE_DOMAIN=your-store.myshopify.com
SHOPIFY_ADMIN_TOKEN=shpat_...

META_APP_ID=...
META_APP_SECRET=...
META_LONG_LIVED_TOKEN=...
META_AD_ACCOUNT_ID=act_...

TIKTOK_APP_ID=...
TIKTOK_SECRET=...
TIKTOK_ACCESS_TOKEN=...
TIKTOK_AD_ACCOUNT_ID=...

DB_URL=postgres://...
TZ=America/New_York
```

Wire these into constructors of adapters/services.

## 14) Observability & Alerts

- Dashboards: spend, revenue, ROAS, profit by platform/creative/time.
- Model: epsilon, Q‑table size, reward trends.
- Alerts: spend spikes, low ROAS, API failures, queue lag.
- Logging: structured logs in `actions_log` and a central logger.

## 15) Testing Strategy

- Unit tests: adapters with mocked HTTP.
- Sandboxes: use test ad accounts where possible.
- Dry runs: shadow mode for 1–2 weeks; compare suggested vs human changes.
- Synthetic backtests: feed historical hourly data into the RL loop offline.

## 16) Rollout Plan (Checklist)

1. Create apps + tokens; verify read access (Shopify, Meta, TikTok).
2. Stand up DB and migrations for the tables above.
3. Implement ETL jobs for Shopify orders and platform insights (hourly).
4. Build featurizer → `AdEnvironmentState` per adset/hour.
5. Implement reward service from Shopify revenue − spend.
6. Implement `MetaAdsAPI` and `TikTokAdsAPI` adapters with rate limiting/backoff.
7. Implement guardrails; add configuration caps.
8. Add `realEnvironment.ts` and `run/real.ts` (shadow mode only).
9. Validate for 1–2 weeks; fix state/reward mismatches.
10. Pilot: canary budgets on a small set of adsets; review daily.
11. Expand safely; add creative rotation; later, A/B targeting.
12. Add nightly retraining + periodic checkpointing.

## 17) Code Map: Where To Plug In

- `src/platforms/meta.ts`: real Instagram/Meta API adapter (extends `AdPlatformAPI`).
- `src/platforms/tiktok.ts`: real TikTok API adapter (extends `AdPlatformAPI`).
- `src/datasources/shopify.ts`: Shopify order fetch + webhooks.
- `src/features/featurizer.ts`: builds `AdEnvironmentState` per adset-hour.
- `src/services/reward.ts`: computes and persists hourly rewards.
- `src/execution/guardrails.ts`: sanity checks and clamps for actions.
- `src/environment/realEnvironment.ts`: production environment loop.
- `src/run/real.ts`: scheduled runner (shadow/pilot modes).

## 18) Security & Compliance

- Store tokens encrypted; rotate regularly; least-privilege scopes.
- Log PII-free; respect ad network policies and rate limits.
- Add approvals for budget increases >X% or over threshold amounts.

---

When ready, I can scaffold the real adapters and a shadow‑mode runner with typed stubs, rate-limiters, and retry/backoff, wired behind feature flags so you can flip between simulator and real mode.

