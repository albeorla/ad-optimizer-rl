# TikTok + Shopify Integration & Migration Guide

This guide consolidates and supersedes prior integration notes and rollout checklists. It describes exactly how to connect this repo to Shopify (revenue) and TikTok Ads (spend + budgets), validate in shadow mode, and run a tightly‑scoped pilot with guardrails.

## Objectives

- Read Shopify revenue and conversions per hour and TikTok ad spend per hour.
- Compute profit/ROAS and use it as reward in shadow mode (no writes).
- Pilot limited hourly budget updates on a canary with strict guardrails.

## Prerequisites

- Shopify store with a Custom App (Admin API token) and store domain.
- TikTok For Business advertiser account + access token; one canary campaign/adgroup ID.
- Timezone finalized for hourly buckets (e.g., `TZ=America/New_York`).

## Configuration (.env)

- `SHOPIFY_STORE_DOMAIN=your-store.myshopify.com`
- `SHOPIFY_API_KEY=shpat_...` (Admin API access token)
- `TIKTOK_API_KEY=...` (or OAuth access token)
- `DAILY_BUDGET_TARGET=30` (USD)
- `COGS_PER_UNIT=15` (or `PRINTFUL_COGS`)
- `TZ=America/New_York`
- Optional cost sensitivity: `LAMBDA_SPEND=0.25`, `LAGRANGE_STEP=0.05`

## Repo Hotspots (where to plug in)

- Shopify datasource: `src/datasources/shopify.ts:32` — implement `getSales(window)` to aggregate hourly revenue + conversions.
- TikTok adapter (read): `src/platforms/realTikTok.ts:43` — implement `getAdMetricsForWindow(window, campaignId?)` to return hourly ad spend.
- TikTok adapter (write for pilot): `src/platforms/realTikTok.ts:31` — implement `updateCampaign(campaignId, params)` for budget updates (pilot only).
- Shadow environment: `src/environment/realShadow.ts:61` — composes Shopify + TikTok metrics into `RewardMetrics` and reward.
- Guardrails: `src/execution/guardrails.ts:1` — enforces daily cap, delta limits, min/max floors.
- Real runner skeleton: `src/run/real.ts:1` — applies guardrails and prints allowed budgets; wire writes in pilot mode.

## Step‑By‑Step Migration

1) Implement Shopify hourly revenue

- In `src/datasources/shopify.ts:32`, replace the stub with a real fetch using Admin API (GraphQL recommended):
  - Query orders created in `[start, end)` with `financial_status` paid/partially_paid.
  - Sum `current_total_price` as `revenue`; count orders as `conversions`.
  - Optional (pilot‑friendly): basic attribution — include orders with UTM source matching TikTok, or maintain a mapping via checkout/pixel metadata.

2) Implement TikTok hourly spend

- In `src/platforms/realTikTok.ts:43`, call TikTok reporting for advertiser/campaign/adgroup within `[start, end)`:
  - Return `{ adSpend }` in USD for the window. If the API provides only daily breakdowns, align the environment window to daily during validation.
  - Keep rate limiting/backoff conservative.

3) Validate in Shadow Mode (read‑only)

- Use the real shadow training loop, which composes Shopify + TikTok:

```
npm run run:shadow -- --episodes=14
```

- Expect non‑zero `revenue`, `adSpend`, and shaped reward prints. No platform writes occur in shadow.

4) Configure Guardrails and Runner

- Guardrails are applied in the real runner:

```
npm run run:real -- \
  --mode=shadow \
  --daily-budget-target=30 \
  --delta-max=0.10 \
  --min-hourly=0.5 \
  --max-hourly=3.0 \
  --canary-list="tiktok:ADGROUP_ID"
```

- Review allowed hourly budgets and reasons (no writes yet).

5) Pilot: Budget‑Only Writes on Canary

- Implement `updateCampaign` in `src/platforms/realTikTok.ts:31` to set budget at the entity you manage (campaign/adgroup).
- In `src/run/real.ts`, when `--mode=pilot`, compute `allowedBudget` via `applyGuardrails` and call the TikTok update for the canary IDs.
- Run pilot during peak hours only (e.g., 18–22 local) with ±10% per‑hour delta, floors $0.50–$3.00, and daily cap $30 combined.

## Safety Guardrails (musts)

- Daily cap: stop increases when projected spend ~= target (`DAILY_BUDGET_TARGET`).
- Per‑hour delta clamp: ±10% (`deltaMax`).
- Min/max hourly floors: enforce in early pilots.
- Freeze: if trailing ROAS < threshold or no sales for N hours, allow min floor only or switch to shadow.
- Canary: restrict writes to an explicit allowlist until stable; keep a kill switch.

## Runbook

- Shadow verification:
  - Configure `.env`, implement Shopify + TikTok reads.
  - `npm run run:shadow -- --episodes=14` (2 weeks of offline episodes at 24 steps/day) or shorter for spot checks.
- Pilot dry‑run with guardrails:
  - `npm run run:real -- --mode=shadow --daily-budget-target=30 --delta-max=0.10 --min-hourly=0.5 --max-hourly=3.0 --canary-list="tiktok:ADGROUP_ID"`
- Pilot live (writes enabled):
  - Implement `updateCampaign` calls and enable `--mode=pilot` with the same flags.

## Open Decisions

- Which TikTok entity level to manage in pilot (campaign vs adgroup)?
- Attribution source of truth (Shopify‑joined revenue vs platform conversions) for reward.
- Peak‑hours window and freeze thresholds.

## Superseded Docs

This guide replaces and consolidates:

- `docs/real_integration.md`
- `docs/low_spend_rollout.md`
- `docs/poc_checklist.md`
- `docs/torchjs_dqn_refactor.md` (Torch.js refactor is out of scope for this TF.js‑based repo)

