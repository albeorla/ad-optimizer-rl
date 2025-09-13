# API Integration Specification

This document defines the integration targets and contract for connecting the RL system to Shopify, Meta (Instagram) Marketing API, and TikTok Business Ads, operating under a strict $30/day budget with cost‑sensitive optimization.

## Goals & KPIs

- Daily budget target: $30 (combined TikTok + Instagram, canary scope)
- Primary KPI: profit/day (Shopify revenue − ad spend − COGS/fees)
- Supporting KPIs: ROAS, CPA, conversions, spend vs budget target
- Safety: zero‑conversion freeze windows, per‑hour delta caps, peak‑hours focus (18–22 local)

## Integration Overview

- Adapters extend `AdPlatformAPI` with real REST calls + rate limiting
- Data flows hourly: ingest → featurize → decide → guardrails → (apply or shadow) → log
- Reward computed from Shopify revenue − spend with cost penalty λ_spend and lag handling

## Shopify Admin API

Purpose: orders, refunds, discounts for revenue/profit; optional pixel/UTM data for attribution.

- Auth: Admin API access token (Custom App). Store securely.
- Interface:
  - Orders (GraphQL preferred). Collect hourly since last watermark.
  - Webhooks: `orders/create`, `orders/updated`, `refunds/create`.
- Core fields: `id`, `createdAt`, `currentTotalPrice`, `currencyCode`, `cancelledAt`, `lineItems`, `discountAllocations`.
- Notes:
  - Map to hourly buckets per timezone.
  - Persist to `orders` and aggregate to `rewards` by adset (via attribution join).
  - Include COGS/fees if available for true profit.

## Meta (Instagram) Marketing API

Purpose: read hourly insights (spend, impressions, clicks, purchases), apply budget updates at adset level.

- Auth: OAuth → long‑lived token; scopes: `ads_management`, `ads_read`, `business_management`.
- Read: insights at adset level (hourly granularity). Fields to request: spend, impressions, clicks, purchases/conversions (for reference), breakdowns as feasible.
- Write: update adset daily budget; adjust bid strategy where allowed.
- Rate limits: implement retry/backoff; limit concurrency; batch requests.
- Mapping:
  - `adset_id` ⇄ internal `adset_external_id`
  - Insights → `insights_hourly`
  - Budget updates logged to `actions_log`

## TikTok Business Ads API

Purpose: read hourly reports (spend, impressions, clicks, conversions), apply budget updates at adgroup level.

- Auth: OAuth; implement HMAC signing if required.
- Read: reporting endpoints for adgroup/ad level; hourly breakdowns.
- Write: update adgroup/adset budget; adjust bid method if permitted.
- Rate limits: retry/backoff, staggered polling, batching.
- Mapping similar to Meta.

## Data & Storage Contract

Tables (logical):
- `insights_hourly(platform, entity_type, entity_id, ts_hour, impressions, clicks, spend, purchases, revenue)`
- `orders(order_id, ts, total, currency, attribution_source, adset_external_id, platform)`
- `rewards(ts_hour, adset_external_id, platform, profit, roas, conversions)`
- `experiences(adset_external_id, platform, ts, state_json, action_json, reward, next_state_json)`
- `actions_log(ts, platform, adset_external_id, action, payload, status, error)`
- `budget_state(date, platform, target_spend, actual_spend, lambda_spend)`

## Reward Function (Cost‑Sensitive)

Baseline (lag‑aware):
- `profit = revenue − spend`
- `reward = profit/1000 + bonuses(ROAS tiers, conversions) − penalties(overspend)`

Cost penalty and constraint:
- `reward ← reward − λ_spend · adSpend` (λ_spend configurable, default 0.25)
- `λ_spend ← max(0, λ_spend + η · (spendToDate − budgetTarget)/budgetTarget)` daily (η default 0.05)

## Guardrails (Enforced in Runner)

- Daily cap: $30 combined. Stop increases once projected daily spend reaches target.
- Per‑hour delta: ±10% (configurable). Clamp at min/max hourly floors.
- Peak hours: only 18–22 local time in pilot.
- No‑sales freeze: if conversions = 0 for N hours (e.g., 12) or trailing ROAS < 1.5, allow only minimum floors or switch to shadow.
- Kill switch: instant rollback; no writes on anomalies.

## Runner (Shadow/Pilot) Flags

- `--mode=shadow|pilot`
- `--daily-budget-target=30`
- `--peak-hours=18-22`
- `--delta-max=0.10`
- `--lambda-spend=0.25`
- `--lagrange-step=0.05`
- `--canary-list=platform:adset_id,...`

## Open Items & Ownership

- Confirm account IDs, app credentials, and canary adsets.
- Decide attribution method (platform vs Shopify‑joined) for reward canonicalization.
- Provide profit parameters (COGS, fees) or accept revenue‑based proxy initially.
- Approve peak‑hours window and freeze thresholds.

