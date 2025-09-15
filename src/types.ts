/**
 * Core domain types shared across agents, environments, and platform adapters.
 *
 * Keep these interfaces stable and documented: encoders and adapters depend on
 * deterministic categorical values and field presence for correct operation.
 */
// DQN-REFAC TODO:
// - Keep categorical domains stable (age groups, creative types, platforms) for deterministic encoding.
// - If adding features, update `docs/torchjs_dqn_refactor.md` and `src/agent/encoding.ts`.

/**
 * Environment state presented to the agent at each decision step.
 */
export interface AdEnvironmentState {
  dayOfWeek: number; // 0-6 (Monday-Sunday)
  hourOfDay: number; // 0-23
  currentBudget: number; // Daily budget in USD
  targetAgeGroup: string; // "18-24", "25-34", "35-44", "45+"
  targetInterests: string[]; // ["fashion", "sports", "music", etc.]
  creativeType: string; // "lifestyle", "product", "discount", "ugc"
  platform: string; // "tiktok", "instagram", "shopify"
  historicalCTR: number; // Click-through rate
  historicalCVR: number; // Conversion rate
  competitorActivity: number; // 0-1 (normalized competitor presence)
  seasonality: number; // 0-1 (seasonal demand factor)
}

/**
 * Action chosen by the agent describing the next hour’s campaign adjustments.
 */
export interface AdAction {
  budgetAdjustment: number; // -50% to +100% multiplier
  targetAgeGroup: string; // New age group to target
  targetInterests: string[]; // New interests to target
  creativeType: string; // Creative strategy to use
  bidStrategy: "CPC" | "CPM" | "CPA"; // Bidding strategy
  platform: "tiktok" | "instagram" | "shopify"; // Platform focus
}

/**
 * Unified performance metrics used for reward calculation.
 *
 * revenue: gross revenue
 * adSpend: media spend for the window
 * profit: revenue - adSpend - cogs
 * roas: revenue / adSpend
 * grossMargin: revenue - cogs
 * marginRoas: (revenue - cogs) / adSpend
 * conversions: units sold (or orders)
 */
export interface RewardMetrics {
  revenue: number;
  adSpend: number;
  profit: number; // revenue - adSpend - cogs
  roas: number; // Return on Ad Spend (revenue / adSpend)
  cogs?: number; // cost of goods sold (optional in older sims)
  grossMargin?: number; // revenue - cogs
  marginRoas?: number; // (revenue - cogs) / adSpend
  conversions: number;
}
