// Core interfaces & types
// DQN-REFAC TODO:
// - Keep categorical domains stable (age groups, creative types, platforms) for deterministic encoding.
// - If adding features, update `docs/torchjs_dqn_refactor.md` and `src/agent/encoding.ts`.

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

export interface AdAction {
  budgetAdjustment: number; // -50% to +100% multiplier
  targetAgeGroup: string; // New age group to target
  targetInterests: string[]; // New interests to target
  creativeType: string; // Creative strategy to use
  bidStrategy: "CPC" | "CPM" | "CPA"; // Bidding strategy
  platform: "tiktok" | "instagram" | "shopify"; // Platform focus
}

export interface RewardMetrics {
  revenue: number;
  adSpend: number;
  profit: number;
  roas: number; // Return on Ad Spend
  conversions: number;
}
