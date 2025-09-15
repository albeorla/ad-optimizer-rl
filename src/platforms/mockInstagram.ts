import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";
import { AdPlatformAPI } from "./base";
// DQN-REFAC TODO:
// - No model changes; keep behavior complementary to TikTok for diversity in training.
// - Revisit shapingStrength to calibrate reward scales if needed.

export class MockInstagramAdsAPI extends AdPlatformAPI {
  private shapingStrength: number;
  private productPrice: number;
  private cogsPerUnit: number;

  constructor(
    shapingStrength: number = 1,
    productPrice: number = 29.99,
    cogsPerUnit: number = 15.0,
  ) {
    super();
    this.shapingStrength = shapingStrength;
    this.productPrice = productPrice;
    this.cogsPerUnit = cogsPerUnit;
  }
  async updateCampaign(campaignId: string, params: any): Promise<any> {
    return { success: true, campaignId, platform: "instagram" };
  }

  async getCampaignMetrics(campaignId: string): Promise<RewardMetrics> {
    return this.generateInstagramMetrics();
  }

  simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction,
  ): RewardMetrics {
    // Budget-driven base impressions
    const budgetAmount = state.currentBudget * action.budgetAdjustment;
    const baseImpressions = budgetAmount * 18; // IG slightly fewer per $ than TikTok

    // Performance multipliers: demographics, creative, time-of-day
    let performanceMultiplier = 1.0;
    if (action.targetAgeGroup === "25-34" || action.targetAgeGroup === "35-44")
      performanceMultiplier *= 1.25 * this.shapingStrength;
    if (
      action.creativeType === "lifestyle" ||
      action.creativeType === "product"
    )
      performanceMultiplier *= 1.3 * this.shapingStrength;
    // Discounts acceptable on IG but neutral
    if (action.creativeType === "discount") performanceMultiplier *= 1.0;
    // Peak hours boost (evening)
    if (state.hourOfDay >= 18 && state.hourOfDay <= 22)
      performanceMultiplier *= 1.5;
    else if (state.hourOfDay >= 0 && state.hourOfDay <= 6)
      performanceMultiplier *= 0.7;
    // Diminishing returns for aggressive budgets
    if (action.budgetAdjustment > 1.5) {
      performanceMultiplier *= Math.max(
        0.5,
        2.0 - action.budgetAdjustment * 0.8,
      );
    }

    const effectiveImpressions = baseImpressions * performanceMultiplier;
    const ctr = 0.022 * performanceMultiplier; // IG CTR
    const clicks = effectiveImpressions * ctr; // expected clicks (not floored)
    const conversionRate = 0.032 * performanceMultiplier; // IG CVR
    const conversions = clicks * conversionRate; // expected conversions
    const adSpend = budgetAmount;
    const variance = 0.9 + Math.random() * 0.2; // ±10%
    const revenueNominal = conversions * this.productPrice;
    const revenue = revenueNominal * variance;
    const units = conversions * variance;
    const cogs = units * this.cogsPerUnit;
    const grossMargin = revenue - cogs;
    const marginRoas = adSpend > 0 ? grossMargin / adSpend : 0;

    return {
      revenue,
      adSpend,
      profit: revenue - adSpend - cogs,
      roas: adSpend > 0 ? revenue / adSpend : 0,
      cogs,
      grossMargin,
      marginRoas,
      conversions: units,
    };
  }

  private generateInstagramMetrics(): RewardMetrics {
    return {
      revenue: Math.random() * 6000,
      adSpend: Math.random() * 1200,
      profit: Math.random() * 4800,
      roas: 2.5 + Math.random() * 3,
      conversions: Math.floor(Math.random() * 120),
    };
  }
}
