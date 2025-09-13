import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";
import { AdPlatformAPI } from "./base";
// DQN-REFAC TODO:
// - No model changes; simulator should continue to produce realistic signals.
// - Keep demographic/creative/time sensitivities consistent with docs and encoder.

export class MockTikTokAdsAPI extends AdPlatformAPI {
  private campaigns: Map<string, any> = new Map();
  private shapingStrength: number;

  constructor(shapingStrength: number = 1) {
    super();
    this.shapingStrength = shapingStrength;
  }

  async updateCampaign(campaignId: string, params: any): Promise<any> {
    await this.simulateLatency();
    this.campaigns.set(campaignId, {
      ...this.campaigns.get(campaignId),
      ...params,
      updatedAt: new Date().toISOString(),
    });
    return { success: true, campaignId };
  }

  async getCampaignMetrics(campaignId: string): Promise<RewardMetrics> {
    await this.simulateLatency();
    return this.generateMockMetrics();
  }

  simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction
  ): RewardMetrics {
    // Budget-driven base impressions ($1 ~ 20 impressions)
    const budgetAmount = state.currentBudget * action.budgetAdjustment;
    const baseImpressions = budgetAmount * 20;

    // Performance multiplier: demographics, creative, time-of-day
    let performanceMultiplier = 1.0;
    // TikTok: young audiences excel
    if (action.targetAgeGroup === "18-24") performanceMultiplier *= 1.5 * this.shapingStrength;
    else if (action.targetAgeGroup === "25-34") performanceMultiplier *= 1.2;
    else if (action.targetAgeGroup === "45+") performanceMultiplier *= 0.8;
    // TikTok: UGC thrives; discounts underperform
    if (action.creativeType === "ugc") performanceMultiplier *= 1.3 * this.shapingStrength;
    else if (action.creativeType === "discount") performanceMultiplier *= 0.8;
    // Peak hours boost (evening)
    if (state.hourOfDay >= 18 && state.hourOfDay <= 22) performanceMultiplier *= 1.5;
    else if (state.hourOfDay >= 0 && state.hourOfDay <= 6) performanceMultiplier *= 0.6;
    // Diminishing returns for aggressive budgets
    if (action.budgetAdjustment > 1.5) {
      performanceMultiplier *= Math.max(0.5, 2.0 - action.budgetAdjustment * 0.8);
    }

    // Calculate realistic metrics
    const effectiveImpressions = baseImpressions * performanceMultiplier;
    const ctr = 0.02 * performanceMultiplier; // 2% base CTR
    const clicks = Math.floor(effectiveImpressions * ctr);
    const conversionRate = 0.03 * performanceMultiplier; // 3% base CVR
    const conversions = Math.floor(clicks * conversionRate);
    const revenueNominal = conversions * 29.99;
    const adSpend = budgetAmount; // actual spend equals budget
    const variance = 0.9 + Math.random() * 0.2; // ±10%
    const revenue = revenueNominal * variance;

    return {
      revenue,
      adSpend,
      profit: revenue - adSpend,
      roas: adSpend > 0 ? revenue / adSpend : 0,
      conversions: Math.floor(conversions * variance),
    };
  }

  private async simulateLatency(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, Math.random() * 100));
  }

  private generateMockMetrics(): RewardMetrics {
    return {
      revenue: Math.random() * 5000,
      adSpend: Math.random() * 1000,
      profit: Math.random() * 4000,
      roas: 2 + Math.random() * 3,
      conversions: Math.floor(Math.random() * 100),
    };
  }
}
