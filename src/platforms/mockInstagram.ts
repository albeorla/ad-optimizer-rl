import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";
import { AdPlatformAPI } from "./base";

export class MockInstagramAdsAPI extends AdPlatformAPI {
  async updateCampaign(campaignId: string, params: any): Promise<any> {
    return { success: true, campaignId, platform: "instagram" };
  }

  async getCampaignMetrics(campaignId: string): Promise<RewardMetrics> {
    return this.generateInstagramMetrics();
  }

  simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction
  ): RewardMetrics {
    let baseConversion = 0.025;

    if (action.creativeType === "lifestyle" || action.creativeType === "product") {
      baseConversion *= 1.3;
    }

    if (action.targetAgeGroup === "25-34" || action.targetAgeGroup === "35-44") {
      baseConversion *= 1.25;
    }

    const impressions = Math.floor(action.budgetAdjustment * 8000 * Math.random());
    const clicks = Math.floor(impressions * state.historicalCTR * 1.1);
    const conversions = Math.floor(clicks * baseConversion);
    const revenue = conversions * 29.99;
    const adSpend = action.budgetAdjustment * state.currentBudget;

    return {
      revenue,
      adSpend,
      profit: revenue - adSpend,
      roas: adSpend > 0 ? revenue / adSpend : 0,
      conversions,
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

