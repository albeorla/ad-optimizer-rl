import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";
import { AdPlatformAPI } from "./base";

export class MockTikTokAdsAPI extends AdPlatformAPI {
  private campaigns: Map<string, any> = new Map();

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
    let baseConversion = 0.02;

    if (action.targetAgeGroup === "18-24") {
      baseConversion *= 1.5;
    } else if (action.targetAgeGroup === "25-34") {
      baseConversion *= 1.2;
    }

    if (action.creativeType === "ugc") {
      baseConversion *= 1.4;
    }

    const impressions = Math.floor(action.budgetAdjustment * 10000 * Math.random());
    const clicks = Math.floor(impressions * state.historicalCTR);
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

