import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";
import { AdPlatformAPI } from "./base";

export interface TikTokAdMetrics {
  adSpend: number;
  impressions?: number;
  clicks?: number;
  cpc?: number;
  cpm?: number;
}

export interface TimeWindow {
  start: Date;
  end: Date;
}

export class RealTikTokAdsAPI extends AdPlatformAPI {
  private apiKey: string | undefined;

  constructor(apiKey?: string) {
    super();
    this.apiKey = apiKey ?? process.env.TIKTOK_API_KEY;
  }

  // In shadow mode, do not write changes. This stub just logs intent.
  async updateCampaign(campaignId: string, params: any): Promise<any> {
    return { success: false, mode: "shadow", campaignId, wouldUpdate: params };
  }

  // Not recommended for real usage; use getAdMetricsForWindow + Shopify data to compute reward.
  async getCampaignMetrics(campaignId: string): Promise<RewardMetrics> {
    return { revenue: 0, adSpend: 0, profit: 0, roas: 0, conversions: 0 };
  }

  // Real data fetcher to get ad spend for a given window.
  // TODO: Implement TikTok Ads API calls with filtering by campaign/adgroup and time granularity.
  async getAdMetricsForWindow(window: TimeWindow, campaignId?: string): Promise<TikTokAdMetrics> {
    return { adSpend: 0 };
  }

  // Real adapters should not simulate performance; the environment must compose real metrics.
  simulatePerformance(state: AdEnvironmentState, action: AdAction): RewardMetrics {
    throw new Error("simulatePerformance is not supported in RealTikTokAdsAPI. Use real metrics composition.");
  }
}

