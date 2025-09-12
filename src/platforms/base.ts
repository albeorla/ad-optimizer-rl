import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";

// Abstract base for ad platform APIs (Dependency Inversion Principle)
export abstract class AdPlatformAPI {
  abstract updateCampaign(campaignId: string, params: any): Promise<any>;
  abstract getCampaignMetrics(campaignId: string): Promise<RewardMetrics>;
  abstract simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction
  ): RewardMetrics;
}

