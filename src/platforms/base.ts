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
// DQN-REFAC TODO:
// - No changes for DQN itself; ensure real adapters produce state fields the encoder expects.
// - Consider adding a method to return feature views for parity between sim and real.
