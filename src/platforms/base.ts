import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";

/**
 * Abstract base for ad platform APIs (Dependency Inversion Principle).
 *
 * Real adapters should implement read/write calls (updateCampaign/getCampaignMetrics).
 * Simulator/mocks implement simulatePerformance for offline training.
 */
export abstract class AdPlatformAPI {
  /** Update campaign settings (budget, targeting, creative). Real adapters only. */
  abstract updateCampaign(campaignId: string, params: any): Promise<any>;
  /** Retrieve aggregated performance metrics from the platform. Real adapters only. */
  abstract getCampaignMetrics(campaignId: string): Promise<RewardMetrics>;
  /**
   * Simulate one step of performance given state and action. Simulator/mocks only.
   * @returns RewardMetrics used for reward shaping and training.
   */
  abstract simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction,
  ): RewardMetrics;
}
// DQN-REFAC NOTE:
// - Ensure real adapters produce state fields the encoder expects.
// - Consider exposing a feature view for parity between sim and real.
