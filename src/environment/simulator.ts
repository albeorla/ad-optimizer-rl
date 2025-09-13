import { AdAction, AdEnvironmentState } from "../types";
import { MockTikTokAdsAPI } from "../platforms/mockTikTok";
import { MockInstagramAdsAPI } from "../platforms/mockInstagram";
import { AdPlatformAPI } from "../platforms/base";
// DQN-REFAC TODO:
// - Keep state fields stable and documented; encoder depends on deterministic categories.
// - If you add/rename features, update `src/agent/encoding.ts` and docs accordingly.
// - Consider exposing a 'feature view' for real adapters to ensure parity with simulator.

export class AdEnvironmentSimulator {
  private platforms: Map<string, AdPlatformAPI>;
  private currentState: AdEnvironmentState;
  private timeStep: number = 0;

  constructor(opts?: { shapingStrength?: number }) {
    const shaping = opts?.shapingStrength ?? 1;
    const entries: Array<[string, AdPlatformAPI]> = [
      ["tiktok", new MockTikTokAdsAPI(shaping)],
      ["instagram", new MockInstagramAdsAPI(shaping)],
    ];
    this.platforms = new Map<string, AdPlatformAPI>(entries);

    this.currentState = this.generateInitialState();
  }

  private generateInitialState(): AdEnvironmentState {
    return {
      dayOfWeek: 0,
      hourOfDay: 12,
      currentBudget: 500,
      targetAgeGroup: "25-34",
      targetInterests: ["fashion", "lifestyle"],
      creativeType: "product",
      platform: "tiktok",
      historicalCTR: 0.02,
      historicalCVR: 0.01,
      competitorActivity: 0.5,
      seasonality: 0.7,
    };
  }

  reset(): AdEnvironmentState {
    this.timeStep = 0;
    this.currentState = this.generateInitialState();
    return this.currentState;
  }

  step(action: AdAction): [AdEnvironmentState, number, boolean, import("../types").RewardMetrics] {
    const platform = this.platforms.get(action.platform);
    if (!platform) throw new Error(`Platform ${action.platform} not found`);

    const metrics = platform.simulatePerformance(this.currentState, action);
    const reward = this.calculateReward(metrics);
    this.currentState = this.updateState(action);
    this.timeStep++;
    const done = this.timeStep >= 24;
    return [this.currentState, reward, done, metrics];
  }

  // Reward shaping to guide learning toward good ROAS and sensible spend
  private calculateReward(metrics: import("../types").RewardMetrics): number {
    let reward = metrics.profit / 1000; // Base reward
    // Bonus for strong ROAS
    if (metrics.roas > 4.0) reward += 1.0;
    else if (metrics.roas > 3.0) reward += 0.5;
    else if (metrics.roas > 2.0) reward += 0.2;
    // Penalty for overspending heavily
    if (metrics.adSpend > 1500) reward -= (metrics.adSpend - 1500) / 6000;
    // Small bonus per conversion
    reward += metrics.conversions * 0.01;
    return reward;
  }

  private updateState(action: AdAction): AdEnvironmentState {
    const newState = { ...this.currentState };
    newState.hourOfDay = (newState.hourOfDay + 1) % 24;
    if (newState.hourOfDay === 0) newState.dayOfWeek = (newState.dayOfWeek + 1) % 7;

    newState.currentBudget = this.currentState.currentBudget * action.budgetAdjustment;
    newState.targetAgeGroup = action.targetAgeGroup;
    newState.targetInterests = action.targetInterests;
    newState.creativeType = action.creativeType;
    newState.platform = action.platform;

    newState.historicalCTR = Math.max(0.001, newState.historicalCTR + (Math.random() - 0.5) * 0.002);
    newState.historicalCVR = Math.max(0.001, newState.historicalCVR + (Math.random() - 0.5) * 0.001);

    newState.competitorActivity = Math.min(
      1,
      Math.max(0, newState.competitorActivity + (Math.random() - 0.5) * 0.1)
    );
    newState.seasonality = 0.7 + 0.3 * Math.sin((this.timeStep / 168) * Math.PI * 2);
    return newState;
  }
}
