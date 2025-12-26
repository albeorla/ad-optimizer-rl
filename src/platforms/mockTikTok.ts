import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";
import { AdPlatformAPI } from "./base";

/**
 * Ad fatigue tracking for realistic performance decay.
 */
interface CreativeFatigueState {
  impressions: number;
  lastSeenHour: number;
  totalClicks: number;
}

/**
 * Synthetic TikTok Ads adapter used by the simulator. Encodes platform-specific
 * multipliers for demographics, creative types, and time-of-day effects.
 *
 * Features:
 * - Ad fatigue simulation: CTR/CVR decay as impressions accumulate
 * - Creative rotation incentives: Fresh creatives perform better
 * - Audience saturation: Diminishing returns on repeated targeting
 * - Time-of-day effects: Peak hours boost engagement
 */
export class MockTikTokAdsAPI extends AdPlatformAPI {
  private campaigns: Map<string, any> = new Map();
  private shapingStrength: number;
  private productPrice: number;
  private cogsPerUnit: number;

  // Ad fatigue tracking: creative type + age group -> fatigue state
  private creativeFatigue: Map<string, CreativeFatigueState> = new Map();

  // Fatigue parameters
  private readonly fatigueDecayRate = 0.99; // CTR decay per 1000 impressions
  private readonly fatigueRecoveryRate = 0.1; // Recovery per hour of not running
  private readonly maxFatigueDecay = 0.5; // Minimum CTR multiplier (50%)

  // Audience saturation tracking
  private audienceImpressions: Map<string, number> = new Map();
  private readonly saturationThreshold = 5000; // Impressions before diminishing returns
  private readonly saturationDecay = 0.8; // Performance at saturation

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

  /**
   * Reset fatigue state (e.g., for new episode or after creative refresh).
   */
  resetFatigue(): void {
    this.creativeFatigue.clear();
    this.audienceImpressions.clear();
  }

  /**
   * Get fatigue multiplier for a creative+audience combination.
   * Returns a value between maxFatigueDecay and 1.0.
   */
  private getFatigueMultiplier(
    creativeType: string,
    ageGroup: string,
    currentHour: number,
  ): number {
    const key = `${creativeType}:${ageGroup}`;
    const state = this.creativeFatigue.get(key);

    if (!state) {
      // Fresh creative - no fatigue
      return 1.0;
    }

    // Calculate fatigue based on impressions
    const impressionFatigue = Math.pow(
      this.fatigueDecayRate,
      state.impressions / 1000,
    );

    // Calculate recovery based on hours since last seen
    const hoursSinceLastSeen = Math.max(0, currentHour - state.lastSeenHour);
    const recovery = Math.min(
      1.0,
      hoursSinceLastSeen * this.fatigueRecoveryRate,
    );

    // Combine fatigue with recovery
    const baseFatigue = Math.max(this.maxFatigueDecay, impressionFatigue);
    const fatigueMultiplier = baseFatigue + (1 - baseFatigue) * recovery;

    return Math.min(1.0, fatigueMultiplier);
  }

  /**
   * Update fatigue state after showing impressions.
   */
  private updateFatigueState(
    creativeType: string,
    ageGroup: string,
    impressions: number,
    currentHour: number,
  ): void {
    const key = `${creativeType}:${ageGroup}`;
    const existing = this.creativeFatigue.get(key);

    if (existing) {
      existing.impressions += impressions;
      existing.lastSeenHour = currentHour;
    } else {
      this.creativeFatigue.set(key, {
        impressions,
        lastSeenHour: currentHour,
        totalClicks: 0,
      });
    }
  }

  /**
   * Get audience saturation multiplier.
   * Performance decreases as the same audience sees more ads.
   */
  private getAudienceSaturationMultiplier(ageGroup: string): number {
    const impressions = this.audienceImpressions.get(ageGroup) || 0;

    if (impressions < this.saturationThreshold) {
      return 1.0;
    }

    // Logarithmic decay beyond threshold
    const overThreshold = impressions - this.saturationThreshold;
    const decayFactor =
      1 - Math.log10(1 + overThreshold / this.saturationThreshold) * 0.2;
    return Math.max(this.saturationDecay, decayFactor);
  }

  /**
   * Update audience saturation state.
   */
  private updateAudienceSaturation(
    ageGroup: string,
    impressions: number,
  ): void {
    const existing = this.audienceImpressions.get(ageGroup) || 0;
    this.audienceImpressions.set(ageGroup, existing + impressions);
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

  /**
   * Generate synthetic RewardMetrics for a single step given state and action.
   * Incorporates ad fatigue and audience saturation for realistic simulation.
   */
  simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction,
  ): RewardMetrics {
    // Budget-driven base impressions ($1 ~ 20 impressions)
    const budgetAmount = state.currentBudget * action.budgetAdjustment;
    const baseImpressions = budgetAmount * 20;

    // Performance multiplier: demographics, creative, time-of-day
    let performanceMultiplier = 1.0;

    // TikTok: young audiences excel
    if (action.targetAgeGroup === "18-24")
      performanceMultiplier *= 1.5 * this.shapingStrength;
    else if (action.targetAgeGroup === "25-34") performanceMultiplier *= 1.2;
    else if (action.targetAgeGroup === "45+") performanceMultiplier *= 0.8;

    // TikTok: UGC thrives; discounts underperform
    if (action.creativeType === "ugc")
      performanceMultiplier *= 1.3 * this.shapingStrength;
    else if (action.creativeType === "discount") performanceMultiplier *= 0.8;

    // Peak hours boost (evening)
    if (state.hourOfDay >= 18 && state.hourOfDay <= 22)
      performanceMultiplier *= 1.5;
    else if (state.hourOfDay >= 0 && state.hourOfDay <= 6)
      performanceMultiplier *= 0.6;

    // Diminishing returns for aggressive budgets
    if (action.budgetAdjustment > 1.5) {
      performanceMultiplier *= Math.max(
        0.5,
        2.0 - action.budgetAdjustment * 0.8,
      );
    }

    // Apply ad fatigue multiplier (teaches agent to rotate creatives)
    const fatigueMultiplier = this.getFatigueMultiplier(
      action.creativeType,
      action.targetAgeGroup,
      state.hourOfDay,
    );
    performanceMultiplier *= fatigueMultiplier;

    // Apply audience saturation multiplier (teaches agent to expand targeting)
    const saturationMultiplier = this.getAudienceSaturationMultiplier(
      action.targetAgeGroup,
    );
    performanceMultiplier *= saturationMultiplier;

    // Calculate realistic metrics
    const effectiveImpressions = baseImpressions * performanceMultiplier;

    // Base rates with fatigue affecting CTR more than CVR
    // (people stop clicking but still buy if they click)
    const baseCtr = 0.02 * performanceMultiplier;
    const baseCvr = 0.03 * Math.sqrt(performanceMultiplier); // Less affected by fatigue

    const clicks = effectiveImpressions * baseCtr;
    const conversions = clicks * baseCvr;
    const revenueNominal = conversions * this.productPrice;
    const adSpend = budgetAmount;

    // Variance ±10%
    const variance = 0.9 + Math.random() * 0.2;
    const revenue = revenueNominal * variance;
    const units = conversions * variance;
    const cogs = units * this.cogsPerUnit;
    const grossMargin = revenue - cogs;
    const marginRoas = adSpend > 0 ? grossMargin / adSpend : 0;

    // Update fatigue and saturation state
    this.updateFatigueState(
      action.creativeType,
      action.targetAgeGroup,
      effectiveImpressions,
      state.hourOfDay,
    );
    this.updateAudienceSaturation(action.targetAgeGroup, effectiveImpressions);

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

  /**
   * Get current fatigue level for a creative+audience (for diagnostics).
   */
  getFatigueLevel(creativeType: string, ageGroup: string): number {
    const key = `${creativeType}:${ageGroup}`;
    const state = this.creativeFatigue.get(key);
    return state?.impressions ?? 0;
  }

  /**
   * Get audience saturation level (for diagnostics).
   */
  getSaturationLevel(ageGroup: string): number {
    return this.audienceImpressions.get(ageGroup) ?? 0;
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
