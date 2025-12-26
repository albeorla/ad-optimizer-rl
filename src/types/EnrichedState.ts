/**
 * Enriched State Space for CMDP-Based Bidding.
 *
 * Standard RL implementations suffer from an impoverished state space that
 * ignores critical campaign context. Research shows that the optimal bid is
 * a function of BOTH the impression's intrinsic value AND the advertiser's
 * remaining resources.
 *
 * This module extends AdEnvironmentState with:
 * 1. Budgetary Context: Remaining budget, time, budget utilization
 * 2. Temporal Features: Time step, day/week fraction, diurnal patterns
 * 3. Competitive Landscape: Win rate, clearing prices, competition intensity
 * 4. Performance Metrics: Current vs target KPIs
 *
 * The enriched state enables the agent to learn:
 * - "End-of-day" behaviors (aggressive bidding if under-utilized)
 * - Pacing strategies (conservative if burn rate too high)
 * - Competition-aware bid shading
 */

import { AdEnvironmentState, RewardMetrics } from '../types';

/**
 * Budgetary context for campaign pacing.
 */
export interface BudgetaryContext {
  /** Total campaign budget (e.g., daily budget) */
  totalBudget: number;
  /** Remaining budget for the current period */
  remainingBudget: number;
  /** Budget utilization ratio (spent / total) */
  utilizationRatio: number;
  /** Time remaining in the campaign period (0-1) */
  timeRemainingRatio: number;
  /** Current spend rate ($ per hour) */
  currentSpendRate: number;
  /** Target spend rate for even pacing */
  targetSpendRate: number;
  /** Spend rate error (current - target) / target */
  spendRateError: number;
  /** Projected end-of-period spend based on current rate */
  projectedSpend: number;
  /** Whether on track for budget utilization (within 10%) */
  isOnTrack: boolean;
}

/**
 * Temporal context for time-aware bidding.
 */
export interface TemporalContext {
  /** Current time step (e.g., hour of day: 0-23) */
  currentStep: number;
  /** Total time steps in the period (e.g., 24 for daily) */
  totalSteps: number;
  /** Fraction of period elapsed (0-1) */
  periodFraction: number;
  /** Day of week (0-6) */
  dayOfWeek: number;
  /** Hour of day (0-23) */
  hourOfDay: number;
  /** Is peak hours (typically 18-22 for social platforms) */
  isPeakHours: boolean;
  /** Is low-engagement hours (typically 2-6) */
  isLowHours: boolean;
  /** Weekend flag */
  isWeekend: boolean;
  /** Cyclical encoding of hour (sin, cos) for smooth transitions */
  hourCyclical: [number, number];
  /** Cyclical encoding of day (sin, cos) */
  dayCyclical: [number, number];
}

/**
 * Competitive landscape context.
 */
export interface CompetitiveContext {
  /** Trailing win rate over last N auctions */
  winRate: number;
  /** Average clearing price (market price) */
  avgClearingPrice: number;
  /** Estimated bid landscape percentile (0-1) */
  bidPercentile: number;
  /** Competition intensity (0-1, higher = more competitive) */
  competitionIntensity: number;
  /** Trend in competition (positive = increasing) */
  competitionTrend: number;
  /** Market price volatility (standard deviation) */
  priceVolatility: number;
  /** Recommended bid shading factor based on competition */
  bidShadingFactor: number;
}

/**
 * Current performance vs KPI targets.
 */
export interface PerformanceContext {
  /** Current CPA vs target CPA ratio */
  cpaRatio: number;
  /** Current ROAS vs target ROAS ratio */
  roasRatio: number;
  /** Current CVR */
  currentCVR: number;
  /** Current CTR */
  currentCTR: number;
  /** Conversions in current period */
  periodConversions: number;
  /** Revenue in current period */
  periodRevenue: number;
  /** Hours since last conversion */
  hoursSinceLastConversion: number;
  /** Performance trend (positive = improving) */
  performanceTrend: number;
}

/**
 * Enriched state combining all context types.
 */
export interface EnrichedAdState extends AdEnvironmentState {
  /** Budgetary context */
  budget: BudgetaryContext;
  /** Temporal context */
  temporal: TemporalContext;
  /** Competitive context */
  competitive: CompetitiveContext;
  /** Performance context */
  performance: PerformanceContext;
}

/**
 * Configuration for building enriched state.
 */
export interface StateEnrichmentConfig {
  /** Daily budget target */
  dailyBudget: number;
  /** Target CPA for the campaign */
  targetCPA: number;
  /** Target ROAS for the campaign */
  targetROAS: number;
  /** Peak hours definition */
  peakHours: [number, number];
  /** Low engagement hours definition */
  lowHours: [number, number];
  /** Window size for trailing metrics (in time steps) */
  trailingWindow: number;
}

const DEFAULT_ENRICHMENT_CONFIG: StateEnrichmentConfig = {
  dailyBudget: 30,
  targetCPA: 15,
  targetROAS: 2.0,
  peakHours: [18, 22],
  lowHours: [2, 6],
  trailingWindow: 24,
};

/**
 * State Enrichment Engine.
 *
 * Maintains running context and produces enriched states from raw observations.
 */
export class StateEnrichmentEngine {
  private config: StateEnrichmentConfig;

  // Running metrics
  private totalSpend = 0;
  private totalRevenue = 0;
  private totalConversions = 0;
  private periodStartTime = 0;

  // Trailing metrics for competition estimation
  private recentBids: Array<{ bid: number; won: boolean; clearingPrice: number }> = [];
  private recentConversions: number[] = []; // timestamps
  private recentPerformance: number[] = []; // ROAS values

  // Current step tracking
  private currentStep = 0;
  private lastConversionTime = 0;

  constructor(config: Partial<StateEnrichmentConfig> = {}) {
    this.config = { ...DEFAULT_ENRICHMENT_CONFIG, ...config };
    this.periodStartTime = Date.now();
  }

  /**
   * Create enriched state from base state and current context.
   */
  enrich(baseState: AdEnvironmentState): EnrichedAdState {
    const budget = this.computeBudgetaryContext();
    const temporal = this.computeTemporalContext(baseState);
    const competitive = this.computeCompetitiveContext();
    const performance = this.computePerformanceContext();

    return {
      ...baseState,
      budget,
      temporal,
      competitive,
      performance,
    };
  }

  /**
   * Update engine with new observation.
   */
  recordObservation(obs: {
    spend?: number;
    revenue?: number;
    conversion?: boolean;
    bid?: number;
    won?: boolean;
    clearingPrice?: number;
  }): void {
    const now = Date.now();

    if (obs.spend) {
      this.totalSpend += obs.spend;
    }

    if (obs.revenue) {
      this.totalRevenue += obs.revenue;
    }

    if (obs.conversion) {
      this.totalConversions++;
      this.lastConversionTime = now;
      this.recentConversions.push(now);
    }

    if (obs.bid !== undefined && obs.won !== undefined) {
      this.recentBids.push({
        bid: obs.bid,
        won: obs.won,
        clearingPrice: obs.clearingPrice ?? obs.bid,
      });
      // Keep trailing window
      if (this.recentBids.length > this.config.trailingWindow * 10) {
        this.recentBids = this.recentBids.slice(-this.config.trailingWindow * 10);
      }
    }

    // Track ROAS for performance trend
    if (this.totalSpend > 0) {
      this.recentPerformance.push(this.totalRevenue / this.totalSpend);
      if (this.recentPerformance.length > this.config.trailingWindow) {
        this.recentPerformance.shift();
      }
    }

    this.cleanupOldData(now);
  }

  /**
   * Advance to next time step.
   */
  advanceStep(): void {
    this.currentStep++;
  }

  /**
   * Reset for new period (e.g., new day).
   */
  resetPeriod(): void {
    this.totalSpend = 0;
    this.totalRevenue = 0;
    this.totalConversions = 0;
    this.currentStep = 0;
    this.periodStartTime = Date.now();
    this.recentBids = [];
    this.recentConversions = [];
    this.recentPerformance = [];
  }

  private computeBudgetaryContext(): BudgetaryContext {
    const remainingBudget = Math.max(0, this.config.dailyBudget - this.totalSpend);
    const utilizationRatio = this.config.dailyBudget > 0
      ? this.totalSpend / this.config.dailyBudget
      : 0;

    const totalSteps = 24; // Daily period
    const timeRemainingRatio = Math.max(0, (totalSteps - this.currentStep) / totalSteps);

    // Calculate spend rate
    const hoursElapsed = Math.max(1, this.currentStep);
    const currentSpendRate = this.totalSpend / hoursElapsed;
    const targetSpendRate = this.config.dailyBudget / totalSteps;
    const spendRateError = targetSpendRate > 0
      ? (currentSpendRate - targetSpendRate) / targetSpendRate
      : 0;

    // Project end-of-period spend
    const projectedSpend = currentSpendRate * totalSteps;

    // On track if within 10% of target trajectory
    const expectedSpend = (this.currentStep / totalSteps) * this.config.dailyBudget;
    const isOnTrack = Math.abs(this.totalSpend - expectedSpend) / this.config.dailyBudget < 0.1;

    return {
      totalBudget: this.config.dailyBudget,
      remainingBudget,
      utilizationRatio,
      timeRemainingRatio,
      currentSpendRate,
      targetSpendRate,
      spendRateError,
      projectedSpend,
      isOnTrack,
    };
  }

  private computeTemporalContext(baseState: AdEnvironmentState): TemporalContext {
    const { hourOfDay, dayOfWeek } = baseState;
    const totalSteps = 24;
    const periodFraction = this.currentStep / totalSteps;

    const [peakStart, peakEnd] = this.config.peakHours;
    const [lowStart, lowEnd] = this.config.lowHours;

    const isPeakHours = hourOfDay >= peakStart && hourOfDay <= peakEnd;
    const isLowHours = hourOfDay >= lowStart && hourOfDay <= lowEnd;
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;

    // Cyclical encodings
    const hourAngle = (2 * Math.PI * hourOfDay) / 24;
    const dayAngle = (2 * Math.PI * dayOfWeek) / 7;

    return {
      currentStep: this.currentStep,
      totalSteps,
      periodFraction,
      dayOfWeek,
      hourOfDay,
      isPeakHours,
      isLowHours,
      isWeekend,
      hourCyclical: [Math.sin(hourAngle), Math.cos(hourAngle)],
      dayCyclical: [Math.sin(dayAngle), Math.cos(dayAngle)],
    };
  }

  private computeCompetitiveContext(): CompetitiveContext {
    // Win rate from recent auctions
    const recentAuctions = this.recentBids.slice(-100);
    const winRate = recentAuctions.length > 0
      ? recentAuctions.filter((a) => a.won).length / recentAuctions.length
      : 0.1; // Default assumption

    // Average clearing price
    const clearingPrices = recentAuctions.filter((a) => a.won).map((a) => a.clearingPrice);
    const avgClearingPrice = clearingPrices.length > 0
      ? clearingPrices.reduce((a, b) => a + b, 0) / clearingPrices.length
      : 0.5;

    // Price volatility
    const priceVolatility = clearingPrices.length > 1
      ? this.standardDeviation(clearingPrices)
      : 0.1;

    // Bid percentile (rough estimate based on win rate)
    // Higher win rate suggests bidding in higher percentile
    const bidPercentile = Math.min(1, winRate * 2);

    // Competition intensity inversely related to win rate
    const competitionIntensity = 1 - winRate;

    // Competition trend (comparing first half vs second half)
    const halfPoint = Math.floor(recentAuctions.length / 2);
    const firstHalfWinRate = this.calculateWinRate(recentAuctions.slice(0, halfPoint));
    const secondHalfWinRate = this.calculateWinRate(recentAuctions.slice(halfPoint));
    const competitionTrend = firstHalfWinRate - secondHalfWinRate; // Negative = increasing competition

    // Bid shading factor (reduce bid when winning too much)
    let bidShadingFactor = 1.0;
    if (winRate > 0.3) {
      bidShadingFactor = 0.85; // Shade bids down
    } else if (winRate < 0.05) {
      bidShadingFactor = 1.15; // Bid more aggressively
    }

    return {
      winRate,
      avgClearingPrice,
      bidPercentile,
      competitionIntensity,
      competitionTrend,
      priceVolatility,
      bidShadingFactor,
    };
  }

  private computePerformanceContext(): PerformanceContext {
    const currentCPA = this.totalConversions > 0
      ? this.totalSpend / this.totalConversions
      : Infinity;
    const cpaRatio = this.config.targetCPA > 0 && isFinite(currentCPA)
      ? currentCPA / this.config.targetCPA
      : 1;

    const currentROAS = this.totalSpend > 0
      ? this.totalRevenue / this.totalSpend
      : 0;
    const roasRatio = this.config.targetROAS > 0
      ? currentROAS / this.config.targetROAS
      : 1;

    // CVR and CTR would require click data - using defaults
    const currentCVR = 0.03; // Would be computed from click/conversion data
    const currentCTR = 0.02; // Would be computed from impression/click data

    // Hours since last conversion
    const now = Date.now();
    const hoursSinceLastConversion = this.lastConversionTime > 0
      ? (now - this.lastConversionTime) / (60 * 60 * 1000)
      : this.currentStep;

    // Performance trend from ROAS history
    let performanceTrend = 0;
    if (this.recentPerformance.length >= 2) {
      const halfPoint = Math.floor(this.recentPerformance.length / 2);
      const firstHalf = this.recentPerformance.slice(0, halfPoint);
      const secondHalf = this.recentPerformance.slice(halfPoint);
      const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
      const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
      performanceTrend = (secondAvg - firstAvg) / Math.max(0.01, firstAvg);
    }

    return {
      cpaRatio,
      roasRatio,
      currentCVR,
      currentCTR,
      periodConversions: this.totalConversions,
      periodRevenue: this.totalRevenue,
      hoursSinceLastConversion,
      performanceTrend,
    };
  }

  private calculateWinRate(auctions: Array<{ won: boolean }>): number {
    if (auctions.length === 0) return 0;
    return auctions.filter((a) => a.won).length / auctions.length;
  }

  private standardDeviation(values: number[]): number {
    if (values.length < 2) return 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map((v) => Math.pow(v - mean, 2));
    return Math.sqrt(squaredDiffs.reduce((a, b) => a + b, 0) / (values.length - 1));
  }

  private cleanupOldData(now: number): void {
    const windowMs = this.config.trailingWindow * 60 * 60 * 1000;
    this.recentConversions = this.recentConversions.filter(
      (t) => now - t < windowMs
    );
  }

  // Getters for diagnostics
  getTotalSpend(): number { return this.totalSpend; }
  getTotalRevenue(): number { return this.totalRevenue; }
  getTotalConversions(): number { return this.totalConversions; }
  getCurrentStep(): number { return this.currentStep; }
}

/**
 * Encode enriched state to feature vector.
 *
 * Extends the base encoding with budgetary, temporal, competitive,
 * and performance features.
 */
export function encodeEnrichedState(state: EnrichedAdState): number[] {
  // Base encoding (from existing encodeState)
  const base = encodeBaseState(state);

  // Budgetary features (7 features)
  const budgetFeatures = [
    state.budget.utilizationRatio,
    state.budget.timeRemainingRatio,
    state.budget.spendRateError,
    state.budget.remainingBudget / Math.max(1, state.budget.totalBudget),
    state.budget.projectedSpend / Math.max(1, state.budget.totalBudget),
    state.budget.isOnTrack ? 1 : 0,
    state.budget.currentSpendRate / Math.max(0.01, state.budget.targetSpendRate),
  ];

  // Temporal features (6 features, using cyclical already in base)
  const temporalFeatures = [
    state.temporal.periodFraction,
    state.temporal.isPeakHours ? 1 : 0,
    state.temporal.isLowHours ? 1 : 0,
    state.temporal.isWeekend ? 1 : 0,
    // Emergency end-of-day signal (last 2 hours)
    state.temporal.currentStep >= 22 ? 1 : 0,
    // Mid-day stability zone
    state.temporal.hourOfDay >= 10 && state.temporal.hourOfDay <= 18 ? 1 : 0,
  ];

  // Competitive features (6 features)
  const competitiveFeatures = [
    state.competitive.winRate,
    state.competitive.competitionIntensity,
    state.competitive.bidPercentile,
    Math.tanh(state.competitive.competitionTrend), // Bounded
    state.competitive.priceVolatility,
    state.competitive.bidShadingFactor - 1, // Center around 0
  ];

  // Performance features (6 features)
  const performanceFeatures = [
    Math.min(2, state.performance.cpaRatio) / 2, // Normalize to 0-1
    Math.min(2, state.performance.roasRatio) / 2,
    state.performance.currentCVR * 10, // Scale up
    state.performance.currentCTR * 10,
    Math.min(1, state.performance.hoursSinceLastConversion / 6), // Cap at 6h
    Math.tanh(state.performance.performanceTrend), // Bounded
  ];

  return [
    ...base,
    ...budgetFeatures,
    ...temporalFeatures,
    ...competitiveFeatures,
    ...performanceFeatures,
  ];
}

/**
 * Get the dimension of the enriched state encoding.
 */
export function getEnrichedStateDimension(): number {
  // Base: 28 (from existing encoding)
  // Budget: 7
  // Temporal: 6
  // Competitive: 6
  // Performance: 6
  return 28 + 7 + 6 + 6 + 6; // = 53
}

// Helper to encode base state (mirrors existing encodeState)
function encodeBaseState(state: AdEnvironmentState): number[] {
  const AGE_GROUPS = ['18-24', '25-34', '35-44', '45+'];
  const CREATIVE_TYPES = ['lifestyle', 'product', 'discount', 'ugc'];
  const PLATFORMS = ['tiktok', 'instagram'];
  const INTEREST_VOCAB = ['fashion', 'sports', 'music', 'tech', 'fitness', 'art', 'travel'];

  const cyclicalPair = (value: number, period: number): [number, number] => {
    const angle = (2 * Math.PI * (value % period)) / period;
    return [Math.sin(angle), Math.cos(angle)];
  };

  const oneHot = (cats: string[], v: string): number[] => {
    return cats.map((c) => (c === v ? 1 : 0));
  };

  const multiHot = (vocab: string[], values: string[]): number[] => {
    const set = new Set(values);
    return vocab.map((t) => (set.has(t) ? 1 : 0));
  };

  const clamp = (x: number, min: number, max: number) => Math.max(min, Math.min(max, x));

  const [sinH, cosH] = cyclicalPair(state.hourOfDay, 24);
  const [sinD, cosD] = cyclicalPair(state.dayOfWeek, 7);

  const budget = [clamp(state.currentBudget / 100, 0, 2.0)];
  const age = oneHot(AGE_GROUPS, state.targetAgeGroup);
  const creative = oneHot(CREATIVE_TYPES, state.creativeType);
  const platform = oneHot(PLATFORMS, state.platform);
  const interests = multiHot(INTEREST_VOCAB, state.targetInterests);
  const hist = [
    clamp(state.historicalCTR, 0, 1),
    clamp(state.historicalCVR, 0, 1),
    clamp(state.competitorActivity, 0, 1),
    clamp(state.seasonality, 0, 1),
  ];

  return [
    sinH, cosH, sinD, cosD,
    ...budget,
    ...age,
    ...creative,
    ...platform,
    ...interests,
    ...hist,
  ];
}
