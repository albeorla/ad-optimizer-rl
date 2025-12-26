/**
 * Delayed Feedback Handling for E-commerce Attribution.
 *
 * In digital advertising, conversions often occur hours or days after the initial
 * click. A user might click an ad at 9 AM but not complete a purchase until 8 PM.
 * Standard RL assumes immediate rewards, which leads to "false negative" training
 * where good actions appear to have zero reward because the conversion hasn't
 * happened yet.
 *
 * This module implements:
 * - AttributionBuffer: Holds pending experiences until attribution resolves
 * - GDFM (Generalized Delayed Feedback Model): Probabilistic delay modeling
 * - Importance Sampling: Corrects for policy drift in delayed samples
 *
 * Reference: Research shows 50%+ of conversions can occur >24h after click.
 */

import { AdAction, AdEnvironmentState } from '../types';

/**
 * A pending experience waiting for attribution resolution.
 */
export interface PendingExperience {
  /** Unique identifier for this experience */
  id: string;
  /** State when action was taken */
  state: AdEnvironmentState;
  /** Action taken by the agent */
  action: AdAction;
  /** Timestamp when the action (click/impression) occurred */
  actionTimestamp: number;
  /** Attribution identifier (e.g., ttclid, gclid, fbclid) */
  attributionId: string;
  /** Policy probability of this action at time of execution */
  actionProbability: number;
  /** Episode/batch ID for tracking */
  episodeId?: number;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

/**
 * A resolved experience ready for training.
 */
export interface ResolvedExperience {
  /** Original pending experience */
  pending: PendingExperience;
  /** Final reward (0 for no conversion, revenue for conversion) */
  reward: number;
  /** Whether a conversion occurred */
  converted: boolean;
  /** Conversion timestamp (if converted) */
  conversionTimestamp?: number;
  /** Observed delay in milliseconds (if converted) */
  delayMs?: number;
  /** Importance weight for off-policy correction */
  importanceWeight: number;
  /** Resolution timestamp */
  resolvedAt: number;
  /** Next state (can be synthetic for terminal states) */
  nextState: AdEnvironmentState;
  /** Whether this is a terminal state */
  done: boolean;
}

/**
 * Configuration for the Attribution Buffer.
 */
export interface AttributionBufferConfig {
  /** Maximum time to wait for attribution (default: 24 hours) */
  attributionWindowMs: number;
  /** Maximum number of pending experiences to hold */
  maxPendingSize: number;
  /** Maximum number of resolved experiences to hold */
  maxResolvedSize: number;
  /** Whether to use importance sampling for policy drift correction */
  useImportanceSampling: boolean;
  /** Maximum importance weight to prevent variance explosion */
  maxImportanceWeight: number;
  /** Minimum importance weight to prevent near-zero weights */
  minImportanceWeight: number;
}

const DEFAULT_CONFIG: AttributionBufferConfig = {
  attributionWindowMs: 24 * 60 * 60 * 1000, // 24 hours
  maxPendingSize: 10000,
  maxResolvedSize: 50000,
  useImportanceSampling: true,
  maxImportanceWeight: 10.0,
  minImportanceWeight: 0.1,
};

/**
 * Attribution Buffer: Manages delayed reward attribution.
 *
 * Flow:
 * 1. Agent takes action → experience goes to "pending" buffer
 * 2. Conversion monitor polls for attribution events
 * 3. When conversion matches attribution ID → move to "resolved" with reward=revenue
 * 4. When attribution window expires with no match → move to "resolved" with reward=0
 * 5. Training samples from "resolved" buffer with importance weighting
 */
export class AttributionBuffer {
  private pending: Map<string, PendingExperience> = new Map();
  private resolved: ResolvedExperience[] = [];
  private config: AttributionBufferConfig;
  private currentPolicyProbabilities: Map<string, number> = new Map();

  constructor(config: Partial<AttributionBufferConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Add a new experience to the pending buffer.
   *
   * @param experience - The experience to track
   */
  addPending(experience: Omit<PendingExperience, 'id'>): string {
    // Generate unique ID
    const id = `exp_${experience.actionTimestamp}_${experience.attributionId}_${Math.random().toString(36).slice(2, 8)}`;

    const pending: PendingExperience = {
      id,
      ...experience,
    };

    // Evict oldest if at capacity
    if (this.pending.size >= this.config.maxPendingSize) {
      const oldestKey = this.pending.keys().next().value;
      if (oldestKey) {
        this.expireExperience(oldestKey, Date.now());
      }
    }

    this.pending.set(id, pending);
    return id;
  }

  /**
   * Record a conversion event and resolve matching experience.
   *
   * @param attributionId - The attribution identifier to match
   * @param conversionData - Conversion details
   */
  recordConversion(
    attributionId: string,
    conversionData: {
      revenue: number;
      conversionTimestamp: number;
      nextState: AdEnvironmentState;
      done?: boolean;
    }
  ): boolean {
    // Find matching pending experience
    let matchedExperience: PendingExperience | undefined;
    let matchedId: string | undefined;

    for (const [id, exp] of this.pending) {
      if (exp.attributionId === attributionId) {
        matchedExperience = exp;
        matchedId = id;
        break;
      }
    }

    if (!matchedExperience || !matchedId) {
      return false;
    }

    const delayMs = conversionData.conversionTimestamp - matchedExperience.actionTimestamp;

    // Check if within attribution window
    if (delayMs > this.config.attributionWindowMs) {
      // Conversion arrived too late - treat as no conversion
      this.expireExperience(matchedId, Date.now());
      return false;
    }

    // Calculate importance weight for off-policy correction
    const importanceWeight = this.calculateImportanceWeight(matchedExperience);

    const resolved: ResolvedExperience = {
      pending: matchedExperience,
      reward: conversionData.revenue,
      converted: true,
      conversionTimestamp: conversionData.conversionTimestamp,
      delayMs,
      importanceWeight,
      resolvedAt: Date.now(),
      nextState: conversionData.nextState,
      done: conversionData.done ?? false,
    };

    this.addResolved(resolved);
    this.pending.delete(matchedId);
    return true;
  }

  /**
   * Process expired pending experiences (no conversion within window).
   *
   * @param currentTime - Current timestamp for expiration check
   * @returns Number of experiences expired
   */
  processExpiredExperiences(currentTime: number): number {
    let expiredCount = 0;

    for (const [id, exp] of this.pending) {
      const elapsed = currentTime - exp.actionTimestamp;
      if (elapsed > this.config.attributionWindowMs) {
        this.expireExperience(id, currentTime);
        expiredCount++;
      }
    }

    return expiredCount;
  }

  /**
   * Expire a single experience (no conversion).
   */
  private expireExperience(id: string, currentTime: number): void {
    const exp = this.pending.get(id);
    if (!exp) return;

    const importanceWeight = this.calculateImportanceWeight(exp);

    // Create synthetic next state (same as current for no-conversion)
    const nextState = { ...exp.state };

    const resolved: ResolvedExperience = {
      pending: exp,
      reward: 0,
      converted: false,
      importanceWeight,
      resolvedAt: currentTime,
      nextState,
      done: true, // No conversion is terminal for this impression
    };

    this.addResolved(resolved);
    this.pending.delete(id);
  }

  /**
   * Add a resolved experience to the training buffer.
   */
  private addResolved(experience: ResolvedExperience): void {
    // Evict oldest if at capacity
    while (this.resolved.length >= this.config.maxResolvedSize) {
      this.resolved.shift();
    }
    this.resolved.push(experience);
  }

  /**
   * Calculate importance weight for off-policy correction.
   *
   * When a reward arrives after significant delay, the current policy π_curr
   * may differ from the policy π_old that generated the action. We correct
   * for this distribution shift using:
   *
   * w = π_curr(a|s) / π_old(a|s)
   *
   * This ensures unbiased value function updates even with stale data.
   */
  private calculateImportanceWeight(experience: PendingExperience): number {
    if (!this.config.useImportanceSampling) {
      return 1.0;
    }

    const stateKey = this.getStateKey(experience.state);
    const actionKey = this.getActionKey(experience.action);
    const policyKey = `${stateKey}_${actionKey}`;

    // Get current policy probability (if tracking is enabled)
    const currentProb = this.currentPolicyProbabilities.get(policyKey) ?? experience.actionProbability;
    const oldProb = experience.actionProbability;

    // Avoid division by zero
    if (oldProb === 0) {
      return this.config.minImportanceWeight;
    }

    const weight = currentProb / oldProb;

    // Clamp to prevent variance explosion
    return Math.max(
      this.config.minImportanceWeight,
      Math.min(this.config.maxImportanceWeight, weight)
    );
  }

  /**
   * Update current policy probabilities for importance sampling.
   * Call this after each policy update during training.
   */
  updatePolicyProbabilities(
    probabilities: Array<{ state: AdEnvironmentState; action: AdAction; probability: number }>
  ): void {
    for (const { state, action, probability } of probabilities) {
      const stateKey = this.getStateKey(state);
      const actionKey = this.getActionKey(action);
      const key = `${stateKey}_${actionKey}`;
      this.currentPolicyProbabilities.set(key, probability);
    }
  }

  /**
   * Sample a batch of resolved experiences for training.
   *
   * @param batchSize - Number of experiences to sample
   * @param weighted - Whether to apply importance weights in sampling
   */
  sampleBatch(batchSize: number, weighted = false): ResolvedExperience[] {
    if (this.resolved.length === 0) {
      return [];
    }

    const actualSize = Math.min(batchSize, this.resolved.length);
    const sampled: ResolvedExperience[] = [];

    if (weighted) {
      // Priority sampling based on importance weights
      const weights = this.resolved.map((e) => e.importanceWeight);
      const totalWeight = weights.reduce((a, b) => a + b, 0);

      const usedIndices = new Set<number>();
      while (sampled.length < actualSize && usedIndices.size < this.resolved.length) {
        const r = Math.random() * totalWeight;
        let cumulative = 0;
        for (let i = 0; i < this.resolved.length; i++) {
          if (usedIndices.has(i)) continue;
          cumulative += weights[i]!;
          if (r <= cumulative) {
            sampled.push(this.resolved[i]!);
            usedIndices.add(i);
            break;
          }
        }
      }
    } else {
      // Uniform random sampling
      const indices = new Set<number>();
      while (indices.size < actualSize) {
        indices.add(Math.floor(Math.random() * this.resolved.length));
      }
      for (const i of indices) {
        sampled.push(this.resolved[i]!);
      }
    }

    return sampled;
  }

  /**
   * Get statistics about the buffer state.
   */
  getStats(): {
    pendingCount: number;
    resolvedCount: number;
    conversionRate: number;
    avgDelayMs: number;
    avgImportanceWeight: number;
  } {
    const conversions = this.resolved.filter((e) => e.converted);
    const avgDelayMs =
      conversions.length > 0
        ? conversions.reduce((sum, e) => sum + (e.delayMs ?? 0), 0) / conversions.length
        : 0;

    const avgImportanceWeight =
      this.resolved.length > 0
        ? this.resolved.reduce((sum, e) => sum + e.importanceWeight, 0) / this.resolved.length
        : 1.0;

    return {
      pendingCount: this.pending.size,
      resolvedCount: this.resolved.length,
      conversionRate: this.resolved.length > 0 ? conversions.length / this.resolved.length : 0,
      avgDelayMs,
      avgImportanceWeight,
    };
  }

  /**
   * Generate state key for probability tracking.
   */
  private getStateKey(state: AdEnvironmentState): string {
    return `${state.dayOfWeek}_${state.hourOfDay}_${state.targetAgeGroup}_${state.platform}`;
  }

  /**
   * Generate action key for probability tracking.
   */
  private getActionKey(action: AdAction): string {
    return `${action.bidStrategy}_${action.targetAgeGroup}_${action.platform}`;
  }

  /**
   * Clear all buffers.
   */
  clear(): void {
    this.pending.clear();
    this.resolved = [];
    this.currentPolicyProbabilities.clear();
  }

  /**
   * Get all pending experiences (for debugging).
   */
  getPending(): PendingExperience[] {
    return Array.from(this.pending.values());
  }

  /**
   * Get all resolved experiences (for debugging).
   */
  getResolved(): ResolvedExperience[] {
    return [...this.resolved];
  }
}

/**
 * Generalized Delayed Feedback Model (GDFM).
 *
 * Models the joint probability of conversion and delay:
 * P(Y=1, D=d) = P(Y=1) * P(D=d | Y=1)
 *
 * Where:
 * - Y ∈ {0,1} is the conversion indicator
 * - D is the delay (time between click and conversion)
 * - P(Y=1) is modeled by any conversion prediction model
 * - P(D=d|Y=1) is often modeled as exponential or Weibull distribution
 *
 * This model allows us to:
 * 1. Estimate true conversion probability even with censored data
 * 2. Predict expected delay for better pacing
 * 3. Weight samples appropriately during training
 */
export class DelayedFeedbackModel {
  /** Learned delay rate parameter (exponential distribution) */
  private delayRate: number = 0.0001; // Default: ~2.7 hour mean delay
  /** Minimum observations before updating parameters */
  private minObservations = 100;
  /** Collected delay observations for parameter estimation */
  private delayObservations: number[] = [];
  /** Running sum for online mean estimation */
  private delaySum = 0;
  private delayCount = 0;

  /**
   * Record an observed delay for parameter estimation.
   *
   * @param delayMs - Observed delay in milliseconds
   */
  recordDelay(delayMs: number): void {
    this.delayObservations.push(delayMs);
    this.delaySum += delayMs;
    this.delayCount++;

    // Update rate parameter periodically
    if (this.delayObservations.length >= this.minObservations) {
      this.updateDelayRate();
    }
  }

  /**
   * Update the delay rate parameter using MLE for exponential distribution.
   */
  private updateDelayRate(): void {
    if (this.delayCount === 0) return;
    const meanDelay = this.delaySum / this.delayCount;
    this.delayRate = 1 / meanDelay;
  }

  /**
   * Compute the survival probability: P(D > elapsed | Y=1).
   *
   * This is the probability that a conversion will still occur
   * given that time `elapsed` has passed without one.
   *
   * @param elapsedMs - Time elapsed since the action
   */
  survivalProbability(elapsedMs: number): number {
    // For exponential distribution: S(t) = exp(-λt)
    return Math.exp(-this.delayRate * elapsedMs);
  }

  /**
   * Compute the hazard rate: instantaneous conversion probability.
   *
   * h(t) = f(t) / S(t) = λ for exponential distribution
   *
   * This is useful for real-time conversion probability estimation.
   */
  hazardRate(): number {
    return this.delayRate;
  }

  /**
   * Compute the expected remaining delay given elapsed time.
   *
   * For exponential distribution (memoryless property):
   * E[D - t | D > t] = 1/λ = E[D]
   */
  expectedRemainingDelay(): number {
    return 1 / this.delayRate;
  }

  /**
   * Compute the loss contribution for a sample (for training).
   *
   * For censored data (no conversion yet, elapsed time = e):
   * L = log(1 - P(Y=1)) + log(P(D > e | Y=1)) * P(Y=1)
   *
   * For observed conversions (delay = d):
   * L = log(P(Y=1)) + log(P(D=d | Y=1))
   *
   * @param conversionProb - Predicted P(Y=1) from your model
   * @param elapsedMs - Time elapsed since action
   * @param converted - Whether conversion occurred
   * @param observedDelayMs - Observed delay if converted
   */
  computeLossContribution(
    conversionProb: number,
    elapsedMs: number,
    converted: boolean,
    observedDelayMs?: number
  ): number {
    const epsilon = 1e-10; // For numerical stability

    if (converted && observedDelayMs !== undefined) {
      // Observed conversion: log-likelihood of conversion and delay
      const logConvProb = Math.log(Math.max(conversionProb, epsilon));
      const logDelayProb = Math.log(this.delayRate) - this.delayRate * observedDelayMs;
      return -(logConvProb + logDelayProb);
    } else {
      // Censored: probability of no conversion OR delayed conversion
      const survivalProb = this.survivalProbability(elapsedMs);
      // P(no conversion OR delay > e) = P(Y=0) + P(Y=1) * P(D > e | Y=1)
      const censoredProb = (1 - conversionProb) + conversionProb * survivalProb;
      return -Math.log(Math.max(censoredProb, epsilon));
    }
  }

  /**
   * Estimate the true conversion probability accounting for censoring.
   *
   * Given observed no-conversion after elapsed time, compute:
   * P(eventually converts | no conversion yet, elapsed time)
   *
   * Uses Bayes' theorem:
   * P(Y=1 | D>e) = P(D>e|Y=1) * P(Y=1) / P(D>e)
   *
   * @param priorConversionProb - Prior P(Y=1) before observing the delay
   * @param elapsedMs - Time elapsed without conversion
   */
  posteriorConversionProbability(priorConversionProb: number, elapsedMs: number): number {
    const survivalProb = this.survivalProbability(elapsedMs);

    // P(D > e) = P(Y=0) + P(Y=1) * P(D > e | Y=1)
    const pNoConversionYet =
      (1 - priorConversionProb) + priorConversionProb * survivalProb;

    // P(Y=1 | D > e) = P(D > e | Y=1) * P(Y=1) / P(D > e)
    if (pNoConversionYet === 0) return 0;
    return (survivalProb * priorConversionProb) / pNoConversionYet;
  }

  /**
   * Get current model statistics.
   */
  getStats(): {
    delayRate: number;
    meanDelayMs: number;
    observationCount: number;
  } {
    return {
      delayRate: this.delayRate,
      meanDelayMs: this.delayCount > 0 ? this.delaySum / this.delayCount : 1 / this.delayRate,
      observationCount: this.delayCount,
    };
  }

  /**
   * Reset the model.
   */
  reset(): void {
    this.delayRate = 0.0001;
    this.delayObservations = [];
    this.delaySum = 0;
    this.delayCount = 0;
  }

  /**
   * Set the delay rate manually (for bootstrapping or transfer).
   */
  setDelayRate(rate: number): void {
    this.delayRate = rate;
  }
}

/**
 * Attribution Reconciliation Service.
 *
 * Reconciles conversion data from multiple sources:
 * - Platform attribution (TikTok, Meta, Google)
 * - Server-side tracking (Shopify CAPI)
 * - First-party data (click ID matching)
 *
 * This is critical because:
 * 1. Client-side pixels are unreliable (ITP, ad blockers)
 * 2. Different platforms may claim the same conversion
 * 3. Attribution windows vary by platform
 */
export interface ConversionEvent {
  /** Attribution ID (ttclid, gclid, fbclid, etc.) */
  attributionId: string;
  /** Revenue from the conversion */
  revenue: number;
  /** Conversion timestamp */
  timestamp: number;
  /** Source of the conversion data */
  source: 'platform' | 'server' | 'first_party';
  /** Platform that claimed the conversion */
  platform?: 'tiktok' | 'instagram' | 'google' | 'meta' | 'shopify';
  /** Order ID for deduplication */
  orderId?: string;
  /** Customer ID for cross-device matching */
  customerId?: string;
  /** Additional metadata */
  metadata?: Record<string, unknown>;
}

export class AttributionReconciler {
  private processedOrders: Set<string> = new Set();
  private attributionBuffer: AttributionBuffer;
  private delayModel: DelayedFeedbackModel;

  constructor(
    attributionBuffer: AttributionBuffer,
    delayModel: DelayedFeedbackModel
  ) {
    this.attributionBuffer = attributionBuffer;
    this.delayModel = delayModel;
  }

  /**
   * Process a conversion event from any source.
   *
   * Handles deduplication, attribution matching, and delay tracking.
   */
  processConversion(
    event: ConversionEvent,
    nextState: AdEnvironmentState
  ): { matched: boolean; deduplicated: boolean } {
    // Deduplicate by order ID
    if (event.orderId && this.processedOrders.has(event.orderId)) {
      return { matched: false, deduplicated: true };
    }
    if (event.orderId) {
      this.processedOrders.add(event.orderId);
    }

    // Try to match to pending experience
    const matched = this.attributionBuffer.recordConversion(event.attributionId, {
      revenue: event.revenue,
      conversionTimestamp: event.timestamp,
      nextState,
      done: true,
    });

    if (matched) {
      // Record delay for model training
      const pending = this.attributionBuffer.getPending().find(
        (p) => p.attributionId === event.attributionId
      );
      if (pending) {
        const delay = event.timestamp - pending.actionTimestamp;
        this.delayModel.recordDelay(delay);
      }
    }

    return { matched, deduplicated: false };
  }

  /**
   * Clear processed orders (e.g., at end of day).
   */
  clearProcessedOrders(): void {
    this.processedOrders.clear();
  }

  /**
   * Get reconciliation statistics.
   */
  getStats(): {
    processedOrderCount: number;
    bufferStats: ReturnType<AttributionBuffer['getStats']>;
    delayStats: ReturnType<DelayedFeedbackModel['getStats']>;
  } {
    return {
      processedOrderCount: this.processedOrders.size,
      bufferStats: this.attributionBuffer.getStats(),
      delayStats: this.delayModel.getStats(),
    };
  }
}
