/**
 * Offline Policy Evaluation (OPE) for Safe RL Deployment.
 *
 * Deploying RL agents directly into production is risky - bad policies waste
 * advertising budget. OPE allows us to estimate a new policy's performance
 * using historical data collected by a different (logging) policy, without
 * actually deploying it.
 *
 * This module implements:
 * - IPS (Inverse Propensity Scoring): Unbiased but high variance
 * - SNIPS (Self-Normalized IPS): Lower variance, slight bias
 * - CIPS (Clipped IPS): Variance control via weight clipping
 * - DR (Doubly Robust): Combines IPS with a value model for robustness
 * - MDA (Mean Directional Accuracy): Measures rank-order preservation
 *
 * Reference: In RTB, rank-ordering matters more than absolute values.
 * OPE should achieve MDA > 80% before deployment is approved.
 */

import { AdAction, AdEnvironmentState } from '../types';

/**
 * A logged experience from the historical policy.
 */
export interface LoggedExperience {
  /** State when action was taken */
  state: AdEnvironmentState;
  /** Action taken by the logging policy */
  action: AdAction;
  /** Observed reward */
  reward: number;
  /** Probability of this action under the logging policy: π_old(a|s) */
  loggingProbability: number;
  /** Optional: predicted reward from a value model (for DR estimator) */
  predictedReward?: number;
  /** Optional: next state for value function estimation */
  nextState?: AdEnvironmentState;
  /** Optional: whether episode terminated */
  done?: boolean;
}

/**
 * Policy interface for OPE evaluation.
 */
export interface EvaluationPolicy {
  /** Get the probability of an action under this policy: π_new(a|s) */
  getActionProbability(state: AdEnvironmentState, action: AdAction): number;
  /** Get the predicted value/reward for a state-action pair (for DR) */
  predictReward?(state: AdEnvironmentState, action: AdAction): number;
}

/**
 * Configuration for OPE estimators.
 */
export interface OPEConfig {
  /** Maximum importance weight for clipping (CIPS) */
  maxWeight: number;
  /** Minimum importance weight (prevents division issues) */
  minWeight: number;
  /** Minimum propensity to consider (filters rare actions) */
  minPropensity: number;
  /** Whether to use weighted sampling for variance reduction */
  useWeightedSampling: boolean;
  /** Confidence level for interval estimation */
  confidenceLevel: number;
}

const DEFAULT_OPE_CONFIG: OPEConfig = {
  maxWeight: 10.0,
  minWeight: 0.01,
  minPropensity: 0.001,
  useWeightedSampling: true,
  confidenceLevel: 0.95,
};

/**
 * OPE result containing estimate and diagnostics.
 */
export interface OPEResult {
  /** Point estimate of policy value */
  estimate: number;
  /** Standard error of the estimate */
  standardError: number;
  /** Lower bound of confidence interval */
  confidenceLower: number;
  /** Upper bound of confidence interval */
  confidenceUpper: number;
  /** Effective sample size (accounting for weighting) */
  effectiveSampleSize: number;
  /** Number of samples used */
  sampleCount: number;
  /** Number of samples filtered due to low propensity */
  filteredCount: number;
  /** Mean importance weight */
  meanWeight: number;
  /** Maximum importance weight observed */
  maxWeightObserved: number;
  /** Variance of importance weights (diagnostic for reliability) */
  weightVariance: number;
}

/**
 * Inverse Propensity Scoring (IPS) Estimator.
 *
 * The fundamental OPE estimator. Re-weights historical rewards based on how
 * likely the new policy would have been to take the same actions.
 *
 * V̂_IPS = (1/n) Σ [π_new(a|s) / π_old(a|s)] * r
 *
 * Properties:
 * - Unbiased (asymptotically)
 * - High variance when policies differ significantly
 * - Can explode when π_old(a|s) is very small
 */
export function calculateIPS(
  logs: LoggedExperience[],
  newPolicy: EvaluationPolicy,
  config: Partial<OPEConfig> = {}
): OPEResult {
  const cfg = { ...DEFAULT_OPE_CONFIG, ...config };
  const weights: number[] = [];
  const weightedRewards: number[] = [];
  let filteredCount = 0;

  for (const log of logs) {
    // Skip if logging probability too low (unreliable)
    if (log.loggingProbability < cfg.minPropensity) {
      filteredCount++;
      continue;
    }

    const newProb = newPolicy.getActionProbability(log.state, log.action);
    const weight = newProb / log.loggingProbability;

    weights.push(weight);
    weightedRewards.push(weight * log.reward);
  }

  if (weights.length === 0) {
    return createEmptyResult(0, filteredCount);
  }

  // IPS estimate: mean of weighted rewards
  const estimate = mean(weightedRewards);

  // Variance and confidence interval
  const variance = sampleVariance(weightedRewards);
  const standardError = Math.sqrt(variance / weights.length);
  const zScore = getZScore(cfg.confidenceLevel);

  // Weight diagnostics
  const meanWeight = mean(weights);
  const maxWeightObserved = Math.max(...weights);
  const weightVariance = sampleVariance(weights);

  // Effective sample size: n / (1 + var(w))
  const effectiveSampleSize = weights.length / (1 + weightVariance / (meanWeight * meanWeight));

  return {
    estimate,
    standardError,
    confidenceLower: estimate - zScore * standardError,
    confidenceUpper: estimate + zScore * standardError,
    effectiveSampleSize,
    sampleCount: weights.length,
    filteredCount,
    meanWeight,
    maxWeightObserved,
    weightVariance,
  };
}

/**
 * Self-Normalized IPS (SNIPS) Estimator.
 *
 * Normalizes by the sum of weights instead of count, significantly reducing
 * variance at the cost of a small bias.
 *
 * V̂_SNIPS = Σ [w_i * r_i] / Σ w_i
 * where w_i = π_new(a|s) / π_old(a|s)
 *
 * Properties:
 * - Lower variance than IPS
 * - Bounded estimate (won't explode)
 * - Slight bias, but acceptable in practice
 * - Standard choice for production OPE in RTB
 */
export function calculateSNIPS(
  logs: LoggedExperience[],
  newPolicy: EvaluationPolicy,
  config: Partial<OPEConfig> = {}
): OPEResult {
  const cfg = { ...DEFAULT_OPE_CONFIG, ...config };
  const weights: number[] = [];
  const weightedRewards: number[] = [];
  let filteredCount = 0;

  for (const log of logs) {
    if (log.loggingProbability < cfg.minPropensity) {
      filteredCount++;
      continue;
    }

    const newProb = newPolicy.getActionProbability(log.state, log.action);
    const weight = newProb / log.loggingProbability;

    weights.push(weight);
    weightedRewards.push(weight * log.reward);
  }

  if (weights.length === 0) {
    return createEmptyResult(0, filteredCount);
  }

  // SNIPS: normalize by sum of weights
  const sumWeights = sum(weights);
  const estimate = sum(weightedRewards) / sumWeights;

  // Variance estimation for SNIPS (Taylor approximation)
  const normalizedWeights = weights.map((w) => w / sumWeights);
  const residuals = logs
    .filter((l) => l.loggingProbability >= cfg.minPropensity)
    .map((l, i) => normalizedWeights[i]! * (l.reward - estimate));
  const variance = weights.length * sum(residuals.map((r) => r * r));
  const standardError = Math.sqrt(variance) / weights.length;
  const zScore = getZScore(cfg.confidenceLevel);

  const meanWeight = mean(weights);
  const maxWeightObserved = Math.max(...weights);
  const weightVariance = sampleVariance(weights);
  const effectiveSampleSize = (sumWeights * sumWeights) / sum(weights.map((w) => w * w));

  return {
    estimate,
    standardError,
    confidenceLower: estimate - zScore * standardError,
    confidenceUpper: estimate + zScore * standardError,
    effectiveSampleSize,
    sampleCount: weights.length,
    filteredCount,
    meanWeight,
    maxWeightObserved,
    weightVariance,
  };
}

/**
 * Clipped IPS (CIPS) Estimator.
 *
 * Clips importance weights to a maximum value, trading bias for variance reduction.
 *
 * w_clipped = min(w, maxWeight)
 *
 * Properties:
 * - Controlled variance (no explosion)
 * - Biased towards logging policy when clipping occurs
 * - Good when policies differ significantly
 */
export function calculateClippedIPS(
  logs: LoggedExperience[],
  newPolicy: EvaluationPolicy,
  config: Partial<OPEConfig> = {}
): OPEResult {
  const cfg = { ...DEFAULT_OPE_CONFIG, ...config };
  const weights: number[] = [];
  const weightedRewards: number[] = [];
  let filteredCount = 0;
  let clippedCount = 0;

  for (const log of logs) {
    if (log.loggingProbability < cfg.minPropensity) {
      filteredCount++;
      continue;
    }

    const newProb = newPolicy.getActionProbability(log.state, log.action);
    let weight = newProb / log.loggingProbability;

    // Clip weight
    if (weight > cfg.maxWeight) {
      weight = cfg.maxWeight;
      clippedCount++;
    }
    if (weight < cfg.minWeight) {
      weight = cfg.minWeight;
    }

    weights.push(weight);
    weightedRewards.push(weight * log.reward);
  }

  if (weights.length === 0) {
    return createEmptyResult(0, filteredCount);
  }

  const estimate = mean(weightedRewards);
  const variance = sampleVariance(weightedRewards);
  const standardError = Math.sqrt(variance / weights.length);
  const zScore = getZScore(cfg.confidenceLevel);

  const meanWeight = mean(weights);
  const maxWeightObserved = Math.max(...weights);
  const weightVariance = sampleVariance(weights);
  const effectiveSampleSize = weights.length / (1 + weightVariance / (meanWeight * meanWeight));

  return {
    estimate,
    standardError,
    confidenceLower: estimate - zScore * standardError,
    confidenceUpper: estimate + zScore * standardError,
    effectiveSampleSize,
    sampleCount: weights.length,
    filteredCount,
    meanWeight,
    maxWeightObserved,
    weightVariance,
  };
}

/**
 * Doubly Robust (DR) Estimator.
 *
 * Combines IPS with a direct reward prediction model. The estimator remains
 * consistent even if either the propensity model OR the reward model is
 * misspecified (but not both).
 *
 * V̂_DR = (1/n) Σ [r̂(s,a) + w * (r - r̂(s,a))]
 *
 * Where:
 * - r̂(s,a) is the predicted reward from a supervised model
 * - w is the importance weight
 * - r is the observed reward
 *
 * Properties:
 * - Lower variance than IPS (uses model as baseline)
 * - Robust to misspecification of either model
 * - Recommended for high-stakes deployment decisions
 */
export function calculateDoublyRobust(
  logs: LoggedExperience[],
  newPolicy: EvaluationPolicy,
  config: Partial<OPEConfig> = {}
): OPEResult {
  const cfg = { ...DEFAULT_OPE_CONFIG, ...config };

  if (!newPolicy.predictReward) {
    throw new Error('DR estimator requires a reward prediction model');
  }

  const contributions: number[] = [];
  const weights: number[] = [];
  let filteredCount = 0;

  for (const log of logs) {
    if (log.loggingProbability < cfg.minPropensity) {
      filteredCount++;
      continue;
    }

    const newProb = newPolicy.getActionProbability(log.state, log.action);
    let weight = newProb / log.loggingProbability;

    // Clip weight for stability
    weight = Math.max(cfg.minWeight, Math.min(cfg.maxWeight, weight));

    // Get predicted reward (baseline)
    const predictedReward = log.predictedReward ?? newPolicy.predictReward(log.state, log.action);

    // DR contribution: baseline + importance-weighted residual
    const residual = log.reward - predictedReward;
    const contribution = predictedReward + weight * residual;

    contributions.push(contribution);
    weights.push(weight);
  }

  if (contributions.length === 0) {
    return createEmptyResult(0, filteredCount);
  }

  const estimate = mean(contributions);
  const variance = sampleVariance(contributions);
  const standardError = Math.sqrt(variance / contributions.length);
  const zScore = getZScore(cfg.confidenceLevel);

  const meanWeight = mean(weights);
  const maxWeightObserved = Math.max(...weights);
  const weightVariance = sampleVariance(weights);
  const effectiveSampleSize = contributions.length / (1 + weightVariance / (meanWeight * meanWeight));

  return {
    estimate,
    standardError,
    confidenceLower: estimate - zScore * standardError,
    confidenceUpper: estimate + zScore * standardError,
    effectiveSampleSize,
    sampleCount: contributions.length,
    filteredCount,
    meanWeight,
    maxWeightObserved,
    weightVariance,
  };
}

/**
 * Mean Directional Accuracy (MDA).
 *
 * In RTB, the rank-ordering of bids matters more than absolute values.
 * MDA measures how often the OPE estimator correctly predicts the direction
 * of performance changes.
 *
 * Given two policies A and B:
 * - MDA = 1 if OPE(A) > OPE(B) and True(A) > True(B)
 * - MDA = 1 if OPE(A) < OPE(B) and True(A) < True(B)
 * - MDA = 0 otherwise
 *
 * A reliable OPE system should achieve MDA > 80% for deployment decisions.
 */
export function calculateMDA(
  pairs: Array<{
    opeEstimateA: number;
    opeEstimateB: number;
    trueValueA: number;
    trueValueB: number;
  }>
): { mda: number; correctCount: number; totalCount: number } {
  if (pairs.length === 0) {
    return { mda: 0, correctCount: 0, totalCount: 0 };
  }

  let correctCount = 0;

  for (const pair of pairs) {
    const opeDiff = pair.opeEstimateA - pair.opeEstimateB;
    const trueDiff = pair.trueValueA - pair.trueValueB;

    // Check if signs match (both positive, both negative, or both zero)
    if ((opeDiff > 0 && trueDiff > 0) || (opeDiff < 0 && trueDiff < 0) || (opeDiff === 0 && trueDiff === 0)) {
      correctCount++;
    }
  }

  return {
    mda: correctCount / pairs.length,
    correctCount,
    totalCount: pairs.length,
  };
}

/**
 * Policy Deployment Decision Helper.
 *
 * Makes a recommendation on whether to deploy a new policy based on
 * OPE results and safety thresholds.
 */
export interface DeploymentRecommendation {
  /** Whether deployment is recommended */
  recommend: boolean;
  /** Confidence in the recommendation (0-1) */
  confidence: number;
  /** Reasons for the recommendation */
  reasons: string[];
  /** Risk level (low, medium, high) */
  riskLevel: 'low' | 'medium' | 'high';
  /** Expected lift over baseline */
  expectedLift: number;
  /** Confidence interval for lift */
  liftConfidenceInterval: [number, number];
}

export function evaluateDeployment(
  newPolicyOPE: OPEResult,
  baselinePolicyOPE: OPEResult,
  config: {
    minEffectiveSampleSize?: number;
    minLift?: number;
    maxWeightVariance?: number;
    minMDA?: number;
    mda?: number;
  } = {}
): DeploymentRecommendation {
  const minESS = config.minEffectiveSampleSize ?? 100;
  const minLift = config.minLift ?? 0.05; // 5% improvement required
  const maxWeightVar = config.maxWeightVariance ?? 10.0;
  const minMDA = config.minMDA ?? 0.8;

  const reasons: string[] = [];
  let recommend = true;
  let riskLevel: 'low' | 'medium' | 'high' = 'low';

  // Calculate expected lift
  const expectedLift =
    baselinePolicyOPE.estimate !== 0
      ? (newPolicyOPE.estimate - baselinePolicyOPE.estimate) / Math.abs(baselinePolicyOPE.estimate)
      : 0;

  // Lift confidence interval (propagate uncertainty)
  const liftSE = Math.sqrt(
    Math.pow(newPolicyOPE.standardError / Math.abs(baselinePolicyOPE.estimate), 2) +
    Math.pow(baselinePolicyOPE.standardError * newPolicyOPE.estimate / Math.pow(baselinePolicyOPE.estimate, 2), 2)
  );
  const zScore = getZScore(0.95);
  const liftConfidenceInterval: [number, number] = [
    expectedLift - zScore * liftSE,
    expectedLift + zScore * liftSE,
  ];

  // Check effective sample size
  if (newPolicyOPE.effectiveSampleSize < minESS) {
    recommend = false;
    riskLevel = 'high';
    reasons.push(`Effective sample size too low: ${newPolicyOPE.effectiveSampleSize.toFixed(0)} < ${minESS}`);
  }

  // Check minimum lift
  if (expectedLift < minLift) {
    recommend = false;
    reasons.push(`Expected lift too low: ${(expectedLift * 100).toFixed(1)}% < ${(minLift * 100).toFixed(1)}%`);
  }

  // Check if lift is statistically significant
  if (liftConfidenceInterval[0] < 0) {
    recommend = false;
    riskLevel = riskLevel === 'high' ? 'high' : 'medium';
    reasons.push(`Lift not statistically significant (CI includes 0)`);
  }

  // Check weight variance (indicator of policy divergence)
  if (newPolicyOPE.weightVariance > maxWeightVar) {
    riskLevel = riskLevel === 'low' ? 'medium' : riskLevel;
    reasons.push(`High weight variance: ${newPolicyOPE.weightVariance.toFixed(2)} (policies may be too different)`);
  }

  // Check MDA if provided
  if (config.mda !== undefined && config.mda < minMDA) {
    recommend = false;
    riskLevel = 'high';
    reasons.push(`MDA too low: ${(config.mda * 100).toFixed(0)}% < ${(minMDA * 100).toFixed(0)}%`);
  }

  // Positive signals
  if (recommend) {
    reasons.push(`Expected lift: ${(expectedLift * 100).toFixed(1)}%`);
    reasons.push(`Statistically significant improvement`);
  }

  // Calculate confidence in recommendation
  const confidence = Math.min(
    1.0,
    (newPolicyOPE.effectiveSampleSize / (2 * minESS)) *
    (1 - newPolicyOPE.weightVariance / (2 * maxWeightVar)) *
    (config.mda ?? 1.0)
  );

  return {
    recommend,
    confidence,
    reasons,
    riskLevel,
    expectedLift,
    liftConfidenceInterval,
  };
}

/**
 * OPE Suite: Run all estimators and compare results.
 */
export interface OPESuiteResult {
  ips: OPEResult;
  snips: OPEResult;
  cips: OPEResult;
  dr?: OPEResult;
  agreement: 'high' | 'medium' | 'low';
  recommendedEstimate: OPEResult;
  recommendation: string;
}

export function runOPESuite(
  logs: LoggedExperience[],
  newPolicy: EvaluationPolicy,
  config: Partial<OPEConfig> = {}
): OPESuiteResult {
  const ips = calculateIPS(logs, newPolicy, config);
  const snips = calculateSNIPS(logs, newPolicy, config);
  const cips = calculateClippedIPS(logs, newPolicy, config);

  let dr: OPEResult | undefined;
  if (newPolicy.predictReward) {
    try {
      dr = calculateDoublyRobust(logs, newPolicy, config);
    } catch {
      // DR not available
    }
  }

  // Check agreement between estimators
  const estimates = [ips.estimate, snips.estimate, cips.estimate];
  if (dr) estimates.push(dr.estimate);

  const meanEstimate = mean(estimates);
  const estimateVariance = sampleVariance(estimates);
  const coefficientOfVariation = Math.sqrt(estimateVariance) / Math.abs(meanEstimate);

  let agreement: 'high' | 'medium' | 'low';
  if (coefficientOfVariation < 0.1) {
    agreement = 'high';
  } else if (coefficientOfVariation < 0.3) {
    agreement = 'medium';
  } else {
    agreement = 'low';
  }

  // Recommend estimator based on diagnostics
  let recommendedEstimate: OPEResult;
  let recommendation: string;

  if (dr && dr.weightVariance < snips.weightVariance) {
    recommendedEstimate = dr;
    recommendation = 'Doubly Robust recommended (lowest variance with reward model)';
  } else if (ips.weightVariance > 5) {
    recommendedEstimate = snips;
    recommendation = 'SNIPS recommended (high weight variance in IPS)';
  } else {
    recommendedEstimate = snips;
    recommendation = 'SNIPS recommended (standard choice for RTB)';
  }

  // Build result - only include dr if defined
  const result: OPESuiteResult = {
    ips,
    snips,
    cips,
    agreement,
    recommendedEstimate,
    recommendation,
  };
  if (dr !== undefined) {
    result.dr = dr;
  }
  return result;
}

// Helper functions

function sum(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0);
}

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return sum(arr) / arr.length;
}

function sampleVariance(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  const squaredDiffs = arr.map((x) => Math.pow(x - m, 2));
  return sum(squaredDiffs) / (arr.length - 1);
}

function getZScore(confidenceLevel: number): number {
  // Common z-scores for two-tailed confidence intervals
  const zScores: Record<number, number> = {
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
  };
  return zScores[confidenceLevel] ?? 1.96;
}

function createEmptyResult(estimate: number, filteredCount: number): OPEResult {
  return {
    estimate,
    standardError: 0,
    confidenceLower: estimate,
    confidenceUpper: estimate,
    effectiveSampleSize: 0,
    sampleCount: 0,
    filteredCount,
    meanWeight: 0,
    maxWeightObserved: 0,
    weightVariance: 0,
  };
}
