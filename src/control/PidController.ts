/**
 * Enhanced PID Controllers for Real-Time Bidding Budget Management.
 *
 * Implements industry-standard feedback control for ad campaign pacing:
 * - PidPacer: Primary multiplier-based pacing controller
 * - DualPidController: Dual-PID for simultaneous budget and CPA constraints
 * - CascadedPidController: Hierarchical control for complex constraints
 *
 * Reference: PID control is essential for smooth budget delivery in RTB systems,
 * preventing "early stop" scenarios and ensuring optimal inventory coverage.
 */

/**
 * Configuration for a single PID controller instance.
 */
export interface PidConfig {
  /** Proportional gain - reacts to current error */
  kp: number;
  /** Integral gain - corrects accumulated historical error */
  ki: number;
  /** Derivative gain - dampens oscillations */
  kd: number;
  /** Maximum allowed integral accumulation (anti-windup) */
  integralMax?: number;
  /** Minimum allowed integral accumulation (anti-windup) */
  integralMin?: number;
  /** Output clamp minimum */
  outputMin?: number;
  /** Output clamp maximum */
  outputMax?: number;
}

/**
 * Default PID configurations for different control objectives.
 */
export const PID_PRESETS = {
  /** Conservative pacing - smooth, slow response */
  conservative: { kp: 0.2, ki: 0.03, kd: 0.1, integralMax: 3.0 },
  /** Balanced pacing - moderate response */
  balanced: { kp: 0.3, ki: 0.05, kd: 0.1, integralMax: 5.0 },
  /** Aggressive pacing - fast response for volatile markets */
  aggressive: { kp: 0.5, ki: 0.08, kd: 0.15, integralMax: 8.0 },
  /** CPA control - slower, more stable for cost optimization */
  cpaControl: { kp: 0.15, ki: 0.02, kd: 0.08, integralMax: 2.0 },
} as const;

/**
 * PID Pacer: Multiplier-based budget pacing controller.
 *
 * Unlike the existing BudgetPIDController that outputs absolute values,
 * this controller outputs a bid multiplier (α) that modulates the RL agent's
 * base bid to ensure smooth budget delivery.
 *
 * Usage: final_bid = rl_bid * pid_multiplier
 *
 * The multiplier approach provides several advantages:
 * 1. Decouples valuation (RL) from delivery (PID)
 * 2. Preserves relative bid rankings from the RL agent
 * 3. Allows dynamic scaling without changing RL training
 */
export class PidPacer {
  private integral = 0;
  private prevError = 0;
  private lastMultiplier = 1.0;

  private kp: number;
  private ki: number;
  private kd: number;
  private integralMax: number;
  private integralMin: number;
  private outputMin: number;
  private outputMax: number;

  constructor(
    private totalBudget: number,
    private campaignDuration: number,
    config: Partial<PidConfig> = PID_PRESETS.balanced
  ) {
    this.kp = config.kp ?? 0.3;
    this.ki = config.ki ?? 0.05;
    this.kd = config.kd ?? 0.1;
    this.integralMax = config.integralMax ?? 5.0;
    this.integralMin = config.integralMin ?? -this.integralMax;
    this.outputMin = config.outputMin ?? 0.5;
    this.outputMax = config.outputMax ?? 2.0;
  }

  /**
   * Compute the bid multiplier based on current spend trajectory.
   *
   * @param currentSpend - Total spend so far in the campaign
   * @param elapsedTime - Time elapsed (same units as campaignDuration)
   * @param dt - Time step size (default: 1)
   * @returns Bid multiplier α ∈ [outputMin, outputMax]
   */
  getMultiplier(currentSpend: number, elapsedTime: number, dt = 1): number {
    // Target spend at current time (linear pacing)
    const targetSpend = (elapsedTime / this.campaignDuration) * this.totalBudget;

    // Positive error = underspending, need to bid higher
    // Negative error = overspending, need to bid lower
    const error = targetSpend - currentSpend;

    // Normalize error relative to budget for stable gains across different budgets
    const normalizedError = error / this.totalBudget;

    // Proportional term
    const pTerm = this.kp * normalizedError;

    // Integral term with anti-windup clamping
    this.integral += normalizedError * dt;
    this.integral = Math.max(
      this.integralMin,
      Math.min(this.integralMax, this.integral)
    );
    const iTerm = this.ki * this.integral;

    // Derivative term (rate of change of error)
    const derivative = (normalizedError - this.prevError) / dt;
    const dTerm = this.kd * derivative;
    this.prevError = normalizedError;

    // Base multiplier is 1.0; adjustment shifts it up or down
    const rawMultiplier = 1.0 + pTerm + iTerm + dTerm;

    // Clamp to safe bounds
    const multiplier = Math.max(
      this.outputMin,
      Math.min(this.outputMax, rawMultiplier)
    );

    this.lastMultiplier = multiplier;
    return multiplier;
  }

  /**
   * Get spend trajectory information for diagnostics.
   */
  getTrajectoryInfo(currentSpend: number, elapsedTime: number): {
    targetSpend: number;
    spendError: number;
    spendErrorPercent: number;
    pacingStatus: 'under' | 'on_track' | 'over';
  } {
    const targetSpend = (elapsedTime / this.campaignDuration) * this.totalBudget;
    const spendError = targetSpend - currentSpend;
    const spendErrorPercent = (spendError / this.totalBudget) * 100;

    let pacingStatus: 'under' | 'on_track' | 'over';
    if (spendErrorPercent > 5) {
      pacingStatus = 'under';
    } else if (spendErrorPercent < -5) {
      pacingStatus = 'over';
    } else {
      pacingStatus = 'on_track';
    }

    return { targetSpend, spendError, spendErrorPercent, pacingStatus };
  }

  /**
   * Reset controller state (e.g., at campaign start or day boundary).
   */
  reset(): void {
    this.integral = 0;
    this.prevError = 0;
    this.lastMultiplier = 1.0;
  }

  /**
   * Update campaign parameters (for dynamic budget changes).
   */
  updateCampaignParams(totalBudget: number, campaignDuration: number): void {
    this.totalBudget = totalBudget;
    this.campaignDuration = campaignDuration;
    // Optionally reset on parameter change
    // this.reset();
  }

  getLastMultiplier(): number {
    return this.lastMultiplier;
  }

  getIntegral(): number {
    return this.integral;
  }

  getDiagnostics(): {
    integral: number;
    prevError: number;
    lastMultiplier: number;
    config: { kp: number; ki: number; kd: number };
  } {
    return {
      integral: this.integral,
      prevError: this.prevError,
      lastMultiplier: this.lastMultiplier,
      config: { kp: this.kp, ki: this.ki, kd: this.kd },
    };
  }
}

/**
 * CPA PID Controller: Controls Cost Per Acquisition.
 *
 * Unlike budget pacing which tracks cumulative spend vs time,
 * CPA control tracks instantaneous cost efficiency:
 * - Target CPA = desired cost per conversion
 * - Actual CPA = recent spend / recent conversions
 * - Error = (Target CPA - Actual CPA) / Target CPA (normalized)
 *
 * High CPA → reduce bids (multiplier < 1)
 * Low CPA → can increase bids (multiplier > 1)
 */
export class CpaPidController {
  private integral = 0;
  private prevError = 0;
  private lastMultiplier = 1.0;

  private kp: number;
  private ki: number;
  private kd: number;
  private integralMax: number;
  private integralMin: number;
  private outputMin: number;
  private outputMax: number;

  constructor(
    private targetCpa: number,
    config: Partial<PidConfig> = PID_PRESETS.cpaControl
  ) {
    this.kp = config.kp ?? 0.15;
    this.ki = config.ki ?? 0.02;
    this.kd = config.kd ?? 0.08;
    this.integralMax = config.integralMax ?? 2.0;
    this.integralMin = config.integralMin ?? -this.integralMax;
    this.outputMin = config.outputMin ?? 0.3;
    this.outputMax = config.outputMax ?? 1.5;
  }

  /**
   * Compute the bid multiplier based on CPA performance.
   *
   * @param recentSpend - Spend in the trailing window
   * @param recentConversions - Conversions in the trailing window
   * @param dt - Time step size
   * @returns Bid multiplier α ∈ [outputMin, outputMax]
   */
  getMultiplier(recentSpend: number, recentConversions: number, dt = 1): number {
    // Handle edge cases
    if (recentConversions === 0) {
      // No conversions - can't compute CPA, apply penalty
      // Reduce bids to conserve budget until we get signal
      this.integral = Math.max(this.integralMin, this.integral - 0.1);
      this.lastMultiplier = Math.max(this.outputMin, this.lastMultiplier * 0.95);
      return this.lastMultiplier;
    }

    if (recentSpend === 0) {
      // No spend - maintain current state
      return this.lastMultiplier;
    }

    const actualCpa = recentSpend / recentConversions;

    // Positive error = actual CPA below target (good), can bid higher
    // Negative error = actual CPA above target (bad), must bid lower
    const error = (this.targetCpa - actualCpa) / this.targetCpa;

    // Proportional term
    const pTerm = this.kp * error;

    // Integral term with anti-windup
    this.integral += error * dt;
    this.integral = Math.max(
      this.integralMin,
      Math.min(this.integralMax, this.integral)
    );
    const iTerm = this.ki * this.integral;

    // Derivative term
    const derivative = (error - this.prevError) / dt;
    const dTerm = this.kd * derivative;
    this.prevError = error;

    // Compute multiplier
    const rawMultiplier = 1.0 + pTerm + iTerm + dTerm;
    const multiplier = Math.max(
      this.outputMin,
      Math.min(this.outputMax, rawMultiplier)
    );

    this.lastMultiplier = multiplier;
    return multiplier;
  }

  /**
   * Get CPA performance diagnostics.
   */
  getCpaInfo(recentSpend: number, recentConversions: number): {
    actualCpa: number;
    targetCpa: number;
    cpaError: number;
    cpaErrorPercent: number;
    performanceStatus: 'efficient' | 'on_target' | 'inefficient';
  } {
    const actualCpa = recentConversions > 0 ? recentSpend / recentConversions : Infinity;
    const cpaError = this.targetCpa - actualCpa;
    const cpaErrorPercent = (cpaError / this.targetCpa) * 100;

    let performanceStatus: 'efficient' | 'on_target' | 'inefficient';
    if (cpaErrorPercent > 10) {
      performanceStatus = 'efficient';
    } else if (cpaErrorPercent < -10) {
      performanceStatus = 'inefficient';
    } else {
      performanceStatus = 'on_target';
    }

    return { actualCpa, targetCpa: this.targetCpa, cpaError, cpaErrorPercent, performanceStatus };
  }

  reset(): void {
    this.integral = 0;
    this.prevError = 0;
    this.lastMultiplier = 1.0;
  }

  setTargetCpa(targetCpa: number): void {
    this.targetCpa = targetCpa;
  }

  getLastMultiplier(): number {
    return this.lastMultiplier;
  }

  getDiagnostics(): {
    integral: number;
    prevError: number;
    lastMultiplier: number;
    targetCpa: number;
  } {
    return {
      integral: this.integral,
      prevError: this.prevError,
      lastMultiplier: this.lastMultiplier,
      targetCpa: this.targetCpa,
    };
  }
}

/**
 * Dual-PID Controller: Simultaneous budget and CPA control.
 *
 * Implements the constraint hierarchy:
 * α_final = min(α_budget, α_cpa)
 *
 * This ensures the system always respects the tightest constraint:
 * - If CPA is good but budget is tight → Budget PID restricts
 * - If budget is ample but CPA is high → CPA PID restricts
 */
export class DualPidController {
  private budgetPid: PidPacer;
  private cpaPid: CpaPidController;
  private lastBudgetMultiplier = 1.0;
  private lastCpaMultiplier = 1.0;
  private lastFinalMultiplier = 1.0;
  private bindingConstraint: 'budget' | 'cpa' | 'none' = 'none';

  constructor(options: {
    totalBudget: number;
    campaignDuration: number;
    targetCpa: number;
    budgetConfig?: Partial<PidConfig>;
    cpaConfig?: Partial<PidConfig>;
  }) {
    this.budgetPid = new PidPacer(
      options.totalBudget,
      options.campaignDuration,
      options.budgetConfig
    );
    this.cpaPid = new CpaPidController(options.targetCpa, options.cpaConfig);
  }

  /**
   * Compute the final bid multiplier considering both constraints.
   *
   * @param context - Current campaign state
   * @returns Final bid multiplier (minimum of budget and CPA multipliers)
   */
  getMultiplier(context: {
    currentSpend: number;
    elapsedTime: number;
    recentSpend: number;
    recentConversions: number;
    dt?: number;
  }): number {
    const dt = context.dt ?? 1;

    // Get multiplier from budget PID
    this.lastBudgetMultiplier = this.budgetPid.getMultiplier(
      context.currentSpend,
      context.elapsedTime,
      dt
    );

    // Get multiplier from CPA PID
    this.lastCpaMultiplier = this.cpaPid.getMultiplier(
      context.recentSpend,
      context.recentConversions,
      dt
    );

    // Apply minimum constraint - respect the tighter limit
    this.lastFinalMultiplier = Math.min(
      this.lastBudgetMultiplier,
      this.lastCpaMultiplier
    );

    // Track which constraint is binding
    if (this.lastBudgetMultiplier < this.lastCpaMultiplier) {
      this.bindingConstraint = 'budget';
    } else if (this.lastCpaMultiplier < this.lastBudgetMultiplier) {
      this.bindingConstraint = 'cpa';
    } else {
      this.bindingConstraint = 'none';
    }

    return this.lastFinalMultiplier;
  }

  /**
   * Get comprehensive diagnostics for both controllers.
   */
  getDiagnostics(): {
    budgetMultiplier: number;
    cpaMultiplier: number;
    finalMultiplier: number;
    bindingConstraint: 'budget' | 'cpa' | 'none';
    budgetPid: ReturnType<PidPacer['getDiagnostics']>;
    cpaPid: ReturnType<CpaPidController['getDiagnostics']>;
  } {
    return {
      budgetMultiplier: this.lastBudgetMultiplier,
      cpaMultiplier: this.lastCpaMultiplier,
      finalMultiplier: this.lastFinalMultiplier,
      bindingConstraint: this.bindingConstraint,
      budgetPid: this.budgetPid.getDiagnostics(),
      cpaPid: this.cpaPid.getDiagnostics(),
    };
  }

  reset(): void {
    this.budgetPid.reset();
    this.cpaPid.reset();
    this.lastBudgetMultiplier = 1.0;
    this.lastCpaMultiplier = 1.0;
    this.lastFinalMultiplier = 1.0;
    this.bindingConstraint = 'none';
  }

  getBudgetPacer(): PidPacer {
    return this.budgetPid;
  }

  getCpaPid(): CpaPidController {
    return this.cpaPid;
  }

  getLastFinalMultiplier(): number {
    return this.lastFinalMultiplier;
  }

  getBindingConstraint(): 'budget' | 'cpa' | 'none' {
    return this.bindingConstraint;
  }
}

/**
 * Bid Modifier: Applies PID multiplier to RL agent's bid decision.
 *
 * This is the integration point between the RL valuation system and the
 * PID pacing system. It ensures that:
 * 1. The RL agent's relative rankings are preserved
 * 2. Budget constraints are respected
 * 3. CPA targets are maintained
 */
export function applyBidModifier(
  rlBid: number,
  pidMultiplier: number,
  constraints?: {
    minBid?: number;
    maxBid?: number;
    minMultiplier?: number;
    maxMultiplier?: number;
  }
): { finalBid: number; wasConstrained: boolean; constraintReason?: string } {
  const minMult = constraints?.minMultiplier ?? 0.1;
  const maxMult = constraints?.maxMultiplier ?? 3.0;
  const minBid = constraints?.minBid ?? 0.01;
  const maxBid = constraints?.maxBid ?? Infinity;

  let constraintReason: string | undefined;
  let wasConstrained = false;

  // Clamp multiplier first
  let effectiveMult = pidMultiplier;
  if (pidMultiplier < minMult) {
    effectiveMult = minMult;
    wasConstrained = true;
    constraintReason = 'min_multiplier';
  } else if (pidMultiplier > maxMult) {
    effectiveMult = maxMult;
    wasConstrained = true;
    constraintReason = 'max_multiplier';
  }

  // Apply multiplier
  let finalBid = rlBid * effectiveMult;

  // Clamp final bid
  if (finalBid < minBid) {
    finalBid = minBid;
    wasConstrained = true;
    constraintReason = constraintReason ?? 'min_bid';
  } else if (finalBid > maxBid) {
    finalBid = maxBid;
    wasConstrained = true;
    constraintReason = constraintReason ?? 'max_bid';
  }

  // Build result - only include constraintReason if defined
  const result: { finalBid: number; wasConstrained: boolean; constraintReason?: string } = {
    finalBid,
    wasConstrained,
  };
  if (constraintReason !== undefined) {
    result.constraintReason = constraintReason;
  }
  return result;
}

/**
 * Adaptive Gain Scheduler: Adjusts PID gains based on campaign phase.
 *
 * Different phases of a campaign may require different control characteristics:
 * - Early phase: More aggressive to establish spend trajectory
 * - Mid phase: Balanced control
 * - Late phase: Conservative to avoid overspend
 * - Emergency phase: Very aggressive if severely underspent
 */
export function computeAdaptiveGains(
  baseConfig: PidConfig,
  context: {
    elapsedRatio: number; // 0-1, fraction of campaign elapsed
    spendRatio: number;   // 0-1, fraction of budget spent
    pacingError: number;  // (target - actual) / target
  }
): PidConfig {
  const { elapsedRatio, spendRatio, pacingError } = context;

  // Phase-based multipliers
  let phaseMultiplier = 1.0;
  if (elapsedRatio < 0.2) {
    // Early phase - slightly aggressive
    phaseMultiplier = 1.2;
  } else if (elapsedRatio > 0.8) {
    // Late phase - more conservative
    phaseMultiplier = 0.8;
  }

  // Emergency adjustment for significant pacing errors
  let emergencyMultiplier = 1.0;
  if (pacingError > 0.3) {
    // Severely underspent - boost gains
    emergencyMultiplier = 1.5;
  } else if (pacingError < -0.3) {
    // Severely overspent - boost gains (to reduce spending faster)
    emergencyMultiplier = 1.3;
  }

  const combinedMultiplier = phaseMultiplier * emergencyMultiplier;

  // Build result - only include optional properties if defined
  const result: PidConfig = {
    kp: baseConfig.kp * combinedMultiplier,
    ki: baseConfig.ki * combinedMultiplier,
    kd: baseConfig.kd * (emergencyMultiplier > 1 ? 0.8 : 1.0), // Reduce D in emergency
  };
  if (baseConfig.integralMax !== undefined) result.integralMax = baseConfig.integralMax;
  if (baseConfig.integralMin !== undefined) result.integralMin = baseConfig.integralMin;
  if (baseConfig.outputMin !== undefined) result.outputMin = baseConfig.outputMin;
  if (baseConfig.outputMax !== undefined) result.outputMax = baseConfig.outputMax;
  return result;
}
