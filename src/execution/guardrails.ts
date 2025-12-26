export type Mode = "shadow" | "pilot";

/**
 * PID Controller for smooth budget transitions.
 *
 * Prevents oscillation by smoothing the transition between current and target budgets.
 * Uses proportional (P), integral (I), and derivative (D) terms:
 * - P: Responds to current error (target - actual)
 * - I: Accumulates past errors (prevents steady-state offset)
 * - D: Anticipates future errors (dampens oscillation)
 */
export class BudgetPIDController {
  private kp: number; // Proportional gain
  private ki: number; // Integral gain
  private kd: number; // Derivative gain

  private integral = 0;
  private previousError = 0;
  private lastOutput = 0;

  // Anti-windup: limit integral accumulation
  private integralMax: number;
  private integralMin: number;

  constructor(options?: {
    kp?: number;
    ki?: number;
    kd?: number;
    integralMax?: number;
  }) {
    // Default gains tuned for hourly budget control
    // Lower gains = smoother but slower response
    // Higher gains = faster but potentially oscillating
    this.kp = options?.kp ?? 0.3; // Proportional: 30% of error
    this.ki = options?.ki ?? 0.05; // Integral: small to prevent windup
    this.kd = options?.kd ?? 0.1; // Derivative: dampen oscillations
    this.integralMax = options?.integralMax ?? 5.0;
    this.integralMin = -this.integralMax;
  }

  /**
   * Compute PID output for budget control.
   *
   * @param targetBudget - Desired hourly budget (from RL agent or pacing)
   * @param currentBudget - Current hourly budget
   * @param dt - Time step (default: 1 hour)
   * @returns Smoothed budget adjustment
   */
  compute(targetBudget: number, currentBudget: number, dt = 1): number {
    const error = targetBudget - currentBudget;

    // Proportional term
    const pTerm = this.kp * error;

    // Integral term with anti-windup
    this.integral += error * dt;
    this.integral = Math.max(
      this.integralMin,
      Math.min(this.integralMax, this.integral),
    );
    const iTerm = this.ki * this.integral;

    // Derivative term (rate of change of error)
    const derivative = (error - this.previousError) / dt;
    const dTerm = this.kd * derivative;
    this.previousError = error;

    // Combine terms
    const output = currentBudget + pTerm + iTerm + dTerm;
    this.lastOutput = output;

    return output;
  }

  /**
   * Reset the controller state (e.g., at start of new day).
   */
  reset(): void {
    this.integral = 0;
    this.previousError = 0;
    this.lastOutput = 0;
  }

  /**
   * Get the last computed output for debugging.
   */
  getLastOutput(): number {
    return this.lastOutput;
  }

  /**
   * Get current integral value for diagnostics.
   */
  getIntegral(): number {
    return this.integral;
  }
}

/**
 * Apply PID-smoothed guardrails to a proposed budget.
 * This is the recommended function for production use.
 */
export function applySmoothedGuardrails(params: {
  cfg: GuardrailConfig;
  ctx: GuardrailContext;
  currentHourlyBudget: number;
  proposedHourlyBudget: number;
  pidController: BudgetPIDController;
}): GuardrailResult {
  const { cfg, ctx, currentHourlyBudget, proposedHourlyBudget, pidController } =
    params;

  // Step 1: Use PID to smooth the transition
  const smoothedBudget = pidController.compute(
    proposedHourlyBudget,
    currentHourlyBudget,
  );

  // Step 2: Apply standard guardrails to the smoothed budget
  return applyGuardrails({
    cfg,
    ctx,
    currentHourlyBudget,
    proposedHourlyBudget: smoothedBudget,
  });
}

export interface GuardrailConfig {
  dailyBudgetTarget: number;
  deltaMax: number;
  minHourly: number;
  maxHourly: number;
}

export interface GuardrailContext {
  currentHour: number;
  projectedDailySpend: number;
  trailingHoursWithoutConversions: number;
  trailingROAS: number;
}

export interface GuardrailResult {
  allowedBudget: number;
  applied: boolean;
  reasons: string[];
  /** Detailed breakdown of constraint applications for debugging. */
  details?: {
    originalProposed: number;
    afterMinMax: number;
    afterDelta: number;
    afterDaily: number;
    remainingBudget: number;
  };
}

/**
 * Clamp a proposed hourly budget using guardrail config and runtime context.
 *
 * Constraint priority (highest to lowest):
 * 1. Daily budget exhaustion: If no budget remains, spend is 0
 * 2. Daily budget cap: Cannot exceed remaining daily budget
 * 3. Delta constraints: Limit hourly change rate (but daily cap takes precedence)
 * 4. Min/max hourly bounds
 *
 * The key insight is that daily budget is a hard constraint (we cannot overspend),
 * while delta constraints are soft (we can violate them to stay within budget).
 */
export function applyGuardrails(params: {
  cfg: GuardrailConfig;
  ctx: GuardrailContext;
  currentHourlyBudget: number;
  proposedHourlyBudget: number;
}): GuardrailResult {
  const { cfg, ctx, currentHourlyBudget, proposedHourlyBudget } = params;
  const reasons: string[] = [];
  const originalProposed = proposedHourlyBudget;

  let allowed = proposedHourlyBudget;

  // Step 1: Apply min/max hourly bounds
  const afterMinMax = Math.max(cfg.minHourly, Math.min(cfg.maxHourly, allowed));
  if (afterMinMax < allowed) {
    allowed = afterMinMax;
    reasons.push("max_hourly_ceiling");
  } else if (afterMinMax > allowed) {
    allowed = afterMinMax;
    reasons.push("min_hourly_floor");
  }

  // Step 2: Apply per-hour delta constraints
  const minDelta = currentHourlyBudget * (1 - cfg.deltaMax);
  const maxDelta = currentHourlyBudget * (1 + cfg.deltaMax);
  let afterDelta = allowed;

  if (allowed < minDelta) {
    afterDelta = minDelta;
    reasons.push("delta_clamp_down");
  } else if (allowed > maxDelta) {
    afterDelta = maxDelta;
    reasons.push("delta_clamp_up");
  }
  allowed = afterDelta;

  // Step 3: Enforce daily budget target (hard constraint, overrides delta)
  const remaining = cfg.dailyBudgetTarget - ctx.projectedDailySpend;

  let afterDaily = allowed;
  if (remaining <= 0) {
    // Daily budget exhausted - must spend 0
    afterDaily = 0;
    if (allowed > 0) {
      reasons.push("daily_target_exhausted");
    }
  } else if (allowed > remaining) {
    // Would exceed daily budget - cap at remaining
    afterDaily = remaining;

    // Check if remaining is below minimum hourly threshold
    if (remaining < cfg.minHourly) {
      // Spend what's left (even if below normal minimum)
      reasons.push("daily_cap_overrode_min_floor");
    } else {
      reasons.push("daily_target_cap");
    }

    // Note: We intentionally do NOT restore delta constraint here.
    // Daily budget is a hard constraint; we cannot overspend.
    // The previous logic that set `allowed = minDelta` after daily cap was a bug
    // because minDelta could exceed remaining budget.
  }
  allowed = afterDaily;

  // Final safety clamp: ensure non-negative
  allowed = Math.max(0, allowed);

  return {
    allowedBudget: Number(allowed.toFixed(2)),
    applied: Math.abs(allowed - proposedHourlyBudget) > 0.001,
    reasons,
    details: {
      originalProposed,
      afterMinMax,
      afterDelta,
      afterDaily: allowed,
      remainingBudget: remaining,
    },
  };
}

/**
 * Calculate the remaining hourly budget capacity for the rest of the day.
 * Useful for pacing algorithms that need to spread budget evenly.
 */
export function calculatePacingBudget(params: {
  cfg: GuardrailConfig;
  ctx: GuardrailContext;
}): { hourlyTarget: number; hoursRemaining: number; canSpend: boolean } {
  const { cfg, ctx } = params;
  const hoursRemaining = Math.max(1, 24 - ctx.currentHour);
  const remaining = cfg.dailyBudgetTarget - ctx.projectedDailySpend;

  if (remaining <= 0) {
    return { hourlyTarget: 0, hoursRemaining, canSpend: false };
  }

  // Even pacing: divide remaining budget by remaining hours
  const evenPace = remaining / hoursRemaining;

  // Clamp to hourly bounds
  const hourlyTarget = Math.max(
    cfg.minHourly,
    Math.min(cfg.maxHourly, evenPace),
  );

  return {
    hourlyTarget: Number(hourlyTarget.toFixed(2)),
    hoursRemaining,
    canSpend: remaining > 0,
  };
}

/**
 * Check if spending should be paused based on performance signals.
 * Returns true if spending should be paused.
 */
export function shouldPauseSpending(params: {
  ctx: GuardrailContext;
  maxHoursWithoutConversion?: number;
  minROASThreshold?: number;
}): { shouldPause: boolean; reason?: string } {
  const {
    ctx,
    maxHoursWithoutConversion = 6,
    minROASThreshold = 0.5,
  } = params;

  // Pause if no conversions for too long
  if (ctx.trailingHoursWithoutConversions >= maxHoursWithoutConversion) {
    return {
      shouldPause: true,
      reason: `no_conversions_for_${ctx.trailingHoursWithoutConversions}_hours`,
    };
  }

  // Pause if ROAS is critically low
  if (ctx.trailingROAS < minROASThreshold && ctx.trailingROAS > 0) {
    return {
      shouldPause: true,
      reason: `low_roas_${ctx.trailingROAS.toFixed(2)}`,
    };
  }

  return { shouldPause: false };
}
