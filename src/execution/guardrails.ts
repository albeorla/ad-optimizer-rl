export type Mode = "shadow" | "pilot";

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
}

/**
 * Clamp a proposed hourly budget using guardrail config and runtime context.
 * Rules applied (in order):
 * - Clamp to [minHourly, maxHourly]
 * - Clamp by per-hour delta: current * (1 ± deltaMax)
 * - Enforce daily target: projectedDailySpend + allowed <= dailyBudgetTarget
 *   - If remaining budget is less than minHourly, allow using the remaining (>=0)
 */
export function applyGuardrails(params: {
  cfg: GuardrailConfig;
  ctx: GuardrailContext;
  currentHourlyBudget: number;
  proposedHourlyBudget: number;
}): GuardrailResult {
  const { cfg, ctx, currentHourlyBudget, proposedHourlyBudget } = params;
  const reasons: string[] = [];

  let allowed = proposedHourlyBudget;
  // Min/max hourly clamp
  if (allowed < cfg.minHourly) {
    allowed = cfg.minHourly;
    reasons.push("min_hourly_floor");
  }
  if (allowed > cfg.maxHourly) {
    allowed = cfg.maxHourly;
    reasons.push("max_hourly_ceiling");
  }

  // Per-hour delta clamp
  const minDelta = currentHourlyBudget * (1 - cfg.deltaMax);
  const maxDelta = currentHourlyBudget * (1 + cfg.deltaMax);
  if (allowed < minDelta) {
    allowed = minDelta;
    reasons.push("delta_clamp_down");
  }
  if (allowed > maxDelta) {
    allowed = maxDelta;
    reasons.push("delta_clamp_up");
  }

  // Enforce daily target
  const remaining = cfg.dailyBudgetTarget - ctx.projectedDailySpend;
  if (remaining <= 0) {
    // No budget left for the day
    const prev = allowed;
    allowed = 0;
    if (prev !== allowed) reasons.push("daily_target_exhausted");
  } else if (ctx.projectedDailySpend + allowed > cfg.dailyBudgetTarget) {
    // Allow at most remaining budget this hour, but consider floors
    const prev = allowed;
    if (remaining < cfg.minHourly) {
      // Allow the leftover even if below floor, but not negative
      allowed = Math.max(0, remaining);
      reasons.push("daily_cap_overrode_min_floor");
    } else {
      allowed = Math.min(allowed, remaining);
      reasons.push("daily_target_cap");
    }
    if (prev !== allowed && allowed < minDelta) {
      // If daily cap conflicts with delta clamp, respect tighter constraint
      allowed = minDelta;
      reasons.push("delta_clamp_after_daily_cap");
    }
  }

  return { allowedBudget: Number(allowed.toFixed(2)), applied: allowed !== proposedHourlyBudget, reasons };
}
