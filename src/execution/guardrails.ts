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
// DQN-REFAC TODO:
// - Guardrails apply post-decision; ensure DQN-selected actions are clamped here.
// - Consider exposing a function to score/clamp budget deltas given policy outputs.
