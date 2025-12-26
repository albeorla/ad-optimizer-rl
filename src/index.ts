// Barrel exports for the library API

// Core types
export { AdEnvironmentState, AdAction, RewardMetrics } from "./types";

// Platform adapters
export { AdPlatformAPI } from "./platforms/base";
export { MockTikTokAdsAPI } from "./platforms/mockTikTok";
export { MockInstagramAdsAPI } from "./platforms/mockInstagram";
export { AdPlatformFactory } from "./platforms/factory";

// RL Agents
export { RLAgent } from "./agent/base";
export { DQNAgent } from "./agent/dqnAgent";
export { DQNAgentNN } from "./agent/dqnAgentNN";
export { CQLAgent, CQLBCAgent } from "./agent/cqlAgent";

// Environments
export { AdEnvironmentSimulator } from "./environment/simulator";

// Training
export { TrainingPipeline } from "./training/pipeline";

// Observers
export { ConsoleLogger } from "./observers/consoleLogger";
export { MetricsCollector } from "./observers/metricsCollector";
export { DiagnosticLogger } from "./observers/diagnosticLogger";

// Control Systems (PID)
export {
  PidPacer,
  CpaPidController,
  DualPidController,
  applyBidModifier,
  computeAdaptiveGains,
  PID_PRESETS,
} from "./control/PidController";

// Delayed Feedback & Attribution
export {
  AttributionBuffer,
  DelayedFeedbackModel,
  AttributionReconciler,
} from "./data/AttributionBuffer";

// Offline Policy Evaluation (OPE)
export {
  calculateIPS,
  calculateSNIPS,
  calculateClippedIPS,
  calculateDoublyRobust,
  calculateMDA,
  evaluateDeployment,
  runOPESuite,
} from "./evaluation/OPE";

// Safety Mechanisms
export {
  CircuitBreaker,
  AnomalyDetector,
  BidValidator,
  SafetyLayer,
  MetricsAggregator,
} from "./execution/SafetyLayer";

// Enriched State Types
export {
  EnrichedAdState,
  BudgetaryContext,
  TemporalContext,
  CompetitiveContext,
  PerformanceContext,
  StateEnrichmentEngine,
  encodeEnrichedState,
  getEnrichedStateDimension,
} from "./types/EnrichedState";

// Existing guardrails
export {
  BudgetPIDController,
  applyGuardrails,
  applySmoothedGuardrails,
  calculatePacingBudget,
  shouldPauseSpending,
} from "./execution/guardrails";

// Encoding utilities
export {
  encodeState,
  ACTIONS,
  actionToIndex,
  indexToAction,
  getActionCount,
  isValidAction,
} from "./agent/encoding";
