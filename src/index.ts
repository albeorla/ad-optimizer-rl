// Barrel exports for the library API
export { AdEnvironmentState, AdAction, RewardMetrics } from "./types";
export { AdPlatformAPI } from "./platforms/base";
export { MockTikTokAdsAPI } from "./platforms/mockTikTok";
export { MockInstagramAdsAPI } from "./platforms/mockInstagram";
export { AdPlatformFactory } from "./platforms/factory";
export { RLAgent } from "./agent/base";
export { DQNAgent } from "./agent/dqnAgent";
export { AdEnvironmentSimulator } from "./environment/simulator";
export { TrainingPipeline } from "./training/pipeline";
export { ConsoleLogger } from "./observers/consoleLogger";
export { MetricsCollector } from "./observers/metricsCollector";
export { DiagnosticLogger } from "./observers/diagnosticLogger";
// DQN-REFAC TODO:
// - When NN agent lands, consider exporting: encodeState/ACTIONS (from agent/encoding), QNet types.
