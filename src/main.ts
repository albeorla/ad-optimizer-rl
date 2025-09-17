/**
 * @fileoverview Main entry point for the T-Shirt Ad Optimization RL System
 *
 * This module orchestrates the training and evaluation of reinforcement learning agents
 * for optimizing T-shirt advertisement campaigns across multiple social media platforms.
 * It supports both tabular Q-learning and neural network-based DQN agents.
 *
 * The system learns to make optimal decisions about:
 * - Platform selection (Instagram, TikTok, etc.)
 * - Budget allocation adjustments
 * - Creative type selection
 * - Target age group demographics
 *
 * @author RL Demo Team
 * @version 1.0.0
 */

// Core agent implementations
import { DQNAgent } from "./agent/dqnAgent";
import { DQNAgentNN } from "./agent/dqnAgentNN";

// Environment and training infrastructure
import { AdEnvironmentSimulator } from "./environment/simulator";
import { TrainingPipeline } from "./training/pipeline";

// Observers for monitoring and logging
import { ConsoleLogger } from "./observers/consoleLogger";
import { MetricsCollector } from "./observers/metricsCollector";
import { DiagnosticLogger } from "./observers/diagnosticLogger";

/**
 * Configuration interface for neural network agent options
 * @interface NNOptions
 */
interface NNOptions {
  /** Initial exploration rate (1.0 = 100% random actions) */
  epsilonStart: number;
  /** Minimum exploration rate (prevents complete exploitation) */
  epsilonMin: number;
  /** Rate at which exploration decreases over time */
  epsilonDecay: number;
  /** Learning rate for neural network optimization */
  lr: number;
  /** Discount factor for future rewards (0-1) */
  gamma: number;
  /** Number of experiences sampled for each training batch */
  batchSize: number;
  /** Frequency of training updates (every N steps) */
  trainFreq: number;
  /** Frequency of target network synchronization */
  targetSync: number;
  /** Maximum capacity of experience replay buffer */
  replayCapacity: number;
}

/**
 * Parsed command-line arguments and environment variables
 * @interface ParsedArgs
 */
interface ParsedArgs {
  /** Number of training episodes to run */
  episodes: number;
  /** Epsilon decay rate for tabular agents */
  epsilonDecay: number;
  /** Learning rate decay for tabular agents */
  lrDecay: number;
  /** Reward shaping strength multiplier */
  shaping: number;
  /** Type of agent to use ("tabular" or "nn") */
  agentKind: string;
  /** Neural network specific configuration options */
  nnOpts: NNOptions;
  /** Mode of operation: simulator, shadow, or pilot */
  mode: "sim" | "shadow" | "pilot";
  /** Skip 24h demo after training */
  noDemo: boolean;
  /** Optional model path to load */
  loadPath?: string;
}

/**
 * Utility function to parse command-line arguments and environment variables
 *
 * Supports both CLI arguments (--arg=value) and environment variables.
 * CLI arguments take precedence over environment variables.
 *
 * @returns {ParsedArgs} Parsed configuration object with sensible defaults
 *
 * @example
 * // CLI usage: node main.js --episodes=100 --agent=nn --lr=0.001
 * // Environment: EPISODES=100 AGENT=nn LR=0.001 node main.js
 */
function parseArgs(): ParsedArgs {
  // Extract command-line arguments (skip 'node' and script name)
  const args = process.argv.slice(2);

  /**
   * Helper function to extract argument value by name
   * @param {string} name - Argument name without '--' prefix
   * @returns {string | undefined} Argument value or undefined if not found
   */
  const getArg = (name: string): string | undefined => {
    const p = args.find((a) => a.startsWith(`--${name}=`));
    return p ? p.split("=")[1] : undefined;
  };

  const _load = getArg("load") ?? process.env.LOAD_MODEL ?? undefined;
  return {
    // Training configuration
    episodes: Number(getArg("episodes") ?? process.env.EPISODES ?? 50),
    epsilonDecay: Number(
      getArg("epsilonDecay") ?? process.env.EPS_DECAY ?? 0.999,
    ),
    lrDecay: Number(getArg("lrDecay") ?? process.env.LR_DECAY ?? 0.99),
    shaping: Number(getArg("shaping") ?? process.env.SHAPING ?? 1),
    agentKind: (
      getArg("agent") ??
      process.env.AGENT ??
      "tabular"
    ).toLowerCase(),
    mode: (
      (getArg("mode") ?? process.env.MODE ?? "sim").toLowerCase() as
        | "sim"
        | "shadow"
        | "pilot"
    ),
    noDemo: (getArg("no-demo") ?? process.env.NO_DEMO ?? "false").toLowerCase() === "true",
    ...(typeof _load === "string" ? { loadPath: _load } : {}),

    // Neural network specific options
    nnOpts: {
      epsilonStart: Number(
        getArg("epsilonStart") ?? process.env.EPS_START ?? 1.0,
      ),
      epsilonMin: Number(getArg("epsilonMin") ?? process.env.EPS_MIN ?? 0.05),
      epsilonDecay: Number(
        getArg("epsilonDecay") ?? process.env.EPS_DECAY ?? 0.995,
      ),
      lr: Number(getArg("lr") ?? process.env.LR ?? 1e-3),
      gamma: Number(getArg("gamma") ?? process.env.GAMMA ?? 0.95),
      batchSize: Number(getArg("batchSize") ?? process.env.BATCH_SIZE ?? 32),
      trainFreq: Number(getArg("trainFreq") ?? process.env.TRAIN_FREQ ?? 1),
      targetSync: Number(
        getArg("targetSync") ?? process.env.TARGET_SYNC ?? 250,
      ),
      replayCapacity: Number(
        getArg("replayCap") ?? process.env.REPLAY_CAP ?? 5000,
      ),
    } as const,
  };
}

/**
 * Factory function to create the appropriate agent based on configuration
 *
 * Supports two agent types:
 * - "tabular": Traditional Q-learning with lookup table
 * - "nn": Deep Q-Network with neural network approximation
 *
 * @param {string} agentKind - Type of agent to create
 * @param {NNOptions} nnOpts - Neural network configuration options
 * @param {number} epsilonDecay - Epsilon decay rate for tabular agents
 * @param {number} lrDecay - Learning rate decay for tabular agents
 * @returns {DQNAgent | DQNAgentNN} Configured agent instance
 *
 * @throws {Error} If agentKind is not supported
 */
function createAgent(
  agentKind: string,
  nnOpts: NNOptions,
  epsilonDecay: number,
  lrDecay: number,
): DQNAgent | DQNAgentNN {
  if (agentKind === "nn") {
    // Create neural network-based DQN agent with experience replay
    return new DQNAgentNN({
      epsilonStart: nnOpts.epsilonStart,
      epsilonMin: nnOpts.epsilonMin,
      epsilonDecay: nnOpts.epsilonDecay,
      lr: nnOpts.lr,
      gamma: nnOpts.gamma,
      batchSize: nnOpts.batchSize,
      trainFreq: nnOpts.trainFreq,
      targetSync: nnOpts.targetSync,
      replayCapacity: nnOpts.replayCapacity,
    });
  } else {
    // Create traditional tabular Q-learning agent
    return new DQNAgent({ epsilonDecay, lrDecay });
  }
}

/**
 * Factory function to create and configure the training environment and pipeline
 *
 * Sets up the complete training infrastructure including:
 * - Ad environment simulator with reward shaping
 * - Training pipeline with agent and environment
 * - Observers for logging, metrics collection, and diagnostics
 *
 * @param {DQNAgent | DQNAgentNN} agent - The RL agent to train
 * @param {number} shaping - Reward shaping strength multiplier
 * @returns {Object} Configuration object containing environment, pipeline, and metrics collector
 */
function setupPipeline(
  agent: DQNAgent | DQNAgentNN,
  shaping: number,
): {
  environment: AdEnvironmentSimulator;
  pipeline: TrainingPipeline;
  metricsCollector: MetricsCollector;
} {
  // Create environment with configurable reward shaping
  const environment = new AdEnvironmentSimulator({ shapingStrength: shaping });

  // Create training pipeline connecting agent and environment
  const pipeline = new TrainingPipeline(agent, environment);

  // Set up observers for monitoring and logging
  const logger = new ConsoleLogger(); // Console output for training progress
  const metricsCollector = new MetricsCollector(); // Performance metrics collection
  const diagnosticLogger = new DiagnosticLogger(); // Detailed diagnostic information

  // Attach observers to pipeline for real-time monitoring
  pipeline.addObserver(logger);
  pipeline.addObserver(metricsCollector);
  pipeline.addObserver(diagnosticLogger);

  return { environment, pipeline, metricsCollector };
}

/**
 * Warm-start the agent with initial heuristics if available
 *
 * Some agents support seeding with domain knowledge to accelerate learning.
 * This function attempts to warm-start the agent, falling back to a simple
 * environment reset if the agent doesn't support heuristics.
 *
 * @param {DQNAgent | DQNAgentNN} agent - The agent to warm-start
 * @param {AdEnvironmentSimulator} environment - The environment to reset
 */
function warmStartAgent(
  agent: DQNAgent | DQNAgentNN,
  environment: AdEnvironmentSimulator,
): void {
  // Check if agent supports heuristic seeding
  if ((agent as any).seedHeuristics) {
    // Warm-start with domain knowledge
    (agent as any).seedHeuristics(environment.reset());
  } else {
    // Fallback to standard environment reset
    environment.reset();
  }
}

/**
 * Print a formatted banner for the application
 *
 * Displays the system name and version information in a visually
 * appealing format to clearly identify the running application.
 */
function printBanner(): void {
  console.log("=".repeat(60));
  console.log("🎯 T-SHIRT AD OPTIMIZATION RL SYSTEM");
  console.log("=".repeat(60));
}

/**
 * Demonstrate the learned policy by running a 24-hour simulation
 *
 * After training, this function showcases the agent's learned behavior
 * by running a complete day simulation with greedy action selection.
 * The agent makes decisions for each hour, showing platform selection,
 * budget adjustments, creative choices, and target demographics.
 *
 * @param {DQNAgent | DQNAgentNN} agent - The trained agent to demonstrate
 * @returns {Promise<void>} Promise that resolves when demonstration is complete
 */
async function demonstratePolicy(agent: DQNAgent | DQNAgentNN): Promise<void> {
  console.log("\n🎮 Demonstrating Learned Policy...\n");

  // Create a fresh environment for policy demonstration
  const testEnv = new AdEnvironmentSimulator();
  let testState = testEnv.reset();
  let totalProfit = 0;

  // Run 24-hour simulation with greedy action selection
  for (let hour = 0; hour < 24; hour++) {
    // Agent selects action based on current state (greedy policy)
    const action = agent.selectAction(testState);

    // Execute action and observe results
    const [nextState, reward] = testEnv.step(action);

    // Accumulate profit (denormalize from normalized reward)
    totalProfit += reward * 1000;

    // Log decision details for each hour
    console.log(
      `Hour ${hour}: Platform=${action.platform}, Budget=${action.budgetAdjustment}x, Creative=${action.creativeType}, Age=${action.targetAgeGroup}`,
    );

    // Update state for next iteration
    testState = nextState;
  }

  // Display final performance summary
  console.log(`\n💰 Total Daily Profit: $${totalProfit.toFixed(2)}`);
}

/**
 * Main application entry point
 *
 * Orchestrates the complete training and evaluation pipeline:
 * 1. Parse command-line arguments and environment variables
 * 2. Create and configure the appropriate RL agent
 * 3. Set up training environment and pipeline with observers
 * 4. Warm-start the agent if possible
 * 5. Train the agent for the specified number of episodes
 * 6. Save the trained model
 * 7. Demonstrate the learned policy
 *
 * @returns {Promise<void>} Promise that resolves when training is complete
 */
async function main(): Promise<void> {
  // Display application banner
  printBanner();

  // Parse configuration from CLI args and environment variables
  const { episodes, epsilonDecay, lrDecay, shaping, agentKind, nnOpts, mode, noDemo, loadPath } =
    parseArgs();

  // Route to mode-specific runners
  if (mode === "shadow") {
    const { shadowTrain } = await import("./run/shadowTraining");
    // Optionally load a model and set epsilon low for evaluation-oriented learning
    const agent = createAgent(agentKind, nnOpts, epsilonDecay, lrDecay);
    if (loadPath) await agent.load(loadPath);
    (agent as any).setEpsilon?.(0.1); // prefer exploitation during shadow
    await shadowTrain(episodes, agent);
    return;
  }

  if (mode === "pilot") {
    // Defer to real runner with guardrails
    const { main: realMain } = await import("./run/real");
    // When piloting, prefer a loaded model and exploitation
    if (loadPath) {
      const agent = createAgent(agentKind, nnOpts, epsilonDecay, lrDecay);
      await agent.load(loadPath);
      (agent as any).setEpsilon?.(0);
    }
    await realMain();
    return;
  }

  // Default: simulator mode
  const agent = createAgent(agentKind, nnOpts, epsilonDecay, lrDecay);
  if (loadPath) await agent.load(loadPath);
  const { environment, pipeline, metricsCollector } = setupPipeline(agent, shaping);
  warmStartAgent(agent, environment);
  await pipeline.train(episodes);
  metricsCollector.printSummary();
  await agent.save("final_model.json");
  if (!noDemo) await demonstratePolicy(agent);
}

// Execute main function if this file is run directly (not imported)
if (require.main === module) {
  main().catch(console.error);
}

// Export main function for programmatic usage
export { main };
