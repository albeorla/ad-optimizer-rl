import { DQNAgent } from "./agent/dqnAgent";
import { DQNAgentNN } from "./agent/dqnAgentNN";
import { AdEnvironmentSimulator } from "./environment/simulator";
import { TrainingPipeline } from "./training/pipeline";
import { ConsoleLogger } from "./observers/consoleLogger";
import { MetricsCollector } from "./observers/metricsCollector";
import { DiagnosticLogger } from "./observers/diagnosticLogger";
// DQN-REFAC TODO:
// - Add CLI flags/env for DQN hyperparams: batchSize, lr, gamma, trainFreq, targetSync, replayCap.
// - Wire NN-based DQNAgent behind a flag or separate class; keep tabular for baseline comparisons.
// - Surface and log NN metrics (loss, epsilon, lr) through observers.

async function main() {
  console.log("=".repeat(60));
  console.log("🎯 T-SHIRT AD OPTIMIZATION RL SYSTEM");
  console.log("=".repeat(60));

  // Parse simple CLI args
  const args = process.argv.slice(2);
  const getArg = (name: string) => {
    const p = args.find((a) => a.startsWith(`--${name}=`));
    return p ? p.split("=")[1] : undefined;
  };
  const episodes = Number(getArg("episodes") ?? process.env.EPISODES ?? 50);
  const epsilonDecay = Number(getArg("epsilonDecay") ?? process.env.EPS_DECAY ?? 0.999);
  const lrDecay = Number(getArg("lrDecay") ?? process.env.LR_DECAY ?? 0.99);
  const shaping = Number(getArg("shaping") ?? process.env.SHAPING ?? 1);
  const agentKind = (getArg("agent") ?? process.env.AGENT ?? "tabular").toLowerCase();

  // Optional NN params (only used if agent=nn)
  const nnOpts = {
    epsilonStart: Number(getArg("epsilonStart") ?? process.env.EPS_START ?? 1.0),
    epsilonMin: Number(getArg("epsilonMin") ?? process.env.EPS_MIN ?? 0.05),
    epsilonDecay: Number(getArg("epsilonDecay") ?? process.env.EPS_DECAY ?? 0.995),
    lr: Number(getArg("lr") ?? process.env.LR ?? 1e-3),
    gamma: Number(getArg("gamma") ?? process.env.GAMMA ?? 0.95),
    batchSize: Number(getArg("batchSize") ?? process.env.BATCH_SIZE ?? 32),
    trainFreq: Number(getArg("trainFreq") ?? process.env.TRAIN_FREQ ?? 1),
    targetSync: Number(getArg("targetSync") ?? process.env.TARGET_SYNC ?? 250),
    replayCapacity: Number(getArg("replayCap") ?? process.env.REPLAY_CAP ?? 5000),
  } as const;

  const agent = agentKind === "nn"
    ? new DQNAgentNN({
        epsilonStart: nnOpts.epsilonStart,
        epsilonMin: nnOpts.epsilonMin,
        epsilonDecay: nnOpts.epsilonDecay,
        lr: nnOpts.lr,
        gamma: nnOpts.gamma,
        batchSize: nnOpts.batchSize,
        trainFreq: nnOpts.trainFreq,
        targetSync: nnOpts.targetSync,
        replayCapacity: nnOpts.replayCapacity,
      })
    : new DQNAgent({ epsilonDecay, lrDecay });
  const environment = new AdEnvironmentSimulator({ shapingStrength: shaping });
  const pipeline = new TrainingPipeline(agent, environment);

  const logger = new ConsoleLogger();
  const metricsCollector = new MetricsCollector();
  pipeline.addObserver(logger);
  pipeline.addObserver(metricsCollector);
  pipeline.addObserver(new DiagnosticLogger());

  // Warm-start seeding by default using environment's initial state (tabular only)
  if ((agent as any).seedHeuristics) {
    (agent as any).seedHeuristics(environment.reset());
  } else {
    // Ensure we reset environment once for a fresh starting state
    environment.reset();
  }
  await pipeline.train(episodes);
  metricsCollector.printSummary();
  agent.save("final_model.json");

  console.log("\n🎮 Demonstrating Learned Policy...\n");
  const testEnv = new AdEnvironmentSimulator();
  let testState = testEnv.reset();
  let totalProfit = 0;
  for (let hour = 0; hour < 24; hour++) {
    const action = agent.selectAction(testState);
    const [nextState, reward] = testEnv.step(action);
    totalProfit += reward * 1000; // Denormalize
    console.log(
      `Hour ${hour}: Platform=${action.platform}, Budget=${action.budgetAdjustment}x, Creative=${action.creativeType}, Age=${action.targetAgeGroup}`
    );
    testState = nextState;
  }
  console.log(`\n💰 Total Daily Profit: $${totalProfit.toFixed(2)}`);
}

if (require.main === module) {
  main().catch(console.error);
}

export { main };
