import { DQNAgent } from "./agent/dqnAgent";
import { DQNAgentNN } from "./agent/dqnAgentNN";
import { AdEnvironmentSimulator } from "./environment/simulator";
import { TrainingPipeline } from "./training/pipeline";
import { ConsoleLogger } from "./observers/consoleLogger";
import { MetricsCollector } from "./observers/metricsCollector";
import { DiagnosticLogger } from "./observers/diagnosticLogger";

// Utility: Parse CLI args and env vars
function parseArgs() {
  const args = process.argv.slice(2);
  const getArg = (name: string) => {
    const p = args.find((a) => a.startsWith(`--${name}=`));
    return p ? p.split("=")[1] : undefined;
  };
  return {
    episodes: Number(getArg("episodes") ?? process.env.EPISODES ?? 50),
    epsilonDecay: Number(getArg("epsilonDecay") ?? process.env.EPS_DECAY ?? 0.999),
    lrDecay: Number(getArg("lrDecay") ?? process.env.LR_DECAY ?? 0.99),
    shaping: Number(getArg("shaping") ?? process.env.SHAPING ?? 1),
    agentKind: (getArg("agent") ?? process.env.AGENT ?? "tabular").toLowerCase(),
    nnOpts: {
      epsilonStart: Number(getArg("epsilonStart") ?? process.env.EPS_START ?? 1.0),
      epsilonMin: Number(getArg("epsilonMin") ?? process.env.EPS_MIN ?? 0.05),
      epsilonDecay: Number(getArg("epsilonDecay") ?? process.env.EPS_DECAY ?? 0.995),
      lr: Number(getArg("lr") ?? process.env.LR ?? 1e-3),
      gamma: Number(getArg("gamma") ?? process.env.GAMMA ?? 0.95),
      batchSize: Number(getArg("batchSize") ?? process.env.BATCH_SIZE ?? 32),
      trainFreq: Number(getArg("trainFreq") ?? process.env.TRAIN_FREQ ?? 1),
      targetSync: Number(getArg("targetSync") ?? process.env.TARGET_SYNC ?? 250),
      replayCapacity: Number(getArg("replayCap") ?? process.env.REPLAY_CAP ?? 5000),
    } as const,
  };
}

// Factory: Create agent based on kind and options
function createAgent(agentKind: string, nnOpts: any, epsilonDecay: number, lrDecay: number) {
  if (agentKind === "nn") {
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
    return new DQNAgent({ epsilonDecay, lrDecay });
  }
}

// Factory: Create and configure environment and pipeline
function setupPipeline(agent: any, shaping: number) {
  const environment = new AdEnvironmentSimulator({ shapingStrength: shaping });
  const pipeline = new TrainingPipeline(agent, environment);

  const logger = new ConsoleLogger();
  const metricsCollector = new MetricsCollector();
  pipeline.addObserver(logger);
  pipeline.addObserver(metricsCollector);
  pipeline.addObserver(new DiagnosticLogger());

  return { environment, pipeline, metricsCollector };
}

// Warm-start agent if possible, else reset environment
function warmStartAgent(agent: any, environment: AdEnvironmentSimulator) {
  if ((agent as any).seedHeuristics) {
    (agent as any).seedHeuristics(environment.reset());
  } else {
    environment.reset();
  }
}

// Print banner
function printBanner() {
  console.log("=".repeat(60));
  console.log("🎯 T-SHIRT AD OPTIMIZATION RL SYSTEM");
  console.log("=".repeat(60));
}

// Demonstrate learned policy
async function demonstratePolicy(agent: any) {
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

async function main() {
  printBanner();

  const { episodes, epsilonDecay, lrDecay, shaping, agentKind, nnOpts } = parseArgs();
  const agent = createAgent(agentKind, nnOpts, epsilonDecay, lrDecay);
  const { environment, pipeline, metricsCollector } = setupPipeline(agent, shaping);

  warmStartAgent(agent, environment);

  await pipeline.train(episodes);
  metricsCollector.printSummary();
  await agent.save("final_model.json");

  await demonstratePolicy(agent);
}

if (require.main === module) {
  main().catch(console.error);
}

export { main };
