import { DQNAgent } from "./agent/dqnAgent";
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

  const agent = new DQNAgent({ epsilonDecay, lrDecay });
  const environment = new AdEnvironmentSimulator({ shapingStrength: shaping });
  const pipeline = new TrainingPipeline(agent, environment);

  const logger = new ConsoleLogger();
  const metricsCollector = new MetricsCollector();
  pipeline.addObserver(logger);
  pipeline.addObserver(metricsCollector);
  pipeline.addObserver(new DiagnosticLogger());

  // Warm-start seeding by default using environment's initial state
  agent.seedHeuristics(environment.reset());
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
