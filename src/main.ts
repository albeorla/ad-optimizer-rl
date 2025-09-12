import { DQNAgent } from "./agent/dqnAgent";
import { AdEnvironmentSimulator } from "./environment/simulator";
import { TrainingPipeline } from "./training/pipeline";
import { ConsoleLogger } from "./observers/consoleLogger";
import { MetricsCollector } from "./observers/metricsCollector";

async function main() {
  console.log("=".repeat(60));
  console.log("🎯 T-SHIRT AD OPTIMIZATION RL SYSTEM");
  console.log("=".repeat(60));

  const agent = new DQNAgent();
  const environment = new AdEnvironmentSimulator();
  const pipeline = new TrainingPipeline(agent, environment);

  const logger = new ConsoleLogger();
  const metricsCollector = new MetricsCollector();
  pipeline.addObserver(logger);
  pipeline.addObserver(metricsCollector);

  await pipeline.train(50);
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

