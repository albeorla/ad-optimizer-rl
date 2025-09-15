import { DQNAgent } from "../agent/dqnAgent";
import { ConsoleLogger } from "../observers/consoleLogger";
import { MetricsCollector } from "../observers/metricsCollector";
import { DiagnosticLogger } from "../observers/diagnosticLogger";
import { RealShadowEnvironment } from "../environment/realShadow";
import { AdEnvironmentState } from "../types";

async function shadowTrain(episodes: number = 10) {
  console.log("=".repeat(60));
  console.log("🕶️ SHADOW-MODE TRAINING (Real Data Scaffolding)");
  console.log("=".repeat(60));

  const agent = new DQNAgent({ epsilonDecay: 0.995, lrDecay: 0.99 });
  const env = new RealShadowEnvironment();

  const logger = new ConsoleLogger();
  const metricsCollector = new MetricsCollector();

  console.log(`\n🚀 Starting Shadow Training for ${episodes} episodes...\n`);
  for (let episode = 0; episode < episodes; episode++) {
    let state: AdEnvironmentState = env.reset();
    let totalReward = 0;
    let stepCount = 0;
    let done = false;
    let revenueSum = 0;
    let adSpendSum = 0;
    const uniqueActions = new Set<string>();

    while (!done) {
      const action = agent.selectAction(state);
      const [nextState, reward, episodeDone, metrics] = await env.step(action);
      agent.update(state, action, reward, nextState);
      totalReward += reward;
      stepCount++;
      state = nextState;
      done = episodeDone;
      revenueSum += metrics.revenue;
      adSpendSum += metrics.adSpend;
      uniqueActions.add(
        JSON.stringify({
          budget: action.budgetAdjustment,
          age: action.targetAgeGroup,
          creative: action.creativeType,
          platform: action.platform,
        }),
      );
    }

    // Diagnostics
    const epsilon = (agent as any).getEpsilon?.() ?? undefined;
    const qTableSize = (agent as any).getQTableSize?.() ?? undefined;
    logger.onEpisodeComplete(episode + 1, totalReward, {
      steps: stepCount,
      finalBudget: state.currentBudget,
      platform: state.platform,
      epsilon,
      qTableSize,
      uniqueActions: uniqueActions.size,
      revenue: revenueSum,
      adSpend: adSpendSum,
    });
    metricsCollector.onEpisodeComplete(episode + 1, totalReward, {
      steps: stepCount,
      finalBudget: state.currentBudget,
      platform: state.platform,
      epsilon,
      qTableSize,
      uniqueActions: uniqueActions.size,
      revenue: revenueSum,
      adSpend: adSpendSum,
    });
    agent.onEpisodeEnd(episode + 1);
  }

  metricsCollector.printSummary();
  agent.save("final_model_shadow.json");
}

if (require.main === module) {
  const episodes = Number(
    process.argv.find((a) => a.startsWith("--episodes="))?.split("=")[1] ?? 10,
  );
  shadowTrain(episodes).catch(console.error);
}

export { shadowTrain };
