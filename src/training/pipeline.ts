import { RLAgent } from "../agent/base";
import { AdEnvironmentSimulator } from "../environment/simulator";
import { TrainingObserver } from "../observers/types";

export interface TrainingConfig {
  numEpisodes: number;
  checkpointFrequency?: number; // Save every N episodes (default: 100)
  evaluationFrequency?: number; // Run eval episode every N episodes (default: 50)
  evaluationEpisodes?: number; // Number of eval episodes to average (default: 3)
  checkpointDir?: string; // Directory for checkpoints (default: current dir)
  earlyStoppingPatience?: number; // Stop if no improvement for N evals (default: disabled)
  earlyStoppingMinDelta?: number; // Minimum improvement to reset patience
}

interface EpisodeMetrics {
  episode: number;
  totalReward: number;
  steps: number;
  revenue: number;
  adSpend: number;
  profit: number;
  roas: number;
  epsilon?: number;
  qTableSize?: number;
  replaySize?: number;
  avgLoss?: number;
  isEvaluation: boolean;
}

/**
 * Training controller that connects an RL agent with an environment and
 * dispatches step/episode events to observers.
 *
 * Features:
 * - Periodic evaluation episodes with epsilon=0
 * - Automatic checkpointing
 * - Early stopping based on evaluation performance
 * - Comprehensive metrics tracking
 */
export class TrainingPipeline {
  private agent: RLAgent;
  private environment: AdEnvironmentSimulator;
  private observers: TrainingObserver[] = [];

  // Training history
  private trainingHistory: EpisodeMetrics[] = [];
  private evaluationHistory: EpisodeMetrics[] = [];
  private bestEvalReward = -Infinity;
  private noImprovementCount = 0;

  constructor(agent: RLAgent, environment: AdEnvironmentSimulator) {
    this.agent = agent;
    this.environment = environment;
  }

  /** Register a training observer for episode-level callbacks. */
  addObserver(observer: TrainingObserver): void {
    this.observers.push(observer);
  }

  /** Notify all observers that an episode has completed. */
  notifyObservers(episode: number, totalReward: number, metrics: any): void {
    for (const observer of this.observers) {
      observer.onEpisodeComplete(episode, totalReward, metrics);
    }
  }

  /**
   * Run a single episode and return metrics.
   * If evaluation=true, uses epsilon=0 (pure exploitation).
   */
  private async runEpisode(
    episodeNum: number,
    evaluation: boolean,
  ): Promise<EpisodeMetrics> {
    let state = this.environment.reset();
    let totalReward = 0;
    let stepCount = 0;
    let done = false;
    let revenueSum = 0;
    let adSpendSum = 0;
    let profitSum = 0;
    const uniqueActions = new Set<string>();

    // Save original epsilon for evaluation mode
    const originalEpsilon = (this.agent as any).epsilon;
    if (evaluation) {
      (this.agent as any).epsilon = 0; // Pure exploitation
    }

    while (!done) {
      const action = this.agent.selectAction(state);
      const [nextState, reward, episodeDone, metrics] =
        this.environment.step(action);

      // Only update agent during training, not evaluation
      if (!evaluation) {
        this.agent.update(state, action, reward, nextState, episodeDone);
      }

      totalReward += reward;
      stepCount++;
      state = nextState;
      done = episodeDone;
      revenueSum += metrics.revenue;
      adSpendSum += metrics.adSpend;
      profitSum += metrics.profit;
      uniqueActions.add(
        JSON.stringify({
          budget: action.budgetAdjustment,
          age: action.targetAgeGroup,
          creative: action.creativeType,
          platform: action.platform,
        }),
      );
    }

    // Restore epsilon after evaluation
    if (evaluation) {
      (this.agent as any).epsilon = originalEpsilon;
    }

    // Collect agent diagnostics
    const epsilon = (this.agent as any).getEpsilon?.() ?? undefined;
    const qTableSize = (this.agent as any).getQTableSize?.() ?? undefined;
    const replaySize = (this.agent as any).getReplaySize?.() ?? undefined;
    const avgLoss = (this.agent as any).getLastAvgLoss?.() ?? undefined;

    return {
      episode: episodeNum,
      totalReward,
      steps: stepCount,
      revenue: revenueSum,
      adSpend: adSpendSum,
      profit: profitSum,
      roas: adSpendSum > 0 ? revenueSum / adSpendSum : 0,
      epsilon,
      qTableSize,
      replaySize,
      avgLoss,
      isEvaluation: evaluation,
    };
  }

  /**
   * Run multiple evaluation episodes and return average metrics.
   */
  private async runEvaluation(
    episodeNum: number,
    numEpisodes: number,
  ): Promise<EpisodeMetrics> {
    const results: EpisodeMetrics[] = [];
    for (let i = 0; i < numEpisodes; i++) {
      const metrics = await this.runEpisode(episodeNum, true);
      results.push(metrics);
    }

    // Average the results - build object conditionally to satisfy exactOptionalPropertyTypes
    const avgMetrics: EpisodeMetrics = {
      episode: episodeNum,
      totalReward:
        results.reduce((s, m) => s + m.totalReward, 0) / results.length,
      steps: Math.round(
        results.reduce((s, m) => s + m.steps, 0) / results.length,
      ),
      revenue: results.reduce((s, m) => s + m.revenue, 0) / results.length,
      adSpend: results.reduce((s, m) => s + m.adSpend, 0) / results.length,
      profit: results.reduce((s, m) => s + m.profit, 0) / results.length,
      roas: results.reduce((s, m) => s + m.roas, 0) / results.length,
      isEvaluation: true,
    };

    // Conditionally add optional fields if they exist
    const first = results[0];
    if (first?.epsilon !== undefined) avgMetrics.epsilon = first.epsilon;
    if (first?.qTableSize !== undefined) avgMetrics.qTableSize = first.qTableSize;
    if (first?.replaySize !== undefined) avgMetrics.replaySize = first.replaySize;
    if (first?.avgLoss !== undefined) avgMetrics.avgLoss = first.avgLoss;

    return avgMetrics;
  }

  /** Run N training episodes with evaluation and checkpointing. */
  async train(config: TrainingConfig): Promise<{
    trainingHistory: EpisodeMetrics[];
    evaluationHistory: EpisodeMetrics[];
    earlyStopped: boolean;
  }> {
    const {
      numEpisodes,
      checkpointFrequency = 100,
      evaluationFrequency = 50,
      evaluationEpisodes = 3,
      checkpointDir = ".",
      earlyStoppingPatience,
      earlyStoppingMinDelta = 0.01,
    } = config;

    console.log(`\n🚀 Starting RL Training for ${numEpisodes} episodes...\n`);
    console.log(
      `   Evaluation every ${evaluationFrequency} episodes (${evaluationEpisodes} eval episodes)`,
    );
    console.log(`   Checkpoints every ${checkpointFrequency} episodes\n`);

    let earlyStopped = false;
    this.trainingHistory = [];
    this.evaluationHistory = [];
    this.bestEvalReward = -Infinity;
    this.noImprovementCount = 0;

    for (let episode = 0; episode < numEpisodes; episode++) {
      // Run training episode
      const metrics = await this.runEpisode(episode + 1, false);
      this.trainingHistory.push(metrics);

      // Notify observers
      this.notifyObservers(episode + 1, metrics.totalReward, {
        steps: metrics.steps,
        finalBudget: metrics.adSpend / Math.max(1, metrics.steps),
        epsilon: metrics.epsilon,
        qTableSize: metrics.qTableSize,
        replaySize: metrics.replaySize,
        avgLoss: metrics.avgLoss,
        uniqueActions: 0,
        revenue: metrics.revenue,
        adSpend: metrics.adSpend,
        profit: metrics.profit,
        roas: metrics.roas,
      });

      // Episode end hook for agent (e.g., learning-rate schedule)
      this.agent.onEpisodeEnd(episode + 1);

      // Periodic evaluation
      if ((episode + 1) % evaluationFrequency === 0) {
        const evalMetrics = await this.runEvaluation(
          episode + 1,
          evaluationEpisodes,
        );
        this.evaluationHistory.push(evalMetrics);

        console.log(
          `\n📊 Evaluation at episode ${episode + 1}: ` +
            `Reward=${evalMetrics.totalReward.toFixed(2)}, ` +
            `Profit=${evalMetrics.profit.toFixed(2)}, ` +
            `ROAS=${evalMetrics.roas.toFixed(2)}`,
        );

        // Early stopping check
        if (earlyStoppingPatience !== undefined) {
          if (
            evalMetrics.totalReward >
            this.bestEvalReward + earlyStoppingMinDelta
          ) {
            this.bestEvalReward = evalMetrics.totalReward;
            this.noImprovementCount = 0;
            // Save best model
            await this.agent.save(`${checkpointDir}/best_model.json`);
            console.log("   📈 New best model saved!");
          } else {
            this.noImprovementCount++;
            console.log(
              `   ⚠️ No improvement (${this.noImprovementCount}/${earlyStoppingPatience})`,
            );
            if (this.noImprovementCount >= earlyStoppingPatience) {
              console.log(`\n🛑 Early stopping triggered at episode ${episode + 1}`);
              earlyStopped = true;
              break;
            }
          }
        }
      }

      // Periodic checkpoint
      if ((episode + 1) % checkpointFrequency === 0) {
        await this.agent.save(
          `${checkpointDir}/checkpoint_${episode + 1}.json`,
        );
        console.log(`💾 Checkpoint saved at episode ${episode + 1}`);
      }
    }

    // Final save
    await this.agent.save(`${checkpointDir}/final_model.json`);
    console.log("\n✅ Training Complete!\n");

    // Print summary
    this.printTrainingSummary();

    return {
      trainingHistory: this.trainingHistory,
      evaluationHistory: this.evaluationHistory,
      earlyStopped,
    };
  }

  /** Print a summary of training performance. */
  private printTrainingSummary(): void {
    if (this.trainingHistory.length === 0) return;

    const last10 = this.trainingHistory.slice(-10);
    const avgReward = last10.reduce((s, m) => s + m.totalReward, 0) / last10.length;
    const avgProfit = last10.reduce((s, m) => s + m.profit, 0) / last10.length;
    const avgRoas = last10.reduce((s, m) => s + m.roas, 0) / last10.length;

    console.log("📈 Training Summary (last 10 episodes):");
    console.log(`   Avg Reward: ${avgReward.toFixed(2)}`);
    console.log(`   Avg Profit: ${avgProfit.toFixed(2)}`);
    console.log(`   Avg ROAS: ${avgRoas.toFixed(2)}`);

    if (this.evaluationHistory.length > 0) {
      const lastEval = this.evaluationHistory[this.evaluationHistory.length - 1]!;
      console.log("\n📊 Last Evaluation:");
      console.log(`   Reward: ${lastEval.totalReward.toFixed(2)}`);
      console.log(`   Profit: ${lastEval.profit.toFixed(2)}`);
      console.log(`   ROAS: ${lastEval.roas.toFixed(2)}`);
      console.log(`   Best Eval Reward: ${this.bestEvalReward.toFixed(2)}`);
    }
  }

  /** Get training history for analysis. */
  getTrainingHistory(): EpisodeMetrics[] {
    return this.trainingHistory;
  }

  /** Get evaluation history for analysis. */
  getEvaluationHistory(): EpisodeMetrics[] {
    return this.evaluationHistory;
  }
}
