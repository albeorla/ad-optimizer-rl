import { AdAction, AdEnvironmentState } from "../types";
import { RLAgent } from "./base";
import {
  ACTIONS,
  BUDGET_STEPS,
  AGE_GROUPS,
  CREATIVE_TYPES,
  PLATFORMS,
  BID_STRATEGIES,
  INTEREST_BUNDLES,
} from "./encoding";

/**
 * Tabular Q-learning agent with ε-greedy policy and experience replay.
 *
 * Uses the deterministic discrete action grid from encoding.ts for consistency
 * between tabular and NN agents. Q-values are stored in a nested Map structure
 * for interpretability and debugging.
 */
export class DQNAgent extends RLAgent {
  private qTable: Map<string, Map<string, number>> = new Map();
  private actionSpace: AdAction[];
  private experienceReplay: Array<{
    state: AdEnvironmentState;
    action: AdAction;
    reward: number;
    nextState: AdEnvironmentState;
    done: boolean;
  }> = [];
  private maxReplaySize: number = 1000;
  private batchSize: number = 32;
  private initialLearningRate: number;
  private lrDecay: number = 0.99;
  private updateCount: number = 0;

  constructor(opts?: {
    learningRate?: number;
    discountFactor?: number;
    epsilonStart?: number;
    epsilonDecay?: number;
    minEpsilon?: number;
    lrDecay?: number;
    maxReplaySize?: number;
    batchSize?: number;
  }) {
    super();
    if (opts?.learningRate !== undefined) this.learningRate = opts.learningRate;
    if (opts?.discountFactor !== undefined)
      this.discountFactor = opts.discountFactor;
    if (opts?.epsilonStart !== undefined) this.epsilon = opts.epsilonStart;
    if (opts?.epsilonDecay !== undefined) this.epsilonDecay = opts.epsilonDecay;
    if (opts?.minEpsilon !== undefined) this.minEpsilon = opts.minEpsilon;
    if (opts?.lrDecay !== undefined) this.lrDecay = opts.lrDecay;
    if (opts?.maxReplaySize !== undefined)
      this.maxReplaySize = opts.maxReplaySize;
    if (opts?.batchSize !== undefined) this.batchSize = opts.batchSize;
    this.initialLearningRate = this.learningRate;
    // Use the deterministic action space from encoding.ts
    this.actionSpace = ACTIONS;
  }

  private pickRandom<T>(arr: readonly T[]): T {
    return arr[Math.floor(Math.random() * arr.length)]!;
  }

  /** Compact, deterministic key for Q-table indexing. */
  private stateToKey(state: AdEnvironmentState): string {
    return JSON.stringify({
      dow: state.dayOfWeek,
      hod: state.hourOfDay,
      budget: Number(state.currentBudget.toFixed(2)),
      age: state.targetAgeGroup,
      creative: state.creativeType,
      platform: state.platform,
    });
  }

  /** Compact, deterministic key for action-specific Q-values. */
  private actionToKey(action: AdAction): string {
    return JSON.stringify({
      budget: action.budgetAdjustment,
      age: action.targetAgeGroup,
      creative: action.creativeType,
      platform: action.platform,
    });
  }

  /** ε-greedy action selection over the tabular Q-values. */
  selectAction(state: AdEnvironmentState): AdAction {
    if (Math.random() < this.epsilon) {
      return this.pickRandom(this.actionSpace);
    } else {
      const stateKey = this.stateToKey(state);
      const stateQValues = this.qTable.get(stateKey);
      if (!stateQValues || stateQValues.size === 0) {
        return this.pickRandom(this.actionSpace);
      }
      let bestAction = this.actionSpace[0]!;
      let bestValue = -Infinity;
      for (const action of this.actionSpace) {
        const actionKey = this.actionToKey(action);
        const qValue = stateQValues.get(actionKey) || 0;
        if (qValue > bestValue) {
          bestValue = qValue;
          bestAction = action;
        }
      }
      return bestAction;
    }
  }

  /** One-step TD update with optional mini-batch replay when buffer is warm. */
  update(
    state: AdEnvironmentState,
    action: AdAction,
    reward: number,
    nextState: AdEnvironmentState,
    done: boolean,
  ): void {
    this.experienceReplay.push({ state, action, reward, nextState, done });
    if (this.experienceReplay.length > this.maxReplaySize)
      this.experienceReplay.shift();

    const stateKey = this.stateToKey(state);
    const actionKey = this.actionToKey(action);
    const nextStateKey = this.stateToKey(nextState);

    if (!this.qTable.has(stateKey)) this.qTable.set(stateKey, new Map());
    if (!this.qTable.has(nextStateKey))
      this.qTable.set(nextStateKey, new Map());

    const currentQ = this.qTable.get(stateKey)!.get(actionKey) || 0;

    let maxNextQ = 0;
    if (!done) {
      const nextStateQValues = this.qTable.get(nextStateKey);
      if (nextStateQValues) {
        for (const qValue of nextStateQValues.values())
          maxNextQ = Math.max(maxNextQ, qValue);
      }
    }

    const newQ =
      currentQ +
      this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);
    this.qTable.get(stateKey)!.set(actionKey, newQ);

    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

    if (this.experienceReplay.length >= 32) this.replayExperience();
  }

  /**
   * Sample a batch from replay and apply TD updates.
   * Uses Fisher-Yates partial shuffle for unique sampling without replacement.
   */
  private replayExperience(): void {
    const n = Math.min(this.batchSize, this.experienceReplay.length);
    if (n === 0) return;

    // Fisher-Yates partial shuffle for unique sampling
    const indices = Array.from(
      { length: this.experienceReplay.length },
      (_, i) => i,
    );
    const batch: typeof this.experienceReplay = [];
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(Math.random() * (indices.length - i));
      [indices[i], indices[j]] = [indices[j]!, indices[i]!];
      batch.push(this.experienceReplay[indices[i]!]!);
    }

    for (const experience of batch) {
      const stateKey = this.stateToKey(experience.state);
      const actionKey = this.actionToKey(experience.action);
      const nextStateKey = this.stateToKey(experience.nextState);
      if (!this.qTable.has(stateKey)) this.qTable.set(stateKey, new Map());
      const currentQ = this.qTable.get(stateKey)!.get(actionKey) || 0;
      let maxNextQ = 0;
      if (!experience.done) {
        const nextStateQValues = this.qTable.get(nextStateKey);
        if (nextStateQValues) {
          for (const qValue of nextStateQValues.values())
            maxNextQ = Math.max(maxNextQ, qValue);
        }
      }
      const tdError =
        experience.reward + this.discountFactor * maxNextQ - currentQ;
      const newQ = currentQ + this.learningRate * tdError;
      this.qTable.get(stateKey)!.set(actionKey, newQ);
    }
    this.updateCount++;
  }

  /** Serialize the Q-table and scheduling params to a JSON file. */
  async save(filepath: string): Promise<void> {
    const data = {
      version: 1,
      timestamp: new Date().toISOString(),
      qTable: Array.from(this.qTable.entries()).map(([state, actions]) => ({
        state,
        actions: Array.from(actions.entries()),
      })),
      hyperparams: {
        epsilon: this.epsilon,
        learningRate: this.learningRate,
        discountFactor: this.discountFactor,
        lrDecay: this.lrDecay,
        minEpsilon: this.minEpsilon,
        epsilonDecay: this.epsilonDecay,
      },
      stats: {
        updateCount: this.updateCount,
        qTableSize: this.getQTableSize(),
        replaySize: this.experienceReplay.length,
      },
    };
    const fs = await import("fs");
    await fs.promises.writeFile(filepath, JSON.stringify(data, null, 2));
    console.log(`Model saved to ${filepath}`);
  }

  /** Load a saved model from JSON file. */
  async load(filepath: string): Promise<void> {
    const fs = await import("fs");
    try {
      const content = await fs.promises.readFile(filepath, "utf8");
      const data = JSON.parse(content);

      // Restore Q-table
      this.qTable.clear();
      for (const entry of data.qTable || []) {
        const stateMap = new Map<string, number>(entry.actions);
        this.qTable.set(entry.state, stateMap);
      }

      // Restore hyperparameters
      if (data.hyperparams) {
        this.epsilon = data.hyperparams.epsilon ?? this.epsilon;
        this.learningRate = data.hyperparams.learningRate ?? this.learningRate;
        this.discountFactor =
          data.hyperparams.discountFactor ?? this.discountFactor;
        this.lrDecay = data.hyperparams.lrDecay ?? this.lrDecay;
        this.minEpsilon = data.hyperparams.minEpsilon ?? this.minEpsilon;
        this.epsilonDecay = data.hyperparams.epsilonDecay ?? this.epsilonDecay;
      }

      console.log(
        `Model loaded from ${filepath} (${this.getQTableSize()} Q-values)`,
      );
    } catch (err) {
      console.error(`Failed to load model from ${filepath}:`, err);
      throw err;
    }
  }

  // Introspection helpers for diagnostics
  getEpsilon(): number {
    return this.epsilon;
  }
  getQTableSize(): number {
    let total = 0;
    for (const m of this.qTable.values()) total += m.size;
    return total;
  }

  /**
   * Warm-start the Q-table with domain knowledge heuristics.
   * Uses valid actions from the deterministic action space.
   */
  seedHeuristics(stateTemplate: AdEnvironmentState): void {
    // Peak engagement hours for social media ads
    const peakHours = [18, 19, 20, 21];
    // High-value action combinations based on domain knowledge
    const goodActions: AdAction[] = [
      // TikTok: Young audience + UGC content works well
      {
        budgetAdjustment: 1.05, // Valid budget from BUDGET_STEPS
        targetAgeGroup: "18-24",
        targetInterests: ["fashion"],
        creativeType: "ugc",
        bidStrategy: "CPC",
        platform: "tiktok",
      },
      // Instagram: 25-34 + lifestyle/product content
      {
        budgetAdjustment: 1.0,
        targetAgeGroup: "25-34",
        targetInterests: ["fashion"],
        creativeType: "product",
        bidStrategy: "CPM",
        platform: "instagram",
      },
      // Conservative budget during testing
      {
        budgetAdjustment: 1.0,
        targetAgeGroup: "25-34",
        targetInterests: ["tech"],
        creativeType: "lifestyle",
        bidStrategy: "CPA",
        platform: "tiktok",
      },
    ];

    // Filter to only valid actions in our action space
    const validActions = goodActions.filter((a) =>
      this.actionSpace.some(
        (validA) =>
          validA.budgetAdjustment === a.budgetAdjustment &&
          validA.targetAgeGroup === a.targetAgeGroup &&
          validA.creativeType === a.creativeType &&
          validA.platform === a.platform &&
          validA.bidStrategy === a.bidStrategy,
      ),
    );

    for (const hod of peakHours) {
      const state = { ...stateTemplate, hourOfDay: hod };
      const sk = this.stateToKey(state);
      if (!this.qTable.has(sk)) this.qTable.set(sk, new Map());
      for (const a of validActions) {
        const ak = this.actionToKey(a);
        // Set a moderate positive value to encourage exploration
        this.qTable.get(sk)!.set(ak, 3.0);
      }
    }

    // Also seed some negative values for known bad combinations
    const badHours = [2, 3, 4]; // Low engagement hours
    for (const hod of badHours) {
      const state = { ...stateTemplate, hourOfDay: hod };
      const sk = this.stateToKey(state);
      if (!this.qTable.has(sk)) this.qTable.set(sk, new Map());
      // Discourage high budget during low-engagement hours
      for (const a of this.actionSpace) {
        if (a.budgetAdjustment > 1.0) {
          const ak = this.actionToKey(a);
          this.qTable.get(sk)!.set(ak, -1.0);
        }
      }
    }
  }

  // Episode end hook: apply learning rate schedule
  onEpisodeEnd(episode: number): void {
    this.learningRate =
      this.initialLearningRate * Math.pow(this.lrDecay, episode);
  }
}
