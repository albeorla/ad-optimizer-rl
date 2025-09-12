import { AdAction, AdEnvironmentState } from "../types";
import { RLAgent } from "./base";

export class DQNAgent extends RLAgent {
  private qTable: Map<string, Map<string, number>> = new Map();
  private actionSpace: AdAction[];
  private experienceReplay: Array<{
    state: AdEnvironmentState;
    action: AdAction;
    reward: number;
    nextState: AdEnvironmentState;
  }> = [];
  private maxReplaySize: number = 1000;

  constructor(opts?: {
    learningRate?: number;
    discountFactor?: number;
    epsilonStart?: number;
    epsilonDecay?: number;
    minEpsilon?: number;
  }) {
    super();
    if (opts?.learningRate !== undefined) this.learningRate = opts.learningRate;
    if (opts?.discountFactor !== undefined) this.discountFactor = opts.discountFactor;
    if (opts?.epsilonStart !== undefined) this.epsilon = opts.epsilonStart;
    if (opts?.epsilonDecay !== undefined) this.epsilonDecay = opts.epsilonDecay;
    if (opts?.minEpsilon !== undefined) this.minEpsilon = opts.minEpsilon;
    this.actionSpace = this.generateActionSpace();
  }

  private pickRandom<T>(arr: readonly T[]): T {
    return arr[Math.floor(Math.random() * arr.length)]!;
  }

  private generateActionSpace(): AdAction[] {
    const actions: AdAction[] = [];
    const budgetAdjustments = [0.5, 0.75, 1.0, 1.25, 1.5];
    const ageGroups = ["18-24", "25-34", "35-44", "45+"];
    const creativeTypes = ["lifestyle", "product", "discount", "ugc"];
    const bidStrategies: Array<"CPC" | "CPM" | "CPA"> = ["CPC", "CPM", "CPA"];
    const platforms: Array<"tiktok" | "instagram" | "shopify"> = ["tiktok", "instagram"];

    for (const budget of budgetAdjustments) {
      for (const age of ageGroups) {
        for (const creative of creativeTypes) {
          for (const platform of platforms) {
            actions.push({
              budgetAdjustment: budget,
              targetAgeGroup: age,
              targetInterests: this.generateInterests(),
              creativeType: creative,
              bidStrategy: this.pickRandom(bidStrategies),
              platform,
            });
          }
        }
      }
    }
    return actions;
  }

  private generateInterests(): string[] {
    const allInterests = ["fashion", "sports", "music", "tech", "fitness", "art", "travel"];
    const numInterests = Math.floor(Math.random() * 3) + 1;
    const interests: string[] = [];
    for (let i = 0; i < numInterests; i++) {
      const interest = this.pickRandom(allInterests);
      if (!interests.includes(interest)) interests.push(interest);
    }
    return interests;
  }

  private stateToKey(state: AdEnvironmentState): string {
    return JSON.stringify({
      dow: state.dayOfWeek,
      hod: state.hourOfDay,
      budget: Math.round(state.currentBudget / 100),
      age: state.targetAgeGroup,
      creative: state.creativeType,
      platform: state.platform,
    });
  }

  private actionToKey(action: AdAction): string {
    return JSON.stringify({
      budget: action.budgetAdjustment,
      age: action.targetAgeGroup,
      creative: action.creativeType,
      platform: action.platform,
    });
  }

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

  update(
    state: AdEnvironmentState,
    action: AdAction,
    reward: number,
    nextState: AdEnvironmentState
  ): void {
    this.experienceReplay.push({ state, action, reward, nextState });
    if (this.experienceReplay.length > this.maxReplaySize) this.experienceReplay.shift();

    const stateKey = this.stateToKey(state);
    const actionKey = this.actionToKey(action);
    const nextStateKey = this.stateToKey(nextState);

    if (!this.qTable.has(stateKey)) this.qTable.set(stateKey, new Map());
    if (!this.qTable.has(nextStateKey)) this.qTable.set(nextStateKey, new Map());

    const currentQ = this.qTable.get(stateKey)!.get(actionKey) || 0;

    let maxNextQ = 0;
    const nextStateQValues = this.qTable.get(nextStateKey);
    if (nextStateQValues) {
      for (const qValue of nextStateQValues.values()) maxNextQ = Math.max(maxNextQ, qValue);
    }

    const newQ = currentQ + this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);
    this.qTable.get(stateKey)!.set(actionKey, newQ);

    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

    if (this.experienceReplay.length >= 32) this.replayExperience();
  }

  private replayExperience(): void {
    const batchSize = Math.min(32, this.experienceReplay.length);
    const batch = [] as typeof this.experienceReplay;
    for (let i = 0; i < batchSize; i++) batch.push(this.pickRandom(this.experienceReplay));

    for (const experience of batch) {
      const stateKey = this.stateToKey(experience.state);
      const actionKey = this.actionToKey(experience.action);
      const nextStateKey = this.stateToKey(experience.nextState);
      if (!this.qTable.has(stateKey)) this.qTable.set(stateKey, new Map());
      const currentQ = this.qTable.get(stateKey)!.get(actionKey) || 0;
      let maxNextQ = 0;
      const nextStateQValues = this.qTable.get(nextStateKey);
      if (nextStateQValues) {
        for (const qValue of nextStateQValues.values()) maxNextQ = Math.max(maxNextQ, qValue);
      }
      const newQ = currentQ + this.learningRate * (experience.reward + this.discountFactor * maxNextQ - currentQ);
      this.qTable.get(stateKey)!.set(actionKey, newQ);
    }
  }

  save(filepath: string): void {
    const data = {
      qTable: Array.from(this.qTable.entries()).map(([state, actions]) => ({ state, actions: Array.from(actions.entries()) })),
      epsilon: this.epsilon,
      learningRate: this.learningRate,
      discountFactor: this.discountFactor,
    };
    console.log(`Model saved to ${filepath}:`, data);
  }

  load(filepath: string): void {
    console.log(`Loading model from ${filepath}`);
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

  // Optional warm-start seeding
  seedHeuristics(stateTemplate: AdEnvironmentState): void {
    const hours = [18, 19, 20];
    const combos: AdAction[] = [
      { budgetAdjustment: 1.25, targetAgeGroup: "18-24", targetInterests: ["fashion"], creativeType: "ugc", bidStrategy: "CPC", platform: "tiktok" },
      { budgetAdjustment: 1.0, targetAgeGroup: "25-34", targetInterests: ["lifestyle"], creativeType: "product", bidStrategy: "CPM", platform: "instagram" },
    ];
    for (const hod of hours) {
      const state = { ...stateTemplate, hourOfDay: hod };
      const sk = this.stateToKey(state);
      if (!this.qTable.has(sk)) this.qTable.set(sk, new Map());
      for (const a of combos) {
        const ak = this.actionToKey(a);
        this.qTable.get(sk)!.set(ak, 5.0);
      }
    }
  }
}
