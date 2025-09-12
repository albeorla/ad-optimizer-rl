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

  constructor() {
    super();
    this.actionSpace = this.generateActionSpace();
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
              bidStrategy: bidStrategies[Math.floor(Math.random() * bidStrategies.length)],
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
      const interest = allInterests[Math.floor(Math.random() * allInterests.length)];
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
      return this.actionSpace[Math.floor(Math.random() * this.actionSpace.length)];
    } else {
      const stateKey = this.stateToKey(state);
      const stateQValues = this.qTable.get(stateKey);
      if (!stateQValues || stateQValues.size === 0) {
        return this.actionSpace[Math.floor(Math.random() * this.actionSpace.length)];
      }
      let bestAction = this.actionSpace[0];
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
    for (let i = 0; i < batchSize; i++) batch.push(this.experienceReplay[Math.floor(Math.random() * this.experienceReplay.length)]);

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
}

