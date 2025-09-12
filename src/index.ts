// ===============================================
// CORE INTERFACES & TYPES
// ===============================================

/**
 * Represents the observable state of our ad campaigns
 */
interface AdEnvironmentState {
  dayOfWeek: number; // 0-6 (Monday-Sunday)
  hourOfDay: number; // 0-23
  currentBudget: number; // Daily budget in USD
  targetAgeGroup: string; // "18-24", "25-34", "35-44", "45+"
  targetInterests: string[]; // ["fashion", "sports", "music", etc.]
  creativeType: string; // "lifestyle", "product", "discount", "ugc"
  platform: string; // "tiktok", "instagram", "shopify"
  historicalCTR: number; // Click-through rate
  historicalCVR: number; // Conversion rate
  competitorActivity: number; // 0-1 (normalized competitor presence)
  seasonality: number; // 0-1 (seasonal demand factor)
}

/**
 * Actions the RL agent can take to modify campaigns
 */
interface AdAction {
  budgetAdjustment: number; // -50% to +100% multiplier
  targetAgeGroup: string; // New age group to target
  targetInterests: string[]; // New interests to target
  creativeType: string; // Creative strategy to use
  bidStrategy: "CPC" | "CPM" | "CPA"; // Bidding strategy
  platform: "tiktok" | "instagram" | "shopify"; // Platform focus
}

/**
 * Reward metrics from the environment
 */
interface RewardMetrics {
  revenue: number;
  adSpend: number;
  profit: number;
  roas: number; // Return on Ad Spend
  conversions: number;
}

// ===============================================
// ABSTRACT CLASSES (Following SOLID principles)
// ===============================================

/**
 * Abstract base for ad platform APIs (Dependency Inversion Principle)
 */
abstract class AdPlatformAPI {
  abstract updateCampaign(campaignId: string, params: any): Promise<any>;
  abstract getCampaignMetrics(campaignId: string): Promise<RewardMetrics>;
  abstract simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction
  ): RewardMetrics;
}

/**
 * Abstract RL Agent base class (Open/Closed Principle)
 */
abstract class RLAgent {
  protected learningRate: number = 0.01;
  protected discountFactor: number = 0.95;
  protected epsilon: number = 1.0; // Exploration rate
  protected epsilonDecay: number = 0.995;
  protected minEpsilon: number = 0.01;

  abstract selectAction(state: AdEnvironmentState): AdAction;
  abstract update(
    state: AdEnvironmentState,
    action: AdAction,
    reward: number,
    nextState: AdEnvironmentState
  ): void;
  abstract save(filepath: string): void;
  abstract load(filepath: string): void;
}

// ===============================================
// CONCRETE IMPLEMENTATIONS
// ===============================================

/**
 * Mock TikTok Ads API
 */
class MockTikTokAdsAPI extends AdPlatformAPI {
  private campaigns: Map<string, any> = new Map();

  async updateCampaign(campaignId: string, params: any): Promise<any> {
    // Simulate API latency
    await this.simulateLatency();

    this.campaigns.set(campaignId, {
      ...this.campaigns.get(campaignId),
      ...params,
      updatedAt: new Date().toISOString(),
    });

    return { success: true, campaignId };
  }

  async getCampaignMetrics(campaignId: string): Promise<RewardMetrics> {
    await this.simulateLatency();

    // Return simulated metrics
    return this.generateMockMetrics();
  }

  simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction
  ): RewardMetrics {
    // TikTok-specific performance simulation
    // Younger demographics perform better on TikTok
    let baseConversion = 0.02;

    if (action.targetAgeGroup === "18-24") {
      baseConversion *= 1.5;
    } else if (action.targetAgeGroup === "25-34") {
      baseConversion *= 1.2;
    }

    // UGC content performs better on TikTok
    if (action.creativeType === "ugc") {
      baseConversion *= 1.4;
    }

    // Simulate conversions and revenue
    const impressions = Math.floor(
      action.budgetAdjustment * 10000 * Math.random()
    );
    const clicks = Math.floor(impressions * state.historicalCTR);
    const conversions = Math.floor(clicks * baseConversion);
    const revenue = conversions * 29.99; // Average t-shirt price
    const adSpend = action.budgetAdjustment * state.currentBudget;

    return {
      revenue,
      adSpend,
      profit: revenue - adSpend,
      roas: adSpend > 0 ? revenue / adSpend : 0,
      conversions,
    };
  }

  private async simulateLatency(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, Math.random() * 100));
  }

  private generateMockMetrics(): RewardMetrics {
    return {
      revenue: Math.random() * 5000,
      adSpend: Math.random() * 1000,
      profit: Math.random() * 4000,
      roas: 2 + Math.random() * 3,
      conversions: Math.floor(Math.random() * 100),
    };
  }
}

/**
 * Mock Instagram Ads API
 */
class MockInstagramAdsAPI extends AdPlatformAPI {
  async updateCampaign(campaignId: string, params: any): Promise<any> {
    // Simulate Instagram-specific campaign update
    return { success: true, campaignId, platform: "instagram" };
  }

  async getCampaignMetrics(campaignId: string): Promise<RewardMetrics> {
    return this.generateInstagramMetrics();
  }

  simulatePerformance(
    state: AdEnvironmentState,
    action: AdAction
  ): RewardMetrics {
    // Instagram favors lifestyle and product imagery
    let baseConversion = 0.025;

    if (
      action.creativeType === "lifestyle" ||
      action.creativeType === "product"
    ) {
      baseConversion *= 1.3;
    }

    // 25-44 age groups perform well on Instagram
    if (
      action.targetAgeGroup === "25-34" ||
      action.targetAgeGroup === "35-44"
    ) {
      baseConversion *= 1.25;
    }

    const impressions = Math.floor(
      action.budgetAdjustment * 8000 * Math.random()
    );
    const clicks = Math.floor(impressions * state.historicalCTR * 1.1); // Slightly better CTR
    const conversions = Math.floor(clicks * baseConversion);
    const revenue = conversions * 29.99;
    const adSpend = action.budgetAdjustment * state.currentBudget;

    return {
      revenue,
      adSpend,
      profit: revenue - adSpend,
      roas: adSpend > 0 ? revenue / adSpend : 0,
      conversions,
    };
  }

  private generateInstagramMetrics(): RewardMetrics {
    return {
      revenue: Math.random() * 6000,
      adSpend: Math.random() * 1200,
      profit: Math.random() * 4800,
      roas: 2.5 + Math.random() * 3,
      conversions: Math.floor(Math.random() * 120),
    };
  }
}

/**
 * Factory Pattern for creating platform APIs
 */
class AdPlatformFactory {
  private static platforms: Map<string, AdPlatformAPI> = new Map([
    ["tiktok", new MockTikTokAdsAPI()],
    ["instagram", new MockInstagramAdsAPI()],
  ]);

  static getPlatform(platform: string): AdPlatformAPI {
    const api = this.platforms.get(platform);
    if (!api) {
      throw new Error(`Platform ${platform} not supported`);
    }
    return api;
  }

  static registerPlatform(name: string, api: AdPlatformAPI): void {
    this.platforms.set(name, api);
  }
}

/**
 * Deep Q-Learning Agent Implementation
 */
class DQNAgent extends RLAgent {
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
    const platforms: Array<"tiktok" | "instagram" | "shopify"> = [
      "tiktok",
      "instagram",
    ];

    // Generate a subset of possible actions to keep the space manageable
    for (const budget of budgetAdjustments) {
      for (const age of ageGroups) {
        for (const creative of creativeTypes) {
          for (const platform of platforms) {
            actions.push({
              budgetAdjustment: budget,
              targetAgeGroup: age,
              targetInterests: this.generateInterests(),
              creativeType: creative,
              bidStrategy:
                bidStrategies[Math.floor(Math.random() * bidStrategies.length)],
              platform: platform,
            });
          }
        }
      }
    }

    return actions;
  }

  private generateInterests(): string[] {
    const allInterests = [
      "fashion",
      "sports",
      "music",
      "tech",
      "fitness",
      "art",
      "travel",
    ];
    const numInterests = Math.floor(Math.random() * 3) + 1;
    const interests: string[] = [];

    for (let i = 0; i < numInterests; i++) {
      const interest =
        allInterests[Math.floor(Math.random() * allInterests.length)];
      if (!interests.includes(interest)) {
        interests.push(interest);
      }
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
    // Epsilon-greedy strategy
    if (Math.random() < this.epsilon) {
      // Exploration: random action
      return this.actionSpace[
        Math.floor(Math.random() * this.actionSpace.length)
      ];
    } else {
      // Exploitation: best known action
      const stateKey = this.stateToKey(state);
      const stateQValues = this.qTable.get(stateKey);

      if (!stateQValues || stateQValues.size === 0) {
        // No knowledge about this state yet
        return this.actionSpace[
          Math.floor(Math.random() * this.actionSpace.length)
        ];
      }

      // Find action with highest Q-value
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
    // Store experience for replay
    this.experienceReplay.push({ state, action, reward, nextState });
    if (this.experienceReplay.length > this.maxReplaySize) {
      this.experienceReplay.shift();
    }

    // Update Q-table
    const stateKey = this.stateToKey(state);
    const actionKey = this.actionToKey(action);
    const nextStateKey = this.stateToKey(nextState);

    // Initialize Q-values if needed
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, new Map());
    }
    if (!this.qTable.has(nextStateKey)) {
      this.qTable.set(nextStateKey, new Map());
    }

    const currentQ = this.qTable.get(stateKey)!.get(actionKey) || 0;

    // Find max Q-value for next state
    let maxNextQ = 0;
    const nextStateQValues = this.qTable.get(nextStateKey);
    if (nextStateQValues) {
      for (const qValue of nextStateQValues.values()) {
        maxNextQ = Math.max(maxNextQ, qValue);
      }
    }

    // Q-learning update rule
    const newQ =
      currentQ +
      this.learningRate * (reward + this.discountFactor * maxNextQ - currentQ);
    this.qTable.get(stateKey)!.set(actionKey, newQ);

    // Decay epsilon
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

    // Experience replay (sample batch and update)
    if (this.experienceReplay.length >= 32) {
      this.replayExperience();
    }
  }

  private replayExperience(): void {
    const batchSize = Math.min(32, this.experienceReplay.length);
    const batch = [];

    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * this.experienceReplay.length);
      batch.push(this.experienceReplay[idx]);
    }

    for (const experience of batch) {
      const stateKey = this.stateToKey(experience.state);
      const actionKey = this.actionToKey(experience.action);
      const nextStateKey = this.stateToKey(experience.nextState);

      if (!this.qTable.has(stateKey)) {
        this.qTable.set(stateKey, new Map());
      }

      const currentQ = this.qTable.get(stateKey)!.get(actionKey) || 0;

      let maxNextQ = 0;
      const nextStateQValues = this.qTable.get(nextStateKey);
      if (nextStateQValues) {
        for (const qValue of nextStateQValues.values()) {
          maxNextQ = Math.max(maxNextQ, qValue);
        }
      }

      const newQ =
        currentQ +
        this.learningRate *
          (experience.reward + this.discountFactor * maxNextQ - currentQ);
      this.qTable.get(stateKey)!.set(actionKey, newQ);
    }
  }

  save(filepath: string): void {
    const data = {
      qTable: Array.from(this.qTable.entries()).map(([state, actions]) => ({
        state,
        actions: Array.from(actions.entries()),
      })),
      epsilon: this.epsilon,
      learningRate: this.learningRate,
      discountFactor: this.discountFactor,
    };
    console.log(`Model saved to ${filepath}:`, data);
  }

  load(filepath: string): void {
    console.log(`Loading model from ${filepath}`);
    // In a real implementation, this would load from disk
  }
}

/**
 * Environment Simulator (Adapter Pattern)
 */
class AdEnvironmentSimulator {
  private platforms: Map<string, AdPlatformAPI>;
  private currentState: AdEnvironmentState;
  private timeStep: number = 0;

  constructor() {
    this.platforms = new Map([
      ["tiktok", new MockTikTokAdsAPI()],
      ["instagram", new MockInstagramAdsAPI()],
    ]);

    this.currentState = this.generateInitialState();
  }

  private generateInitialState(): AdEnvironmentState {
    return {
      dayOfWeek: 0,
      hourOfDay: 12,
      currentBudget: 500,
      targetAgeGroup: "25-34",
      targetInterests: ["fashion", "lifestyle"],
      creativeType: "product",
      platform: "tiktok",
      historicalCTR: 0.02,
      historicalCVR: 0.01,
      competitorActivity: 0.5,
      seasonality: 0.7,
    };
  }

  reset(): AdEnvironmentState {
    this.timeStep = 0;
    this.currentState = this.generateInitialState();
    return this.currentState;
  }

  step(action: AdAction): [AdEnvironmentState, number, boolean] {
    // Apply action to environment
    const platform = this.platforms.get(action.platform);
    if (!platform) {
      throw new Error(`Platform ${action.platform} not found`);
    }

    // Simulate performance based on action
    const metrics = platform.simulatePerformance(this.currentState, action);

    // Calculate reward (profit with some normalization)
    const reward = metrics.profit / 1000; // Normalize to reasonable range

    // Update state based on action and time progression
    this.currentState = this.updateState(action);
    this.timeStep++;

    // Episode ends after 24 hours (24 steps)
    const done = this.timeStep >= 24;

    return [this.currentState, reward, done];
  }

  private updateState(action: AdAction): AdEnvironmentState {
    const newState = { ...this.currentState };

    // Progress time
    newState.hourOfDay = (newState.hourOfDay + 1) % 24;
    if (newState.hourOfDay === 0) {
      newState.dayOfWeek = (newState.dayOfWeek + 1) % 7;
    }

    // Update based on action
    newState.currentBudget =
      this.currentState.currentBudget * action.budgetAdjustment;
    newState.targetAgeGroup = action.targetAgeGroup;
    newState.targetInterests = action.targetInterests;
    newState.creativeType = action.creativeType;
    newState.platform = action.platform;

    // Simulate CTR/CVR drift
    newState.historicalCTR = Math.max(
      0.001,
      newState.historicalCTR + (Math.random() - 0.5) * 0.002
    );
    newState.historicalCVR = Math.max(
      0.001,
      newState.historicalCVR + (Math.random() - 0.5) * 0.001
    );

    // Update competitive and seasonal factors
    newState.competitorActivity = Math.min(
      1,
      Math.max(0, newState.competitorActivity + (Math.random() - 0.5) * 0.1)
    );
    newState.seasonality =
      0.7 + 0.3 * Math.sin((this.timeStep / 168) * Math.PI * 2); // Weekly cycle

    return newState;
  }
}

/**
 * Observer Pattern for Training Monitoring
 */
interface TrainingObserver {
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void;
}

class ConsoleLogger implements TrainingObserver {
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void {
    console.log(
      `Episode ${episode} | Total Reward: ${totalReward.toFixed(2)} | Metrics:`,
      metrics
    );
  }
}

class MetricsCollector implements TrainingObserver {
  private history: Array<{
    episode: number;
    totalReward: number;
    metrics: any;
  }> = [];

  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void {
    this.history.push({ episode, totalReward, metrics });
  }

  getHistory() {
    return this.history;
  }

  printSummary(): void {
    if (this.history.length === 0) return;

    const last10 = this.history.slice(-10);
    const avgReward =
      last10.reduce((sum, h) => sum + h.totalReward, 0) / last10.length;

    console.log("\n=== Training Summary ===");
    console.log(`Average Reward (last 10 episodes): ${avgReward.toFixed(2)}`);
    console.log(
      `Best Episode: ${
        this.history.reduce((best, h) =>
          h.totalReward > best.totalReward ? h : best
        ).episode
      }`
    );
    console.log(`Total Episodes: ${this.history.length}`);
  }
}

/**
 * Main Training Pipeline (Command Pattern)
 */
class TrainingPipeline {
  private agent: RLAgent;
  private environment: AdEnvironmentSimulator;
  private observers: TrainingObserver[] = [];

  constructor(agent: RLAgent, environment: AdEnvironmentSimulator) {
    this.agent = agent;
    this.environment = environment;
  }

  addObserver(observer: TrainingObserver): void {
    this.observers.push(observer);
  }

  notifyObservers(episode: number, totalReward: number, metrics: any): void {
    for (const observer of this.observers) {
      observer.onEpisodeComplete(episode, totalReward, metrics);
    }
  }

  async train(numEpisodes: number): Promise<void> {
    console.log(`\n🚀 Starting RL Training for ${numEpisodes} episodes...\n`);

    for (let episode = 0; episode < numEpisodes; episode++) {
      let state = this.environment.reset();
      let totalReward = 0;
      let stepCount = 0;
      let done = false;

      while (!done) {
        // Agent selects action
        const action = this.agent.selectAction(state);

        // Environment responds
        const [nextState, reward, episodeDone] = this.environment.step(action);

        // Agent learns
        this.agent.update(state, action, reward, nextState);

        // Update tracking
        totalReward += reward;
        stepCount++;
        state = nextState;
        done = episodeDone;
      }

      // Notify observers
      this.notifyObservers(episode + 1, totalReward, {
        steps: stepCount,
        finalBudget: state.currentBudget,
        platform: state.platform,
      });

      // Periodic model saving
      if ((episode + 1) % 100 === 0) {
        this.agent.save(`model_checkpoint_${episode + 1}.json`);
      }
    }

    console.log("\n✅ Training Complete!\n");
  }
}

// ===============================================
// MAIN EXECUTION
// ===============================================

async function main() {
  console.log("=".repeat(60));
  console.log("🎯 T-SHIRT AD OPTIMIZATION RL SYSTEM");
  console.log("=".repeat(60));

  // Initialize components
  const agent = new DQNAgent();
  const environment = new AdEnvironmentSimulator();
  const pipeline = new TrainingPipeline(agent, environment);

  // Add observers
  const logger = new ConsoleLogger();
  const metricsCollector = new MetricsCollector();
  pipeline.addObserver(logger);
  pipeline.addObserver(metricsCollector);

  // Run training
  await pipeline.train(50); // Train for 50 episodes

  // Print summary
  metricsCollector.printSummary();

  // Save final model
  agent.save("final_model.json");

  // Demonstrate learned policy
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

// Execute if running directly
if (require.main === module) {
  main().catch(console.error);
}

// ===============================================
// EXPORT FOR TESTING
// ===============================================

export {
  AdEnvironmentState,
  AdAction,
  RewardMetrics,
  AdPlatformAPI,
  RLAgent,
  DQNAgent,
  AdEnvironmentSimulator,
  TrainingPipeline,
  MockTikTokAdsAPI,
  MockInstagramAdsAPI,
  AdPlatformFactory,
  ConsoleLogger,
  MetricsCollector,
};
