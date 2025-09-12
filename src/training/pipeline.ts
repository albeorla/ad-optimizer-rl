import { RLAgent } from "../agent/base";
import { AdEnvironmentSimulator } from "../environment/simulator";
import { TrainingObserver } from "../observers/types";

export class TrainingPipeline {
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
        const action = this.agent.selectAction(state);
        const [nextState, reward, episodeDone] = this.environment.step(action);
        this.agent.update(state, action, reward, nextState);
        totalReward += reward;
        stepCount++;
        state = nextState;
        done = episodeDone;
      }

      this.notifyObservers(episode + 1, totalReward, {
        steps: stepCount,
        finalBudget: state.currentBudget,
        platform: state.platform,
      });

      if ((episode + 1) % 100 === 0) {
        this.agent.save(`model_checkpoint_${episode + 1}.json`);
      }
    }

    console.log("\n✅ Training Complete!\n");
  }
}

