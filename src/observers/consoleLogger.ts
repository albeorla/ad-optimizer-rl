import { TrainingObserver } from "./types";
// DQN-REFAC TODO:
// - Print NN-specific metrics when available (avgLoss, epsilon, lr, qMax).
// - Keep output compact; align with training pipeline metrics keys.

export class ConsoleLogger implements TrainingObserver {
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void {
    console.log(
      `Episode ${episode} | Total Reward: ${totalReward.toFixed(2)} | Metrics:`,
      metrics,
    );
  }
}
