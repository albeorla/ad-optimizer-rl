import { TrainingObserver } from "./types";

export class ConsoleLogger implements TrainingObserver {
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void {
    console.log(
      `Episode ${episode} | Total Reward: ${totalReward.toFixed(2)} | Metrics:`,
      metrics
    );
  }
}

