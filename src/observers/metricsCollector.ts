import { TrainingObserver } from "./types";
// DQN-REFAC TODO:
// - Track batch-level stats: avgLoss, maxTD, qValue stats, replay size, targetSync count.
// - Expose rolling windows for dashboarding and anomaly detection.

export class MetricsCollector implements TrainingObserver {
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
          h.totalReward > best.totalReward ? h : best,
        ).episode
      }`,
    );
    console.log(`Total Episodes: ${this.history.length}`);
  }
}
