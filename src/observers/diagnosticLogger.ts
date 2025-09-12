import { TrainingObserver } from "./types";

export class DiagnosticLogger implements TrainingObserver {
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void {
    const eps = metrics?.epsilon ?? "n/a";
    const table = metrics?.qTableSize ?? "n/a";
    const unique = metrics?.uniqueActions ?? "n/a";
    const revenue = metrics?.revenue?.toFixed ? metrics.revenue.toFixed(2) : metrics?.revenue;
    const adSpend = metrics?.adSpend?.toFixed ? metrics.adSpend.toFixed(2) : metrics?.adSpend;
    console.log(`Episode ${episode}:`);
    console.log(`  - Exploration Rate (epsilon): ${eps}`);
    console.log(`  - Q-Table Size: ${table}`);
    console.log(`  - Unique Actions Taken: ${unique}`);
    if (revenue !== undefined && adSpend !== undefined) {
      console.log(`  - Profit Breakdown: +$${revenue} -$${adSpend}`);
    }
    console.log(`  - Total Reward: ${totalReward.toFixed(2)}`);
  }
}

