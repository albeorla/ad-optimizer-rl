export interface TrainingObserver {
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void;
}

