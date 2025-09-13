export interface TrainingObserver {
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void;
}
// DQN-REFAC TODO:
// - Consider a richer metrics type including: avgLoss, maxTD, epsilon, lr, targetSyncs.
// - Add optional hooks for onBatchTrained if we want real-time loss logging.
