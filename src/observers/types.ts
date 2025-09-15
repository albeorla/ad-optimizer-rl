/** Interface for training observers to receive episode-level callbacks. */
export interface TrainingObserver {
  /** Called after each episode with scalar reward and arbitrary metrics. */
  onEpisodeComplete(episode: number, totalReward: number, metrics: any): void;
}
// DQN-REFAC NOTE:
// - Consider a richer metrics type including: avgLoss, maxTD, epsilon, lr, targetSyncs.
// - Add optional hooks for onBatchTrained if we want real-time loss logging.
