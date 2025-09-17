import { AdAction, AdEnvironmentState } from "../types";
// DQN-REFAC TODO:
// - Base interface is sufficient; no signature changes required.
// - Subclasses will shift from table updates to neural TD training.
// - Consider adding optional hooks: onBatchTrained(loss), onTargetSync().

/**
 * Base class for RL agents.
 *
 * Subclasses implement action selection, learning update, and serialization.
 * Provides basic epsilon/learning-rate controls for runtime adjustments.
 */
export abstract class RLAgent {
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
    nextState: AdEnvironmentState,
    done: boolean,
  ): void;
  abstract save(filepath: string): void | Promise<void>;
  abstract load(filepath: string): void | Promise<void>;

  /** Optional lifecycle hook: override in subclasses if needed */
  onEpisodeEnd(_episode: number): void {}

  // Optional control hooks
  setEpsilon(value: number): void {
    this.epsilon = Math.max(0, Math.min(1, value));
  }
  setLearningRate(value: number): void {
    this.learningRate = Math.max(0, value);
  }
}
