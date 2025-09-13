import { AdAction, AdEnvironmentState } from "../types";
// DQN-REFAC TODO:
// - Base interface is sufficient; no signature changes required.
// - Subclasses will shift from table updates to neural TD training.
// - Consider adding optional hooks: onBatchTrained(loss), onTargetSync().

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
    nextState: AdEnvironmentState
  ): void;
  abstract save(filepath: string): void;
  abstract load(filepath: string): void;

  // Optional lifecycle hook: override in subclasses if needed
  onEpisodeEnd(_episode: number): void {}
}
