import { AdAction, AdEnvironmentState } from "../types";
import { RLAgent } from "./base";
import { ACTIONS, encodeState, actionToIndex, indexToAction } from "./encoding";
import { DQNHyperparams, QNet, QNetTorch } from "./nn/qnet";
import { ReplayBuffer, Transition } from "./replay";

type Vec = ReadonlyArray<number> & number[]; // helpful alias

export interface DQNAgentNNOptions {
  epsilonStart?: number;
  epsilonMin?: number;
  epsilonDecay?: number;
  lr?: number;
  gamma?: number;
  batchSize?: number;
  trainFreq?: number;
  targetSync?: number;
  replayCapacity?: number;
  // New options for improved training
  useDoubleDQN?: boolean; // Use Double DQN to reduce overestimation
  useHuberLoss?: boolean; // Use Huber loss instead of MSE
  huberDelta?: number; // Huber loss delta (default: 1.0)
  gradientClip?: number; // Gradient clipping value (default: 10.0)
  tau?: number; // Soft update coefficient (0-1, default: 1.0 for hard update)
  warmupSteps?: number; // Steps before training starts (default: batchSize)
}

/**
 * Neural Network-based DQN Agent with modern improvements.
 *
 * Features:
 * - Double DQN for reduced Q-value overestimation
 * - Huber loss for robustness to outliers
 * - Gradient clipping for training stability
 * - Soft target network updates option
 * - Warmup period before training begins
 */
export class DQNAgentNN extends RLAgent {
  private qNet: QNet | undefined;
  private targetNet: QNet | undefined;
  private hp: DQNHyperparams;
  private stepCounter = 0;
  private trainCounter = 0;
  private replay: ReplayBuffer<Vec>;
  private lastAvgLoss: number | undefined;
  private useDoubleDQN: boolean;
  private tau: number;
  private warmupSteps: number;

  constructor(opts?: DQNAgentNNOptions) {
    super();
    // Hyperparameters with sensible defaults
    const lr = opts?.lr ?? 1e-3;
    const gamma = opts?.gamma ?? 0.95;
    const batchSize = opts?.batchSize ?? 32;
    const trainFreq = opts?.trainFreq ?? 1;
    const targetSync = opts?.targetSync ?? 250;
    const replayCapacity = opts?.replayCapacity ?? 5000;
    this.epsilon = opts?.epsilonStart ?? 1.0;
    this.minEpsilon = opts?.epsilonMin ?? 0.05;
    this.epsilonDecay = opts?.epsilonDecay ?? 0.995;
    this.discountFactor = gamma;
    this.learningRate = lr;

    // New options with defaults
    this.useDoubleDQN = opts?.useDoubleDQN ?? true; // Enable by default
    this.tau = opts?.tau ?? 1.0; // Hard update by default
    this.warmupSteps = opts?.warmupSteps ?? batchSize;

    this.hp = {
      lr,
      gamma,
      batchSize,
      trainFreq,
      targetSync,
      replayCapacity,
      epsilonStart: this.epsilon,
      epsilonMin: this.minEpsilon,
      epsilonDecay: this.epsilonDecay,
      useDoubleDQN: this.useDoubleDQN,
      useHuberLoss: opts?.useHuberLoss ?? true,
      huberDelta: opts?.huberDelta ?? 1.0,
      gradientClip: opts?.gradientClip ?? 10.0,
      tau: this.tau,
    };

    this.replay = new ReplayBuffer<Vec>(replayCapacity);
  }

  private ensureInitialized(inputSize: number): void {
    if (this.qNet && this.targetNet) return;
    const A = ACTIONS.length;
    const netOptions = {
      useHuberLoss: this.hp.useHuberLoss ?? true,
      huberDelta: this.hp.huberDelta ?? 1.0,
      gradientClip: this.hp.gradientClip ?? 10.0,
    };
    this.qNet = new QNetTorch(inputSize, A, 128, 64, this.learningRate, netOptions);
    this.targetNet = new QNetTorch(inputSize, A, 128, 64, this.learningRate, netOptions);
    this.targetNet.copyFrom(this.qNet);
  }

  private pickRandom<T>(arr: readonly T[]): T {
    return arr[Math.floor(Math.random() * arr.length)]!;
  }

  private bestActionIndex(qValues: number[]): number {
    let best = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < qValues.length; i++) {
      if (qValues[i]! > bestVal) {
        bestVal = qValues[i]!;
        best = i;
      }
    }
    return best;
  }

  selectAction(state: AdEnvironmentState): AdAction {
    // ε-greedy
    if (Math.random() < this.epsilon) {
      return this.pickRandom(ACTIONS);
    }
    const s = encodeState(state) as Vec;
    this.ensureInitialized(s.length);
    const q = this.qNet!.forward([s])[0] ?? [];
    const aIdx =
      q.length === ACTIONS.length
        ? this.bestActionIndex(q)
        : Math.floor(Math.random() * ACTIONS.length);
    return indexToAction(aIdx);
  }

  update(
    state: AdEnvironmentState,
    action: AdAction,
    reward: number,
    nextState: AdEnvironmentState,
    done: boolean,
  ): void {
    const s = encodeState(state) as Vec;
    const sp = encodeState(nextState) as Vec;
    this.ensureInitialized(s.length);
    const aIdx = actionToIndex(action);
    if (aIdx < 0) {
      // Fallback if action not found in grid (should not happen with deterministic grid)
      console.warn(`Action not found in grid: ${JSON.stringify(action)}`);
      return;
    }

    const t: Transition<Vec> = { s, aIdx, r: reward, sp, done };
    this.replay.push(t);
    this.stepCounter++;

    // Decay epsilon per step
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

    // Wait for warmup period before training
    if (this.stepCounter < this.warmupSteps) return;

    // Train periodically
    if (
      this.replay.size() >= this.hp.batchSize &&
      this.stepCounter % this.hp.trainFreq === 0
    ) {
      this.trainBatch();
      this.trainCounter++;

      // Target network update
      if (this.trainCounter % this.hp.targetSync === 0) {
        if (this.tau < 1.0 && this.targetNet!.softUpdate) {
          // Soft update: gradually blend weights
          this.targetNet!.softUpdate(this.qNet!, this.tau);
        } else {
          // Hard update: full copy
          this.targetNet!.copyFrom(this.qNet!);
        }
      }
    }
  }

  /**
   * Train on a batch using Double DQN or standard DQN.
   *
   * Double DQN: Uses online network to select actions, target network to evaluate.
   * This reduces overestimation bias compared to standard DQN.
   */
  private trainBatch(): void {
    const batch = this.replay.sample(this.hp.batchSize);
    if (batch.length === 0) return;

    const A = ACTIONS.length;
    const X: number[][] = batch.map((b) => b.s.slice());
    const Xp: number[][] = batch.map((b) => b.sp.slice());

    // Compute TD targets
    const y: number[] = new Array(batch.length);

    if (this.useDoubleDQN) {
      // Double DQN: use online network to select best action,
      // target network to evaluate that action
      const qOnline = this.qNet!.forward(Xp); // [B, A] - for action selection
      const qTarget = this.targetNet!.forward(Xp); // [B, A] - for value estimation

      for (let i = 0; i < batch.length; i++) {
        const b = batch[i]!;
        if (b.done) {
          y[i] = b.r;
        } else {
          // Select best action using online network
          const qOnlineRow = qOnline[i] ?? new Array(A).fill(0);
          let bestAction = 0;
          let bestValue = -Infinity;
          for (let a = 0; a < A; a++) {
            if (qOnlineRow[a]! > bestValue) {
              bestValue = qOnlineRow[a]!;
              bestAction = a;
            }
          }
          // Evaluate that action using target network
          const qTargetRow = qTarget[i] ?? new Array(A).fill(0);
          y[i] = b.r + this.hp.gamma * qTargetRow[bestAction]!;
        }
      }
    } else {
      // Standard DQN: use target network for both selection and evaluation
      const Qp = this.targetNet!.forward(Xp); // [B, A]

      for (let i = 0; i < batch.length; i++) {
        const b = batch[i]!;
        if (b.done) {
          y[i] = b.r;
        } else {
          const qpi = Qp[i] ?? new Array(A).fill(0);
          const maxQp = Math.max(...qpi);
          y[i] = b.r + this.hp.gamma * maxQp;
        }
      }
    }

    // Train network on (states, action indices, targets)
    const actionsIdx = batch.map((b) => b.aIdx);
    this.lastAvgLoss = this.qNet!.trainOnBatch(X, actionsIdx, y);
  }

  async save(filepath: string): Promise<void> {
    if (this.qNet && (this.qNet as any).save) {
      await (this.qNet as any).save(filepath);
    }
  }
  async load(filepath: string): Promise<void> {
    if (this.qNet && (this.qNet as any).load) {
      await (this.qNet as any).load(filepath);
    }
  }

  // Introspection helpers
  getEpsilon(): number {
    return this.epsilon;
  }
  getReplaySize(): number {
    return this.replay.size();
  }
  getLastAvgLoss(): number | undefined {
    return this.lastAvgLoss;
  }
}
