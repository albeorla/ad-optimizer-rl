import { AdAction, AdEnvironmentState } from "../types";
import { RLAgent } from "./base";
import { ACTIONS, encodeState, actionToIndex, indexToAction } from "./encoding";
import { DQNHyperparams, QNet, QNetSimple } from "./nn/qnet";
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
}

export class DQNAgentNN extends RLAgent {
  private qNet: QNet | undefined;
  private targetNet: QNet | undefined;
  private hp: DQNHyperparams;
  private stepCounter = 0;
  private trainCounter = 0;
  private replay: ReplayBuffer<Vec>;
  private lastAvgLoss: number | undefined;

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
    };

    this.replay = new ReplayBuffer<Vec>(replayCapacity);
  }

  private ensureInitialized(inputSize: number): void {
    if (this.qNet && this.targetNet) return;
    const A = ACTIONS.length;
    this.qNet = new QNetSimple(inputSize, A, 128, 64, this.learningRate);
    this.targetNet = new QNetSimple(inputSize, A, 128, 64, this.learningRate);
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
    const aIdx = q.length === ACTIONS.length ? this.bestActionIndex(q) : Math.floor(Math.random() * ACTIONS.length);
    return indexToAction(aIdx);
  }

  update(state: AdEnvironmentState, action: AdAction, reward: number, nextState: AdEnvironmentState): void {
    const s = encodeState(state) as Vec;
    const sp = encodeState(nextState) as Vec;
    this.ensureInitialized(s.length);
    const aIdx = actionToIndex(action);
    if (aIdx < 0) {
      // Fallback if action not found in grid (should not happen with deterministic grid)
      return;
    }

    const t: Transition<Vec> = { s, aIdx, r: reward, sp, done: false };
    this.replay.push(t);
    this.stepCounter++;

    // Decay epsilon per step
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

    // Train periodically
    if (this.replay.size() >= this.hp.batchSize && this.stepCounter % this.hp.trainFreq === 0) {
      this.trainBatch();
      this.trainCounter++;
      if (this.trainCounter % this.hp.targetSync === 0) {
        this.targetNet.copyFrom(this.qNet);
      }
    }
  }

  private trainBatch(): void {
    const batch = this.replay.sample(this.hp.batchSize);
    if (batch.length === 0) return;

    // Build state and next-state batches
    const X: number[][] = batch.map((b) => b.s.slice());
    const Xp: number[][] = batch.map((b) => b.sp.slice());
    const Qp = this.targetNet!.forward(Xp); // [B, A]

    // Compute simple TD targets (no grad yet — placeholder)
    const A = ACTIONS.length;
    const y: number[] = new Array(batch.length).fill(0);
    const qsa: number[] = new Array(batch.length).fill(0);
    for (let i = 0; i < batch.length; i++) {
      const b = batch[i]!;
      const qpi = Qp[i] ?? new Array(A).fill(0);
      const maxQp = qpi.reduce((m, v) => (v > m ? v : m), -Infinity);
      y[i] = b.r + this.hp.gamma * (b.done ? 0 : maxQp);
      // qsa is only for reporting loss; QNet will recompute internally during train
      qsa[i] = 0;
    }
    // Train network on (states, action indices, targets)
    const actionsIdx = batch.map((b) => b.aIdx);
    this.lastAvgLoss = this.qNet!.trainOnBatch(X, actionsIdx, y);
  }

  save(filepath: string): void {
    // Placeholder: just log; real impl should persist weights/optimizer state
    console.log(`[DQNAgentNN] save() → ${filepath} (placeholder)`);
  }
  load(filepath: string): void {
    console.log(`[DQNAgentNN] load() ← ${filepath} (placeholder)`);
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
