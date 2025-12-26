/**
 * Conservative Q-Learning (CQL) Agent for Safe Offline RL.
 *
 * In production ad tech, exploration is expensive - every suboptimal bid wastes
 * real advertising budget. CQL addresses this by learning pessimistic value
 * estimates that lower-bound the true policy value, enabling safe deployment.
 *
 * Key Insight: Standard Q-learning overestimates OOD (Out-of-Distribution)
 * actions because max_a Q(s,a) cherry-picks positive noise for rarely-seen
 * actions. CQL penalizes Q-values for actions not in the training data.
 *
 * CQL Loss = Standard TD Loss + α * CQL Regularizer
 *
 * CQL Regularizer = E_s[log Σ_a exp(Q(s,a)) - E_{a~π_β}[Q(s,a)]]
 *
 * This pushes down Q-values for all actions (logsumexp term) while pushing up
 * Q-values for actions seen in the dataset (expectation under behavior policy).
 *
 * References:
 * - Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning"
 * - Levine et al., "Offline Reinforcement Learning: Tutorial, Review, and
 *   Perspectives on Open Problems"
 */

import { AdAction, AdEnvironmentState } from '../types';
import { RLAgent } from './base';
import { ACTIONS, encodeState, actionToIndex, indexToAction } from './encoding';
import { QNetTorch, QNet } from './nn/qnet';
import { ReplayBuffer, Transition, PrioritizedReplayBuffer } from './replay';

type Vec = ReadonlyArray<number> & number[];

/**
 * Configuration for CQL Agent.
 */
export interface CQLAgentOptions {
  // Epsilon-greedy parameters
  epsilonStart?: number;
  epsilonMin?: number;
  epsilonDecay?: number;

  // Learning parameters
  lr?: number;
  gamma?: number;
  batchSize?: number;
  trainFreq?: number;
  targetSync?: number;
  replayCapacity?: number;

  // CQL-specific parameters
  /** CQL regularization strength (α). Higher = more conservative. */
  cqlAlpha?: number;
  /** Temperature for logsumexp (τ). Lower = sharper penalty on max Q. */
  cqlTemperature?: number;
  /** Minimum Q-value allowed (prevents collapse). */
  minQValue?: number;
  /** Whether to use importance sampling for CQL. */
  useCqlImportanceSampling?: boolean;
  /** Lagrange multiplier for automatic α tuning. */
  cqlLagrangeThreshold?: number;
  /** Whether to adaptively tune α. */
  adaptiveAlpha?: boolean;

  // Double DQN
  useDoubleDQN?: boolean;

  // Huber loss
  useHuberLoss?: boolean;
  huberDelta?: number;

  // Gradient clipping
  gradientClip?: number;

  // Target network
  tau?: number;

  // Warmup
  warmupSteps?: number;

  // Prioritized replay
  usePrioritizedReplay?: boolean;
  priorityAlpha?: number;
  priorityBetaStart?: number;
  priorityBetaEnd?: number;
}

/**
 * CQL Training Metrics.
 */
export interface CQLMetrics {
  tdLoss: number;
  cqlLoss: number;
  totalLoss: number;
  meanQValue: number;
  maxQValue: number;
  minQValue: number;
  cqlAlpha: number;
  conservatismGap: number;
}

/**
 * Conservative Q-Learning Agent.
 *
 * Extends DQN with CQL regularization for safe offline learning.
 * The agent learns pessimistic Q-values that provide a lower bound
 * on the true policy value, enabling safe deployment without
 * risky online exploration.
 */
export class CQLAgent extends RLAgent {
  private qNet: QNet | undefined;
  private targetNet: QNet | undefined;
  private replay: ReplayBuffer<Vec> | PrioritizedReplayBuffer<Vec>;

  // Counters
  private stepCounter = 0;
  private trainCounter = 0;

  // Hyperparameters
  private lr: number;
  private gamma: number;
  private batchSize: number;
  private trainFreq: number;
  private targetSync: number;
  private warmupSteps: number;

  // CQL parameters
  private cqlAlpha: number;
  private cqlTemperature: number;
  private minQValue: number;
  private useCqlImportanceSampling: boolean;
  private adaptiveAlpha: boolean;
  private cqlLagrangeThreshold: number;
  private logAlpha: number; // For adaptive alpha

  // Double DQN
  private useDoubleDQN: boolean;

  // Target network
  private tau: number;

  // Network options
  private netOptions: { useHuberLoss: boolean; huberDelta: number; gradientClip: number };

  // Metrics
  private lastMetrics: CQLMetrics | undefined;

  // Prioritized replay
  private usePrioritizedReplay: boolean;
  private priorityBeta: number;
  private priorityBetaEnd: number;

  constructor(opts: CQLAgentOptions = {}) {
    super();

    // Epsilon-greedy
    this.epsilon = opts.epsilonStart ?? 0.3; // Lower exploration for offline
    this.minEpsilon = opts.epsilonMin ?? 0.01;
    this.epsilonDecay = opts.epsilonDecay ?? 0.999;

    // Learning
    this.lr = opts.lr ?? 3e-4;
    this.gamma = opts.gamma ?? 0.99;
    this.batchSize = opts.batchSize ?? 256; // Larger batches for stability
    this.trainFreq = opts.trainFreq ?? 1;
    this.targetSync = opts.targetSync ?? 100;
    this.warmupSteps = opts.warmupSteps ?? this.batchSize;
    this.learningRate = this.lr;
    this.discountFactor = this.gamma;

    // CQL-specific
    this.cqlAlpha = opts.cqlAlpha ?? 1.0;
    this.cqlTemperature = opts.cqlTemperature ?? 1.0;
    this.minQValue = opts.minQValue ?? -100.0;
    this.useCqlImportanceSampling = opts.useCqlImportanceSampling ?? false;
    this.adaptiveAlpha = opts.adaptiveAlpha ?? false;
    this.cqlLagrangeThreshold = opts.cqlLagrangeThreshold ?? 10.0;
    this.logAlpha = Math.log(this.cqlAlpha);

    // Double DQN
    this.useDoubleDQN = opts.useDoubleDQN ?? true;

    // Target network
    this.tau = opts.tau ?? 0.005; // Soft update by default for CQL

    // Network options
    this.netOptions = {
      useHuberLoss: opts.useHuberLoss ?? true,
      huberDelta: opts.huberDelta ?? 1.0,
      gradientClip: opts.gradientClip ?? 1.0, // Tighter clipping for CQL
    };

    // Replay buffer
    this.usePrioritizedReplay = opts.usePrioritizedReplay ?? true;
    this.priorityBeta = opts.priorityBetaStart ?? 0.4;
    this.priorityBetaEnd = opts.priorityBetaEnd ?? 1.0;

    const replayCapacity = opts.replayCapacity ?? 100000;
    if (this.usePrioritizedReplay) {
      this.replay = new PrioritizedReplayBuffer<Vec>(replayCapacity, opts.priorityAlpha ?? 0.6);
    } else {
      this.replay = new ReplayBuffer<Vec>(replayCapacity);
    }
  }

  private ensureInitialized(inputSize: number): void {
    if (this.qNet && this.targetNet) return;
    const A = ACTIONS.length;
    this.qNet = new QNetTorch(inputSize, A, 256, 128, this.lr, this.netOptions);
    this.targetNet = new QNetTorch(inputSize, A, 256, 128, this.lr, this.netOptions);
    this.targetNet.copyFrom(this.qNet);
  }

  selectAction(state: AdEnvironmentState): AdAction {
    // CQL uses lower exploration since it's designed for offline data
    if (Math.random() < this.epsilon) {
      return ACTIONS[Math.floor(Math.random() * ACTIONS.length)]!;
    }

    const s = encodeState(state) as Vec;
    this.ensureInitialized(s.length);

    const q = this.qNet!.forward([s])[0] ?? [];
    if (q.length !== ACTIONS.length) {
      return ACTIONS[Math.floor(Math.random() * ACTIONS.length)]!;
    }

    // Select action with highest Q-value
    let bestIdx = 0;
    let bestQ = -Infinity;
    for (let i = 0; i < q.length; i++) {
      if (q[i]! > bestQ) {
        bestQ = q[i]!;
        bestIdx = i;
      }
    }

    return indexToAction(bestIdx);
  }

  update(
    state: AdEnvironmentState,
    action: AdAction,
    reward: number,
    nextState: AdEnvironmentState,
    done: boolean
  ): void {
    const s = encodeState(state) as Vec;
    const sp = encodeState(nextState) as Vec;
    this.ensureInitialized(s.length);

    const aIdx = actionToIndex(action);
    if (aIdx < 0) return;

    const t: Transition<Vec> = { s, aIdx, r: reward, sp, done };
    this.replay.push(t);
    this.stepCounter++;

    // Decay epsilon
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);

    // Warmup period
    if (this.stepCounter < this.warmupSteps) return;

    // Train periodically
    if (this.replay.size() >= this.batchSize && this.stepCounter % this.trainFreq === 0) {
      this.trainBatch();
      this.trainCounter++;

      // Target network update (soft)
      if (this.trainCounter % this.targetSync === 0) {
        this.softUpdateTarget();
      }

      // Anneal priority beta
      if (this.usePrioritizedReplay) {
        const progress = Math.min(1.0, this.trainCounter / 100000);
        this.priorityBeta = this.priorityBeta + progress * (this.priorityBetaEnd - this.priorityBeta);
      }
    }
  }

  /**
   * Train on a batch with CQL regularization.
   */
  private trainBatch(): void {
    let batch: Transition<Vec>[];
    let importanceWeights: number[] | undefined;

    if (this.usePrioritizedReplay) {
      const prb = this.replay as PrioritizedReplayBuffer<Vec>;
      const sampled = prb.sample(this.batchSize);
      batch = sampled.transitions;
      importanceWeights = sampled.weights;
    } else {
      batch = (this.replay as ReplayBuffer<Vec>).sample(this.batchSize);
    }

    if (batch.length === 0) return;

    const A = ACTIONS.length;
    const X: number[][] = batch.map((b) => b.s.slice());
    const Xp: number[][] = batch.map((b) => b.sp.slice());

    // Forward pass
    const qValues = this.qNet!.forward(X); // [B, A]
    const qValuesNext = this.targetNet!.forward(Xp); // [B, A]

    // Compute TD targets
    const tdTargets: number[] = new Array(batch.length);
    const tdErrors: number[] = new Array(batch.length);

    if (this.useDoubleDQN) {
      const qOnlineNext = this.qNet!.forward(Xp);
      for (let i = 0; i < batch.length; i++) {
        const b = batch[i]!;
        if (b.done) {
          tdTargets[i] = b.r;
        } else {
          // Double DQN: select action with online, evaluate with target
          const qOnlineRow = qOnlineNext[i] ?? [];
          let bestAction = 0;
          let bestValue = -Infinity;
          for (let a = 0; a < A; a++) {
            if ((qOnlineRow[a] ?? 0) > bestValue) {
              bestValue = qOnlineRow[a] ?? 0;
              bestAction = a;
            }
          }
          const qTargetRow = qValuesNext[i] ?? [];
          tdTargets[i] = b.r + this.gamma * (qTargetRow[bestAction] ?? 0);
        }
        tdErrors[i] = Math.abs(tdTargets[i]! - (qValues[i]?.[b.aIdx] ?? 0));
      }
    } else {
      for (let i = 0; i < batch.length; i++) {
        const b = batch[i]!;
        if (b.done) {
          tdTargets[i] = b.r;
        } else {
          const qRow = qValuesNext[i] ?? [];
          const maxQ = Math.max(...qRow.map((q) => q ?? -Infinity));
          tdTargets[i] = b.r + this.gamma * maxQ;
        }
        tdErrors[i] = Math.abs(tdTargets[i]! - (qValues[i]?.[b.aIdx] ?? 0));
      }
    }

    // Compute CQL loss components
    const cqlMetrics = this.computeCQLLoss(qValues, batch);

    // Combined loss (network handles TD loss internally)
    // We add CQL penalty to TD targets to incorporate conservatism
    const conservativeTargets = tdTargets.map((target, i) => {
      const cqlPenalty = this.cqlAlpha * cqlMetrics.cqlLossPerSample[i]!;
      return target - cqlPenalty;
    });

    // Train network
    const actionsIdx = batch.map((b) => b.aIdx);
    const tdLoss = this.qNet!.trainOnBatch(X, actionsIdx, conservativeTargets, importanceWeights);

    // Update priorities for PER
    if (this.usePrioritizedReplay) {
      const prb = this.replay as PrioritizedReplayBuffer<Vec>;
      // Use combined TD + CQL error for priorities
      const combinedErrors = tdErrors.map(
        (td, i) => td + this.cqlAlpha * Math.abs(cqlMetrics.cqlLossPerSample[i]!)
      );
      prb.updatePriorities(batch.map((_, i) => i), combinedErrors);
    }

    // Adaptive alpha tuning
    if (this.adaptiveAlpha) {
      this.updateAdaptiveAlpha(cqlMetrics.conservatismGap);
    }

    // Store metrics
    this.lastMetrics = {
      tdLoss,
      cqlLoss: cqlMetrics.cqlLoss,
      totalLoss: tdLoss + this.cqlAlpha * cqlMetrics.cqlLoss,
      meanQValue: cqlMetrics.meanQ,
      maxQValue: cqlMetrics.maxQ,
      minQValue: cqlMetrics.minQ,
      cqlAlpha: this.cqlAlpha,
      conservatismGap: cqlMetrics.conservatismGap,
    };
  }

  /**
   * Compute CQL regularization loss.
   *
   * CQL Loss = E_s[log Σ_a exp(Q(s,a)/τ) - E_{a~β}[Q(s,a)]]
   *
   * The first term (logsumexp) pushes down all Q-values.
   * The second term pushes up Q-values for actions in the dataset.
   * The difference creates a conservative gap.
   */
  private computeCQLLoss(
    qValues: number[][],
    batch: Transition<Vec>[]
  ): {
    cqlLoss: number;
    cqlLossPerSample: number[];
    conservatismGap: number;
    meanQ: number;
    maxQ: number;
    minQ: number;
  } {
    const cqlLossPerSample: number[] = [];
    let totalLogsumexp = 0;
    let totalDataQ = 0;
    let allQs: number[] = [];

    for (let i = 0; i < batch.length; i++) {
      const qRow = qValues[i] ?? [];
      const b = batch[i]!;

      // Logsumexp over all actions (soft maximum)
      const scaledQ = qRow.map((q) => (q ?? 0) / this.cqlTemperature);
      const maxScaledQ = Math.max(...scaledQ);
      const logsumexp =
        maxScaledQ +
        Math.log(scaledQ.reduce((sum, q) => sum + Math.exp(q - maxScaledQ), 0)) *
        this.cqlTemperature;

      // Q-value for the action actually taken (in dataset)
      const dataQ = qRow[b.aIdx] ?? 0;

      // CQL loss for this sample: push down logsumexp, push up data Q
      const cqlLossSample = logsumexp - dataQ;
      cqlLossPerSample.push(cqlLossSample);

      totalLogsumexp += logsumexp;
      totalDataQ += dataQ;
      allQs = allQs.concat(qRow.filter((q): q is number => q !== undefined));
    }

    const n = batch.length;
    const cqlLoss = (totalLogsumexp - totalDataQ) / n;
    const conservatismGap = totalLogsumexp / n - totalDataQ / n;

    return {
      cqlLoss,
      cqlLossPerSample,
      conservatismGap,
      meanQ: allQs.length > 0 ? allQs.reduce((a, b) => a + b, 0) / allQs.length : 0,
      maxQ: allQs.length > 0 ? Math.max(...allQs) : 0,
      minQ: allQs.length > 0 ? Math.min(...allQs) : 0,
    };
  }

  /**
   * Update adaptive alpha using Lagrange dual gradient.
   *
   * If conservatism gap is too high, reduce alpha.
   * If conservatism gap is too low, increase alpha.
   */
  private updateAdaptiveAlpha(conservatismGap: number): void {
    const alphaLr = 3e-4;
    const alphaGradient = conservatismGap - this.cqlLagrangeThreshold;
    this.logAlpha = this.logAlpha + alphaLr * alphaGradient;
    this.cqlAlpha = Math.max(0.0, Math.exp(this.logAlpha));
  }

  /**
   * Soft update target network.
   */
  private softUpdateTarget(): void {
    if (this.targetNet!.softUpdate) {
      this.targetNet!.softUpdate(this.qNet!, this.tau);
    } else {
      this.targetNet!.copyFrom(this.qNet!);
    }
  }

  /**
   * Load experiences from offline dataset.
   *
   * For pure offline RL, call this method to populate the replay buffer
   * with historical data before training.
   */
  loadOfflineData(
    experiences: Array<{
      state: AdEnvironmentState;
      action: AdAction;
      reward: number;
      nextState: AdEnvironmentState;
      done: boolean;
    }>
  ): number {
    let loaded = 0;
    for (const exp of experiences) {
      const s = encodeState(exp.state) as Vec;
      const sp = encodeState(exp.nextState) as Vec;
      const aIdx = actionToIndex(exp.action);

      if (aIdx >= 0) {
        this.ensureInitialized(s.length);
        const t: Transition<Vec> = {
          s,
          aIdx,
          r: exp.reward,
          sp,
          done: exp.done,
        };
        this.replay.push(t);
        loaded++;
      }
    }
    return loaded;
  }

  /**
   * Train for a fixed number of gradient steps on offline data.
   *
   * For pure offline RL, call loadOfflineData first, then trainOffline.
   */
  trainOffline(gradientSteps: number, progressCallback?: (step: number, metrics: CQLMetrics) => void): void {
    for (let step = 0; step < gradientSteps; step++) {
      if (this.replay.size() >= this.batchSize) {
        this.trainBatch();
        this.trainCounter++;

        if (this.trainCounter % this.targetSync === 0) {
          this.softUpdateTarget();
        }

        if (progressCallback && this.lastMetrics) {
          progressCallback(step, this.lastMetrics);
        }
      }
    }
  }

  // Introspection methods
  getMetrics(): CQLMetrics | undefined {
    return this.lastMetrics;
  }

  getEpsilon(): number {
    return this.epsilon;
  }

  getReplaySize(): number {
    return this.replay.size();
  }

  getCqlAlpha(): number {
    return this.cqlAlpha;
  }

  setCqlAlpha(alpha: number): void {
    this.cqlAlpha = alpha;
    this.logAlpha = Math.log(alpha);
  }

  async save(filepath: string): Promise<void> {
    if (this.qNet && (this.qNet as QNetTorch).save) {
      await (this.qNet as QNetTorch).save(filepath);
    }
  }

  async load(filepath: string): Promise<void> {
    if (this.qNet && (this.qNet as QNetTorch).load) {
      await (this.qNet as QNetTorch).load(filepath);
    }
  }
}

/**
 * CQL with Behavior Cloning (CQL+BC).
 *
 * Adds an explicit behavior cloning term to encourage the policy to stay
 * close to the data distribution. Useful when the dataset is high-quality.
 *
 * Loss = TD Loss + α * CQL Loss + β * BC Loss
 *
 * BC Loss = -log π(a_data | s)
 */
export class CQLBCAgent extends CQLAgent {
  private bcWeight: number;

  constructor(opts: CQLAgentOptions & { bcWeight?: number } = {}) {
    super(opts);
    this.bcWeight = opts.bcWeight ?? 0.1;
  }

  // BC-specific training could be added here
  // For now, inherits CQL behavior with the option to add BC
}
