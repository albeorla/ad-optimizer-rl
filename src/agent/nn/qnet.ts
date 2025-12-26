/**
 * Q-Network interfaces for Deep Q-Learning.
 *
 * Implementations can use TensorFlow.js, a custom MLP, or other backends.
 * All implementations should support Double DQN and Huber loss for stability.
 */

export interface QNet {
  /** Forward pass: returns Q-values per action for each state in the batch. */
  forward(batchStates: number[][]): number[][];

  /**
   * Train on a batch of (states, action indices, TD targets).
   * @param states - Encoded state vectors [B, inputSize]
   * @param actionsIdx - Action indices [B]
   * @param targets - TD targets [B]
   * @param weights - Optional importance sampling weights for PER [B]
   * @returns Average loss value
   */
  trainOnBatch(
    states: number[][],
    actionsIdx: number[],
    targets: number[],
    weights?: number[],
  ): number;

  /**
   * Compute TD errors for a batch (used for prioritized replay).
   * @returns Array of TD errors [B]
   */
  computeTDErrors?(
    states: number[][],
    actionsIdx: number[],
    targets: number[],
  ): number[];

  /** Copy parameters from another network (target sync). */
  copyFrom(source: QNet): void;

  /** Soft update: target = tau * source + (1 - tau) * target. */
  softUpdate?(source: QNet, tau: number): void;

  getInputSize(): number;
  getActionCount(): number;

  /** Serialize model weights to file. */
  save(path: string): Promise<void>;
  /** Load model weights from file. */
  load(path: string): Promise<void>;
}

export interface OptimizerLike {
  step(): void;
  zeroGrad(): void;
}

export interface DQNHyperparams {
  lr: number;
  gamma: number;
  batchSize: number;
  trainFreq: number; // env steps between train steps
  targetSync: number; // train steps between target copies
  replayCapacity: number;
  epsilonStart: number;
  epsilonMin: number;
  epsilonDecay: number;
  // New hyperparameters for improved stability
  useDoubleDQN?: boolean; // Use Double DQN for reduced overestimation
  useHuberLoss?: boolean; // Use Huber loss instead of MSE
  huberDelta?: number; // Huber loss delta (default: 1.0)
  gradientClip?: number; // Gradient clipping value (default: 10.0)
  tau?: number; // Soft update coefficient (default: 1.0 for hard update)
}

// Placeholder to unblock code completion; replace with real Torch.js-backed class.
export class PlaceholderQNet implements QNet {
  private inputSize = 1;
  private actionCount = 1;
  constructor(inputSize?: number, actionCount?: number) {
    if (inputSize) this.inputSize = inputSize;
    if (actionCount) this.actionCount = actionCount;
  }
  forward(batchStates: number[][]): number[][] {
    // Return zeros to keep shape expectations; replace with real forward
    return batchStates.map(() => Array(this.actionCount).fill(0));
  }
  trainOnBatch(
    _states: number[][],
    _actionsIdx: number[],
    _targets: number[],
  ): number {
    return 0;
  }
  copyFrom(_source: QNet): void {}
  getInputSize(): number {
    return this.inputSize;
  }
  getActionCount(): number {
    return this.actionCount;
  }
  async save(_path: string): Promise<void> {}
  async load(_path: string): Promise<void> {}
}

// Minimal in-repo MLP to enable real training without external deps.
// Replace with a Torch.js-backed implementation later.
export class QNetSimple implements QNet {
  private W1: number[][];
  private b1: number[];
  private W2: number[][];
  private b2: number[];
  private W3: number[][];
  private b3: number[];
  private lr: number;

  constructor(
    private inputSize: number,
    private actionCount: number,
    hidden1 = 128,
    hidden2 = 64,
    lr = 1e-3,
  ) {
    this.lr = lr;
    const rand = (n: number, m: number) =>
      Array.from({ length: n }, () =>
        Array.from(
          { length: m },
          () => (Math.random() - 0.5) * Math.sqrt(2 / (n + m)),
        ),
      );
    const zeros = (n: number) => Array.from({ length: n }, () => 0);
    this.W1 = rand(hidden1, inputSize);
    this.b1 = zeros(hidden1);
    this.W2 = rand(hidden2, hidden1);
    this.b2 = zeros(hidden2);
    this.W3 = rand(actionCount, hidden2);
    this.b3 = zeros(actionCount);
  }

  getInputSize(): number {
    return this.inputSize;
  }
  getActionCount(): number {
    return this.actionCount;
  }

  forward(batchStates: number[][]): number[][] {
    const relu = (x: number) => (x > 0 ? x : 0);
    const out: number[][] = [];
    for (const x of batchStates) {
      const a1 = this.affineRelu(this.W1, this.b1, x, relu);
      const a2 = this.affineRelu(this.W2, this.b2, a1, relu);
      const q = this.affine(this.W3, this.b3, a2);
      out.push(q);
    }
    return out;
  }

  trainOnBatch(
    states: number[][],
    actionsIdx: number[],
    targets: number[],
  ): number {
    const relu = (x: number) => (x > 0 ? x : 0);
    const drelu = (x: number) => (x > 0 ? 1 : 0);
    const H1 = this.W1.length;
    const H2 = this.W2.length;
    const A = this.actionCount;
    const B = states.length;
    // Accumulate grads
    const gW1 = this.zerosLike(this.W1);
    const gb1 = new Array(H1).fill(0);
    const gW2 = this.zerosLike(this.W2);
    const gb2 = new Array(H2).fill(0);
    const gW3 = this.zerosLike(this.W3);
    const gb3 = new Array(A).fill(0);

    let lossSum = 0;
    for (let i = 0; i < B; i++) {
      const x = states[i]!;
      const aIdx = actionsIdx[i]!;
      const y = targets[i]!;
      // Forward with caches for this sample
      const z1 = this.affine(this.W1, this.b1, x);
      const a1 = z1.map(relu);
      const z2 = this.affine(this.W2, this.b2, a1);
      const a2 = z2.map(relu);
      const q = this.affine(this.W3, this.b3, a2);
      const qsa = q[aIdx]!;
      const diff = qsa - y;
      lossSum += diff * diff;
      // dL/dq is zero for all except action index
      const dQ = new Array(A).fill(0);
      dQ[aIdx] = (2 * diff) / B; // average over batch
      // Backprop to W3, b3
      // dz3 = dQ since output is linear
      for (let o = 0; o < A; o++) {
        const dz = dQ[o]!;
        gb3[o]! += dz;
        const row = gW3[o]!;
        for (let k = 0; k < a2.length; k++) row[k]! += dz * a2[k]!;
      }
      // Backprop to a2
      const da2 = new Array(H2).fill(0);
      for (let o = 0; o < A; o++) {
        const dz = dQ[o]!;
        const Wrow = this.W3[o]!;
        for (let k = 0; k < H2; k++) da2[k]! += dz * Wrow[k]!;
      }
      // Through ReLU at z2
      const dz2 = da2.map((v, k) => v * drelu(z2[k]!));
      // Grads W2, b2
      for (let j = 0; j < H2; j++) {
        gb2[j]! += dz2[j]!;
        const row = gW2[j]!;
        for (let k = 0; k < H1; k++) row[k]! += dz2[j]! * a1[k]!;
      }
      // Backprop to a1
      const da1 = new Array(H1).fill(0);
      for (let j = 0; j < H2; j++) {
        const dz = dz2[j]!;
        const Wrow = this.W2[j]!;
        for (let k = 0; k < H1; k++) da1[k]! += dz * Wrow[k]!;
      }
      // Through ReLU at z1
      const dz1 = da1.map((v, k) => v * drelu(z1[k]!));
      // Grads W1, b1
      for (let h = 0; h < H1; h++) {
        gb1[h]! += dz1[h]!;
        const row = gW1[h]!;
        for (let k = 0; k < this.inputSize; k++) row[k]! += dz1[h]! * x[k]!;
      }
    }

    // Gradient step (SGD) with simple clipping
    const clip = (v: number, c = 1) => Math.max(-c, Math.min(c, v));
    for (let i = 0; i < this.W1.length; i++) {
      for (let j = 0; j < this.W1[i]!.length; j++)
        this.W1[i]![j]! -= this.lr * clip(gW1[i]![j]!);
      this.b1[i]! -= this.lr * clip(gb1[i]!);
    }
    for (let i = 0; i < this.W2.length; i++) {
      for (let j = 0; j < this.W2[i]!.length; j++)
        this.W2[i]![j]! -= this.lr * clip(gW2[i]![j]!);
      this.b2[i]! -= this.lr * clip(gb2[i]!);
    }
    for (let i = 0; i < this.W3.length; i++) {
      for (let j = 0; j < this.W3[i]!.length; j++)
        this.W3[i]![j]! -= this.lr * clip(gW3[i]![j]!);
      this.b3[i]! -= this.lr * clip(gb3[i]!);
    }

    return lossSum / B;
  }

  copyFrom(source: QNet): void {
    const s = source as any as QNetSimple;
    this.W1 = s.clone2D(s.W1);
    this.b1 = s.b1.slice();
    this.W2 = s.clone2D(s.W2);
    this.b2 = s.b2.slice();
    this.W3 = s.clone2D(s.W3);
    this.b3 = s.b3.slice();
  }

  async save(path: string): Promise<void> {
    const payload = {
      W1: this.W1,
      b1: this.b1,
      W2: this.W2,
      b2: this.b2,
      W3: this.W3,
      b3: this.b3,
    };
    const fs = await import("fs");
    await fs.promises.writeFile(path, JSON.stringify(payload));
  }
  async load(path: string): Promise<void> {
    const fs = await import("fs");
    const txt = await fs.promises.readFile(path, "utf8");
    const obj = JSON.parse(txt);
    this.W1 = obj.W1;
    this.b1 = obj.b1;
    this.W2 = obj.W2;
    this.b2 = obj.b2;
    this.W3 = obj.W3;
    this.b3 = obj.b3;
  }

  // Helpers
  private affine(W: number[][], b: number[], x: number[]): number[] {
    const out = new Array(W.length).fill(0);
    for (let i = 0; i < W.length; i++) {
      let s = b[i]!;
      const Wi = W[i]!;
      for (let j = 0; j < Wi.length; j++) s += Wi[j]! * x[j]!;
      out[i] = s;
    }
    return out;
  }
  private affineRelu(
    W: number[][],
    b: number[],
    x: number[],
    relu: (x: number) => number,
  ): number[] {
    const z = this.affine(W, b, x);
    return z.map(relu);
  }
  private zerosLike(W: number[][]): number[][] {
    return W.map((row) => row.map(() => 0));
  }
  private clone2D(W: number[][]): number[][] {
    return W.map((r) => r.slice());
  }
}

/**
 * TensorFlow.js-backed Q-Network with improved training stability.
 *
 * Features:
 * - Huber loss (smooth L1) for robustness to outliers
 * - Gradient clipping to prevent exploding gradients
 * - Soft update support for gradual target network updates
 * - Importance sampling weight support for prioritized replay
 */
export class QNetTorch implements QNet {
  private model!: import("@tensorflow/tfjs").LayersModel;
  private optimizer!: import("@tensorflow/tfjs").Optimizer;
  private useHuberLoss: boolean;
  private huberDelta: number;
  private gradientClip: number;

  constructor(
    private inputSize: number,
    private actionCount: number,
    hidden1 = 128,
    hidden2 = 64,
    private lr = 1e-3,
    options?: {
      useHuberLoss?: boolean;
      huberDelta?: number;
      gradientClip?: number;
    },
  ) {
    this.useHuberLoss = options?.useHuberLoss ?? true;
    this.huberDelta = options?.huberDelta ?? 1.0;
    this.gradientClip = options?.gradientClip ?? 10.0;
    this.init(hidden1, hidden2);
  }

  private init(hidden1: number, hidden2: number): void {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    const model = tf.sequential();

    // Input layer with He initialization for ReLU
    model.add(
      tf.layers.dense({
        units: hidden1,
        activation: "relu",
        inputShape: [this.inputSize],
        kernelInitializer: "heNormal",
      }),
    );

    // Hidden layer
    model.add(
      tf.layers.dense({
        units: hidden2,
        activation: "relu",
        kernelInitializer: "heNormal",
      }),
    );

    // Output layer (linear for Q-values)
    model.add(
      tf.layers.dense({
        units: this.actionCount,
        activation: "linear",
        kernelInitializer: tf.initializers.randomUniform({
          minval: -0.003,
          maxval: 0.003,
        }),
      }),
    );

    this.model = model;
    // Use Adam optimizer with default betas (good for RL)
    this.optimizer = tf.train.adam(this.lr);
  }

  getInputSize(): number {
    return this.inputSize;
  }
  getActionCount(): number {
    return this.actionCount;
  }

  forward(batchStates: number[][]): number[][] {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    if (!this.model) throw new Error("QNetTorch not initialized");
    return tf.tidy(() => {
      const xs = tf.tensor2d(batchStates, [batchStates.length, this.inputSize]);
      const out = this.model.predict(xs) as import("@tensorflow/tfjs").Tensor2D;
      return out.arraySync() as number[][];
    });
  }

  /**
   * Train on a batch with Huber loss and gradient clipping.
   * Supports importance sampling weights for prioritized replay.
   */
  trainOnBatch(
    states: number[][],
    actionsIdx: number[],
    targets: number[],
    weights?: number[],
  ): number {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    if (!this.model || !this.optimizer)
      throw new Error("QNetTorch not initialized");

    const B = states.length;
    let lossVal = 0;

    // Use tidy to automatically clean up tensors
    tf.tidy(() => {
      const xs = tf.tensor2d(states, [B, this.inputSize]);
      const a = tf.tensor1d(actionsIdx, "int32");
      const y = tf.tensor1d(targets, "float32");
      const onehot = tf.oneHot(a, this.actionCount).toFloat();

      // Importance sampling weights (default to 1.0 if not provided)
      const w = weights
        ? tf.tensor1d(weights, "float32")
        : tf.ones([B], "float32");

      const trainFn = () => {
        const q = this.model.apply(xs, {
          training: true,
        }) as import("@tensorflow/tfjs").Tensor2D;
        const qsa = tf.sum(tf.mul(q, onehot), 1);
        const diff = tf.sub(qsa, y);

        let loss: import("@tensorflow/tfjs").Tensor;
        if (this.useHuberLoss) {
          // Huber loss (smooth L1): more robust to outliers
          const absDiff = tf.abs(diff);
          const quadratic = tf.minimum(absDiff, this.huberDelta);
          const linear = tf.sub(absDiff, quadratic);
          loss = tf.add(
            tf.mul(tf.scalar(0.5), tf.square(quadratic)),
            tf.mul(tf.scalar(this.huberDelta), linear),
          );
        } else {
          // Standard MSE loss
          loss = tf.square(diff);
        }

        // Apply importance sampling weights and take mean
        const weightedLoss = tf.mul(loss, w);
        return tf.mean(weightedLoss) as unknown as import("@tensorflow/tfjs").Scalar;
      };

      // Compute gradients and apply clipping
      const { value, grads } = this.optimizer.computeGradients(trainFn);
      lossVal = (value as import("@tensorflow/tfjs").Scalar).arraySync() as number;

      // Clip gradients by global norm
      const gradValues = Object.values(grads);
      const clippedGrads = this.clipGradientsByNorm(tf, gradValues, this.gradientClip);

      // Apply clipped gradients
      const namedGrads: { [key: string]: import("@tensorflow/tfjs").Tensor } = {};
      const gradKeys = Object.keys(grads);
      for (let i = 0; i < gradKeys.length; i++) {
        namedGrads[gradKeys[i]!] = clippedGrads[i]!;
      }
      this.optimizer.applyGradients(namedGrads);

      // Clean up
      value.dispose();
      clippedGrads.forEach((g) => g.dispose());
    });

    return lossVal;
  }

  /**
   * Clip gradients by global norm to prevent exploding gradients.
   */
  private clipGradientsByNorm(
    tf: typeof import("@tensorflow/tfjs"),
    grads: (import("@tensorflow/tfjs").Tensor | null)[],
    clipNorm: number,
  ): import("@tensorflow/tfjs").Tensor[] {
    // Calculate global norm
    let globalNorm = tf.scalar(0);
    for (const g of grads) {
      if (g) {
        globalNorm = tf.add(globalNorm, tf.sum(tf.square(g)));
      }
    }
    globalNorm = tf.sqrt(globalNorm);

    // Clip factor
    const clipFactor = tf.div(
      tf.scalar(clipNorm),
      tf.maximum(globalNorm, tf.scalar(clipNorm)),
    );

    // Apply clipping
    const clipped: import("@tensorflow/tfjs").Tensor[] = [];
    for (const g of grads) {
      if (g) {
        clipped.push(tf.mul(g, clipFactor));
      } else {
        clipped.push(tf.zeros([1]));
      }
    }
    return clipped;
  }

  /**
   * Compute TD errors for prioritized experience replay.
   */
  computeTDErrors(
    states: number[][],
    actionsIdx: number[],
    targets: number[],
  ): number[] {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    return tf.tidy(() => {
      const xs = tf.tensor2d(states, [states.length, this.inputSize]);
      const a = tf.tensor1d(actionsIdx, "int32");
      const y = tf.tensor1d(targets, "float32");
      const onehot = tf.oneHot(a, this.actionCount).toFloat();
      const q = this.model.predict(xs) as import("@tensorflow/tfjs").Tensor2D;
      const qsa = tf.sum(tf.mul(q, onehot), 1);
      const tdErrors = tf.sub(qsa, y);
      return tdErrors.arraySync() as number[];
    });
  }

  /** Hard copy: copy all weights from source network. */
  copyFrom(source: QNet): void {
    if ((source as any).model && this.model) {
      const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
      const srcWeights = (source as any).model.getWeights() as import("@tensorflow/tfjs").Tensor[];
      const clones = srcWeights.map((w) => tf.clone(w));
      this.model.setWeights(clones);
      clones.forEach((t) => t.dispose());
    }
  }

  /**
   * Soft update: target = tau * source + (1 - tau) * target.
   * This provides smoother, more stable target network updates.
   */
  softUpdate(source: QNet, tau: number): void {
    if ((source as any).model && this.model) {
      const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
      const srcWeights = (source as any).model.getWeights() as import("@tensorflow/tfjs").Tensor[];
      const tgtWeights = this.model.getWeights();

      const newWeights = srcWeights.map((srcW, i) => {
        const tgtW = tgtWeights[i]!;
        return tf.add(tf.mul(srcW, tau), tf.mul(tgtW, 1 - tau));
      });

      this.model.setWeights(newWeights);

      // Clean up intermediate tensors
      srcWeights.forEach((t) => t.dispose());
      tgtWeights.forEach((t) => t.dispose());
      newWeights.forEach((t) => t.dispose());
    }
  }

  async save(path: string): Promise<void> {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    if (!this.model) throw new Error("QNetTorch not initialized");
    const fs = await import("fs");
    const weights = this.model.getWeights();
    const serial = await Promise.all(
      weights.map(async (t) => ({
        shape: t.shape,
        data: Array.from(await t.data()),
      })),
    );
    const payload = {
      version: 2,
      inputSize: this.inputSize,
      actionCount: this.actionCount,
      lr: this.lr,
      useHuberLoss: this.useHuberLoss,
      huberDelta: this.huberDelta,
      gradientClip: this.gradientClip,
      weights: serial,
    };
    await fs.promises.writeFile(path, JSON.stringify(payload));
  }

  async load(path: string): Promise<void> {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    if (!this.model) throw new Error("QNetTorch not initialized");
    const fs = await import("fs");
    const txt = await fs.promises.readFile(path, "utf8");
    const obj = JSON.parse(txt) as {
      weights: { shape: number[]; data: number[] }[];
      useHuberLoss?: boolean;
      huberDelta?: number;
      gradientClip?: number;
    };

    // Restore hyperparameters if present
    if (obj.useHuberLoss !== undefined) this.useHuberLoss = obj.useHuberLoss;
    if (obj.huberDelta !== undefined) this.huberDelta = obj.huberDelta;
    if (obj.gradientClip !== undefined) this.gradientClip = obj.gradientClip;

    const tensors = obj.weights.map((w) =>
      tf.tensor(w.data, w.shape as [number, ...number[]]),
    );
    this.model.setWeights(tensors);
    tensors.forEach((t) => t.dispose());
  }
}
