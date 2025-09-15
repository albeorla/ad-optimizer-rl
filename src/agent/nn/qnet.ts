// Torch.js Q-Network interfaces (stubs)
// Implementations will wrap Torch.js models; this file defines the surface area.

export interface QNet {
  // Forward pass: returns Q-values per action for each state in the batch
  forward(batchStates: number[][]): number[][];
  // Train on a batch of (states, action indices, TD targets); returns avg loss
  trainOnBatch(
    states: number[][],
    actionsIdx: number[],
    targets: number[],
  ): number;
  // Copy parameters from another network (target sync)
  copyFrom(source: QNet): void;
  // Shapes
  getInputSize(): number;
  getActionCount(): number;
  // Serialize/deserialize model weights
  save(path: string): Promise<void>;
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

// Torch.js-style implementation using TensorFlow.js backend.
// Note: We depend on '@tensorflow/tfjs' to run in Node without native bindings.
export class QNetTorch implements QNet {
  private model!: import("@tensorflow/tfjs").LayersModel;
  private optimizer!: import("@tensorflow/tfjs").Optimizer;

  constructor(
    private inputSize: number,
    private actionCount: number,
    hidden1 = 128,
    hidden2 = 64,
    private lr = 1e-3,
  ) {
    this.init(hidden1, hidden2);
  }

  private init(hidden1: number, hidden2: number): void {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    const model = tf.sequential();
    model.add(
      tf.layers.dense({
        units: hidden1,
        activation: "relu",
        inputShape: [this.inputSize],
      }),
    );
    model.add(tf.layers.dense({ units: hidden2, activation: "relu" }));
    model.add(
      tf.layers.dense({ units: this.actionCount, activation: "linear" }),
    );
    this.model = model;
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
      const arr = out.arraySync() as number[][];
      return arr;
    });
  }

  trainOnBatch(
    states: number[][],
    actionsIdx: number[],
    targets: number[],
  ): number {
    const tf = require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
    if (!this.model || !this.optimizer)
      throw new Error("QNetTorch not initialized");
    const B = states.length;
    let lossVal = 0;
    tf.tidy(() => {
      const xs = tf.tensor2d(states, [B, this.inputSize]);
      const a = tf.tensor1d(actionsIdx, "int32");
      const y = tf.tensor1d(targets, "float32");
      const onehot = tf.oneHot(a, this.actionCount).toFloat();
      const trainFn = () => {
        const q = this.model.apply(xs, {
          training: true,
        }) as import("@tensorflow/tfjs").Tensor2D; // [B, A]
        const qsa = tf.sum(tf.mul(q, onehot), 1); // [B]
        const diff = tf.sub(qsa, y);
        const loss = tf.mean(tf.square(diff));
        return loss as unknown as import("@tensorflow/tfjs").Scalar;
      };
      const out = this.optimizer.minimize(
        trainFn,
        true,
      ) as import("@tensorflow/tfjs").Scalar;
      lossVal = out.arraySync() as number;
      out.dispose();
    });
    return lossVal;
  }

  copyFrom(source: QNet): void {
    // Hard sync: copy weights from source if it is also QNetTorch/QNetSimple
    if ((source as any).model && this.model) {
      const tf =
        require("@tensorflow/tfjs") as typeof import("@tensorflow/tfjs");
      const srcWeights = (
        source as any
      ).model.getWeights() as import("@tensorflow/tfjs").Tensor[];
      const clones = srcWeights.map((w) => tf.clone(w));
      this.model.setWeights(clones);
      clones.forEach((t) => t.dispose());
      return;
    }
    // Fallback: approximate via forward pass is not meaningful for weight copy; no-op.
  }

  async save(path: string): Promise<void> {
    // Save weight arrays to JSON for portability without tfjs-node file I/O
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
      inputSize: this.inputSize,
      actionCount: this.actionCount,
      lr: this.lr,
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
    };
    const tensors = obj.weights.map((w) =>
      tf.tensor(w.data, w.shape as [number, ...number[]]),
    );
    this.model.setWeights(tensors);
    tensors.forEach((t) => t.dispose());
  }
}
