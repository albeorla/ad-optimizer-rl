/** One transition tuple for DQN replay buffers. */
export interface Transition<S extends readonly number[]> {
  s: S;
  aIdx: number;
  r: number;
  sp: S;
  done: boolean;
}

/**
 * Prioritized transition with TD-error based priority.
 * Used for Prioritized Experience Replay (PER).
 */
export interface PrioritizedTransition<S extends readonly number[]>
  extends Transition<S> {
  priority: number;
  index: number;
}

/**
 * Fixed-size ring buffer for off-policy replay sampling.
 * Uses Fisher-Yates partial shuffle for efficient unique sampling without replacement.
 */
export class ReplayBuffer<S extends readonly number[]> {
  private buf: Transition<S>[] = [];
  private next = 0;

  constructor(private capacity: number) {
    if (capacity <= 0) throw new Error("ReplayBuffer capacity must be > 0");
  }

  size(): number {
    return this.buf.length;
  }

  isFull(): boolean {
    return this.buf.length >= this.capacity;
  }

  push(t: Transition<S>): void {
    if (this.buf.length < this.capacity) {
      this.buf.push(t);
    } else {
      this.buf[this.next] = t;
    }
    this.next = (this.next + 1) % this.capacity;
  }

  /**
   * Sample unique transitions using Fisher-Yates partial shuffle.
   * This ensures no duplicates in the sampled batch, which is important
   * for stable gradient estimates in DQN training.
   */
  sample(batchSize: number): Transition<S>[] {
    const n = Math.min(batchSize, this.buf.length);
    if (n === 0) return [];

    // Create array of indices for Fisher-Yates
    const indices = new Array(this.buf.length);
    for (let i = 0; i < this.buf.length; i++) {
      indices[i] = i;
    }

    // Fisher-Yates partial shuffle: only shuffle first n elements
    const out: Transition<S>[] = new Array(n);
    for (let i = 0; i < n; i++) {
      // Pick random index from remaining elements [i, length)
      const j = i + Math.floor(Math.random() * (this.buf.length - i));
      // Swap indices[i] and indices[j]
      const temp = indices[i]!;
      indices[i] = indices[j]!;
      indices[j] = temp;
      // Add the selected transition
      out[i] = this.buf[indices[i]!]!;
    }

    return out;
  }

  /** Get all transitions (for debugging/analysis). */
  getAll(): Transition<S>[] {
    return this.buf.slice();
  }

  /** Clear the buffer. */
  clear(): void {
    this.buf = [];
    this.next = 0;
  }
}

/**
 * Prioritized Experience Replay buffer using a sum-tree for O(log n) sampling.
 * Transitions are sampled with probability proportional to their TD-error priority.
 */
export class PrioritizedReplayBuffer<S extends readonly number[]> {
  private buf: (Transition<S> | null)[] = [];
  private priorities: number[] = [];
  private next = 0;
  private count = 0;
  private maxPriority = 1.0;

  // PER hyperparameters
  private alpha: number; // priority exponent (0 = uniform, 1 = full prioritization)
  private beta: number; // importance sampling exponent (annealed from beta0 to 1)
  private betaIncrement: number;
  private epsilon = 1e-6; // small constant to avoid zero priority

  constructor(
    private capacity: number,
    alpha = 0.6,
    beta0 = 0.4,
    betaIncrement = 1e-4,
  ) {
    if (capacity <= 0) throw new Error("ReplayBuffer capacity must be > 0");
    this.alpha = alpha;
    this.beta = beta0;
    this.betaIncrement = betaIncrement;
    this.buf = new Array(capacity).fill(null);
    this.priorities = new Array(capacity).fill(0);
  }

  size(): number {
    return this.count;
  }

  isFull(): boolean {
    return this.count >= this.capacity;
  }

  /** Add transition with max priority (ensures new experiences are sampled). */
  push(t: Transition<S>): void {
    this.buf[this.next] = t;
    this.priorities[this.next] = Math.pow(this.maxPriority, this.alpha);
    this.next = (this.next + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  /**
   * Sample transitions proportionally to their priorities.
   * Returns transitions with importance sampling weights for unbiased learning.
   */
  sample(batchSize: number): {
    transitions: Transition<S>[];
    indices: number[];
    weights: number[];
  } {
    const n = Math.min(batchSize, this.count);
    if (n === 0) return { transitions: [], indices: [], weights: [] };

    // Calculate priority sum
    let prioritySum = 0;
    for (let i = 0; i < this.count; i++) {
      prioritySum += this.priorities[i]!;
    }

    // Stratified sampling: divide range into n segments
    const segmentSize = prioritySum / n;
    const transitions: Transition<S>[] = [];
    const indices: number[] = [];
    const weights: number[] = [];

    // Calculate max weight for normalization
    const minProbability = Math.min(...this.priorities.slice(0, this.count)) / prioritySum;
    const maxWeight = Math.pow(this.count * minProbability, -this.beta);

    for (let i = 0; i < n; i++) {
      // Sample uniformly within segment
      const low = segmentSize * i;
      const high = segmentSize * (i + 1);
      const target = low + Math.random() * (high - low);

      // Find transition via linear scan (could use sum-tree for O(log n))
      let cumSum = 0;
      let idx = 0;
      for (let j = 0; j < this.count; j++) {
        cumSum += this.priorities[j]!;
        if (cumSum >= target) {
          idx = j;
          break;
        }
      }

      const probability = this.priorities[idx]! / prioritySum;
      const weight = Math.pow(this.count * probability, -this.beta) / maxWeight;

      transitions.push(this.buf[idx]!);
      indices.push(idx);
      weights.push(weight);
    }

    // Anneal beta towards 1
    this.beta = Math.min(1.0, this.beta + this.betaIncrement);

    return { transitions, indices, weights };
  }

  /** Update priorities after learning (based on TD errors). */
  updatePriorities(indices: number[], tdErrors: number[]): void {
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i]!;
      const priority = Math.pow(Math.abs(tdErrors[i]!) + this.epsilon, this.alpha);
      this.priorities[idx] = priority;
      this.maxPriority = Math.max(this.maxPriority, Math.abs(tdErrors[i]!) + this.epsilon);
    }
  }

  /** Get current beta value (for logging). */
  getBeta(): number {
    return this.beta;
  }

  clear(): void {
    this.buf = new Array(this.capacity).fill(null);
    this.priorities = new Array(this.capacity).fill(0);
    this.next = 0;
    this.count = 0;
    this.maxPriority = 1.0;
  }
}
