// Torch.js Q-Network interfaces (stubs)
// Implementations will wrap Torch.js models; this file defines the surface area.

export interface QNet {
  // Forward pass: returns Q-values per action for each state in the batch
  forward(batchStates: number[][]): number[][];
  // Copy parameters from another network (target sync)
  copyFrom(source: QNet): void;
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
  forward(batchStates: number[][]): number[][] {
    // Return zeros to keep shape expectations; replace with real forward
    const A = 1; // caller should replace with actual action count
    return batchStates.map(() => Array(A).fill(0));
  }
  copyFrom(_source: QNet): void {}
  async save(_path: string): Promise<void> {}
  async load(_path: string): Promise<void> {}
}

