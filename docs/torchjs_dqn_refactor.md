# Torch.js DQN Refactoring Tutorial

This tutorial shows how to refactor a table-based Q-learning agent into a Deep Q-Network (DQN) using Torch.js. It is written for this repository’s current agent (`src/agent/dqnAgent.ts`), which uses a Q-table, and outlines the changes to transition to a neural function approximator.

## Conceptual Checklist

- Replace Q-table with a Q-network: approximate Q(s, a) via a neural net.
- Engineer state features and map actions to fixed indices for the network.
- Add experience replay and a target network for stable learning.
- Train with TD targets: optimize MSE loss over mini-batches.
- Schedule exploration (epsilon) and learning rate; clamp gradients.
- Persist models, evaluate offline, and add guardrails for deployment.

---

## 1) Starting Point (Current Code)

The existing `DQNAgent` in `src/agent/dqnAgent.ts` keeps a `Map`-based Q-table and updates entries using the tabular Q-learning rule:

```
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]
```

Action selection is ε-greedy and there is a simple replay pass that re-applies the tabular update to sampled experiences. To scale beyond small, discrete state spaces, we will replace Q-table lookups with a neural network that estimates Q-values for all actions given a state vector.

## 2) Define State Features and Action Indexing

- State encoder: Convert `AdEnvironmentState` into a fixed-length numeric vector. For example: one-hot day-of-week (7), one-hot hour (24) or cyclical sin/cos (2), normalized budget (1), one-hot age group (4), one-hot creative type (4), one-hot platform (2), and normalized historical metrics (e.g., CTR, CVR). Keep this deterministic and documented.
- Action indexing: Keep the existing discrete action generation but map each `AdAction` to a stable index `0..A-1`. Store the mapping once at construction for reproducibility.

Result: `stateVec: Float32Array` with size `S`, and `numActions: A` with a mapping `actionIndex ↔ AdAction`.

## 3) Build the Q-Network (Torch.js)

- Input: `S` features. Output: `A` Q-values (one per discrete action).
- Architecture: MLP with 2–3 hidden layers (e.g., 128 → 64), ReLU activations.
- Optimizer/Loss: Adam + mean squared error (MSE) on TD targets.
- Device: CPU is fine to start; abstract for future GPU/WebGPU.

Pseudo-TypeScript sketch (API-agnostic Torch.js style):

```ts
// Shapes: S = state feature size, A = number of actions
const qNet = new TorchSequential([
  Linear(S, 128), ReLU(),
  Linear(128, 64), ReLU(),
  Linear(64, A),
]);
const targetNet = qNet.clone().freezeGrad();
const optimizer = torch.optim.adam(qNet.parameters(), { lr: 1e-3 });
```

## 4) Experience Replay and Target Network

- Replay buffer: fixed-capacity ring buffer storing `(s, a, r, s', done)`.
- Mini-batches: sample uniformly (or with prioritization later) to decorrelate updates.
- Target network: copy `qNet` weights to `targetNet` every N updates (e.g., 1000) to stabilize targets.

## 5) Training Step (TD Targets)

For a mini-batch of size `B`:

1. Encode states to `X: [B, S]` and next-states to `X': [B, S]`.
2. Compute `Q = qNet(X)` and `Q' = targetNet(X')`.
3. For each sample i, pick `q_sa = Q[i, a_i]` and `y_i = r_i + γ · (1 − done_i) · max_a' Q'[i, a']`.
4. Loss = MSE(q_sa, y_i) across the batch. Backprop, step optimizer, and optionally clip gradients.

Pseudo-code:

```ts
function trainStep(batch) {
  const X = encodeStates(batch.states);     // [B, S]
  const Xp = encodeStates(batch.nextStates);// [B, S]
  const Q = qNet.forward(X);                // [B, A]
  const Qp = targetNet.forward(Xp);         // [B, A]

  const maxQp = Qp.max(1);                  // [B]
  const y = batch.rewards + gamma * (1 - batch.dones) * maxQp; // [B]

  const q_sa = Q.gather(1, batch.actionsIdx); // [B]
  const loss = mseLoss(q_sa, y.detach());

  optimizer.zeroGrad();
  loss.backward();
  torch.nn.utils.clipGradNorm_(qNet.parameters(), 5.0);
  optimizer.step();
}
```

## 6) Exploration, Scheduling, and Persistence

- ε-greedy: decay ε from 1.0 → 0.05 over episodes/steps; keep a non-zero floor for ongoing exploration in training.
- LR schedule: optional exponential decay or step schedule; avoid decaying too quickly.
- Checkpoints: save `qNet` weights, replay buffer watermark, ε, and optimizer state; load on resume.

## 7) Integration Steps in This Repo

Target files:

- `src/agent/dqnAgent.ts`: Replace Q-table members with Torch.js model, target network, optimizer, and a replay buffer storing tensors (or raw JS objects encoded on batch creation).
- `src/environment/simulator.ts`: No change needed beyond ensuring state encoding can access the required fields.
- `src/training/pipeline.ts`: Call `agent.update(...)` as before; inside the agent perform batching and periodic target sync.

Key method changes:

- `selectAction(state)`: encode state → tensor → `qNet.forward` → argmax over actions (or random with probability ε).
- `update(s, a, r, s')`: push to replay; if buffer ≥ warmup and step % trainFreq == 0, sample a batch and `trainStep`.
- `save/load(path)`: persist/load model weights and optimizer state; leave a JSON manifest for metadata.

## 8) Validation and Safety

- Unit tests: sanity-check increasing returns on a stationary toy environment; verify shapes and no NaNs.
- Offline eval: run policies with ε=0 on historical/simulated data; compare to baseline Q-table results.
- Guardrails: keep the existing budget clamps and no-sales freezes; learning upgrades should not bypass safety.

## 9) Migration Tips

- Start with a small network and small action set to validate the plumbing; then expand action combinatorics.
- Normalize inputs; track feature drift in production.
- Log TD error statistics and ε, LR schedules to help debug learning dynamics.

---

Appendix: Minimal Type Sketches

```ts
type EncodedState = Float32Array; // length S

interface Transition {
  s: EncodedState;
  a: number;       // action index
  r: number;
  sp: EncodedState;
  done: 0 | 1;
}

class ReplayBuffer {
  constructor(capacity: number) { /* ... */ }
  push(t: Transition) { /* ... */ }
  sample(batchSize: number): Transition[] { /* ... */ }
}
```

