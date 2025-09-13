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

## A Concise Guide to Implementing a DQN Agent

This guide outlines the key steps to transition from a Q-table to a neural network–based Deep Q-Network (DQN) agent using Torch.js.

1) The Goal: From Table to Network
- Replace the Q-table (lookups) with a Q-network that approximates Q(s, a), enabling generalization in large/continuous state spaces.

2) State & Action Encoding
- State encoding: deterministic function mapping a state object to a fixed-length numeric vector (use one-hot for categorical features; normalize or scale numeric features; consider cyclical encodings for time).
- Action encoding: define a stable, finite action set and map each action to an integer index; the network output dimension equals the number of actions.

3) Building the Q-Network
- Create `qNet` (MLP with 2–3 hidden layers, ReLU) and a frozen `targetNet` clone.
- Use Adam optimizer and MSE loss on TD targets.

4) Stabilizing Training
- Experience replay: random mini-batches from a fixed-capacity buffer of (s, a, r, s', done).
- Target network: periodically copy weights from `qNet` to `targetNet` to stabilize targets.

5) The Training Step
- For each batch, compute predictions and TD targets `y = r + γ·(1−done)·max_a' Q_target(s', a')`.
- Minimize MSE between `q_sa` and `y`, backpropagate, and step optimizer (optionally clip gradients).

6) Managing Learning Over Time
- ε-greedy exploration: start high (e.g., 1.0) and decay to a floor (e.g., 0.05).
- Learning rate scheduling: optionally decay LR from 1e-3 toward a smaller value.

7) Integration and Persistence
- Integrate into `dqnAgent.ts`: replace table operations with network forward/backward, replay, and target sync.
- Implement `save/load` for model weights, optimizer state, and training metadata (e.g., ε).

## 1) Starting Point (Current Code)

The existing `DQNAgent` in `src/agent/dqnAgent.ts` keeps a `Map`-based Q-table and updates entries using the tabular Q-learning rule:

```
Q(s, a) ← Q(s, a) + α [ r + γ max_a' Q(s', a') − Q(s, a) ]
```

Action selection is ε-greedy and there is a simple replay pass that re-applies the tabular update to sampled experiences. To scale beyond small, discrete state spaces, we will replace Q-table lookups with a neural network that estimates Q-values for all actions given a state vector.

## 2) Define State Features and Action Indexing

This repo’s `AdEnvironmentState` includes: `dayOfWeek`, `hourOfDay`, `currentBudget`, `targetAgeGroup`, `targetInterests`, `creativeType`, `platform`, `historicalCTR`, `historicalCVR`, `competitorActivity`, `seasonality`.

Below is a deterministic encoder tailored to these features. It uses cyclical encodings for time, multi-hot for interests, one-hot for categories, and normalized scalars for continuous values.

### State Encoding (Repo Features)

```ts
// Fixed categorical orderings for determinism
const AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"] as const;
const CREATIVE_TYPES = ["lifestyle", "product", "discount", "ugc"] as const;
const PLATFORMS = ["tiktok", "instagram"] as const; // actionable platforms

// Canonical interest vocabulary observed in codebase
const INTEREST_VOCAB = [
  "fashion",
  "sports",
  "music",
  "tech",
  "fitness",
  "art",
  "travel",
] as const;

type AdEnvironmentState = {
  dayOfWeek: number;      // 0-6
  hourOfDay: number;      // 0-23
  currentBudget: number;  // USD
  targetAgeGroup: string;
  targetInterests: string[];
  creativeType: string;
  platform: string;       // "tiktok" | "instagram" | "shopify"
  historicalCTR: number;  // 0..1
  historicalCVR: number;  // 0..1
  competitorActivity: number; // 0..1
  seasonality: number;        // 0..1
};

function cyclicalPair(value: number, period: number): [number, number] {
  const angle = (2 * Math.PI * (value % period)) / period;
  return [Math.sin(angle), Math.cos(angle)];
}

function oneHot<T extends readonly string[]>(cats: T, v: string): number[] {
  return cats.map((c) => (c === v ? 1 : 0));
}

function multiHot<T extends readonly string[]>(vocab: T, values: string[]): number[] {
  const set = new Set(values);
  return vocab.map((t) => (set.has(t) ? 1 : 0));
}

function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x));
}

// Adjust divisor/log-scale to your expected budget range
function normalizeBudget(usd: number): number {
  return clamp(usd / 100, 0, 2.0); // e.g., $0–$200 → 0–2
}

export function encodeState(state: AdEnvironmentState): number[] {
  // Time (cyclical): hour + day → 4 dims
  const [sinH, cosH] = cyclicalPair(state.hourOfDay, 24);
  const [sinD, cosD] = cyclicalPair(state.dayOfWeek, 7);

  // Budget (normalized) → 1 dim
  const budget = [normalizeBudget(state.currentBudget)];

  // Categorical (one-hot): age(4) + creative(4) + platform(2) → 10 dims
  const age = oneHot(AGE_GROUPS as unknown as string[], state.targetAgeGroup);
  const creative = oneHot(CREATIVE_TYPES as unknown as string[], state.creativeType);
  const platform = oneHot(PLATFORMS as unknown as string[], state.platform);

  // Interests (multi-hot) → |INTEREST_VOCAB| dims
  const interests = multiHot(INTEREST_VOCAB as unknown as string[], state.targetInterests);

  // Historical + market (already 0..1-ish) → 4 dims
  const hist = [
    clamp(state.historicalCTR, 0, 1),
    clamp(state.historicalCVR, 0, 1),
    clamp(state.competitorActivity, 0, 1),
    clamp(state.seasonality, 0, 1),
  ];

  return [
    sinH, cosH, sinD, cosD,
    ...budget,
    ...age,
    ...creative,
    ...platform,
    ...interests,
    ...hist,
  ];
}

// Example feature count S = 4 (time) + 1 (budget) + 10 (cats) + 7 (interests) + 4 (hist) = 26
```

Notes:
- Determinism comes from fixed ordering of categories and the interest vocabulary.
- If budgets vary widely, prefer log-scale or standardization per account.
- You may replace cyclical encodings with one-hot if preferred; keep ordering documented.

### Action Indexing (Repo Actions)

Define a fixed, finite action grid and map each combination to an index. Keep the grid manageable (consider interest bundles) and deterministic.

```ts
// Discrete action grid (fixed ordering)
const BUDGET_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5] as const;
const AGE_ACTIONS = AGE_GROUPS; // reuse
const CREATIVE_ACTIONS = CREATIVE_TYPES; // reuse
const PLAT_ACTIONS = PLATFORMS; // actionable platforms
const BID_STRATEGIES = ["CPC", "CPM", "CPA"] as const;

// Optional: compress interests into a few canonical bundles
const INTEREST_BUNDLES = [
  [],
  ["fashion"],
  ["sports"],
  ["tech"],
] as const;

type AdAction = {
  budgetAdjustment: number;
  targetAgeGroup: (typeof AGE_ACTIONS)[number];
  targetInterests: string[];
  creativeType: (typeof CREATIVE_ACTIONS)[number];
  bidStrategy: (typeof BID_STRATEGIES)[number];
  platform: (typeof PLAT_ACTIONS)[number];
};

// Build the actions list once; index = array position
export const ACTIONS: AdAction[] = [];
for (const b of BUDGET_STEPS)
  for (const age of AGE_ACTIONS)
    for (const cr of CREATIVE_ACTIONS)
      for (const pf of PLAT_ACTIONS)
        for (const bid of BID_STRATEGIES)
          for (const ib of INTEREST_BUNDLES)
            ACTIONS.push({
              budgetAdjustment: b,
              targetAgeGroup: age,
              targetInterests: ib as string[],
              creativeType: cr,
              bidStrategy: bid,
              platform: pf,
            });

export function actionToIndex(a: AdAction): number {
  return ACTIONS.findIndex(
    (x) =>
      x.budgetAdjustment === a.budgetAdjustment &&
      x.targetAgeGroup === a.targetAgeGroup &&
      x.creativeType === a.creativeType &&
      x.platform === a.platform &&
      x.bidStrategy === a.bidStrategy &&
      JSON.stringify(x.targetInterests) === JSON.stringify(a.targetInterests)
  );
}

export function indexToAction(idx: number): AdAction {
  return ACTIONS[idx];
}
```

Result: a deterministic `stateVec` of size `S` (e.g., 26 above) and an output dimension `A = ACTIONS.length` with stable mapping `index ↔ action`.

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

---

## Implementation Status (Repo)

- Q-network: `QNetTorch` in `src/agent/nn/qnet.ts` provides a Torch.js-style interface backed by TensorFlow.js for now (MLP: 128→64→A, Adam, MSE).
- Agent: `DQNAgentNN` in `src/agent/dqnAgentNN.ts` uses deterministic `encodeState` and stable `ACTIONS` from `src/agent/encoding.ts`, with experience replay, ε-decay, TD targets, and hard target sync.

## Using the NN Agent

- Select agent:
  - CLI: `npm start -- --agent=nn`
  - Env: `AGENT=nn npm start`

- Useful flags (defaults):
  - `--episodes (50)`, `--batchSize (32)`, `--gamma (0.95)`, `--lr (0.001)`
  - `--trainFreq (1)`, `--targetSync (250)`, `--replayCap (5000)`
  - `--epsilonStart (1.0)`, `--epsilonMin (0.05)`, `--epsilonDecay (0.995)`

- Example:
```
npm start -- \
  --agent=nn \
  --episodes=200 \
  --batchSize=64 \
  --gamma=0.97 \
  --lr=0.0005 \
  --targetSync=500 \
  --replayCap=20000
```

## Saving and Loading

- Save: `await agent.save('model.json')` (when using `DQNAgentNN`).
- Load: `await agent.load('model.json')` before training/inference.
- Current serializer writes weights to JSON for portability (no native deps).

## Backend Performance

- Default: `@tensorflow/tfjs` pure JS backend (portable, slower). Consider `@tensorflow/tfjs-node` for faster training; if adopted, only the `QNet` implementation is affected.

## Roadmap to Torch.js Backend

- Swap `QNetTorch` internals to a Torch.js/LibTorch binding while keeping the same `QNet` interface.
- Validate parity by comparing outputs/loss on fixed batches; keep save/load stable.
