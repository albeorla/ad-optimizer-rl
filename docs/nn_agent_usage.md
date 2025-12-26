# Neural Network Agent Usage Guide

This document covers the neural network-based DQN agent (`DQNAgentNN`) and the Conservative Q-Learning agent (`CQLAgent`).

## Quick Start

### Basic NN Agent

```bash
# Run with neural network DQN
npm start -- --agent=nn

# With custom hyperparameters
npm start -- \
  --agent=nn \
  --episodes=200 \
  --batchSize=64 \
  --gamma=0.97 \
  --lr=0.0005 \
  --targetSync=500 \
  --replayCap=20000
```

### Conservative Q-Learning (Offline RL)

For safe deployment from historical data:

```typescript
import { CQLAgent } from 'ad-optimizer-rl';

const agent = new CQLAgent({
  epsilonStart: 0.3,      // Lower exploration for offline
  epsilonMin: 0.01,
  epsilonDecay: 0.999,
  lr: 3e-4,
  gamma: 0.99,
  batchSize: 256,
  cqlAlpha: 1.0,          // CQL regularization strength
  cqlTemperature: 1.0,    // Softmax temperature
  adaptiveAlpha: true,    // Auto-tune alpha
  useDoubleDQN: true,
  gradientClip: 1.0,
});
```

## Command-Line Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | `tabular` | Agent type: `tabular` or `nn` |
| `--episodes` | `50` | Number of training episodes |
| `--batchSize` | `32` | Mini-batch size for training |
| `--gamma` | `0.95` | Discount factor (0-1) |
| `--lr` | `0.001` | Learning rate |
| `--trainFreq` | `1` | Training frequency (every N steps) |
| `--targetSync` | `250` | Target network sync frequency |
| `--replayCap` | `5000` | Experience replay buffer capacity |
| `--epsilonStart` | `1.0` | Initial exploration rate |
| `--epsilonMin` | `0.05` | Minimum exploration rate |
| `--epsilonDecay` | `0.995` | Exploration decay rate |
| `--load` | - | Path to pre-trained model to load |
| `--no-demo` | `false` | Skip 24h policy demonstration |

## Network Architecture

The Q-network (`src/agent/nn/qnet.ts`) uses a fully-connected architecture:

```
Input Layer (38 dimensions - encoded state)
    |
Dense Layer (128 units, ReLU activation)
    |
Dense Layer (64 units, ReLU activation)
    |
Dense Layer (32 units, ReLU activation)
    |
Output Layer (288 units - Q-values for each action)
```

### State Encoding

The state is encoded into a 38-dimensional feature vector (see `src/agent/encoding.ts`):

- **Temporal (4)**: sin/cos of hour and day (cyclical encoding)
- **Budget (1)**: Normalized current budget
- **Demographics (4)**: One-hot age group
- **Creative (4)**: One-hot creative type
- **Platform (2)**: One-hot platform
- **Interests (7)**: Multi-hot interest flags
- **Performance (4)**: CTR, CVR, competition, seasonality

### Action Space

288 discrete actions combining:
- Budget multipliers: 0.95, 1.0, 1.05
- Platforms: TikTok, Instagram
- Creatives: lifestyle, product, discount, UGC
- Age groups: 18-24, 25-34, 35-44, 45+
- Bid strategies: CPC, CPM, CPA

## Model Persistence

```typescript
// Save trained model
await agent.save('model.json');

// Load pre-trained model
await agent.load('model.json');

// Load and continue training
npm start -- --agent=nn --load=model.json --episodes=100
```

## Training Features

### Double DQN

Reduces Q-value overestimation by using separate networks for action selection and evaluation:

```
y = r + γ Q_target(s', argmax_a Q_online(s', a))
```

### Experience Replay

Stores transitions `(s, a, r, s', done)` and samples random batches for training:
- Breaks temporal correlations
- Improves sample efficiency
- Stabilizes learning

### Target Network

Slowly updated copy of the Q-network for computing stable TD targets:
- Synced every `targetSync` steps
- Prevents training oscillation

### Huber Loss

Robust to outliers, clips large errors:

```
L = 0.5 * (y - Q)²     if |y - Q| < 1
L = |y - Q| - 0.5      otherwise
```

## Backend Options

```bash
# Default (portable, slower)
npm start -- --agent=nn

# With native backend (faster, requires native dependencies)
# In package.json, add: "@tensorflow/tfjs-node": "^4.x"
# Then training uses native bindings automatically
```

## Hyperparameter Tuning Tips

1. **Learning Rate**: Start with 1e-3, decrease if unstable
2. **Batch Size**: 32-128 typical; larger for more stable gradients
3. **Replay Capacity**: 5000-50000; larger for more diverse sampling
4. **Target Sync**: 200-1000; higher for more stability
5. **Epsilon Decay**: 0.99-0.999; slower for more exploration
6. **Gamma**: 0.95-0.99; higher values weight future rewards more

## Reference Files

| File | Purpose |
|------|---------|
| `src/agent/dqnAgentNN.ts` | Neural network DQN agent |
| `src/agent/cqlAgent.ts` | Conservative Q-Learning agent |
| `src/agent/nn/qnet.ts` | Q-network architecture |
| `src/agent/encoding.ts` | State/action encoding |
| `src/agent/replay.ts` | Experience replay buffer |

---

*Last updated: December 2024*
