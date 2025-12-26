# Ad-Optimizer-RL Architecture

A production-grade reinforcement learning system for real-time bidding in digital advertising.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Ad-Optimizer-RL                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   State     │    │     RL      │    │   Safety    │    │  Platform   │  │
│  │ Enrichment  │───▶│   Agent     │───▶│   Layer     │───▶│    API      │  │
│  │   Engine    │    │  (DQN/CQL)  │    │             │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│         ▲                  │                  │                  │         │
│         │                  │                  │                  │         │
│         │           ┌──────▼──────┐    ┌──────▼──────┐           │         │
│         │           │    PID      │    │   Circuit   │           │         │
│         │           │ Controller  │    │   Breaker   │           │         │
│         │           └─────────────┘    └─────────────┘           │         │
│         │                                                        │         │
│  ┌──────┴──────────────────────────────────────────────────────┴──────┐   │
│  │                        Attribution Buffer                          │   │
│  │              (Delayed Feedback Model / GDFM)                       │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    Offline Policy Evaluation                        │   │
│  │              (IPS / SNIPS / Doubly Robust / MDA)                   │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Overview

### 1. RL Agents (`/src/agent/`)

| Agent | Description | Use Case |
|-------|-------------|----------|
| `DQNAgent` | Tabular Q-learning with experience replay | Small state spaces, fast training |
| `DQNAgentNN` | Neural network DQN with Double DQN | Large state spaces, generalization |
| `CQLAgent` | Conservative Q-Learning for offline RL | Safe deployment from historical data |

**Key Features:**
- Double DQN to reduce Q-value overestimation
- Huber loss for robustness to outliers
- Gradient clipping for training stability
- Prioritized experience replay
- Soft target network updates

### 2. Control Systems (`/src/control/`)

#### PID Controllers
```typescript
import { DualPidController, applyBidModifier } from 'ad-optimizer-rl';

const controller = new DualPidController({
  totalBudget: 100,
  campaignDuration: 24,
  targetCpa: 10,
});

const multiplier = controller.getMultiplier({
  currentSpend: 50,
  elapsedTime: 12,
  recentSpend: 50,
  recentConversions: 5,
});

const { finalBid } = applyBidModifier(rlBid, multiplier);
```

**Components:**
- `PidPacer`: Budget pacing with multiplier output
- `CpaPidController`: CPA-constrained bidding
- `DualPidController`: Simultaneous budget + CPA control

### 3. Delayed Feedback (`/src/data/`)

Handles the reality that conversions occur hours/days after ad clicks.

```typescript
import { AttributionBuffer, DelayedFeedbackModel } from 'ad-optimizer-rl';

const buffer = new AttributionBuffer({
  attributionWindowMs: 24 * 60 * 60 * 1000, // 24 hours
  useImportanceSampling: true,
});

// Record click
buffer.addPending({
  state, action,
  actionTimestamp: Date.now(),
  attributionId: ttclid,
  actionProbability: 0.1,
});

// Later: record conversion
buffer.recordConversion(ttclid, {
  revenue: 29.99,
  conversionTimestamp: Date.now(),
  nextState,
});

// Sample for training with importance weights
const batch = buffer.sampleBatch(32);
```

### 4. Safety Mechanisms (`/src/execution/`)

#### Circuit Breaker
```typescript
import { SafetyLayer } from 'ad-optimizer-rl';

const safety = new SafetyLayer({
  circuitConfig: { failureThreshold: 5 },
  anomalyThresholds: { minROAS: 0.5, maxCPA: 20 },
});

const result = safety.processAction(action, metrics, currentBudget);
if (!result.allowed) {
  // Use safe mode action
  console.log('Entering safe mode:', result.alerts);
}
```

**Protection Layers:**
1. **Circuit Breaker**: Auto-fallback on repeated failures
2. **Anomaly Detection**: Alert on win rate, ROAS, CPA anomalies
3. **Bid Validation**: Hard constraints on actions
4. **Safe Mode**: Conservative fallback strategy

### 5. Offline Policy Evaluation (`/src/evaluation/`)

Validate new policies before deployment using historical data.

```typescript
import { runOPESuite, evaluateDeployment } from 'ad-optimizer-rl';

const suite = runOPESuite(historicalLogs, newPolicy);

console.log('IPS Estimate:', suite.ips.estimate);
console.log('SNIPS Estimate:', suite.snips.estimate);
console.log('Estimator Agreement:', suite.agreement);

const deployment = evaluateDeployment(suite.recommendedEstimate, baselineOPE);
if (deployment.recommend) {
  console.log('Safe to deploy! Expected lift:', deployment.expectedLift);
}
```

### 6. State Enrichment (`/src/types/EnrichedState.ts`)

Extends the base state with contextual information:

```typescript
import { StateEnrichmentEngine, encodeEnrichedState } from 'ad-optimizer-rl';

const engine = new StateEnrichmentEngine({
  dailyBudget: 100,
  targetCPA: 15,
  targetROAS: 2.0,
});

// Record observations
engine.recordObservation({ spend: 5.0, conversion: true, revenue: 29.99 });

// Get enriched state
const enrichedState = engine.enrich(baseState);

// Encode for neural network (53 dimensions)
const features = encodeEnrichedState(enrichedState);
```

**Added Context:**
- Budgetary: remaining budget, spend rate, pacing error
- Temporal: peak hours, end-of-day signals
- Competitive: win rate, bid shading factor
- Performance: CPA/ROAS vs targets, trend

## Data Flow

### Training Loop (Offline)
```
Historical Logs → Attribution Buffer → GDFM Processing → CQL Agent Training
                                                                    ↓
                                            Offline Policy Evaluation
                                                                    ↓
                                            Deployment Decision (MDA > 80%?)
```

### Inference Loop (Online)
```
Bid Request → State Enrichment → RL Agent → PID Modifier → Safety Layer → Bid Response
                                                                ↓
                                                        Circuit Breaker Check
                                                                ↓
                                                        Anomaly Detection
```

## Key Design Decisions

### 1. Dual-PID Architecture
**Problem:** RL optimizes immediate reward but struggles with budget pacing.
**Solution:** Separate valuation (RL) from delivery (PID).

### 2. Conservative Q-Learning
**Problem:** Standard DQN overestimates untested actions.
**Solution:** CQL penalizes OOD actions, providing value lower bounds.

### 3. Importance Sampling for Delayed Feedback
**Problem:** Policy changes while waiting for conversions.
**Solution:** Weight delayed experiences by `π_new / π_old`.

### 4. Self-Normalized IPS for OPE
**Problem:** Standard IPS has high variance.
**Solution:** SNIPS normalizes by weight sum, trading slight bias for stability.

## Configuration

### Environment Variables
```bash
# Platform configuration
ALLOWED_PLATFORMS=tiktok,instagram
DISABLE_INSTAGRAM=false
LOCKED_CREATIVE_TYPE=ugc

# Budget defaults
DAILY_BUDGET_TARGET=30
TSHIRT_PRICE=29.99
PRINTFUL_COGS=15.00
```

### Agent Hyperparameters
```typescript
const agent = new CQLAgent({
  // Exploration
  epsilonStart: 0.3,
  epsilonMin: 0.01,
  epsilonDecay: 0.999,

  // Learning
  lr: 3e-4,
  gamma: 0.99,
  batchSize: 256,

  // CQL-specific
  cqlAlpha: 1.0,
  cqlTemperature: 1.0,
  adaptiveAlpha: true,

  // Safety
  useDoubleDQN: true,
  gradientClip: 1.0,
});
```

## Testing

```bash
# Run test suite
npx ts-node tests/control.test.ts
npx ts-node tests/ope.test.ts

# Type check
npx tsc --noEmit
```

## File Structure

```
src/
├── agent/
│   ├── base.ts           # RLAgent abstract class
│   ├── dqnAgent.ts       # Tabular Q-learning
│   ├── dqnAgentNN.ts     # Neural network DQN
│   ├── cqlAgent.ts       # Conservative Q-Learning
│   ├── encoding.ts       # State/action encoding
│   ├── replay.ts         # Experience replay buffers
│   └── nn/
│       └── qnet.ts       # Q-network implementations
├── control/
│   └── PidController.ts  # PID controllers for pacing
├── data/
│   └── AttributionBuffer.ts  # Delayed feedback handling
├── evaluation/
│   └── OPE.ts            # Offline policy evaluation
├── execution/
│   ├── guardrails.ts     # Budget constraints
│   └── SafetyLayer.ts    # Circuit breakers, anomaly detection
├── types/
│   └── EnrichedState.ts  # Extended state representation
├── environment/
│   ├── simulator.ts      # Offline training environment
│   └── realShadow.ts     # Shadow mode environment
├── platforms/
│   ├── mockTikTok.ts     # TikTok simulator
│   └── realTikTok.ts     # Real TikTok API (scaffold)
└── index.ts              # Public API exports
```

## References

- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning"
- Bottou et al., "Counterfactual Reasoning and Learning Systems"
- Thomas & Brunskill, "Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning"
