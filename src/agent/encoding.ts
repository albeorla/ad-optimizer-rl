// Deterministic state and action encoders for DQN
// See docs/torchjs_dqn_refactor.md for rationale and details.

import { AdAction, AdEnvironmentState } from "../types";

// Fixed categorical orderings for determinism
export const AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"] as const;
export const CREATIVE_TYPES = ["lifestyle", "product", "discount", "ugc"] as const;
export const PLATFORMS = ["tiktok", "instagram"] as const; // actionable platforms only

// Canonical interest vocabulary
export const INTEREST_VOCAB = [
  "fashion",
  "sports",
  "music",
  "tech",
  "fitness",
  "art",
  "travel",
] as const;

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

// Adjust based on expected budget range; consider log scaling for wide ranges
function normalizeBudget(usd: number): number {
  return clamp(usd / 100, 0, 2.0); // e.g., $0–$200 → 0–2
}

export function encodeState(state: AdEnvironmentState): number[] {
  const [sinH, cosH] = cyclicalPair(state.hourOfDay, 24);
  const [sinD, cosD] = cyclicalPair(state.dayOfWeek, 7);

  const budget = [normalizeBudget(state.currentBudget)];
  const age = oneHot(AGE_GROUPS as unknown as string[], state.targetAgeGroup);
  const creative = oneHot(CREATIVE_TYPES as unknown as string[], state.creativeType);
  const platform = oneHot(PLATFORMS as unknown as string[], state.platform);
  const interests = multiHot(INTEREST_VOCAB as unknown as string[], state.targetInterests);
  const hist = [
    clamp(state.historicalCTR, 0, 1),
    clamp(state.historicalCVR, 0, 1),
    clamp(state.competitorActivity, 0, 1),
    clamp(state.seasonality, 0, 1),
  ];

  return [sinH, cosH, sinD, cosD, ...budget, ...age, ...creative, ...platform, ...interests, ...hist];
}

// Discrete action grid (keep ordering stable)
export const BUDGET_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5] as const;
export const BID_STRATEGIES = ["CPC", "CPM", "CPA"] as const;
// Optional: reduce interests to canonical bundles to limit action space
export const INTEREST_BUNDLES: ReadonlyArray<ReadonlyArray<string>> = [
  [],
  ["fashion"],
  ["sports"],
  ["tech"],
];

export const ACTIONS: AdAction[] = [];
for (const b of BUDGET_STEPS)
  for (const age of AGE_GROUPS)
    for (const cr of CREATIVE_TYPES)
      for (const pf of PLATFORMS)
        for (const bid of BID_STRATEGIES)
          for (const ib of INTEREST_BUNDLES)
            ACTIONS.push({
              budgetAdjustment: b,
              targetAgeGroup: age as any,
              targetInterests: ib.slice() as string[],
              creativeType: cr as any,
              bidStrategy: bid as any,
              platform: pf as any,
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
  if (ACTIONS.length === 0) throw new Error("ACTIONS not initialized");
  const clamped = Math.max(0, Math.min(ACTIONS.length - 1, idx));
  return ACTIONS[clamped]!;
}

// DQN-REFAC NOTE:
// - These encoders provide the deterministic mapping required by the NN.
// - Keep category orders/vocabulary stable across training and production.
