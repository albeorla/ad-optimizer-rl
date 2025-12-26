/**
 * Deterministic state and action encoders for DQN.
 *
 * See docs/torchjs_dqn_refactor.md for rationale and details.
 * Keep category orders and vocabularies stable across training and production.
 */

import { AdAction, AdEnvironmentState } from "../types";

// Fixed categorical orderings for determinism
export const AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"] as const;
export const CREATIVE_TYPES = [
  "lifestyle",
  "product",
  "discount",
  "ugc",
] as const;
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

function multiHot<T extends readonly string[]>(
  vocab: T,
  values: string[],
): number[] {
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

/** Encode AdEnvironmentState into a normalized feature vector. */
export function encodeState(state: AdEnvironmentState): number[] {
  const [sinH, cosH] = cyclicalPair(state.hourOfDay, 24);
  const [sinD, cosD] = cyclicalPair(state.dayOfWeek, 7);

  const budget = [normalizeBudget(state.currentBudget)];
  const age = oneHot(AGE_GROUPS as unknown as string[], state.targetAgeGroup);
  const creative = oneHot(
    CREATIVE_TYPES as unknown as string[],
    state.creativeType,
  );
  const platform = oneHot(PLATFORMS as unknown as string[], state.platform);
  const interests = multiHot(
    INTEREST_VOCAB as unknown as string[],
    state.targetInterests,
  );
  const hist = [
    clamp(state.historicalCTR, 0, 1),
    clamp(state.historicalCVR, 0, 1),
    clamp(state.competitorActivity, 0, 1),
    clamp(state.seasonality, 0, 1),
  ];

  return [
    sinH,
    cosH,
    sinD,
    cosD,
    ...budget,
    ...age,
    ...creative,
    ...platform,
    ...interests,
    ...hist,
  ];
}

// Discrete action grid (keep ordering stable)
// Tighter budget multipliers for small-budget regimes (e.g., ~$30/day)
/** Small, safe multiplicative budget adjustments for hourly control. */
export const BUDGET_STEPS = [0.95, 1.0, 1.05] as const;
export const BID_STRATEGIES = ["CPC", "CPM", "CPA"] as const;
// Optional: reduce interests to canonical bundles to limit action space
export const INTEREST_BUNDLES: ReadonlyArray<ReadonlyArray<string>> = [
  [],
  ["fashion"],
  ["sports"],
  ["tech"],
];

// Optionally narrow action space via environment variables
const ENV_PLATFORMS = (process.env.ALLOWED_PLATFORMS || "")
  .split(",")
  .map((s) => s.trim().toLowerCase())
  .filter((s) => s === "tiktok" || s === "instagram");
const DISABLE_IG =
  (process.env.DISABLE_INSTAGRAM || "").toLowerCase() === "true";
const selectedPlatforms = (
  ENV_PLATFORMS.length ? ENV_PLATFORMS : Array.from(PLATFORMS)
).filter((p) => (DISABLE_IG ? p !== "instagram" : true));
const LOCKED_CREATIVE = process.env.LOCKED_CREATIVE_TYPE;
const selectedCreatives =
  LOCKED_CREATIVE &&
  (CREATIVE_TYPES as readonly string[]).includes(LOCKED_CREATIVE)
    ? [LOCKED_CREATIVE]
    : Array.from(CREATIVE_TYPES);

/**
 * Create a deterministic, canonical key for an action.
 * Interests are sorted to ensure order-independence.
 */
function actionToKey(a: AdAction): string {
  // Sort interests for canonical representation (order-independent)
  const sortedInterests = [...a.targetInterests].sort();
  return `${a.budgetAdjustment}|${a.targetAgeGroup}|${a.creativeType}|${a.platform}|${a.bidStrategy}|${sortedInterests.join(",")}`;
}

/** Deterministic discrete action grid built at module load. */
export const ACTIONS: AdAction[] = [];

/** O(1) action-to-index lookup map (built once at module load). */
const ACTION_INDEX_MAP = new Map<string, number>();

// Build both ACTIONS array and index map together
for (const b of BUDGET_STEPS)
  for (const age of AGE_GROUPS)
    for (const cr of selectedCreatives as any)
      for (const pf of selectedPlatforms as any)
        for (const bid of BID_STRATEGIES)
          for (const ib of INTEREST_BUNDLES) {
            const action: AdAction = {
              budgetAdjustment: b,
              targetAgeGroup: age as any,
              targetInterests: ib.slice() as string[],
              creativeType: cr as any,
              bidStrategy: bid as any,
              platform: pf as any,
            };
            const key = actionToKey(action);
            ACTION_INDEX_MAP.set(key, ACTIONS.length);
            ACTIONS.push(action);
          }

/**
 * O(1) lookup for action index in the ACTIONS grid.
 * Uses hash map instead of linear search for performance.
 * Returns -1 if action not found.
 */
export function actionToIndex(a: AdAction): number {
  const key = actionToKey(a);
  const idx = ACTION_INDEX_MAP.get(key);
  return idx !== undefined ? idx : -1;
}

/** Safe reverse mapping from index to action in the ACTIONS grid. */
export function indexToAction(idx: number): AdAction {
  if (ACTIONS.length === 0) throw new Error("ACTIONS not initialized");
  const clamped = Math.max(0, Math.min(ACTIONS.length - 1, idx));
  return ACTIONS[clamped]!;
}

/** Get the total number of actions in the action space. */
export function getActionCount(): number {
  return ACTIONS.length;
}

/** Check if an action exists in the action space. */
export function isValidAction(a: AdAction): boolean {
  return actionToIndex(a) >= 0;
}

// DQN-REFAC NOTE:
// - These encoders provide the deterministic mapping required by the NN.
// - Keep category orders/vocabulary stable across training and production.
// - Action index lookup is now O(1) via hash map instead of O(n) linear search.
