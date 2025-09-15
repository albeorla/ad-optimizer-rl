import { AdAction, AdEnvironmentState } from "../types";
import { MockTikTokAdsAPI } from "../platforms/mockTikTok";
import { MockInstagramAdsAPI } from "../platforms/mockInstagram";
import { AdPlatformAPI } from "../platforms/base";
// DQN-REFAC TODO:
// - Keep state fields stable and documented; encoder depends on deterministic categories.
// - If you add/rename features, update `src/agent/encoding.ts` and docs accordingly.
// - Consider exposing a 'feature view' for real adapters to ensure parity with simulator.

export class AdEnvironmentSimulator {
  private platforms: Map<string, AdPlatformAPI>;
  private currentState: AdEnvironmentState;
  private timeStep: number = 0;

  private minHourly = 0.5;
  private maxHourly = 3.0;
  private productPrice = 29.99;
  private cogsPerUnit = 15.0; // Printful base cost per shirt
  private allowedPlatforms: Array<"tiktok" | "instagram"> = [
    "tiktok",
    "instagram",
  ];
  private lockedCreativeType: string | undefined = undefined;
  private dailyBudgetTarget: number = Number(
    process.env.DAILY_BUDGET_TARGET || "30",
  );

  constructor(opts?: {
    shapingStrength?: number;
    minHourly?: number;
    maxHourly?: number;
    productPrice?: number;
    cogsPerUnit?: number;
    allowedPlatforms?: Array<"tiktok" | "instagram">;
    lockedCreativeType?: string;
    dailyBudgetTarget?: number;
  }) {
    const shaping = opts?.shapingStrength ?? 1;
    if (opts?.minHourly !== undefined) this.minHourly = opts.minHourly;
    if (opts?.maxHourly !== undefined) this.maxHourly = opts.maxHourly;
    this.productPrice = Number(
      opts?.productPrice ??
        process.env.TSHIRT_PRICE ??
        process.env.PRODUCT_PRICE ??
        this.productPrice,
    );
    this.cogsPerUnit = Number(
      opts?.cogsPerUnit ??
        process.env.PRINTFUL_COGS ??
        process.env.COGS_PER_UNIT ??
        this.cogsPerUnit,
    );
    // Allow locking platform set via opts or env (comma-separated)
    const envAllowed = (process.env.ALLOWED_PLATFORMS || "")
      .split(",")
      .map((s) => s.trim().toLowerCase())
      .filter((s) => s === "tiktok" || s === "instagram") as Array<
      "tiktok" | "instagram"
    >;
    this.allowedPlatforms =
      opts?.allowedPlatforms ??
      (envAllowed.length ? envAllowed : this.allowedPlatforms);
    // If Instagram is flagged disabled explicitly, filter it out
    if ((process.env.DISABLE_INSTAGRAM || "").toLowerCase() === "true") {
      this.allowedPlatforms = this.allowedPlatforms.filter(
        (p) => p !== "instagram",
      );
    }
    this.lockedCreativeType = (opts?.lockedCreativeType ??
      process.env.LOCKED_CREATIVE_TYPE) as string | undefined;
    this.dailyBudgetTarget = Number(
      opts?.dailyBudgetTarget ??
        process.env.DAILY_BUDGET_TARGET ??
        this.dailyBudgetTarget,
    );

    const entries: Array<[string, AdPlatformAPI]> = [];
    if (this.allowedPlatforms.includes("tiktok")) {
      entries.push([
        "tiktok",
        new MockTikTokAdsAPI(shaping, this.productPrice, this.cogsPerUnit),
      ]);
    }
    if (this.allowedPlatforms.includes("instagram")) {
      entries.push([
        "instagram",
        new MockInstagramAdsAPI(shaping, this.productPrice, this.cogsPerUnit),
      ]);
    }
    this.platforms = new Map<string, AdPlatformAPI>(entries);

    this.currentState = this.generateInitialState();
  }

  private generateInitialState(): AdEnvironmentState {
    return {
      dayOfWeek: 0,
      hourOfDay: 12,
      // Start near $1/hour for small-budget regimes
      currentBudget: 1.0,
      targetAgeGroup: "25-34",
      targetInterests: ["fashion", "lifestyle"],
      creativeType: "product",
      platform: "tiktok",
      historicalCTR: 0.02,
      historicalCVR: 0.01,
      competitorActivity: 0.5,
      seasonality: 0.7,
    };
  }

  reset(): AdEnvironmentState {
    this.timeStep = 0;
    this.currentState = this.generateInitialState();
    return this.currentState;
  }

  step(
    action: AdAction,
  ): [AdEnvironmentState, number, boolean, import("../types").RewardMetrics] {
    // Enforce allowed platform(s)
    let platformKey = action.platform;
    if (!this.allowedPlatforms.includes(platformKey as any)) {
      // Coerce to first allowed platform
      platformKey = this.allowedPlatforms[0] ?? "tiktok";
    }
    // Optionally lock creative to a single asset/type
    const effectiveCreative = this.lockedCreativeType ?? action.creativeType;
    const sanitizedAction: AdAction = {
      ...action,
      platform: platformKey as any,
      creativeType: effectiveCreative,
    };

    const platform = this.platforms.get(platformKey);
    if (!platform) throw new Error(`Platform ${action.platform} not found`);

    // Clamp the proposed budget for this hour and adjust the action multiplier accordingly
    const preBudget = this.currentState.currentBudget;
    const desired = preBudget * sanitizedAction.budgetAdjustment;
    const clamped = Math.max(this.minHourly, Math.min(this.maxHourly, desired));
    const adjMultiplier = preBudget > 0 ? clamped / preBudget : 1.0;
    const adjustedAction: AdAction = {
      ...sanitizedAction,
      budgetAdjustment: adjMultiplier,
    };

    const metrics = platform.simulatePerformance(
      this.currentState,
      adjustedAction,
    );
    const reward = this.calculateReward(metrics);
    this.currentState = this.updateState({
      ...adjustedAction,
      budgetAdjustment: adjMultiplier,
    });
    this.timeStep++;
    const done = this.timeStep >= 24;
    return [this.currentState, reward, done, metrics];
  }

  // Reward shaping to guide learning toward good ROAS and sensible spend
  private calculateReward(metrics: import("../types").RewardMetrics): number {
    // Base reward on net profit (assumes platform metrics.profit already includes COGS)
    let reward = metrics.profit / 1000; // normalize
    // Bonus for strong margin-based ROAS = (revenue - cogs) / adSpend
    const grossMargin =
      metrics.grossMargin ?? metrics.revenue - (metrics.cogs ?? 0);
    const marginRoas =
      metrics.marginRoas ??
      (metrics.adSpend > 0 ? grossMargin / metrics.adSpend : 0);
    if (marginRoas > 2.0) reward += 1.0;
    else if (marginRoas > 1.5) reward += 0.5;
    else if (marginRoas > 1.2) reward += 0.2;
    // Penalty for exceeding hourly allowance based on daily target (default $30 → ~$1.25/hr)
    const hourlyCap = this.dailyBudgetTarget / 24;
    const excess = metrics.adSpend - hourlyCap;
    if (excess > 0) reward -= excess / 2; // moderate penalty on normalized scale
    // Small bonus per conversion
    reward += metrics.conversions * 0.01;
    return reward;
  }

  private updateState(action: AdAction): AdEnvironmentState {
    const newState = { ...this.currentState };
    newState.hourOfDay = (newState.hourOfDay + 1) % 24;
    if (newState.hourOfDay === 0)
      newState.dayOfWeek = (newState.dayOfWeek + 1) % 7;

    // Apply clamped budget to next state to avoid compounding growth
    const desired = this.currentState.currentBudget * action.budgetAdjustment;
    newState.currentBudget = Math.max(
      this.minHourly,
      Math.min(this.maxHourly, desired),
    );
    newState.targetAgeGroup = action.targetAgeGroup;
    newState.targetInterests = action.targetInterests;
    newState.creativeType = this.lockedCreativeType ?? action.creativeType;
    newState.platform = (
      this.allowedPlatforms.includes(action.platform as any)
        ? action.platform
        : (this.allowedPlatforms[0] ?? "tiktok")
    ) as any;

    newState.historicalCTR = Math.max(
      0.001,
      newState.historicalCTR + (Math.random() - 0.5) * 0.002,
    );
    newState.historicalCVR = Math.max(
      0.001,
      newState.historicalCVR + (Math.random() - 0.5) * 0.001,
    );

    newState.competitorActivity = Math.min(
      1,
      Math.max(0, newState.competitorActivity + (Math.random() - 0.5) * 0.1),
    );
    newState.seasonality =
      0.7 + 0.3 * Math.sin((this.timeStep / 168) * Math.PI * 2);
    return newState;
  }
}
