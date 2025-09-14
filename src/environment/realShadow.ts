import { AdAction, AdEnvironmentState, RewardMetrics } from "../types";
import { RealShopifyDataSource, TimeWindow as Win } from "../datasources/shopify";
import { RealTikTokAdsAPI } from "../platforms/realTikTok";

export class RealShadowEnvironment {
  private shopify: RealShopifyDataSource;
  private tiktok: RealTikTokAdsAPI;
  private currentState: AdEnvironmentState;
  private timeStep = 0;
  private minHourly = 0.5;
  private maxHourly = 3.0;
  private cogsPerUnit = Number(process.env.PRINTFUL_COGS ?? process.env.COGS_PER_UNIT ?? 15);
  private dailyBudgetTarget = Number(process.env.DAILY_BUDGET_TARGET ?? 30);
  private lockedCreativeType: string | undefined = process.env.LOCKED_CREATIVE_TYPE;

  constructor(shopify?: RealShopifyDataSource, tiktok?: RealTikTokAdsAPI) {
    this.shopify = shopify ?? new RealShopifyDataSource();
    this.tiktok = tiktok ?? new RealTikTokAdsAPI();
    this.currentState = this.generateInitialState();
  }

  private generateInitialState(): AdEnvironmentState {
    const now = new Date();
    return {
      dayOfWeek: now.getDay(),
      hourOfDay: now.getHours(),
      currentBudget: 1.0,
      targetAgeGroup: "25-34",
      targetInterests: ["fashion"],
      creativeType: this.lockedCreativeType ?? "product",
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

  // Compose real metrics from Shopify (revenue, conversions) and TikTok (ad spend)
  private async composeMetrics(window: Win): Promise<RewardMetrics> {
    const sales = await this.shopify.getSales(window);
    const ad = await this.tiktok.getAdMetricsForWindow(window);
    const cogs = sales.conversions * this.cogsPerUnit;
    const grossMargin = sales.revenue - cogs;
    const adSpend = ad.adSpend || 0;
    const profit = sales.revenue - adSpend - cogs;
    const roas = adSpend > 0 ? sales.revenue / adSpend : 0;
    const marginRoas = adSpend > 0 ? grossMargin / adSpend : 0;
    return {
      revenue: sales.revenue,
      adSpend,
      profit,
      roas,
      cogs,
      grossMargin,
      marginRoas,
      conversions: sales.conversions,
    };
  }

  private calculateReward(metrics: RewardMetrics): number {
    let reward = metrics.profit / 1000;
    const marginRoas = metrics.marginRoas ?? (metrics.adSpend > 0 ? (metrics.grossMargin ?? 0) / metrics.adSpend : 0);
    if (marginRoas > 2.0) reward += 1.0;
    else if (marginRoas > 1.5) reward += 0.5;
    else if (marginRoas > 1.2) reward += 0.2;
    const hourlyCap = this.dailyBudgetTarget / 24;
    const excess = metrics.adSpend - hourlyCap;
    if (excess > 0) reward -= excess / 2;
    reward += (metrics.conversions || 0) * 0.01;
    return reward;
  }

  async step(action: AdAction): Promise<[AdEnvironmentState, number, boolean, RewardMetrics]> {
    // Clamp budget change locally; do not write to real platform in shadow mode
    const preBudget = this.currentState.currentBudget;
    const desired = preBudget * action.budgetAdjustment;
    const nextBudget = Math.max(this.minHourly, Math.min(this.maxHourly, desired));

    // Build 1-hour window ending now (or rolling synthetic hour for demo)
    const end = new Date();
    const start = new Date(end.getTime() - 60 * 60 * 1000);
    const metrics = await this.composeMetrics({ start, end });
    const reward = this.calculateReward(metrics);

    // Advance time
    const newState = { ...this.currentState };
    newState.hourOfDay = (newState.hourOfDay + 1) % 24;
    if (newState.hourOfDay === 0) newState.dayOfWeek = (newState.dayOfWeek + 1) % 7;
    newState.currentBudget = nextBudget;
    newState.targetAgeGroup = action.targetAgeGroup;
    newState.targetInterests = action.targetInterests;
    newState.creativeType = this.lockedCreativeType ?? action.creativeType;
    newState.platform = "tiktok";
    this.currentState = newState;
    this.timeStep++;
    const done = this.timeStep >= 24;
    return [this.currentState, reward, done, metrics];
  }
}

