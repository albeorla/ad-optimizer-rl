import { RewardMetrics } from "../types";

/** Closed-open time window [start, end) to query data for. */
export interface TimeWindow {
  start: Date;
  end: Date;
}

/** Minimal Shopify aggregation for revenue and attributed conversions. */
export interface ShopifySales {
  revenue: number; // total gross sales in USD
  conversions: number; // count of orders attributable to ads (approx)
}

/**
 * Shopify Admin API client (scaffold).
 *
 * Replace stubs with real HTTP requests using API creds. Responsible for
 * aggregating revenue and unit counts per time window.
 */
export class RealShopifyDataSource {
  private apiKey: string | undefined;
  private storeDomain: string | undefined;

  constructor(opts?: { apiKey?: string; storeDomain?: string }) {
    this.apiKey = opts?.apiKey ?? process.env.SHOPIFY_API_KEY;
    this.storeDomain = opts?.storeDomain ?? process.env.SHOPIFY_STORE_DOMAIN;
  }

  // Scaffolding: fetch aggregated revenue and conversions for a window.
  // TODO: implement with Shopify Admin API + attribution via UTM/pixel mapping.
  async getSales(window: TimeWindow): Promise<ShopifySales> {
    // Placeholder: return zeros to keep shadow loop running.
    // Replace with real HTTP requests and pagination.
    return { revenue: 0, conversions: 0 };
  }
}
