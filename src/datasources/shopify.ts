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

/** Product inventory information. */
export interface ProductInventory {
  productId: string;
  variantId: string;
  sku: string;
  title: string;
  inventoryQuantity: number;
  availableForSale: boolean;
}

/** Inventory status with recommendations. */
export interface InventoryStatus {
  products: ProductInventory[];
  lowStockProducts: ProductInventory[]; // quantity < threshold
  outOfStockProducts: ProductInventory[]; // quantity = 0
  shouldPauseAds: boolean; // true if critical inventory issues
  reason?: string;
}

/**
 * Shopify Admin API client with inventory awareness.
 *
 * Replace stubs with real HTTP requests using API creds. Responsible for
 * aggregating revenue, unit counts, and inventory levels.
 *
 * Key features:
 * - Sales data aggregation per time window
 * - Inventory level tracking
 * - Low stock alerts and ad pause recommendations
 */
export class RealShopifyDataSource {
  private apiKey: string | undefined;
  private storeDomain: string | undefined;
  private accessToken: string | undefined;

  // Inventory thresholds
  private lowStockThreshold: number;
  private criticalStockThreshold: number;

  // Cache for inventory data (avoid excessive API calls)
  private inventoryCache: Map<string, ProductInventory> = new Map();
  private cacheExpiry = 0;
  private cacheTTL = 5 * 60 * 1000; // 5 minutes

  constructor(opts?: {
    apiKey?: string;
    storeDomain?: string;
    accessToken?: string;
    lowStockThreshold?: number;
    criticalStockThreshold?: number;
  }) {
    this.apiKey = opts?.apiKey ?? process.env.SHOPIFY_API_KEY;
    this.storeDomain = opts?.storeDomain ?? process.env.SHOPIFY_STORE_DOMAIN;
    this.accessToken = opts?.accessToken ?? process.env.SHOPIFY_ACCESS_TOKEN;
    this.lowStockThreshold = opts?.lowStockThreshold ?? 10;
    this.criticalStockThreshold = opts?.criticalStockThreshold ?? 3;
  }

  /**
   * Fetch aggregated revenue and conversions for a time window.
   * TODO: implement with Shopify Admin API + attribution via UTM/pixel mapping.
   */
  async getSales(window: TimeWindow): Promise<ShopifySales> {
    // Placeholder: return zeros to keep shadow loop running.
    // Replace with real HTTP requests and pagination.
    // Example: GET /admin/api/2024-01/orders.json?created_at_min=...&created_at_max=...
    return { revenue: 0, conversions: 0 };
  }

  /**
   * Fetch inventory levels for all products or specific product IDs.
   * Returns structured inventory data with low stock warnings.
   */
  async getInventoryStatus(productIds?: string[]): Promise<InventoryStatus> {
    // Check cache first
    if (Date.now() < this.cacheExpiry && this.inventoryCache.size > 0) {
      return this.buildInventoryStatus(
        Array.from(this.inventoryCache.values()),
      );
    }

    // Placeholder implementation - replace with real API calls
    // Example: GET /admin/api/2024-01/products.json?fields=id,title,variants
    const products = await this.fetchProductInventory(productIds);

    // Update cache
    this.inventoryCache.clear();
    for (const product of products) {
      this.inventoryCache.set(product.productId, product);
    }
    this.cacheExpiry = Date.now() + this.cacheTTL;

    return this.buildInventoryStatus(products);
  }

  /**
   * Check if ads should be paused for a specific product.
   * Returns true if inventory is critically low or out of stock.
   */
  async shouldPauseAdsForProduct(productId: string): Promise<{
    shouldPause: boolean;
    reason?: string;
    inventory?: ProductInventory;
  }> {
    const status = await this.getInventoryStatus([productId]);
    const product = status.products.find((p) => p.productId === productId);

    if (!product) {
      return { shouldPause: true, reason: "product_not_found" };
    }

    if (!product.availableForSale || product.inventoryQuantity <= 0) {
      return {
        shouldPause: true,
        reason: "out_of_stock",
        inventory: product,
      };
    }

    if (product.inventoryQuantity <= this.criticalStockThreshold) {
      return {
        shouldPause: true,
        reason: "critical_low_stock",
        inventory: product,
      };
    }

    return { shouldPause: false, inventory: product };
  }

  /**
   * Get inventory-adjusted bid multiplier.
   * Reduces bids as inventory gets low to avoid over-promising.
   */
  getInventoryBidMultiplier(inventoryQuantity: number): number {
    if (inventoryQuantity <= 0) {
      return 0; // Don't bid at all
    }

    if (inventoryQuantity <= this.criticalStockThreshold) {
      return 0.25; // Minimal bidding
    }

    if (inventoryQuantity <= this.lowStockThreshold) {
      // Linear scaling from 0.5 to 1.0 based on stock level
      const ratio =
        (inventoryQuantity - this.criticalStockThreshold) /
        (this.lowStockThreshold - this.criticalStockThreshold);
      return 0.5 + 0.5 * ratio;
    }

    return 1.0; // Full bidding
  }

  /**
   * Internal: Fetch product inventory from Shopify API.
   * TODO: Replace with real API implementation.
   */
  private async fetchProductInventory(
    productIds?: string[],
  ): Promise<ProductInventory[]> {
    // Placeholder: return mock data for development
    // In production, this would call:
    // GET /admin/api/2024-01/products.json
    // And then: GET /admin/api/2024-01/inventory_levels.json
    return [
      {
        productId: "mock-product-1",
        variantId: "mock-variant-1",
        sku: "TSHIRT-BLK-M",
        title: "Black T-Shirt (M)",
        inventoryQuantity: 50,
        availableForSale: true,
      },
      {
        productId: "mock-product-2",
        variantId: "mock-variant-2",
        sku: "TSHIRT-WHT-L",
        title: "White T-Shirt (L)",
        inventoryQuantity: 5,
        availableForSale: true,
      },
    ];
  }

  /**
   * Internal: Build inventory status from product list.
   */
  private buildInventoryStatus(products: ProductInventory[]): InventoryStatus {
    const lowStockProducts = products.filter(
      (p) =>
        p.inventoryQuantity > 0 &&
        p.inventoryQuantity <= this.lowStockThreshold,
    );

    const outOfStockProducts = products.filter(
      (p) => p.inventoryQuantity <= 0 || !p.availableForSale,
    );

    // Determine if ads should be paused globally
    const status: InventoryStatus = {
      products,
      lowStockProducts,
      outOfStockProducts,
      shouldPauseAds: false,
    };

    if (outOfStockProducts.length === products.length) {
      status.shouldPauseAds = true;
      status.reason = "all_products_out_of_stock";
    } else if (
      outOfStockProducts.length > 0 &&
      outOfStockProducts.length >= products.length * 0.5
    ) {
      status.shouldPauseAds = true;
      status.reason = "majority_products_out_of_stock";
    }

    return status;
  }

  /**
   * Clear the inventory cache (e.g., after inventory update).
   */
  clearCache(): void {
    this.inventoryCache.clear();
    this.cacheExpiry = 0;
  }

  /**
   * Check if the datasource is configured with API credentials.
   */
  isConfigured(): boolean {
    return !!(this.apiKey || this.accessToken) && !!this.storeDomain;
  }
}
