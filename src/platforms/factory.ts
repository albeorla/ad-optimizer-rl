import { AdPlatformAPI } from "./base";
import { MockTikTokAdsAPI } from "./mockTikTok";
import { MockInstagramAdsAPI } from "./mockInstagram";
/** Simple registry/factory for platform adapters. */
export class AdPlatformFactory {
  private static platforms: Map<string, AdPlatformAPI> = new Map<
    string,
    AdPlatformAPI
  >([
    ["tiktok", new MockTikTokAdsAPI() as AdPlatformAPI],
    ["instagram", new MockInstagramAdsAPI() as AdPlatformAPI],
  ]);

  static getPlatform(platform: string): AdPlatformAPI {
    const api = this.platforms.get(platform);
    if (!api) {
      throw new Error(`Platform ${platform} not supported`);
    }
    return api;
  }

  static registerPlatform(name: string, api: AdPlatformAPI): void {
    this.platforms.set(name, api);
  }
}
