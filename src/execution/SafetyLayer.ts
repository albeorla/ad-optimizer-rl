/**
 * Safety Layer: Circuit Breakers and Anomaly Detection for RTB.
 *
 * In production ad tech, the RL agent's output must pass through multiple
 * safety checks before being executed. A single bad decision can waste
 * significant budget or damage campaign performance.
 *
 * This module implements:
 * - Circuit Breakers: Automatic fallback when anomalies are detected
 * - Anomaly Detection: Statistical monitoring of key metrics
 * - Safe Mode: Fallback to conservative bidding strategy
 * - Bid Validation: Hard constraints on bid values
 * - Performance Monitoring: Trailing metrics with alerting
 *
 * The safety layer sits between the RL agent and the platform API:
 * RL Agent → Safety Layer → Platform API
 */

import { AdAction, RewardMetrics } from '../types';

/**
 * Circuit breaker states.
 */
export type CircuitState = 'closed' | 'open' | 'half_open';

/**
 * Configuration for the circuit breaker.
 */
export interface CircuitBreakerConfig {
  /** Number of failures before opening the circuit */
  failureThreshold: number;
  /** Time to wait before attempting half-open (ms) */
  recoveryTimeMs: number;
  /** Number of successes in half-open before closing */
  successThreshold: number;
  /** Time window for counting failures (ms) */
  failureWindowMs: number;
}

/**
 * Anomaly detection thresholds.
 */
export interface AnomalyThresholds {
  /** Win rate below this triggers alert */
  minWinRate: number;
  /** Win rate above this triggers alert (overpaying) */
  maxWinRate: number;
  /** ROAS below this triggers alert */
  minROAS: number;
  /** CPA above this triggers alert */
  maxCPA: number;
  /** Spend rate deviation from target (%) */
  maxSpendDeviation: number;
  /** Maximum consecutive hours without conversion */
  maxHoursNoConversion: number;
  /** Click-through rate below this triggers alert */
  minCTR: number;
  /** Conversion rate below this triggers alert */
  minCVR: number;
}

/**
 * Safe mode bidding configuration.
 */
export interface SafeModeConfig {
  /** Base bid in safe mode */
  baseBid: number;
  /** Maximum bid multiplier in safe mode */
  maxMultiplier: number;
  /** Preferred bid strategy in safe mode */
  bidStrategy: 'CPC' | 'CPM' | 'CPA';
  /** Target age groups in safe mode */
  targetAgeGroups: string[];
  /** Whether to pause completely or continue with reduced bids */
  pauseCompletely: boolean;
}

/**
 * Alert severity levels.
 */
export type AlertSeverity = 'info' | 'warning' | 'critical';

/**
 * Safety alert structure.
 */
export interface SafetyAlert {
  id: string;
  timestamp: number;
  severity: AlertSeverity;
  type: string;
  message: string;
  metric?: string;
  currentValue?: number;
  threshold?: number;
  recommendation: string;
}

/**
 * Validation result for a bid action.
 */
export interface BidValidationResult {
  isValid: boolean;
  originalAction: AdAction;
  modifiedAction?: AdAction;
  violations: string[];
  warnings: string[];
}

/**
 * Safety layer status.
 */
export interface SafetyStatus {
  circuitState: CircuitState;
  isInSafeMode: boolean;
  activeAlerts: SafetyAlert[];
  metrics: TrailingMetrics;
  lastCheck: number;
}

/**
 * Trailing metrics for monitoring.
 */
export interface TrailingMetrics {
  winRate: number;
  roas: number;
  cpa: number;
  ctr: number;
  cvr: number;
  spendRate: number;
  targetSpendRate: number;
  hoursWithoutConversion: number;
  impressions: number;
  clicks: number;
  conversions: number;
  spend: number;
  revenue: number;
}

const DEFAULT_CIRCUIT_CONFIG: CircuitBreakerConfig = {
  failureThreshold: 5,
  recoveryTimeMs: 5 * 60 * 1000, // 5 minutes
  successThreshold: 3,
  failureWindowMs: 60 * 1000, // 1 minute
};

const DEFAULT_ANOMALY_THRESHOLDS: AnomalyThresholds = {
  minWinRate: 0.01, // 1% - bidding too low
  maxWinRate: 0.50, // 50% - overpaying
  minROAS: 0.5, // Below 0.5 is losing money
  maxCPA: Infinity, // Set based on campaign target
  maxSpendDeviation: 0.30, // 30% deviation from target
  maxHoursNoConversion: 6,
  minCTR: 0.001, // 0.1%
  minCVR: 0.001, // 0.1%
};

const DEFAULT_SAFE_MODE: SafeModeConfig = {
  baseBid: 0.50,
  maxMultiplier: 1.2,
  bidStrategy: 'CPC',
  targetAgeGroups: ['25-34'], // Typically highest converting
  pauseCompletely: false,
};

/**
 * Circuit Breaker: Prevents cascading failures.
 *
 * States:
 * - CLOSED: Normal operation, monitoring for failures
 * - OPEN: Failures exceeded threshold, blocking operations
 * - HALF_OPEN: Testing if system has recovered
 */
export class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failureTimestamps: number[] = [];
  private successCount = 0;
  private lastStateChange = Date.now();
  private config: CircuitBreakerConfig;

  constructor(config: Partial<CircuitBreakerConfig> = {}) {
    this.config = { ...DEFAULT_CIRCUIT_CONFIG, ...config };
  }

  /**
   * Record a successful operation.
   */
  recordSuccess(): void {
    if (this.state === 'half_open') {
      this.successCount++;
      if (this.successCount >= this.config.successThreshold) {
        this.transitionTo('closed');
      }
    }
    // In closed state, success is normal - no action needed
  }

  /**
   * Record a failed operation.
   */
  recordFailure(): void {
    const now = Date.now();

    if (this.state === 'half_open') {
      // Any failure in half-open goes back to open
      this.transitionTo('open');
      return;
    }

    if (this.state === 'closed') {
      // Add to failure window
      this.failureTimestamps.push(now);

      // Clean old failures outside window
      const windowStart = now - this.config.failureWindowMs;
      this.failureTimestamps = this.failureTimestamps.filter((t) => t > windowStart);

      // Check threshold
      if (this.failureTimestamps.length >= this.config.failureThreshold) {
        this.transitionTo('open');
      }
    }
  }

  /**
   * Check if operations are allowed.
   */
  isAllowed(): boolean {
    this.checkRecovery();
    return this.state !== 'open';
  }

  /**
   * Get current circuit state.
   */
  getState(): CircuitState {
    this.checkRecovery();
    return this.state;
  }

  /**
   * Force circuit to specific state (for manual intervention).
   */
  forceState(state: CircuitState): void {
    this.transitionTo(state);
  }

  /**
   * Reset the circuit breaker.
   */
  reset(): void {
    this.transitionTo('closed');
    this.failureTimestamps = [];
    this.successCount = 0;
  }

  private transitionTo(newState: CircuitState): void {
    this.state = newState;
    this.lastStateChange = Date.now();
    this.successCount = 0;
    this.failureTimestamps = [];
  }

  private checkRecovery(): void {
    if (this.state === 'open') {
      const elapsed = Date.now() - this.lastStateChange;
      if (elapsed >= this.config.recoveryTimeMs) {
        this.transitionTo('half_open');
      }
    }
  }

  getDiagnostics(): {
    state: CircuitState;
    recentFailures: number;
    successCount: number;
    timeSinceStateChange: number;
  } {
    return {
      state: this.state,
      recentFailures: this.failureTimestamps.length,
      successCount: this.successCount,
      timeSinceStateChange: Date.now() - this.lastStateChange,
    };
  }
}

/**
 * Anomaly Detector: Statistical monitoring of campaign metrics.
 */
export class AnomalyDetector {
  private thresholds: AnomalyThresholds;
  private alertHistory: SafetyAlert[] = [];
  private metricsHistory: TrailingMetrics[] = [];
  private maxHistorySize = 1000;

  constructor(thresholds: Partial<AnomalyThresholds> = {}) {
    this.thresholds = { ...DEFAULT_ANOMALY_THRESHOLDS, ...thresholds };
  }

  /**
   * Check metrics for anomalies.
   */
  checkMetrics(metrics: TrailingMetrics): SafetyAlert[] {
    const alerts: SafetyAlert[] = [];
    const now = Date.now();

    // Win rate checks
    if (metrics.winRate < this.thresholds.minWinRate && metrics.impressions > 100) {
      alerts.push({
        id: `alert_${now}_winrate_low`,
        timestamp: now,
        severity: 'warning',
        type: 'win_rate_low',
        message: `Win rate critically low: ${(metrics.winRate * 100).toFixed(2)}%`,
        metric: 'winRate',
        currentValue: metrics.winRate,
        threshold: this.thresholds.minWinRate,
        recommendation: 'Increase bid multiplier or review targeting',
      });
    }

    if (metrics.winRate > this.thresholds.maxWinRate && metrics.impressions > 100) {
      alerts.push({
        id: `alert_${now}_winrate_high`,
        timestamp: now,
        severity: 'warning',
        type: 'win_rate_high',
        message: `Win rate too high (overpaying): ${(metrics.winRate * 100).toFixed(2)}%`,
        metric: 'winRate',
        currentValue: metrics.winRate,
        threshold: this.thresholds.maxWinRate,
        recommendation: 'Reduce bid multiplier - winning too many auctions',
      });
    }

    // ROAS check
    if (metrics.roas < this.thresholds.minROAS && metrics.spend > 10) {
      alerts.push({
        id: `alert_${now}_roas_low`,
        timestamp: now,
        severity: 'critical',
        type: 'roas_low',
        message: `ROAS below threshold: ${metrics.roas.toFixed(2)}`,
        metric: 'roas',
        currentValue: metrics.roas,
        threshold: this.thresholds.minROAS,
        recommendation: 'Review campaign targeting and creative performance',
      });
    }

    // CPA check
    if (metrics.cpa > this.thresholds.maxCPA && metrics.conversions > 0) {
      alerts.push({
        id: `alert_${now}_cpa_high`,
        timestamp: now,
        severity: 'critical',
        type: 'cpa_high',
        message: `CPA exceeds target: $${metrics.cpa.toFixed(2)}`,
        metric: 'cpa',
        currentValue: metrics.cpa,
        threshold: this.thresholds.maxCPA,
        recommendation: 'Reduce bids or improve conversion optimization',
      });
    }

    // Spend rate deviation
    const spendDeviation = Math.abs(metrics.spendRate - metrics.targetSpendRate) / metrics.targetSpendRate;
    if (spendDeviation > this.thresholds.maxSpendDeviation && metrics.targetSpendRate > 0) {
      alerts.push({
        id: `alert_${now}_spend_deviation`,
        timestamp: now,
        severity: 'warning',
        type: 'spend_deviation',
        message: `Spend rate ${spendDeviation > 0 ? 'above' : 'below'} target by ${(spendDeviation * 100).toFixed(1)}%`,
        metric: 'spendRate',
        currentValue: metrics.spendRate,
        threshold: metrics.targetSpendRate,
        recommendation: 'Adjust PID controller or review pacing strategy',
      });
    }

    // Conversion drought
    if (metrics.hoursWithoutConversion >= this.thresholds.maxHoursNoConversion) {
      alerts.push({
        id: `alert_${now}_no_conversions`,
        timestamp: now,
        severity: 'critical',
        type: 'no_conversions',
        message: `No conversions for ${metrics.hoursWithoutConversion} hours`,
        metric: 'hoursWithoutConversion',
        currentValue: metrics.hoursWithoutConversion,
        threshold: this.thresholds.maxHoursNoConversion,
        recommendation: 'Consider pausing campaign or switching to safe mode',
      });
    }

    // CTR check
    if (metrics.ctr < this.thresholds.minCTR && metrics.impressions > 1000) {
      alerts.push({
        id: `alert_${now}_ctr_low`,
        timestamp: now,
        severity: 'warning',
        type: 'ctr_low',
        message: `CTR below threshold: ${(metrics.ctr * 100).toFixed(3)}%`,
        metric: 'ctr',
        currentValue: metrics.ctr,
        threshold: this.thresholds.minCTR,
        recommendation: 'Review ad creative and targeting relevance',
      });
    }

    // Store in history
    for (const alert of alerts) {
      this.alertHistory.push(alert);
      if (this.alertHistory.length > this.maxHistorySize) {
        this.alertHistory.shift();
      }
    }

    // Store metrics
    this.metricsHistory.push(metrics);
    if (this.metricsHistory.length > this.maxHistorySize) {
      this.metricsHistory.shift();
    }

    return alerts;
  }

  /**
   * Set target CPA threshold (campaign-specific).
   */
  setTargetCPA(targetCPA: number): void {
    this.thresholds.maxCPA = targetCPA * 1.5; // Allow 50% overage before alert
  }

  /**
   * Get recent alerts.
   */
  getRecentAlerts(count = 10): SafetyAlert[] {
    return this.alertHistory.slice(-count);
  }

  /**
   * Get critical alerts (active).
   */
  getCriticalAlerts(): SafetyAlert[] {
    const oneHourAgo = Date.now() - 60 * 60 * 1000;
    return this.alertHistory.filter(
      (a) => a.severity === 'critical' && a.timestamp > oneHourAgo
    );
  }

  /**
   * Clear alert history.
   */
  clearAlerts(): void {
    this.alertHistory = [];
  }

  /**
   * Check for specific anomaly pattern (e.g., sustained low ROAS).
   */
  detectPattern(
    metricName: keyof TrailingMetrics,
    condition: (value: number) => boolean,
    minOccurrences: number
  ): boolean {
    const recentMetrics = this.metricsHistory.slice(-minOccurrences);
    if (recentMetrics.length < minOccurrences) return false;

    return recentMetrics.every((m) => {
      const value = m[metricName];
      return typeof value === 'number' && condition(value);
    });
  }
}

/**
 * Bid Validator: Hard constraints on bid actions.
 */
export class BidValidator {
  private maxBid: number;
  private minBid: number;
  private allowedPlatforms: Set<string>;
  private allowedBidStrategies: Set<string>;
  private maxBudgetMultiplier: number;

  constructor(options: {
    maxBid?: number;
    minBid?: number;
    allowedPlatforms?: string[];
    allowedBidStrategies?: string[];
    maxBudgetMultiplier?: number;
  } = {}) {
    this.maxBid = options.maxBid ?? 10.0;
    this.minBid = options.minBid ?? 0.01;
    this.allowedPlatforms = new Set(options.allowedPlatforms ?? ['tiktok', 'instagram']);
    this.allowedBidStrategies = new Set(options.allowedBidStrategies ?? ['CPC', 'CPM', 'CPA']);
    this.maxBudgetMultiplier = options.maxBudgetMultiplier ?? 2.0;
  }

  /**
   * Validate and potentially modify a bid action.
   */
  validate(action: AdAction, currentBudget: number): BidValidationResult {
    const violations: string[] = [];
    const warnings: string[] = [];
    let modifiedAction = { ...action };
    let needsModification = false;

    // Check platform
    if (!this.allowedPlatforms.has(action.platform)) {
      violations.push(`Invalid platform: ${action.platform}`);
      modifiedAction.platform = 'tiktok'; // Default fallback
      needsModification = true;
    }

    // Check bid strategy
    if (!this.allowedBidStrategies.has(action.bidStrategy)) {
      violations.push(`Invalid bid strategy: ${action.bidStrategy}`);
      modifiedAction.bidStrategy = 'CPC'; // Default fallback
      needsModification = true;
    }

    // Check budget adjustment bounds
    const proposedBudget = currentBudget * action.budgetAdjustment;
    if (proposedBudget > currentBudget * this.maxBudgetMultiplier) {
      warnings.push(`Budget adjustment too aggressive: ${action.budgetAdjustment}`);
      modifiedAction.budgetAdjustment = this.maxBudgetMultiplier;
      needsModification = true;
    }

    if (action.budgetAdjustment < 0.5) {
      warnings.push(`Budget adjustment too aggressive (reduction): ${action.budgetAdjustment}`);
      modifiedAction.budgetAdjustment = 0.5;
      needsModification = true;
    }

    // Check for empty targets
    if (!action.targetAgeGroup) {
      violations.push('Missing target age group');
      modifiedAction.targetAgeGroup = '25-34';
      needsModification = true;
    }

    if (!action.targetInterests || action.targetInterests.length === 0) {
      warnings.push('No target interests specified');
    }

    // Build result - only include modifiedAction if needed
    const result: BidValidationResult = {
      isValid: violations.length === 0,
      originalAction: action,
      violations,
      warnings,
    };
    if (needsModification) {
      result.modifiedAction = modifiedAction;
    }
    return result;
  }

  /**
   * Set maximum allowed bid.
   */
  setMaxBid(maxBid: number): void {
    this.maxBid = maxBid;
  }

  /**
   * Add allowed platform.
   */
  addPlatform(platform: string): void {
    this.allowedPlatforms.add(platform);
  }
}

/**
 * Safety Layer: Main integration point for all safety mechanisms.
 */
export class SafetyLayer {
  private circuitBreaker: CircuitBreaker;
  private anomalyDetector: AnomalyDetector;
  private bidValidator: BidValidator;
  private safeModeConfig: SafeModeConfig;

  private isInSafeMode = false;
  private safeModeReason: string | undefined;
  private safeModeStartTime: number | undefined;

  constructor(options: {
    circuitConfig?: Partial<CircuitBreakerConfig>;
    anomalyThresholds?: Partial<AnomalyThresholds>;
    bidValidatorOptions?: {
      maxBid?: number;
      minBid?: number;
      allowedPlatforms?: string[];
      allowedBidStrategies?: string[];
      maxBudgetMultiplier?: number;
    };
    safeModeConfig?: Partial<SafeModeConfig>;
  } = {}) {
    this.circuitBreaker = new CircuitBreaker(options.circuitConfig);
    this.anomalyDetector = new AnomalyDetector(options.anomalyThresholds);
    this.bidValidator = new BidValidator(options.bidValidatorOptions);
    this.safeModeConfig = { ...DEFAULT_SAFE_MODE, ...options.safeModeConfig };
  }

  /**
   * Process an action through all safety checks.
   */
  processAction(
    action: AdAction,
    metrics: TrailingMetrics,
    currentBudget: number
  ): {
    action: AdAction;
    allowed: boolean;
    inSafeMode: boolean;
    alerts: SafetyAlert[];
    validation: BidValidationResult;
  } {
    // Check circuit breaker
    if (!this.circuitBreaker.isAllowed()) {
      return {
        action: this.getSafeModeAction(),
        allowed: false,
        inSafeMode: true,
        alerts: [{
          id: `circuit_open_${Date.now()}`,
          timestamp: Date.now(),
          severity: 'critical',
          type: 'circuit_open',
          message: 'Circuit breaker is open - using safe mode',
          recommendation: 'Wait for circuit to recover or manually reset',
        }],
        validation: {
          isValid: false,
          originalAction: action,
          violations: ['Circuit breaker open'],
          warnings: [],
        },
      };
    }

    // Check for anomalies
    const alerts = this.anomalyDetector.checkMetrics(metrics);
    const criticalAlerts = alerts.filter((a) => a.severity === 'critical');

    // Enter safe mode if critical alerts
    if (criticalAlerts.length > 0 && !this.isInSafeMode) {
      this.enterSafeMode(criticalAlerts[0]?.message ?? 'Critical alert detected');
    }

    // Validate the action
    const validation = this.bidValidator.validate(action, currentBudget);

    // If in safe mode, use safe action
    if (this.isInSafeMode) {
      if (this.safeModeConfig.pauseCompletely) {
        return {
          action: { ...action, budgetAdjustment: 0 },
          allowed: false,
          inSafeMode: true,
          alerts,
          validation,
        };
      }
      return {
        action: this.getSafeModeAction(),
        allowed: true,
        inSafeMode: true,
        alerts,
        validation,
      };
    }

    // Use validated action
    const finalAction = validation.modifiedAction ?? action;

    return {
      action: finalAction,
      allowed: validation.isValid || validation.modifiedAction !== undefined,
      inSafeMode: false,
      alerts,
      validation,
    };
  }

  /**
   * Record operation outcome.
   */
  recordOutcome(success: boolean): void {
    if (success) {
      this.circuitBreaker.recordSuccess();
    } else {
      this.circuitBreaker.recordFailure();
    }
  }

  /**
   * Enter safe mode.
   */
  enterSafeMode(reason: string): void {
    this.isInSafeMode = true;
    this.safeModeReason = reason;
    this.safeModeStartTime = Date.now();
  }

  /**
   * Exit safe mode.
   */
  exitSafeMode(): void {
    this.isInSafeMode = false;
    this.safeModeReason = undefined;
    this.safeModeStartTime = undefined;
  }

  /**
   * Check if should exit safe mode based on recovery.
   */
  checkSafeModeRecovery(metrics: TrailingMetrics): boolean {
    if (!this.isInSafeMode) return false;

    // Recovery conditions
    const goodROAS = metrics.roas >= 1.0;
    const hasConversions = metrics.conversions > 0;
    const stableSpend = Math.abs(metrics.spendRate - metrics.targetSpendRate) < 0.1;

    if (goodROAS && hasConversions && stableSpend) {
      this.exitSafeMode();
      return true;
    }

    return false;
  }

  /**
   * Get safe mode action.
   */
  private getSafeModeAction(): AdAction {
    return {
      budgetAdjustment: 0.8, // Conservative spend
      targetAgeGroup: this.safeModeConfig.targetAgeGroups[0] ?? '25-34',
      targetInterests: ['fashion'], // Generic high-intent interest
      creativeType: 'product', // Direct product showcase
      bidStrategy: this.safeModeConfig.bidStrategy,
      platform: 'tiktok',
    };
  }

  /**
   * Get current status.
   */
  getStatus(): SafetyStatus {
    return {
      circuitState: this.circuitBreaker.getState(),
      isInSafeMode: this.isInSafeMode,
      activeAlerts: this.anomalyDetector.getCriticalAlerts(),
      metrics: {
        winRate: 0,
        roas: 0,
        cpa: 0,
        ctr: 0,
        cvr: 0,
        spendRate: 0,
        targetSpendRate: 0,
        hoursWithoutConversion: 0,
        impressions: 0,
        clicks: 0,
        conversions: 0,
        spend: 0,
        revenue: 0,
      },
      lastCheck: Date.now(),
    };
  }

  /**
   * Reset all safety mechanisms.
   */
  reset(): void {
    this.circuitBreaker.reset();
    this.anomalyDetector.clearAlerts();
    this.exitSafeMode();
  }

  // Expose sub-components for advanced usage
  getCircuitBreaker(): CircuitBreaker {
    return this.circuitBreaker;
  }

  getAnomalyDetector(): AnomalyDetector {
    return this.anomalyDetector;
  }

  getBidValidator(): BidValidator {
    return this.bidValidator;
  }
}

/**
 * Real-time monitoring metrics aggregator.
 */
export class MetricsAggregator {
  private windowMs: number;
  private impressions: Array<{ timestamp: number; won: boolean }> = [];
  private clicks: Array<{ timestamp: number }> = [];
  private conversions: Array<{ timestamp: number; revenue: number }> = [];
  private spend: Array<{ timestamp: number; amount: number }> = [];

  private lastConversionTime = 0;
  private targetSpendRate = 0;

  constructor(windowMs = 60 * 60 * 1000) { // 1 hour default
    this.windowMs = windowMs;
  }

  recordImpression(won: boolean): void {
    this.impressions.push({ timestamp: Date.now(), won });
    this.cleanup();
  }

  recordClick(): void {
    this.clicks.push({ timestamp: Date.now() });
    this.cleanup();
  }

  recordConversion(revenue: number): void {
    this.conversions.push({ timestamp: Date.now(), revenue });
    this.lastConversionTime = Date.now();
    this.cleanup();
  }

  recordSpend(amount: number): void {
    this.spend.push({ timestamp: Date.now(), amount });
    this.cleanup();
  }

  setTargetSpendRate(rate: number): void {
    this.targetSpendRate = rate;
  }

  private cleanup(): void {
    const cutoff = Date.now() - this.windowMs;
    this.impressions = this.impressions.filter((e) => e.timestamp > cutoff);
    this.clicks = this.clicks.filter((e) => e.timestamp > cutoff);
    this.conversions = this.conversions.filter((e) => e.timestamp > cutoff);
    this.spend = this.spend.filter((e) => e.timestamp > cutoff);
  }

  getMetrics(): TrailingMetrics {
    const now = Date.now();
    const totalImpressions = this.impressions.length;
    const wonImpressions = this.impressions.filter((e) => e.won).length;
    const totalClicks = this.clicks.length;
    const totalConversions = this.conversions.length;
    const totalSpend = this.spend.reduce((sum, e) => sum + e.amount, 0);
    const totalRevenue = this.conversions.reduce((sum, e) => sum + e.revenue, 0);

    const hoursWithoutConversion = this.lastConversionTime > 0
      ? (now - this.lastConversionTime) / (60 * 60 * 1000)
      : 0;

    return {
      winRate: totalImpressions > 0 ? wonImpressions / totalImpressions : 0,
      roas: totalSpend > 0 ? totalRevenue / totalSpend : 0,
      cpa: totalConversions > 0 ? totalSpend / totalConversions : Infinity,
      ctr: wonImpressions > 0 ? totalClicks / wonImpressions : 0,
      cvr: totalClicks > 0 ? totalConversions / totalClicks : 0,
      spendRate: totalSpend / (this.windowMs / (60 * 60 * 1000)), // Per hour
      targetSpendRate: this.targetSpendRate,
      hoursWithoutConversion,
      impressions: totalImpressions,
      clicks: totalClicks,
      conversions: totalConversions,
      spend: totalSpend,
      revenue: totalRevenue,
    };
  }

  reset(): void {
    this.impressions = [];
    this.clicks = [];
    this.conversions = [];
    this.spend = [];
    this.lastConversionTime = 0;
  }
}
