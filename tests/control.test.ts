/**
 * Test Suite for PID Controllers and Budget Pacing.
 */

import {
  PidPacer,
  CpaPidController,
  DualPidController,
  applyBidModifier,
  computeAdaptiveGains,
  PID_PRESETS,
} from '../src/control/PidController';

describe('PidPacer', () => {
  describe('basic pacing', () => {
    it('should return multiplier of 1.0 when on track', () => {
      const pacer = new PidPacer(100, 24);
      // Spent $50 at hour 12 = exactly on track
      const multiplier = pacer.getMultiplier(50, 12);
      expect(multiplier).toBeCloseTo(1.0, 1);
    });

    it('should increase multiplier when underspending', () => {
      const pacer = new PidPacer(100, 24);
      // Spent $25 at hour 12 = significantly underspending
      const multiplier = pacer.getMultiplier(25, 12);
      expect(multiplier).toBeGreaterThan(1.0);
    });

    it('should decrease multiplier when overspending', () => {
      const pacer = new PidPacer(100, 24);
      // Spent $75 at hour 12 = significantly overspending
      const multiplier = pacer.getMultiplier(75, 12);
      expect(multiplier).toBeLessThan(1.0);
    });

    it('should respect output bounds', () => {
      const pacer = new PidPacer(100, 24, { outputMin: 0.5, outputMax: 2.0 });
      // Extreme underspend
      const mult1 = pacer.getMultiplier(0, 20);
      expect(mult1).toBeLessThanOrEqual(2.0);

      pacer.reset();
      // Extreme overspend
      const mult2 = pacer.getMultiplier(100, 5);
      expect(mult2).toBeGreaterThanOrEqual(0.5);
    });

    it('should accumulate integral error over time', () => {
      const pacer = new PidPacer(100, 24);
      // Consistently underspend
      pacer.getMultiplier(10, 6);
      pacer.getMultiplier(20, 12);
      const mult = pacer.getMultiplier(30, 18);
      // Integral should push multiplier higher
      expect(mult).toBeGreaterThan(1.2);
    });
  });

  describe('trajectory info', () => {
    it('should correctly identify underspend status', () => {
      const pacer = new PidPacer(100, 24);
      const info = pacer.getTrajectoryInfo(25, 12);
      expect(info.pacingStatus).toBe('under');
      expect(info.spendErrorPercent).toBeGreaterThan(0);
    });

    it('should correctly identify overspend status', () => {
      const pacer = new PidPacer(100, 24);
      const info = pacer.getTrajectoryInfo(75, 12);
      expect(info.pacingStatus).toBe('over');
      expect(info.spendErrorPercent).toBeLessThan(0);
    });
  });

  describe('reset', () => {
    it('should reset integral and derivative state', () => {
      const pacer = new PidPacer(100, 24);
      pacer.getMultiplier(10, 12);
      pacer.getMultiplier(20, 14);
      pacer.reset();
      expect(pacer.getIntegral()).toBe(0);
      expect(pacer.getLastMultiplier()).toBe(1.0);
    });
  });
});

describe('CpaPidController', () => {
  describe('CPA control', () => {
    it('should return multiplier > 1 when CPA is below target', () => {
      const controller = new CpaPidController(10); // Target CPA = $10
      // Actual CPA = $5 (excellent)
      const multiplier = controller.getMultiplier(50, 10);
      expect(multiplier).toBeGreaterThan(1.0);
    });

    it('should return multiplier < 1 when CPA is above target', () => {
      const controller = new CpaPidController(10); // Target CPA = $10
      // Actual CPA = $20 (bad)
      const multiplier = controller.getMultiplier(200, 10);
      expect(multiplier).toBeLessThan(1.0);
    });

    it('should handle zero conversions gracefully', () => {
      const controller = new CpaPidController(10);
      const multiplier = controller.getMultiplier(100, 0);
      // Should reduce bids when no conversions
      expect(multiplier).toBeLessThan(1.0);
    });

    it('should correctly report CPA info', () => {
      const controller = new CpaPidController(10);
      const info = controller.getCpaInfo(100, 20); // CPA = $5
      expect(info.actualCpa).toBe(5);
      expect(info.performanceStatus).toBe('efficient');
    });
  });
});

describe('DualPidController', () => {
  it('should use minimum of budget and CPA multipliers', () => {
    const controller = new DualPidController({
      totalBudget: 100,
      campaignDuration: 24,
      targetCpa: 10,
    });

    // Budget on track but CPA too high
    const result = controller.getMultiplier({
      currentSpend: 50,
      elapsedTime: 12,
      recentSpend: 100,
      recentConversions: 5, // CPA = $20, above target
    });

    // CPA constraint should be binding
    expect(controller.getBindingConstraint()).toBe('cpa');
    expect(result).toBeLessThan(1.0);
  });

  it('should track binding constraint correctly', () => {
    const controller = new DualPidController({
      totalBudget: 100,
      campaignDuration: 24,
      targetCpa: 10,
    });

    // Significantly underspending but CPA is good
    controller.getMultiplier({
      currentSpend: 10,
      elapsedTime: 12,
      recentSpend: 10,
      recentConversions: 2, // CPA = $5, great
    });

    // Budget constraint should be binding (needs to spend more)
    expect(controller.getBindingConstraint()).toBe('budget');
  });

  it('should provide comprehensive diagnostics', () => {
    const controller = new DualPidController({
      totalBudget: 100,
      campaignDuration: 24,
      targetCpa: 10,
    });

    controller.getMultiplier({
      currentSpend: 50,
      elapsedTime: 12,
      recentSpend: 50,
      recentConversions: 5,
    });

    const diag = controller.getDiagnostics();
    expect(diag).toHaveProperty('budgetMultiplier');
    expect(diag).toHaveProperty('cpaMultiplier');
    expect(diag).toHaveProperty('finalMultiplier');
    expect(diag).toHaveProperty('bindingConstraint');
  });
});

describe('applyBidModifier', () => {
  it('should apply multiplier correctly', () => {
    const result = applyBidModifier(1.0, 1.5);
    expect(result.finalBid).toBe(1.5);
    expect(result.wasConstrained).toBe(false);
  });

  it('should respect minimum bid constraint', () => {
    const result = applyBidModifier(0.005, 1.0, { minBid: 0.01 });
    expect(result.finalBid).toBe(0.01);
    expect(result.wasConstrained).toBe(true);
    expect(result.constraintReason).toBe('min_bid');
  });

  it('should respect maximum bid constraint', () => {
    const result = applyBidModifier(10, 2.0, { maxBid: 15 });
    expect(result.finalBid).toBe(15);
    expect(result.wasConstrained).toBe(true);
  });

  it('should clamp extreme multipliers', () => {
    const result = applyBidModifier(1.0, 10.0, { maxMultiplier: 3.0 });
    expect(result.finalBid).toBe(3.0);
    expect(result.wasConstrained).toBe(true);
  });
});

describe('computeAdaptiveGains', () => {
  const baseConfig = PID_PRESETS.balanced;

  it('should increase gains in early phase', () => {
    const gains = computeAdaptiveGains(baseConfig, {
      elapsedRatio: 0.1,
      spendRatio: 0.1,
      pacingError: 0,
    });
    expect(gains.kp).toBeGreaterThan(baseConfig.kp);
  });

  it('should decrease gains in late phase', () => {
    const gains = computeAdaptiveGains(baseConfig, {
      elapsedRatio: 0.9,
      spendRatio: 0.9,
      pacingError: 0,
    });
    expect(gains.kp).toBeLessThan(baseConfig.kp);
  });

  it('should boost gains during emergency underspend', () => {
    const gains = computeAdaptiveGains(baseConfig, {
      elapsedRatio: 0.5,
      spendRatio: 0.2,
      pacingError: 0.4, // Severely underspending
    });
    expect(gains.kp).toBeGreaterThan(baseConfig.kp * 1.2);
  });
});

// Test runner setup
function describe(name: string, fn: () => void) {
  console.log(`\n=== ${name} ===`);
  fn();
}

function it(name: string, fn: () => void) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
  } catch (e) {
    console.error(`  ✗ ${name}`);
    console.error(`    ${(e as Error).message}`);
  }
}

function expect(actual: any) {
  return {
    toBe: (expected: any) => {
      if (actual !== expected) throw new Error(`Expected ${expected}, got ${actual}`);
    },
    toBeCloseTo: (expected: number, precision: number) => {
      const diff = Math.abs(actual - expected);
      const threshold = Math.pow(10, -precision);
      if (diff > threshold) throw new Error(`Expected ${expected} ± ${threshold}, got ${actual}`);
    },
    toBeGreaterThan: (expected: number) => {
      if (actual <= expected) throw new Error(`Expected > ${expected}, got ${actual}`);
    },
    toBeLessThan: (expected: number) => {
      if (actual >= expected) throw new Error(`Expected < ${expected}, got ${actual}`);
    },
    toBeGreaterThanOrEqual: (expected: number) => {
      if (actual < expected) throw new Error(`Expected >= ${expected}, got ${actual}`);
    },
    toBeLessThanOrEqual: (expected: number) => {
      if (actual > expected) throw new Error(`Expected <= ${expected}, got ${actual}`);
    },
    toHaveProperty: (prop: string) => {
      if (!(prop in actual)) throw new Error(`Expected property ${prop}`);
    },
  };
}

// Run tests
console.log('Running PID Controller Tests...\n');
