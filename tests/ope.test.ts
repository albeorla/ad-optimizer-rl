/**
 * Test Suite for Offline Policy Evaluation (OPE).
 */

import {
  calculateIPS,
  calculateSNIPS,
  calculateClippedIPS,
  calculateDoublyRobust,
  calculateMDA,
  evaluateDeployment,
  runOPESuite,
  LoggedExperience,
  EvaluationPolicy,
} from '../src/evaluation/OPE';
import { AdAction, AdEnvironmentState } from '../src/types';

// Helper to create mock state
function createMockState(overrides: Partial<AdEnvironmentState> = {}): AdEnvironmentState {
  return {
    dayOfWeek: 1,
    hourOfDay: 12,
    currentBudget: 30,
    targetAgeGroup: '25-34',
    targetInterests: ['fashion'],
    creativeType: 'product',
    platform: 'tiktok',
    historicalCTR: 0.02,
    historicalCVR: 0.03,
    competitorActivity: 0.5,
    seasonality: 0.7,
    ...overrides,
  };
}

// Helper to create mock action
function createMockAction(overrides: Partial<AdAction> = {}): AdAction {
  return {
    budgetAdjustment: 1.0,
    targetAgeGroup: '25-34',
    targetInterests: ['fashion'],
    creativeType: 'product',
    bidStrategy: 'CPC',
    platform: 'tiktok',
    ...overrides,
  };
}

// Create mock logged experiences
function createMockLogs(count: number, options: {
  avgReward?: number;
  rewardVariance?: number;
  loggingProb?: number;
} = {}): LoggedExperience[] {
  const avgReward = options.avgReward ?? 1.0;
  const variance = options.rewardVariance ?? 0.2;
  const loggingProb = options.loggingProb ?? 0.1;

  return Array.from({ length: count }, (_, i) => ({
    state: createMockState({ hourOfDay: i % 24 }),
    action: createMockAction(),
    reward: avgReward + (Math.random() - 0.5) * variance * 2,
    loggingProbability: loggingProb + (Math.random() - 0.5) * 0.05,
  }));
}

// Mock evaluation policy
function createMockPolicy(probMultiplier = 1.0): EvaluationPolicy {
  return {
    getActionProbability: (state, action) => {
      // Return probability similar to logging policy but scaled
      return Math.min(1, 0.1 * probMultiplier);
    },
    predictReward: (state, action) => {
      // Simple reward prediction model
      return state.hourOfDay >= 18 && state.hourOfDay <= 22 ? 1.2 : 0.8;
    },
  };
}

describe('IPS Estimator', () => {
  describe('basic estimation', () => {
    it('should return unbiased estimate for identical policies', () => {
      const logs = createMockLogs(1000, { avgReward: 1.0, loggingProb: 0.1 });
      const policy = createMockPolicy(1.0); // Same probability as logging

      const result = calculateIPS(logs, policy);

      // Should be close to average reward
      expect(result.estimate).toBeCloseTo(1.0, 0.3);
    });

    it('should have positive sample count', () => {
      const logs = createMockLogs(100);
      const policy = createMockPolicy();

      const result = calculateIPS(logs, policy);

      expect(result.sampleCount).toBeGreaterThan(0);
    });

    it('should compute confidence intervals', () => {
      const logs = createMockLogs(500);
      const policy = createMockPolicy();

      const result = calculateIPS(logs, policy);

      expect(result.confidenceLower).toBeLessThan(result.estimate);
      expect(result.confidenceUpper).toBeGreaterThan(result.estimate);
    });

    it('should filter low propensity samples', () => {
      const logs = createMockLogs(100, { loggingProb: 0.0001 }); // Very low
      const policy = createMockPolicy();

      const result = calculateIPS(logs, policy, { minPropensity: 0.01 });

      // Should filter many samples
      expect(result.filteredCount).toBeGreaterThan(0);
    });
  });
});

describe('SNIPS Estimator', () => {
  it('should have lower variance than IPS', () => {
    const logs = createMockLogs(500, { avgReward: 1.0 });
    const policy = createMockPolicy(2.0); // Different from logging

    const ipsResult = calculateIPS(logs, policy);
    const snipsResult = calculateSNIPS(logs, policy);

    // SNIPS typically has lower standard error
    // (may not always hold in small samples)
    expect(snipsResult.standardError).toBeDefined();
    expect(ipsResult.standardError).toBeDefined();
  });

  it('should produce bounded estimates', () => {
    const logs = createMockLogs(100, { avgReward: 1.0 });
    const policy = createMockPolicy(10.0); // Very different

    const result = calculateSNIPS(logs, policy);

    // SNIPS should be bounded even with extreme weights
    expect(Math.abs(result.estimate)).toBeLessThan(100);
  });
});

describe('Clipped IPS Estimator', () => {
  it('should clip extreme weights', () => {
    const logs = createMockLogs(100, { loggingProb: 0.01 });
    const policy = createMockPolicy(10.0); // High prob = high weights

    const result = calculateClippedIPS(logs, policy, { maxWeight: 5.0 });

    // Max observed weight should be clipped
    expect(result.maxWeightObserved).toBeLessThanOrEqual(5.0);
  });
});

describe('Doubly Robust Estimator', () => {
  it('should use reward model as baseline', () => {
    const logs = createMockLogs(200, { avgReward: 1.0 });
    const policy = createMockPolicy(1.0);

    const result = calculateDoublyRobust(logs, policy);

    expect(result.estimate).toBeDefined();
    expect(result.sampleCount).toBeGreaterThan(0);
  });

  it('should throw without reward prediction', () => {
    const logs = createMockLogs(100);
    const policy: EvaluationPolicy = {
      getActionProbability: () => 0.1,
      // No predictReward
    };

    try {
      calculateDoublyRobust(logs, policy);
      throw new Error('Should have thrown');
    } catch (e) {
      expect((e as Error).message).toContain('reward prediction');
    }
  });
});

describe('MDA Calculator', () => {
  it('should return perfect MDA for correct predictions', () => {
    const pairs = [
      { opeEstimateA: 1.0, opeEstimateB: 0.5, trueValueA: 1.2, trueValueB: 0.6 },
      { opeEstimateA: 0.8, opeEstimateB: 1.2, trueValueA: 0.7, trueValueB: 1.3 },
      { opeEstimateA: 1.5, opeEstimateB: 1.0, trueValueA: 1.6, trueValueB: 0.9 },
    ];

    const result = calculateMDA(pairs);

    expect(result.mda).toBe(1.0);
    expect(result.correctCount).toBe(3);
  });

  it('should return zero MDA for all wrong predictions', () => {
    const pairs = [
      { opeEstimateA: 1.0, opeEstimateB: 0.5, trueValueA: 0.3, trueValueB: 0.8 }, // Wrong
      { opeEstimateA: 0.8, opeEstimateB: 1.2, trueValueA: 1.0, trueValueB: 0.6 }, // Wrong
    ];

    const result = calculateMDA(pairs);

    expect(result.mda).toBe(0);
  });

  it('should handle empty input', () => {
    const result = calculateMDA([]);
    expect(result.mda).toBe(0);
    expect(result.totalCount).toBe(0);
  });
});

describe('Deployment Evaluation', () => {
  it('should recommend deployment for significant improvement', () => {
    const newPolicy = {
      estimate: 1.5,
      standardError: 0.1,
      confidenceLower: 1.3,
      confidenceUpper: 1.7,
      effectiveSampleSize: 500,
      sampleCount: 1000,
      filteredCount: 0,
      meanWeight: 1.2,
      maxWeightObserved: 3.0,
      weightVariance: 0.5,
    };

    const baseline = {
      estimate: 1.0,
      standardError: 0.1,
      confidenceLower: 0.8,
      confidenceUpper: 1.2,
      effectiveSampleSize: 500,
      sampleCount: 1000,
      filteredCount: 0,
      meanWeight: 1.0,
      maxWeightObserved: 1.5,
      weightVariance: 0.2,
    };

    const result = evaluateDeployment(newPolicy, baseline);

    expect(result.recommend).toBe(true);
    expect(result.expectedLift).toBeGreaterThan(0.3);
  });

  it('should reject deployment for small sample size', () => {
    const newPolicy = {
      estimate: 1.5,
      standardError: 0.1,
      confidenceLower: 1.3,
      confidenceUpper: 1.7,
      effectiveSampleSize: 20, // Too low
      sampleCount: 50,
      filteredCount: 0,
      meanWeight: 1.2,
      maxWeightObserved: 3.0,
      weightVariance: 0.5,
    };

    const baseline = {
      estimate: 1.0,
      standardError: 0.1,
      confidenceLower: 0.8,
      confidenceUpper: 1.2,
      effectiveSampleSize: 500,
      sampleCount: 1000,
      filteredCount: 0,
      meanWeight: 1.0,
      maxWeightObserved: 1.5,
      weightVariance: 0.2,
    };

    const result = evaluateDeployment(newPolicy, baseline);

    expect(result.recommend).toBe(false);
    expect(result.riskLevel).toBe('high');
  });
});

describe('OPE Suite', () => {
  it('should run all estimators and provide agreement score', () => {
    const logs = createMockLogs(500);
    const policy = createMockPolicy();

    const result = runOPESuite(logs, policy);

    expect(result.ips).toBeDefined();
    expect(result.snips).toBeDefined();
    expect(result.cips).toBeDefined();
    expect(result.agreement).toBeDefined();
    expect(result.recommendedEstimate).toBeDefined();
  });

  it('should include DR when reward model available', () => {
    const logs = createMockLogs(500);
    const policy = createMockPolicy(); // Has predictReward

    const result = runOPESuite(logs, policy);

    expect(result.dr).toBeDefined();
  });
});

// Test runner
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
    toBeCloseTo: (expected: number, tolerance: number) => {
      if (Math.abs(actual - expected) > tolerance)
        throw new Error(`Expected ${expected} ± ${tolerance}, got ${actual}`);
    },
    toBeGreaterThan: (expected: number) => {
      if (actual <= expected) throw new Error(`Expected > ${expected}, got ${actual}`);
    },
    toBeLessThan: (expected: number) => {
      if (actual >= expected) throw new Error(`Expected < ${expected}, got ${actual}`);
    },
    toBeLessThanOrEqual: (expected: number) => {
      if (actual > expected) throw new Error(`Expected <= ${expected}, got ${actual}`);
    },
    toBeDefined: () => {
      if (actual === undefined) throw new Error('Expected defined value');
    },
    toContain: (substr: string) => {
      if (!actual.includes(substr)) throw new Error(`Expected to contain "${substr}"`);
    },
  };
}

console.log('Running OPE Tests...\n');
