import { describe, expect, it } from 'vitest';
import {
  DOUBLE_ZERO,
  POCKET_COUNT,
  ROULETTE_ODDS,
  WHEEL_ORDER,
  betCovers,
  describeBet,
  isValidBet,
  pocketAngle,
  pocketColor,
  pocketLabel,
  settle,
  spin,
  type RouletteBet,
} from '../../src/casino/Roulette';
import { Random } from '../../src/core/Random';

describe('Roulette wheel', () => {
  it('has 38 pockets with the American colour layout', () => {
    expect(POCKET_COUNT).toBe(38);
    expect(WHEEL_ORDER).toHaveLength(38);
    expect(new Set(WHEEL_ORDER).size).toBe(38);
    expect(pocketColor(0)).toBe('green');
    expect(pocketColor(DOUBLE_ZERO)).toBe('green');
    expect(pocketLabel(DOUBLE_ZERO)).toBe('00');
    expect(pocketColor(1)).toBe('red');
    expect(pocketColor(2)).toBe('black');
    expect(pocketColor(10)).toBe('black');
    expect(pocketColor(19)).toBe('red');
    expect(pocketColor(28)).toBe('black');
    let reds = 0;
    for (let n = 1; n <= 36; n++) if (pocketColor(n) === 'red') reds++;
    expect(reds).toBe(18);
    expect(pocketAngle(0)).toBe(0);
    expect(pocketAngle(28)).toBeCloseTo((Math.PI * 2) / 38);
  });

  it('spin only produces valid pockets', () => {
    const rng = new Random(3);
    const seen = new Set<number>();
    for (let i = 0; i < 2000; i++) {
      const p = spin(() => rng.next());
      expect(p >= 0 && p < POCKET_COUNT).toBe(true);
      seen.add(p);
    }
    expect(seen.size).toBe(38);
    expect(spin(() => 0.9999999)).toBe(DOUBLE_ZERO);
  });
});

describe('Roulette bets', () => {
  it('validates inside bets against the layout grid', () => {
    expect(isValidBet({ kind: 'straight', pockets: [17], amount: 10 })).toBe(true);
    expect(isValidBet({ kind: 'straight', pockets: [38], amount: 10 })).toBe(false);
    expect(isValidBet({ kind: 'straight', pockets: [17], amount: 0 })).toBe(false);
    expect(isValidBet({ kind: 'split', pockets: [17, 20], amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'split', pockets: [17, 18], amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'split', pockets: [18, 19], amount: 5 })).toBe(false);
    expect(isValidBet({ kind: 'split', pockets: [0, DOUBLE_ZERO], amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'split', pockets: [DOUBLE_ZERO, 3], amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'split', pockets: [0, 3], amount: 5 })).toBe(false);
    expect(isValidBet({ kind: 'street', pockets: [4, 5, 6], amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'street', pockets: [5, 6, 7], amount: 5 })).toBe(false);
    expect(isValidBet({ kind: 'corner', pockets: [1, 2, 4, 5], amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'corner', pockets: [2, 3, 5, 6], amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'corner', pockets: [3, 4, 6, 7], amount: 5 })).toBe(false);
    expect(isValidBet({ kind: 'corner', pockets: [34, 35, 37, 38], amount: 5 })).toBe(false);
    expect(isValidBet({ kind: 'dozen', index: 2, amount: 5 })).toBe(true);
    expect(isValidBet({ kind: 'red', amount: 25 })).toBe(true);
  });

  it('pays every bet type at the correct odds', () => {
    expect(ROULETTE_ODDS.straight).toBe(35);
    const bets: RouletteBet[] = [
      { kind: 'straight', pockets: [17], amount: 10 },
      { kind: 'split', pockets: [17, 20], amount: 10 },
      { kind: 'street', pockets: [16, 17, 18], amount: 10 },
      { kind: 'corner', pockets: [16, 17, 19, 20], amount: 10 },
      { kind: 'black', amount: 10 },
      { kind: 'odd', amount: 10 },
      { kind: 'low', amount: 10 },
      { kind: 'dozen', index: 1, amount: 10 },
      { kind: 'column', index: 1, amount: 10 },
      { kind: 'red', amount: 10 },
      { kind: 'even', amount: 10 },
      { kind: 'high', amount: 10 },
      { kind: 'straight', pockets: [18], amount: 10 },
    ];
    const s = settle(bets, 17);
    expect(s.perBet).toEqual([360, 180, 120, 90, 20, 20, 20, 30, 30, 0, 0, 0, 0]);
    expect(s.wagered).toBe(130);
    expect(s.returned).toBe(870);
    expect(s.net).toBe(740);
  });

  it('a lone straight bet on the winning number nets 35x', () => {
    const s = settle([{ kind: 'straight', pockets: [17], amount: 10 }], 17);
    expect(s.net).toBe(350);
    expect(settle([{ kind: 'straight', pockets: [17], amount: 10 }], 16).net).toBe(-10);
  });

  it('zero and double zero lose every outside bet', () => {
    const outside: RouletteBet[] = [
      { kind: 'red', amount: 10 },
      { kind: 'black', amount: 10 },
      { kind: 'odd', amount: 10 },
      { kind: 'even', amount: 10 },
      { kind: 'low', amount: 10 },
      { kind: 'high', amount: 10 },
      { kind: 'dozen', index: 0, amount: 10 },
      { kind: 'column', index: 0, amount: 10 },
    ];
    for (const zero of [0, DOUBLE_ZERO]) {
      const s = settle(outside, zero);
      expect(s.returned).toBe(0);
      expect(s.net).toBe(-80);
    }
    expect(settle([{ kind: 'straight', pockets: [DOUBLE_ZERO], amount: 5 }], DOUBLE_ZERO).returned).toBe(180);
    expect(betCovers({ kind: 'column', index: 2, amount: 1 }, 36)).toBe(true);
    expect(betCovers({ kind: 'dozen', index: 2, amount: 1 }, 24)).toBe(false);
    expect(() => settle([], 38)).toThrow();
  });

  it('describes bets for the UI', () => {
    expect(describeBet({ kind: 'straight', pockets: [DOUBLE_ZERO], amount: 1 })).toBe('Straight 00');
    expect(describeBet({ kind: 'corner', pockets: [1, 2, 4, 5], amount: 1 })).toBe('Corner 1-2-4-5');
    expect(describeBet({ kind: 'dozen', index: 1, amount: 1 })).toBe('Dozen 13-24');
    expect(describeBet({ kind: 'high', amount: 1 })).toBe('19-36');
    expect(describeBet({ kind: 'red', amount: 1 })).toBe('Red');
  });
});
