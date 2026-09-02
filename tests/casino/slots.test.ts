import { describe, expect, it } from 'vitest';
import {
  REEL_WEIGHTS,
  SLOT_BETS,
  SLOT_PAYTABLE,
  SLOT_SYMBOLS,
  evaluate,
  expectedReturn,
  isSlotSymbol,
  isValidBet,
  pickSymbol,
  reelStrip,
  spin,
  type SlotSymbol,
} from '../../src/casino/Slots';
import { Random } from '../../src/core/Random';

describe('Slots', () => {
  it('accepts only the three bet sizes', () => {
    expect(SLOT_BETS).toEqual([10, 50, 100]);
    expect(isValidBet(10)).toBe(true);
    expect(isValidBet(25)).toBe(false);
    expect(() => spin(25, () => 0.5)).toThrow();
    expect(() => evaluate(['bar', 'bar', 'bar'], 1)).toThrow();
  });

  it('pays the paytable multipliers', () => {
    expect(evaluate(['diamond', 'diamond', 'diamond'], 10)).toMatchObject({ multiplier: 500, payout: 5000, jackpot: true });
    expect(evaluate(['gator', 'gator', 'gator'], 100).payout).toBe(100 * SLOT_PAYTABLE.threeGators);
    expect(evaluate(['seven', 'seven', 'seven'], 50).payout).toBe(2500);
    expect(evaluate(['bar', 'bar', 'bar'], 10).payout).toBe(100);
    expect(evaluate(['cherry', 'cherry', 'cherry'], 10).payout).toBe(50);
    expect(evaluate(['cherry', 'cherry', 'bar'], 10).payout).toBe(20);
    expect(evaluate(['cherry', 'bar', 'cherry'], 10).payout).toBe(20);
    expect(evaluate(['bar', 'cherry', 'seven'], 10).payout).toBe(0);
    expect(evaluate(['bar', 'seven', 'gator'], 10)).toMatchObject({ multiplier: 0, payout: 0, jackpot: false });
    expect(evaluate(['diamond', 'diamond', 'gator'], 100).payout).toBe(0);
  });

  it('picks symbols by per-reel weight and honours the injected rng', () => {
    expect(pickSymbol(() => 0, 0)).toBe('cherry');
    expect(pickSymbol(() => 0.9999, 0)).toBe('gator');
    const rng = new Random(11);
    for (let reel = 0; reel < 3; reel++) {
      const counts: Record<SlotSymbol, number> = { cherry: 0, bar: 0, seven: 0, diamond: 0, gator: 0 };
      const n = 20000;
      for (let i = 0; i < n; i++) counts[pickSymbol(() => rng.next(), reel)]++;
      const total = SLOT_SYMBOLS.reduce((s, sym) => s + REEL_WEIGHTS[reel][sym], 0);
      for (const sym of SLOT_SYMBOLS) {
        const expected = (REEL_WEIGHTS[reel][sym] / total) * n;
        expect(Math.abs(counts[sym] - expected)).toBeLessThan(expected * 0.2 + 40);
      }
    }
  });

  it('spin is deterministic for a given rng sequence and a scripted rng can force the jackpot', () => {
    const a = new Random(5);
    const b = new Random(5);
    const ra = spin(50, () => a.next());
    const rb = spin(50, () => b.next());
    expect(ra.reels).toEqual(rb.reels);
    expect(ra.payout).toBe(rb.payout);
    expect(ra.bet).toBe(50);
    // Diamond is the fourth symbol on every reel; aim the rng at its weight band.
    const diamondRoll = (reel: number): number => {
      const w = REEL_WEIGHTS[reel];
      const total = SLOT_SYMBOLS.reduce((s, sym) => s + w[sym], 0);
      return (w.cherry + w.bar + w.seven + 0.5) / total;
    };
    let reel = 0;
    expect(spin(10, () => diamondRoll(reel++)).payout).toBe(5000);
  });

  it('reel strips contain every symbol in weight proportion', () => {
    for (let reel = 0; reel < 3; reel++) {
      const strip = reelStrip(reel);
      const total = SLOT_SYMBOLS.reduce((s, sym) => s + REEL_WEIGHTS[reel][sym], 0);
      expect(strip).toHaveLength(total);
      for (const sym of SLOT_SYMBOLS) expect(strip.filter((s) => s === sym)).toHaveLength(REEL_WEIGHTS[reel][sym]);
    }
    expect(isSlotSymbol('gator')).toBe(true);
    expect(isSlotSymbol('lemon')).toBe(false);
  });

  it('house edge is between 5% and 12% analytically and empirically', () => {
    const rtp = expectedReturn();
    expect(rtp).toBeGreaterThan(0.88);
    expect(rtp).toBeLessThan(0.95);
    const rng = new Random(42);
    let wagered = 0;
    let returned = 0;
    for (let i = 0; i < 60000; i++) {
      const r = spin(10, () => rng.next());
      wagered += 10;
      returned += r.payout;
    }
    // The 500x jackpot makes the sample variance large; allow a wide band around the exact value.
    expect(Math.abs(returned / wagered - rtp)).toBeLessThan(0.25);
  });
});
