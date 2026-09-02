import { describe, expect, it } from 'vitest';
import {
  ARMOR_PRICE,
  BUYABLE_WEAPONS,
  CHOP,
  FOOD,
  RESPRAY,
  WEAPON_PRICES,
  chopHealthMultiplier,
  chopValue,
  resprayCost,
} from '../../src/data/prices';
import { STORES, STORE_BY_ID, storeDef } from '../../src/data/shops';
import { VEHICLES } from '../../src/data/vehicles';
import { WEAPONS } from '../../src/data/weapons';

describe('weapon prices', () => {
  it('matches the design table', () => {
    expect(WEAPON_PRICES.pistol).toEqual({ price: 500, ammo: 30 });
    expect(WEAPON_PRICES.smg).toEqual({ price: 1200, ammo: 60 });
    expect(WEAPON_PRICES.shotgun).toEqual({ price: 1500, ammo: 80 });
    expect(WEAPON_PRICES.rifle).toEqual({ price: 3000, ammo: 100 });
    expect(WEAPON_PRICES.sniper).toEqual({ price: 5000, ammo: 150 });
    expect(ARMOR_PRICE).toBe(500);
  });

  it('covers every non-fist weapon in the weapon table, cheapest first', () => {
    const ids = Object.keys(WEAPONS).filter((id) => id !== 'fists');
    expect([...BUYABLE_WEAPONS].sort()).toEqual(ids.sort());
    for (let i = 1; i < BUYABLE_WEAPONS.length; i++) {
      expect(WEAPON_PRICES[BUYABLE_WEAPONS[i]].price).toBeGreaterThan(WEAPON_PRICES[BUYABLE_WEAPONS[i - 1]].price);
    }
  });
});

describe('food', () => {
  it('heals what it costs to heal', () => {
    expect(FOOD.hotdog).toMatchObject({ price: 10, heal: 25 });
    expect(FOOD.waffle).toMatchObject({ price: 15, heal: 50 });
    expect(FOOD.snack).toMatchObject({ price: 5, heal: 15 });
  });
});

describe('respray cost', () => {
  it('is $1000 per star with a $1000 floor', () => {
    expect(RESPRAY).toEqual({ perStar: 1000, min: 1000 });
    expect(resprayCost(0)).toBe(1000);
    expect(resprayCost(1)).toBe(1000);
    expect(resprayCost(2)).toBe(2000);
    expect(resprayCost(5)).toBe(5000);
    expect(resprayCost(2.7)).toBe(2000);
    expect(resprayCost(-3)).toBe(1000);
  });
});

describe('chop valuation', () => {
  it('pays full at full health and nothing below 20%', () => {
    expect(chopHealthMultiplier(1)).toBe(1);
    expect(chopHealthMultiplier(CHOP.minHealth)).toBeCloseTo(CHOP.minMultiplier);
    expect(chopHealthMultiplier(0.19)).toBe(0);
    expect(chopHealthMultiplier(0)).toBe(0);
  });

  it('is linear between 20% and 100% health', () => {
    expect(chopHealthMultiplier(0.6)).toBeCloseTo(0.7);
    expect(chopHealthMultiplier(0.8)).toBeCloseTo(0.85);
  });

  it('uses the archetype chop value and doubles the wanted archetype', () => {
    expect(chopValue(VEHICLES.sports.chopValue, 1, false)).toBe(4000);
    expect(chopValue(VEHICLES.sports.chopValue, 1, true)).toBe(8000);
    expect(chopValue(VEHICLES.sports.chopValue, 0.2, false)).toBe(1600);
    expect(chopValue(VEHICLES.sports.chopValue, 0.1, true)).toBe(0);
    expect(chopValue(VEHICLES.sedan.chopValue, 0.9, false)).toBe(Math.round((800 * 0.925) / 10) * 10);
  });

  it('rounds payouts to $10', () => {
    for (const health of [0.21, 0.37, 0.5, 0.83, 0.99]) {
      expect(chopValue(2500, health, false) % 10).toBe(0);
    }
  });
});

describe('store table', () => {
  it('has unique ids that match their landmarks and the documented kinds', () => {
    const ids = STORES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(STORE_BY_ID.size).toBe(STORES.length);
    expect(STORES.filter((s) => s.kind === 'convenience')).toHaveLength(4);
    expect(storeDef('gunstore')?.kind).toBe('gun');
    expect(storeDef('respray')?.hasInterior).toBe(false);
    expect(storeDef('chop_shop')?.hasInterior).toBe(false);
  });

  it('only robbable stores carry register cash, and guarded shells carry guards', () => {
    for (const s of STORES) {
      if (s.registerCash) {
        expect(s.registerCash.min).toBeGreaterThanOrEqual(200);
        expect(s.registerCash.max).toBeLessThanOrEqual(800);
        expect(s.clerkType).toBe('clerk');
      }
      if (s.kind === 'jewelry' || s.kind === 'bank') {
        expect(s.registerCash).toBeNull();
        expect(s.guards).toBe(2);
      }
      expect(s.shotgunChance).toBeGreaterThanOrEqual(0);
      expect(s.shotgunChance).toBeLessThanOrEqual(1);
    }
  });
});
