import { describe, expect, it } from 'vitest';
import { WEAPONS, WEAPON_ORDER, shotInterval } from '../../src/data/weapons';
import { explosionFalloff, partMultiplier } from '../../src/systems/damage/falloff';
import { Inventory } from '../../src/systems/weapons/Inventory';

describe('weapon table', () => {
  it('matches SYSTEM_PROMPT §8', () => {
    expect(WEAPON_ORDER).toEqual(['fists', 'pistol', 'smg', 'shotgun', 'rifle', 'sniper']);
    expect(WEAPONS.pistol).toMatchObject({ damage: 25, rpm: 300, magazine: 12, range: 60, spreadDeg: 1.5, price: 500, ammoPrice: 30 });
    // Prices mirror data/prices.ts (WEAPON_PRICES), the authoritative store table.
    expect(WEAPONS.smg).toMatchObject({ damage: 18, rpm: 800, magazine: 30, range: 45, spreadDeg: 3, price: 1200, ammoPrice: 60, automatic: true });
    expect(WEAPONS.shotgun).toMatchObject({ damage: 12, pellets: 8, rpm: 70, magazine: 8, range: 20, spreadDeg: 8, price: 1500, ammoPrice: 80 });
    expect(WEAPONS.rifle).toMatchObject({ damage: 30, rpm: 600, magazine: 30, range: 90, spreadDeg: 2, price: 3000, ammoPrice: 100, automatic: true });
    expect(WEAPONS.sniper).toMatchObject({ damage: 200, rpm: 40, magazine: 5, range: 300, spreadDeg: 0, price: 5000, ammoPrice: 150 });
    expect(WEAPONS.fists).toMatchObject({ damage: 15, range: 1.5, rpm: 0, magazine: 0 });
    for (const id of WEAPON_ORDER) {
      const def = WEAPONS[id];
      expect(def.slot).toBe(WEAPON_ORDER.indexOf(id) + 1);
      if (id !== 'fists') expect(def.reloadSeconds).toBeGreaterThanOrEqual(1.5);
      expect(def.reloadSeconds).toBeLessThanOrEqual(2.5);
    }
  });

  it('derives fire intervals from rpm', () => {
    expect(shotInterval(WEAPONS.smg)).toBeCloseTo(0.075, 10);
    expect(shotInterval(WEAPONS.fists)).toBe(Infinity);
    // 2 s of SMG fire ≈ 26.7 rounds
    expect(Math.floor(2 / shotInterval(WEAPONS.smg))).toBe(26);
  });
});

describe('damage math', () => {
  it('applies the headshot multiplier only where it applies', () => {
    expect(25 * partMultiplier('head', 3, true)).toBe(75);
    expect(25 * partMultiplier('body', 3, true)).toBe(25);
    expect(25 * partMultiplier('head', 3, false)).toBe(25);
    expect(25 * partMultiplier(undefined, 3, true)).toBe(25);
  });

  it('explosion falloff is full inside a third of the radius and zero beyond it', () => {
    expect(explosionFalloff(0, 6, 1 / 3)).toBe(1);
    expect(explosionFalloff(2, 6, 1 / 3)).toBe(1);
    expect(explosionFalloff(4, 6, 1 / 3)).toBeCloseTo(0.5, 10);
    expect(explosionFalloff(6, 6, 1 / 3)).toBe(0);
    expect(explosionFalloff(12, 6, 1 / 3)).toBe(0);
    expect(explosionFalloff(3, 6, 1 / 3)).toBeGreaterThan(0.7);
  });
});

describe('inventory', () => {
  it('starts with fists and fills the magazine first', () => {
    const inv = new Inventory();
    expect(inv.has('fists')).toBe(true);
    expect(inv.has('pistol')).toBe(false);
    expect(inv.give('pistol', 48)).toBe(true);
    expect(inv.slot('pistol')).toEqual({ mag: 12, reserve: 36 });
    expect(inv.give('pistol', 12)).toBe(false);
    expect(inv.slot('pistol')).toEqual({ mag: 12, reserve: 48 });
  });

  it('reloads from reserve and reports when it cannot', () => {
    const inv = new Inventory();
    inv.give('shotgun', 10);
    const s = inv.slot('shotgun')!;
    s.mag = 3;
    expect(inv.canReload('shotgun')).toBe(true);
    expect(inv.completeReload('shotgun')).toBe(2);
    expect(s).toEqual({ mag: 5, reserve: 0 });
    expect(inv.canReload('shotgun')).toBe(false);
    expect(inv.canReload('fists')).toBe(false);
  });

  it('cycles only through owned weapons and round-trips through save data', () => {
    const inv = new Inventory();
    inv.give('smg', 60);
    inv.give('sniper', 5);
    inv.current = 'fists';
    expect(inv.cycle(1)).toBe('smg');
    inv.current = 'smg';
    expect(inv.cycle(1)).toBe('sniper');
    inv.current = 'sniper';
    expect(inv.cycle(1)).toBe('fists');
    expect(inv.cycle(-1)).toBe('smg');
    const copy = new Inventory();
    copy.deserialize(JSON.parse(JSON.stringify(inv.serialize())));
    expect(copy.current).toBe('sniper');
    expect(copy.summary()).toEqual({ fists: 0, smg: 30, sniper: 0 });
    copy.deserialize({ current: 'rifle', slots: [['bogus', 1, 1], ['pistol', 99, -4]] });
    expect(copy.current).toBe('fists');
    expect(copy.slot('pistol')).toEqual({ mag: 12, reserve: 0 });
  });
});
