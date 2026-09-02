import type { WeaponId } from '../core/peers';

/**
 * Price table for everything the player can buy or sell. Kept separate from the
 * weapon/vehicle tables so balancing the economy is a single-file edit.
 * All prices in dollars, all heals in hit points.
 */

export type BuyableWeaponId = Exclude<WeaponId, 'fists'>;

export interface WeaponPrice {
  /** Cost to buy the weapon (comes with one clip). */
  price: number;
  /** Cost of one clip of ammo. */
  ammo: number;
}

export const WEAPON_PRICES: Readonly<Record<BuyableWeaponId, WeaponPrice>> = {
  pistol: { price: 500, ammo: 30 },
  smg: { price: 1200, ammo: 60 },
  shotgun: { price: 1500, ammo: 80 },
  rifle: { price: 3000, ammo: 100 },
  sniper: { price: 5000, ammo: 150 },
};

/** Gun-store shelf order (cheapest first). */
export const BUYABLE_WEAPONS: readonly BuyableWeaponId[] = ['pistol', 'smg', 'shotgun', 'rifle', 'sniper'];

export const ARMOR_PRICE = 500;
export const ARMOR_AMOUNT = 100;

export type FoodId = 'hotdog' | 'waffle' | 'snack';

export interface FoodDef {
  id: FoodId;
  name: string;
  price: number;
  heal: number;
}

export const FOOD: Readonly<Record<FoodId, FoodDef>> = {
  hotdog: { id: 'hotdog', name: 'Hot Dog', price: 10, heal: 25 },
  waffle: { id: 'waffle', name: 'Waffle Stack', price: 15, heal: 50 },
  snack: { id: 'snack', name: 'Gas Station Snack', price: 5, heal: 15 },
};

export const RESPRAY = {
  perStar: 1000,
  min: 1000,
} as const;

/** $1000 per wanted star, never below the minimum (a plain recolour still costs). */
export function resprayCost(wantedLevel: number): number {
  const stars = Math.max(0, Math.floor(wantedLevel));
  return Math.max(RESPRAY.min, stars * RESPRAY.perStar);
}

export const CHOP = {
  /** Health fraction at or below which the chop shop refuses the wreck. */
  minHealth: 0.2,
  /** Multiplier applied at exactly `minHealth`; 1.0 at full health, linear in between. */
  minMultiplier: 0.4,
  /** Bonus for the archetype the chop shop is looking for today. */
  wantedMultiplier: 2,
} as const;

export function chopHealthMultiplier(health01: number): number {
  if (health01 < CHOP.minHealth) return 0;
  const t = Math.min(1, (health01 - CHOP.minHealth) / (1 - CHOP.minHealth));
  return CHOP.minMultiplier + (1 - CHOP.minMultiplier) * t;
}

/** Payout for a vehicle: archetype base × health multiplier × (2 if it is today's wanted archetype). Rounded to $10. */
export function chopValue(baseValue: number, health01: number, isWanted: boolean): number {
  const raw = baseValue * chopHealthMultiplier(health01) * (isWanted ? CHOP.wantedMultiplier : 1);
  return Math.round(raw / 10) * 10;
}
