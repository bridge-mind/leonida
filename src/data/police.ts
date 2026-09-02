import type { CrimeType } from '../core/Events';
import type { WeaponId } from '../core/peers';

/**
 * Wanted / police data (SYSTEM_PROMPT §9, TEAM-PLAN Agent 8 brief). Numbers here are
 * game design; runtime tunables (timers, ranges) live in the systems' config.ts files.
 */

export type WitnessRule = 'none' | 'civilian';

export interface CrimeDef {
  /** Stars added when the crime raises the level from 0. */
  stars: number;
  /** Stars added when the player is already wanted (0 = only resets the evasion timer). */
  starsWhenWanted: number;
  /** 'civilian': counts only if a ped/cop saw it or the player is already wanted; 'none': always counts. */
  witness: WitnessRule;
  /** Seconds during which repeats of this crime are ignored (ramming, helicopter hits). */
  cooldown: number;
  /** Whether the crime position becomes a hot scene. */
  hotScene: boolean;
}

export const CRIME_TABLE: Record<CrimeType, CrimeDef> = {
  shot_fired: { stars: 1, starsWhenWanted: 0, witness: 'civilian', cooldown: 0, hotScene: true },
  kill_ped: { stars: 1, starsWhenWanted: 2, witness: 'civilian', cooldown: 0, hotScene: true },
  kill_cop: { stars: 2, starsWhenWanted: 2, witness: 'none', cooldown: 0, hotScene: true },
  carjack: { stars: 1, starsWhenWanted: 0, witness: 'civilian', cooldown: 0, hotScene: true },
  steal_police_vehicle: { stars: 2, starsWhenWanted: 1, witness: 'none', cooldown: 5, hotScene: true },
  rob_ped: { stars: 1, starsWhenWanted: 0, witness: 'civilian', cooldown: 0, hotScene: true },
  rob_store: { stars: 2, starsWhenWanted: 2, witness: 'none', cooldown: 0, hotScene: true },
  hit_cop_vehicle: { stars: 1, starsWhenWanted: 0, witness: 'none', cooldown: 8, hotScene: false },
  hit_cop: { stars: 1, starsWhenWanted: 0, witness: 'none', cooldown: 8, hotScene: false },
  explosion: { stars: 2, starsWhenWanted: 2, witness: 'civilian', cooldown: 0, hotScene: true },
  attack_helicopter: { stars: 1, starsWhenWanted: 1, witness: 'none', cooldown: 6, hotScene: false },
  kill_swat: { stars: 1, starsWhenWanted: 1, witness: 'none', cooldown: 0, hotScene: false },
};

export const MAX_WANTED_LEVEL = 6;

export interface DispatchTier {
  /** Police cruisers (2 cops each) kept in pursuit. */
  cruisers: number;
  /** SWAT vans (4 SWAT each). */
  swatVans: number;
  helicopters: number;
  roadblock: boolean;
  /** Roadblocks get a spike strip. */
  spikes: boolean;
  /** Foot cops per roadblock. */
  roadblockCops: number;
  copWeapon: WeaponId;
  swatWeapon: WeaponId;
  /** Cruisers ram the player's car. */
  ramming: boolean;
  /** Cops shoot instead of trying to arrest (unless provoked). */
  lethal: boolean;
  /** Added to every cop's accuracy at this tier. */
  accuracyBonus: number;
}

/** Response ladder indexed by wanted level (index 0 = not wanted). */
export const DISPATCH_TABLE: readonly DispatchTier[] = [
  { cruisers: 0, swatVans: 0, helicopters: 0, roadblock: false, spikes: false, roadblockCops: 0, copWeapon: 'pistol', swatWeapon: 'rifle', ramming: false, lethal: false, accuracyBonus: 0 },
  { cruisers: 1, swatVans: 0, helicopters: 0, roadblock: false, spikes: false, roadblockCops: 0, copWeapon: 'pistol', swatWeapon: 'rifle', ramming: false, lethal: false, accuracyBonus: 0 },
  { cruisers: 2, swatVans: 0, helicopters: 0, roadblock: false, spikes: false, roadblockCops: 0, copWeapon: 'pistol', swatWeapon: 'rifle', ramming: true, lethal: false, accuracyBonus: 0 },
  { cruisers: 3, swatVans: 0, helicopters: 0, roadblock: true, spikes: false, roadblockCops: 2, copWeapon: 'smg', swatWeapon: 'rifle', ramming: true, lethal: true, accuracyBonus: 0.05 },
  { cruisers: 3, swatVans: 1, helicopters: 1, roadblock: true, spikes: false, roadblockCops: 2, copWeapon: 'shotgun', swatWeapon: 'rifle', ramming: true, lethal: true, accuracyBonus: 0.08 },
  { cruisers: 4, swatVans: 2, helicopters: 2, roadblock: true, spikes: true, roadblockCops: 3, copWeapon: 'shotgun', swatWeapon: 'rifle', ramming: true, lethal: true, accuracyBonus: 0.12 },
  { cruisers: 4, swatVans: 3, helicopters: 2, roadblock: true, spikes: true, roadblockCops: 3, copWeapon: 'rifle', swatWeapon: 'rifle', ramming: true, lethal: true, accuracyBonus: 0.2 },
];

export function dispatchTier(level: number): DispatchTier {
  const idx = Math.max(0, Math.min(DISPATCH_TABLE.length - 1, Math.round(level)));
  return DISPATCH_TABLE[idx];
}

/** Seconds between shots per weapon for NPC cops (cadence, not the weapon's true rpm). */
export const COP_FIRE_INTERVAL: Record<WeaponId, number> = {
  fists: 1,
  pistol: 0.8,
  smg: 0.3,
  shotgun: 1.4,
  rifle: 0.45,
  sniper: 2.2,
};
