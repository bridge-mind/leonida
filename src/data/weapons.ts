import type { WeaponId } from '../core/peers';
import type { SoundId } from '../systems/audio/sounds';

/**
 * Weapon table from SYSTEM_PROMPT §8. Units: meters, seconds, degrees only for `spreadDeg`
 * (converted to radians in the weapon system). `spreadDeg` is the full cone angle.
 */
export interface WeaponDef {
  id: WeaponId;
  /** 1-based wheel slot / number key. */
  slot: number;
  name: string;
  /** Short HUD label. */
  label: string;
  /** Damage per bullet (per pellet for the shotgun). */
  damage: number;
  /** Rounds per minute; 0 for melee. */
  rpm: number;
  /** Magazine size; 0 for melee. */
  magazine: number;
  range: number;
  spreadDeg: number;
  /** Mirror of `WEAPON_PRICES` in data/prices.ts, which is authoritative (what the store charges). Keep in sync. */
  price: number;
  /** Price of one magazine of ammo (mirror of `WEAPON_PRICES[id].ammo`). */
  ammoPrice: number;
  reloadSeconds: number;
  pellets: number;
  automatic: boolean;
  /** Camera pitch kick per shot in radians. */
  recoil: number;
  sounds: { fire: SoundId; reload: SoundId | null; empty: SoundId | null };
}

export const WEAPONS: Record<WeaponId, WeaponDef> = {
  fists: {
    id: 'fists',
    slot: 1,
    name: 'Fists',
    label: 'FISTS',
    damage: 15,
    rpm: 0,
    magazine: 0,
    range: 1.5,
    spreadDeg: 0,
    price: 0,
    ammoPrice: 0,
    reloadSeconds: 0,
    pellets: 1,
    automatic: false,
    recoil: 0,
    sounds: { fire: 'weapon.melee.swing', reload: null, empty: null },
  },
  pistol: {
    id: 'pistol',
    slot: 2,
    name: 'Pistol',
    label: 'PISTOL',
    damage: 25,
    rpm: 300,
    magazine: 12,
    range: 60,
    spreadDeg: 1.5,
    price: 500,
    ammoPrice: 30,
    reloadSeconds: 1.5,
    pellets: 1,
    automatic: false,
    recoil: 0.022,
    sounds: { fire: 'weapon.pistol.fire', reload: 'weapon.pistol.reload', empty: 'weapon.pistol.empty' },
  },
  smg: {
    id: 'smg',
    slot: 3,
    name: 'SMG',
    label: 'SMG',
    damage: 18,
    rpm: 800,
    magazine: 30,
    range: 45,
    spreadDeg: 3,
    price: 1200,
    ammoPrice: 60,
    reloadSeconds: 1.8,
    pellets: 1,
    automatic: true,
    recoil: 0.009,
    sounds: { fire: 'weapon.smg.fire', reload: 'weapon.smg.reload', empty: 'weapon.smg.empty' },
  },
  shotgun: {
    id: 'shotgun',
    slot: 4,
    name: 'Shotgun',
    label: 'SHTGN',
    damage: 12,
    rpm: 70,
    magazine: 8,
    range: 20,
    spreadDeg: 8,
    price: 1500,
    ammoPrice: 80,
    reloadSeconds: 2.5,
    pellets: 8,
    automatic: false,
    recoil: 0.06,
    sounds: { fire: 'weapon.shotgun.fire', reload: 'weapon.shotgun.reload', empty: 'weapon.shotgun.empty' },
  },
  rifle: {
    id: 'rifle',
    slot: 5,
    name: 'Assault Rifle',
    label: 'AR',
    damage: 30,
    rpm: 600,
    magazine: 30,
    range: 90,
    spreadDeg: 2,
    price: 3000,
    ammoPrice: 100,
    reloadSeconds: 2.2,
    pellets: 1,
    automatic: true,
    recoil: 0.014,
    sounds: { fire: 'weapon.rifle.fire', reload: 'weapon.rifle.reload', empty: 'weapon.rifle.empty' },
  },
  sniper: {
    id: 'sniper',
    slot: 6,
    name: 'Sniper Rifle',
    label: 'SNPR',
    damage: 200,
    rpm: 40,
    magazine: 5,
    range: 300,
    spreadDeg: 0,
    price: 5000,
    ammoPrice: 150,
    reloadSeconds: 2.5,
    pellets: 1,
    automatic: false,
    recoil: 0.05,
    sounds: { fire: 'weapon.sniper.fire', reload: 'weapon.sniper.reload', empty: 'weapon.sniper.empty' },
  },
};

/** Wheel slots 1–6 in order. */
export const WEAPON_ORDER: WeaponId[] = ['fists', 'pistol', 'smg', 'shotgun', 'rifle', 'sniper'];

export function isWeaponId(id: string): id is WeaponId {
  return Object.prototype.hasOwnProperty.call(WEAPONS, id);
}

/** Weapons the driver can fire out of a car window. */
export const VEHICLE_WEAPONS: ReadonlySet<WeaponId> = new Set<WeaponId>(['pistol', 'smg']);

/** Seconds between shots for a weapon (Infinity for melee, which has its own combo timing). */
export function shotInterval(def: WeaponDef): number {
  return def.rpm > 0 ? 60 / def.rpm : Infinity;
}
