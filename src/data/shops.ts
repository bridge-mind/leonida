import type { PedTypeId } from './peds';
import type { FoodId } from './prices';

/**
 * Store definitions. One entry per commercial landmark the player can use.
 * `id` doubles as the interior id for stores that have an interior, so the
 * interiors framework, the shop system and the robbery system all speak the same key.
 */

export type StoreKind = 'gun' | 'convenience' | 'gas' | 'food' | 'safehouse' | 'respray' | 'chopshop' | 'jewelry' | 'bank';

export interface StoreDef {
  id: string;
  name: string;
  /** Landmark in src/data/landmarks.ts that hosts the door / drive-in volume. */
  landmarkId: string;
  kind: StoreKind;
  /** Cash in the register for a hold-up; null = cannot be robbed by the generic robbery flow. */
  registerCash: { min: number; max: number } | null;
  /** Ped type standing behind the counter; null = unstaffed. */
  clerkType: PedTypeId | null;
  /** Chance per visit that the clerk keeps a shotgun under the counter. */
  shotgunChance: number;
  /** Food item sold at the counter, if any. */
  food: FoodId | null;
  /** Armed guards posted inside. */
  guards: number;
  /** Whether the store has a walk-in interior (false = drive-in volume only). */
  hasInterior: boolean;
}

const SHOTGUN = 0.25;

export const STORES: readonly StoreDef[] = [
  {
    id: 'gunstore',
    name: 'Bullet Barn',
    landmarkId: 'gunstore',
    kind: 'gun',
    registerCash: { min: 400, max: 800 },
    clerkType: 'clerk',
    shotgunChance: SHOTGUN,
    food: null,
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'store_habana',
    name: '24/7 Little Habana',
    landmarkId: 'store_habana',
    kind: 'convenience',
    registerCash: { min: 200, max: 600 },
    clerkType: 'clerk',
    shotgunChance: SHOTGUN,
    food: 'hotdog',
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'store_gellhorn',
    name: '24/7 Gellhorn',
    landmarkId: 'store_gellhorn',
    kind: 'convenience',
    registerCash: { min: 200, max: 600 },
    clerkType: 'clerk',
    shotgunChance: SHOTGUN,
    food: 'hotdog',
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'store_downtown',
    name: '24/7 Downtown',
    landmarkId: 'store_downtown',
    kind: 'convenience',
    registerCash: { min: 300, max: 800 },
    clerkType: 'clerk',
    shotgunChance: SHOTGUN,
    food: 'hotdog',
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'store_beach',
    name: '24/7 Beach',
    landmarkId: 'store_beach',
    kind: 'convenience',
    registerCash: { min: 200, max: 600 },
    clerkType: 'clerk',
    shotgunChance: SHOTGUN,
    food: 'hotdog',
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'gas_station',
    name: 'Xero Gas',
    landmarkId: 'gas_station',
    kind: 'gas',
    registerCash: { min: 200, max: 500 },
    clerkType: 'clerk',
    shotgunChance: SHOTGUN,
    food: 'snack',
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'waffle_hut',
    name: 'Waffle Hut',
    landmarkId: 'waffle_hut',
    kind: 'food',
    registerCash: { min: 200, max: 500 },
    clerkType: 'clerk',
    shotgunChance: 0,
    food: 'waffle',
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'safehouse',
    name: 'Safehouse',
    landmarkId: 'safehouse',
    kind: 'safehouse',
    registerCash: null,
    clerkType: null,
    shotgunChance: 0,
    food: null,
    guards: 0,
    hasInterior: true,
  },
  {
    id: 'jewelry',
    name: 'Vangelico Jewelers',
    landmarkId: 'jewelry',
    kind: 'jewelry',
    registerCash: null,
    clerkType: 'clerk',
    shotgunChance: 0,
    food: null,
    guards: 2,
    hasInterior: true,
  },
  {
    id: 'bank',
    name: 'Fleeca Bank',
    landmarkId: 'bank',
    kind: 'bank',
    registerCash: null,
    clerkType: 'clerk',
    shotgunChance: 0,
    food: null,
    guards: 2,
    hasInterior: true,
  },
  {
    id: 'respray',
    name: 'Vice Sprays',
    landmarkId: 'respray',
    kind: 'respray',
    registerCash: null,
    clerkType: null,
    shotgunChance: 0,
    food: null,
    guards: 0,
    hasInterior: false,
  },
  {
    id: 'chop_shop',
    name: 'Chop Shop',
    landmarkId: 'chop_shop',
    kind: 'chopshop',
    registerCash: null,
    clerkType: null,
    shotgunChance: 0,
    food: null,
    guards: 0,
    hasInterior: false,
  },
];

export const STORE_BY_ID: ReadonlyMap<string, StoreDef> = new Map(STORES.map((s) => [s.id, s]));

export function storeDef(id: string): StoreDef | undefined {
  return STORE_BY_ID.get(id);
}
