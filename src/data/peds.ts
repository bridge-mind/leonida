import type { WeaponId } from '../core/peers';
import { PED_PALETTES, type PedPalette } from '../gfx/PedBuilder';
import type { DistrictId } from './districts';

/**
 * Pedestrian archetypes (SYSTEM_PROMPT §7). Spawn weight for a type in a district is
 * `districtWeights[district] * (night ? nightWeight : dayWeight)`; types with no entry
 * for a district never spawn there ambiently. Cops, SWAT, guards, clerks and the two
 * protagonists are only spawned on request by other systems.
 */
export type PedTypeId =
  | 'beachgoer'
  | 'tourist'
  | 'business'
  | 'jogger'
  | 'elderly'
  | 'clubber'
  | 'worker'
  | 'gang'
  | 'dealer'
  | 'cop'
  | 'swat'
  | 'guard'
  | 'clerk'
  | 'lucia'
  | 'jason';

export interface PedTypeDef {
  id: PedTypeId;
  /** One is picked at random per spawn. */
  palettes: readonly PedPalette[];
  walkSpeed: number;
  runSpeed: number;
  districtWeights: Partial<Record<DistrictId, number>>;
  dayWeight: number;
  nightWeight: number;
  /** Probability of fighting back when hit with melee (gang = 1). */
  fightChance: number;
  /** Weapon carried by default (gang peds roll for it in the spawner). */
  defaultWeapon: WeaponId | null;
  isCop: boolean;
  /** Spawned by the ambient spawner; false = only via `peds.spawn`. */
  ambient: boolean;
}

const SKIN = { pale: 0xe8c3a0, tan: 0xd9a77c, olive: 0xc48b62, brown: 0xb9805a, dark: 0x8f5a3a, deep: 0x5c3a24 } as const;
const HAIR = { black: 0x1d1410, brown: 0x5a3b22, blond: 0xd8b56b, grey: 0xb8b8b8, red: 0x9a3b1e, pink: 0xff5fb0 } as const;

function variants(shirts: readonly number[], pants: readonly number[], shoes: number, skins: readonly number[], hairs: readonly number[]): PedPalette[] {
  const out: PedPalette[] = [];
  for (let i = 0; i < shirts.length; i++) {
    out.push({
      skin: skins[i % skins.length],
      hair: hairs[i % hairs.length],
      shirt: shirts[i],
      pants: pants[i % pants.length],
      shoes,
    });
  }
  return out;
}

const ALL_SKINS = [SKIN.pale, SKIN.tan, SKIN.olive, SKIN.brown, SKIN.dark, SKIN.deep];
const ALL_HAIR = [HAIR.black, HAIR.brown, HAIR.blond, HAIR.black, HAIR.red, HAIR.brown];

export const PED_TYPES: Record<PedTypeId, PedTypeDef> = {
  beachgoer: {
    id: 'beachgoer',
    palettes: variants([0xff6fa3, 0x35e6ff, 0xfff05a, 0x7dff8a, 0xffffff], [0x35e6ff, 0xff6fa3, 0xfff0d0, 0x1a1a1a, 0xff8a1f], 0xfff0d0, ALL_SKINS, [HAIR.blond, HAIR.brown, HAIR.black, HAIR.pink]),
    walkSpeed: 1.2,
    runSpeed: 5.5,
    districtWeights: { viceBeach: 3 },
    dayWeight: 1,
    nightWeight: 0.3,
    fightChance: 0.1,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  tourist: {
    id: 'tourist',
    palettes: variants([0xffd166, 0x06d6a0, 0xef476f, 0x118ab2, 0xf78c6b], [0xf5efe0, 0x8b7355, 0x2c3444, 0xf5efe0], 0xffffff, ALL_SKINS, ALL_HAIR),
    walkSpeed: 1.1,
    runSpeed: 5.0,
    districtWeights: { viceBeach: 2, downtown: 1.2, littleHabana: 0.4 },
    dayWeight: 1,
    nightWeight: 0.5,
    fightChance: 0.05,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  business: {
    id: 'business',
    palettes: variants([0x1f1f2a, 0x2b2b3a, 0x3a3a4a, 0xf2f2f2, 0x4a2a3a], [0x1a1a24, 0x2a2a34, 0x1a1a24, 0x2c3444], 0x101010, ALL_SKINS, [HAIR.black, HAIR.brown, HAIR.grey, HAIR.blond]),
    walkSpeed: 1.6,
    runSpeed: 5.5,
    districtWeights: { downtown: 3, viceBeach: 0.4 },
    dayWeight: 1,
    nightWeight: 0.35,
    fightChance: 0.1,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  jogger: {
    id: 'jogger',
    palettes: variants([0xff3d7f, 0x39ff88, 0x2bd9ff, 0xffe14d], [0x1a1a1a, 0x2c3444, 0x1a1a1a, 0x3a1a4a], 0xffffff, ALL_SKINS, [HAIR.blond, HAIR.black, HAIR.brown]),
    walkSpeed: 3.4,
    runSpeed: 6.0,
    districtWeights: { viceBeach: 1.5, downtown: 0.3 },
    dayWeight: 1,
    nightWeight: 0,
    fightChance: 0.1,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  elderly: {
    id: 'elderly',
    palettes: variants([0xd9c4a5, 0xb8c9d9, 0xe8a25a, 0xf5efe0, 0x8fbf8f], [0xc9c0a8, 0x8b7355, 0x2c3444], 0x3a2a1a, ALL_SKINS, [HAIR.grey, HAIR.grey, HAIR.grey]),
    walkSpeed: 0.8,
    runSpeed: 3.0,
    districtWeights: { littleHabana: 2.5, portGellhorn: 2.5, viceBeach: 0.4, downtown: 0.3 },
    dayWeight: 1,
    nightWeight: 0.2,
    fightChance: 0,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  clubber: {
    id: 'clubber',
    palettes: variants([0xff2bd6, 0x2bfff0, 0xb52bff, 0xffffff, 0x000000], [0x000000, 0x1a1a24, 0xff2bd6, 0x000000], 0x000000, ALL_SKINS, [HAIR.pink, HAIR.black, HAIR.blond, HAIR.black]),
    walkSpeed: 1.2,
    runSpeed: 5.5,
    districtWeights: { viceBeach: 3, downtown: 0.6 },
    dayWeight: 0,
    nightWeight: 1,
    fightChance: 0.15,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  worker: {
    id: 'worker',
    palettes: variants([0xff8a1f, 0xffd21f, 0x8a8a8a, 0x3a4a7a], [0x3a4a7a, 0x3a4a7a, 0x2c3444, 0x5a4a3a], 0x3a2a1a, ALL_SKINS, ALL_HAIR),
    walkSpeed: 1.3,
    runSpeed: 5.3,
    districtWeights: { docks: 3, portGellhorn: 1.5, littleHabana: 0.8, downtown: 0.3 },
    dayWeight: 1,
    nightWeight: 0.5,
    fightChance: 0.2,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  gang: {
    id: 'gang',
    palettes: variants([0x7a1f2a, 0x1a1a1a, 0x7a1f2a, 0x3a3a3a], [0x1a1a24, 0x2c3444, 0x1a1a1a], 0x101010, [SKIN.brown, SKIN.dark, SKIN.olive, SKIN.deep], [HAIR.red, HAIR.red, HAIR.black, HAIR.red]),
    walkSpeed: 1.2,
    runSpeed: 5.5,
    districtWeights: { docks: 2 },
    dayWeight: 1,
    nightWeight: 1.4,
    fightChance: 1,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  dealer: {
    id: 'dealer',
    palettes: variants([0x2a2a2a, 0x4a3a2a, 0x2a3a2a], [0x1a1a1a, 0x2c3444], 0x101010, [SKIN.olive, SKIN.dark, SKIN.tan], [HAIR.black, HAIR.brown]),
    walkSpeed: 1.0,
    runSpeed: 5.5,
    districtWeights: { docks: 0.6 },
    dayWeight: 0.5,
    nightWeight: 1,
    fightChance: 0.6,
    defaultWeapon: null,
    isCop: false,
    ambient: true,
  },
  cop: {
    id: 'cop',
    palettes: [PED_PALETTES.cop, { ...PED_PALETTES.cop, skin: SKIN.dark }, { ...PED_PALETTES.cop, skin: SKIN.pale, hair: HAIR.blond }],
    walkSpeed: 1.5,
    runSpeed: 6.0,
    districtWeights: {},
    dayWeight: 0,
    nightWeight: 0,
    fightChance: 1,
    defaultWeapon: 'pistol',
    isCop: true,
    ambient: false,
  },
  swat: {
    id: 'swat',
    palettes: [{ skin: SKIN.tan, hair: 0x101010, shirt: 0x121417, pants: 0x121417, shoes: 0x050505 }],
    walkSpeed: 1.5,
    runSpeed: 6.0,
    districtWeights: {},
    dayWeight: 0,
    nightWeight: 0,
    fightChance: 1,
    defaultWeapon: 'rifle',
    isCop: true,
    ambient: false,
  },
  guard: {
    id: 'guard',
    palettes: [{ skin: SKIN.olive, hair: HAIR.black, shirt: 0xf2f2f2, pants: 0x1a1a24, shoes: 0x101010 }, { skin: SKIN.dark, hair: HAIR.black, shirt: 0x2a2a34, pants: 0x1a1a24, shoes: 0x101010 }],
    walkSpeed: 1.3,
    runSpeed: 5.5,
    districtWeights: {},
    dayWeight: 0,
    nightWeight: 0,
    fightChance: 1,
    defaultWeapon: 'pistol',
    isCop: false,
    ambient: false,
  },
  clerk: {
    id: 'clerk',
    palettes: [{ skin: SKIN.tan, hair: HAIR.brown, shirt: 0x2f8f5f, pants: 0x2c3444, shoes: 0x1a1a1a }, { skin: SKIN.brown, hair: HAIR.black, shirt: 0xd94a4a, pants: 0x1a1a24, shoes: 0x1a1a1a }],
    walkSpeed: 1.2,
    runSpeed: 5.0,
    districtWeights: {},
    dayWeight: 0,
    nightWeight: 0,
    fightChance: 0,
    defaultWeapon: null,
    isCop: false,
    ambient: false,
  },
  lucia: {
    id: 'lucia',
    palettes: [PED_PALETTES.lucia],
    walkSpeed: 2.0,
    runSpeed: 7.0,
    districtWeights: {},
    dayWeight: 0,
    nightWeight: 0,
    fightChance: 1,
    defaultWeapon: 'pistol',
    isCop: false,
    ambient: false,
  },
  jason: {
    id: 'jason',
    palettes: [PED_PALETTES.jason],
    walkSpeed: 2.0,
    runSpeed: 7.0,
    districtWeights: {},
    dayWeight: 0,
    nightWeight: 0,
    fightChance: 1,
    defaultWeapon: 'pistol',
    isCop: false,
    ambient: false,
  },
};

export const PED_TYPE_IDS: readonly PedTypeId[] = Object.keys(PED_TYPES) as PedTypeId[];

export function isPedTypeId(id: string): id is PedTypeId {
  return id in PED_TYPES;
}
