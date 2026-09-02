/**
 * The six districts of Vice City, Leonida (SYSTEM_PROMPT §3). Rectangles are exact;
 * +X is east, +Z is south, origin at the map center. Districts without a z range in
 * the spec use the natural neighbours' bounds.
 */
export type DistrictId = 'viceBeach' | 'downtown' | 'littleHabana' | 'portGellhorn' | 'docks' | 'grassrivers';

export type BuildingStyle = 'artDeco' | 'highRise' | 'mixedUse' | 'suburban' | 'industrial' | 'swamp';

export interface Rect {
  x0: number;
  z0: number;
  x1: number;
  z1: number;
}

export interface DistrictDef {
  id: DistrictId;
  name: string;
  rect: Rect;
  style: BuildingStyle;
  /** Ground and lot slab colors (sRGB hex). */
  groundColor: number;
  lotColor: number;
  /** Facade colors picked per building. */
  palette: readonly number[];
  /** Floors [min, max] for regular buildings. */
  floors: readonly [number, number];
  /** Lot depth from the sidewalk and lot width range along the street. */
  lotDepth: number;
  lotWidth: readonly [number, number];
  /** Probability a lot stays empty (parking / yard). */
  emptyLotChance: number;
  /** Relative spawn densities for later systems (1 = downtown baseline). */
  pedDensity: number;
  trafficDensity: number;
  /** Ambient bed id (see src/systems/audio/sounds.ts). */
  ambient: string;
  /** Minimap fill color. */
  mapColor: number;
}

export const DISTRICTS: readonly DistrictDef[] = [
  {
    id: 'viceBeach',
    name: 'Vice Beach',
    rect: { x0: 500, z0: -650, x1: 900, z1: 650 },
    style: 'artDeco',
    groundColor: 0xb9b3a4,
    lotColor: 0xd8d2c2,
    palette: [0xf7a8c4, 0x8fe3dc, 0xfff1c9, 0xb8f2c8, 0xd9c4f5, 0xffcf9e, 0xffffff, 0x9ad4ff],
    floors: [3, 6],
    lotDepth: 26,
    lotWidth: [18, 34],
    emptyLotChance: 0.12,
    pedDensity: 1.3,
    trafficDensity: 0.9,
    ambient: 'ambient.beach.loop',
    mapColor: 0xd8cfb4,
  },
  {
    id: 'downtown',
    name: 'Downtown',
    rect: { x0: -200, z0: -400, x1: 400, z1: 300 },
    style: 'highRise',
    groundColor: 0x8d8d90,
    lotColor: 0x9c9c9f,
    palette: [0x9fb3cc, 0x8899b0, 0xbfc8d4, 0x6f86a8, 0xa8b8c8, 0xd0d6de, 0x7d94b4, 0xe4e8ee],
    floors: [12, 36],
    lotDepth: 34,
    lotWidth: [26, 44],
    emptyLotChance: 0.08,
    pedDensity: 1.0,
    trafficDensity: 1.0,
    ambient: 'ambient.city.loop',
    mapColor: 0xb4b4ba,
  },
  {
    id: 'littleHabana',
    name: 'Little Habana',
    rect: { x0: -700, z0: -400, x1: -200, z1: 300 },
    style: 'mixedUse',
    groundColor: 0xa39a88,
    lotColor: 0xc2b7a3,
    palette: [0xf2d38a, 0xe8a25a, 0xd96b4a, 0xf5efe0, 0x8fbf8f, 0xe6c47a, 0xc95a5a, 0xf0e2c8],
    floors: [2, 3],
    lotDepth: 20,
    lotWidth: [12, 22],
    emptyLotChance: 0.1,
    pedDensity: 0.8,
    trafficDensity: 0.7,
    ambient: 'ambient.habana.loop',
    mapColor: 0xd2c3a4,
  },
  {
    id: 'portGellhorn',
    name: 'Port Gellhorn',
    rect: { x0: -700, z0: -900, x1: 350, z1: -450 },
    style: 'suburban',
    groundColor: 0x6f8f4d,
    lotColor: 0x7ea25a,
    palette: [0xf4f0e6, 0xd9d3c3, 0xb8c9d9, 0xc9c0a8, 0xe0e6dc, 0xd4b896, 0xa9b7c6, 0xe9d8c0],
    floors: [1, 2],
    lotDepth: 26,
    lotWidth: [16, 26],
    emptyLotChance: 0.15,
    pedDensity: 0.4,
    trafficDensity: 0.5,
    ambient: 'ambient.suburb.loop',
    mapColor: 0x9db47f,
  },
  {
    id: 'docks',
    name: 'The Docks',
    rect: { x0: -700, z0: 400, x1: 0, z1: 900 },
    style: 'industrial',
    groundColor: 0x6b6b68,
    lotColor: 0x7a7a76,
    palette: [0x8b8f94, 0x6f7378, 0x9c5b3c, 0x4f6f8f, 0xa8a8a0, 0x5f5f5f, 0x8f6f4f, 0x7f8f7f],
    floors: [2, 4],
    lotDepth: 44,
    lotWidth: [30, 60],
    emptyLotChance: 0.25,
    pedDensity: 0.25,
    trafficDensity: 0.4,
    ambient: 'ambient.docks.loop',
    mapColor: 0x9a9a94,
  },
  {
    id: 'grassrivers',
    name: 'Grassrivers',
    rect: { x0: -1000, z0: -900, x1: -800, z1: 900 },
    style: 'swamp',
    groundColor: 0x4f5a34,
    lotColor: 0x5b6a3a,
    palette: [0x8b6b4a, 0x6f5a3f, 0x9c8a6a, 0x5f4f3a],
    floors: [1, 1],
    lotDepth: 14,
    lotWidth: [8, 12],
    emptyLotChance: 0.6,
    pedDensity: 0.1,
    trafficDensity: 0.15,
    ambient: 'ambient.swamp.loop',
    mapColor: 0x6a7a48,
  },
];

export const DISTRICT_BY_ID: Record<DistrictId, DistrictDef> = Object.fromEntries(
  DISTRICTS.map((d) => [d.id, d]),
) as Record<DistrictId, DistrictDef>;

export function rectContains(r: Rect, x: number, z: number): boolean {
  return x >= r.x0 && x <= r.x1 && z >= r.z0 && z <= r.z1;
}

/** Squared distance from a point to a rectangle (0 when inside). */
export function rectDistanceSq(r: Rect, x: number, z: number): number {
  const dx = x < r.x0 ? r.x0 - x : x > r.x1 ? x - r.x1 : 0;
  const dz = z < r.z0 ? r.z0 - z : z > r.z1 ? z - r.z1 : 0;
  return dx * dx + dz * dz;
}
