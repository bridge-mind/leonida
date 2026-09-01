/**
 * Vice City's road network as axis-aligned lines. The RoadGraph builder turns these
 * into nodes (intersections, ends, ramp breakpoints) and edges. Keep collinear lines
 * non-overlapping; adjacent lines share their endpoint node.
 */
export type RoadKind = 'street' | 'avenue' | 'highway' | 'dirt' | 'bridge';
export type RoadSurface = 'asphalt' | 'dirt' | 'bridge';

export interface RoadKindDef {
  width: number;
  lanesPerDirection: 1 | 2;
  speedLimit: number;
  isHighway: boolean;
  sidewalks: boolean;
  parking: boolean;
  surface: RoadSurface;
}

export const ROAD_KINDS: Record<RoadKind, RoadKindDef> = {
  street: { width: 12, lanesPerDirection: 1, speedLimit: 14, isHighway: false, sidewalks: true, parking: true, surface: 'asphalt' },
  avenue: { width: 20, lanesPerDirection: 2, speedLimit: 18, isHighway: false, sidewalks: true, parking: true, surface: 'asphalt' },
  highway: { width: 20, lanesPerDirection: 2, speedLimit: 28, isHighway: true, sidewalks: false, parking: false, surface: 'asphalt' },
  dirt: { width: 8, lanesPerDirection: 1, speedLimit: 10, isHighway: false, sidewalks: false, parking: false, surface: 'dirt' },
  bridge: { width: 20, lanesPerDirection: 2, speedLimit: 22, isHighway: false, sidewalks: true, parking: false, surface: 'bridge' },
};

export interface RoadLineDef {
  /** 'x' runs east–west at constant z; 'z' runs north–south at constant x. */
  axis: 'x' | 'z';
  at: number;
  from: number;
  to: number;
  kind: RoadKind;
  /** Elevation keyframes along the line; a node is inserted at each. Missing = flat at y 0. */
  profile?: { at: number; y: number }[];
  name?: string;
}

const H = 8; // bridge deck height (matches Config.world.bridgeDeckHeight)

function bridgeProfile(from: number): { at: number; y: number }[] {
  // Ramp up over land/shore, level deck across the channel, ramp down onto the island.
  return [
    { at: from, y: 0 },
    { at: from + 60, y: H },
    { at: from + 150, y: H },
    { at: from + 210, y: 0 },
  ];
}

const HWY_W = -760;
const HWY_E = 320;
const HWY_N = -800;
const HWY_S = 850;

export const ROAD_LINES: readonly RoadLineDef[] = [
  // --- Highway loop around the mainland ---
  { axis: 'z', at: HWY_W, from: HWY_N, to: HWY_S, kind: 'highway', name: 'Gellhorn Expressway (W)' },
  { axis: 'z', at: HWY_E, from: HWY_N, to: HWY_S, kind: 'highway', name: 'Bayshore Expressway' },
  { axis: 'x', at: HWY_N, from: HWY_W, to: HWY_E, kind: 'highway', name: 'Penitentiary Parkway' },
  { axis: 'x', at: HWY_S, from: HWY_W, to: HWY_E, kind: 'highway', name: 'Harbor Freeway' },

  // --- Main arteries ---
  { axis: 'z', at: -20, from: HWY_N, to: HWY_S, kind: 'avenue', name: 'Leonida Avenue' },
  { axis: 'x', at: 0, from: HWY_W, to: HWY_E, kind: 'avenue', name: 'Calle Ocho' },
  { axis: 'z', at: 160, from: -400, to: 300, kind: 'avenue', name: 'Palms Boulevard' },

  // --- Little Habana + Downtown grid (z -400..300) ---
  ...[-400, -300, -200, -100, 100, 200, 300].map<RoadLineDef>((z) => ({ axis: 'x', at: z, from: HWY_W, to: HWY_E, kind: 'street' })),
  ...[-700, -600, -500, -400, -300, -200, -110, 70, 250].map<RoadLineDef>((x) => ({ axis: 'z', at: x, from: -400, to: 300, kind: 'street' })),

  // --- Port Gellhorn (z -800..-400) ---
  { axis: 'x', at: -660, from: HWY_W, to: HWY_E, kind: 'street' },
  { axis: 'x', at: -530, from: HWY_W, to: HWY_E, kind: 'street' },
  ...[-600, -400, -200, 160].map<RoadLineDef>((x) => ({ axis: 'z', at: x, from: HWY_N, to: -400, kind: 'street' })),
  { axis: 'z', at: -200, from: -830, to: HWY_N, kind: 'street', name: 'Penitentiary Gate Road' },

  // --- Docks + south-east (z 300..850) ---
  { axis: 'x', at: 450, from: HWY_W, to: HWY_E, kind: 'street' },
  { axis: 'x', at: 600, from: HWY_W, to: HWY_E, kind: 'street' },
  { axis: 'x', at: 750, from: HWY_W, to: HWY_E, kind: 'street' },
  ...[-600, -400, -200, 160].map<RoadLineDef>((x) => ({ axis: 'z', at: x, from: 300, to: HWY_S, kind: 'street' })),
  { axis: 'z', at: -100, from: HWY_S, to: 880, kind: 'street', name: 'Marina Way' },

  // --- Bridges to Vice Beach ---
  { axis: 'x', at: 0, from: HWY_E, to: 560, kind: 'bridge', profile: bridgeProfile(HWY_E), name: 'Vice Causeway' },
  { axis: 'x', at: 600, from: HWY_E, to: 560, kind: 'bridge', profile: bridgeProfile(HWY_E), name: 'South Point Bridge' },

  // --- Vice Beach (x 560..860, z -640..640) ---
  { axis: 'z', at: 560, from: -640, to: 640, kind: 'street', name: 'Bay Road' },
  { axis: 'z', at: 660, from: -640, to: 640, kind: 'street', name: 'Washington Ave' },
  { axis: 'z', at: 760, from: -640, to: 640, kind: 'street', name: 'Collins Ave' },
  { axis: 'z', at: 860, from: -640, to: 640, kind: 'street', name: 'Ocean Drive' },
  ...[-600, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 600].map<RoadLineDef>((z) => ({
    axis: 'x',
    at: z,
    from: 560,
    to: 860,
    kind: 'street',
  })),

  // --- Grassrivers dirt roads ---
  { axis: 'x', at: -600, from: -900, to: HWY_W, kind: 'dirt', name: 'Mud Club Trail' },
  { axis: 'x', at: -200, from: -960, to: HWY_W, kind: 'dirt', name: 'Airboat Road' },
  { axis: 'x', at: 300, from: -960, to: HWY_W, kind: 'dirt', name: 'Reed Road' },
  { axis: 'z', at: -900, from: -200, to: 300, kind: 'dirt', name: 'Gator Lane' },
];

/** Half width of the road of a given kind. */
export function roadHalfWidth(kind: RoadKind): number {
  return ROAD_KINDS[kind].width / 2;
}
