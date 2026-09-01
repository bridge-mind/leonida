import type { V3 } from '../core/entities';
import type { BlipKind } from '../core/GameState';
import type { DistrictId } from './districts';

export type LandmarkStyle =
  | 'apartment'
  | 'hospital'
  | 'police'
  | 'prison'
  | 'casino'
  | 'gunstore'
  | 'store'
  | 'gasStation'
  | 'diner'
  | 'jewelry'
  | 'bank'
  | 'respray'
  | 'chopShop'
  | 'marina'
  | 'club';

export interface LandmarkDef {
  id: string;
  name: string;
  kind: BlipKind;
  district: DistrictId;
  style: LandmarkStyle;
  /** Building center at ground level. */
  position: V3;
  /** Footprint (x extent, z extent) and height. */
  footprint: { w: number; d: number; h: number };
  /** Door marker on the sidewalk in front of the entrance. */
  door: V3;
  /** Heading (radians, rig convention: 0 faces +Z) the player faces when stepping out. */
  doorHeading: number;
  /** Interior registered with `game.interiors`, or null for exterior-only landmarks. */
  interiorId: string | null;
}

type Quadrant = 'NE' | 'NW' | 'SE' | 'SW';
type Face = 'N' | 'S' | 'E' | 'W';

const SIDEWALK = 4;
const DOOR_OFF_WALL = 1.4;
const DOOR_FROM_CORNER = 1.8;
const HEADING: Record<Face, number> = { W: -Math.PI / 2, E: Math.PI / 2, N: Math.PI, S: 0 };

/**
 * Places a building flush against both sidewalks of a street corner so its door
 * marker is within 15 m of the intersection node (GPS/spawn friendliness).
 */
function corner(
  node: [number, number],
  quadrant: Quadrant,
  w: number,
  d: number,
  h: number,
  doorFace: Face,
  halfNS = 6,
  halfEW = 6,
): Pick<LandmarkDef, 'position' | 'footprint' | 'door' | 'doorHeading'> {
  const sx = quadrant[1] === 'E' ? 1 : -1;
  const sz = quadrant[0] === 'S' ? 1 : -1;
  const xInner = node[0] + sx * (halfNS + SIDEWALK);
  const zInner = node[1] + sz * (halfEW + SIDEWALK);
  const position = { x: xInner + (sx * w) / 2, y: 0, z: zInner + (sz * d) / 2 };
  let door: V3;
  switch (doorFace) {
    case 'W':
      door = { x: xInner - DOOR_OFF_WALL, y: 0, z: zInner + sz * DOOR_FROM_CORNER };
      break;
    case 'E':
      door = { x: xInner + DOOR_OFF_WALL, y: 0, z: zInner + sz * DOOR_FROM_CORNER };
      break;
    case 'N':
      door = { x: xInner + sx * DOOR_FROM_CORNER, y: 0, z: zInner - DOOR_OFF_WALL };
      break;
    case 'S':
      door = { x: xInner + sx * DOOR_FROM_CORNER, y: 0, z: zInner + DOOR_OFF_WALL };
      break;
  }
  return { position, footprint: { w, d, h }, door, doorHeading: HEADING[doorFace] };
}

export const LANDMARKS: readonly LandmarkDef[] = [
  {
    id: 'safehouse',
    name: 'Safehouse',
    kind: 'safehouse',
    district: 'littleHabana',
    style: 'apartment',
    interiorId: 'safehouse',
    ...corner([-400, 100], 'SE', 30, 20, 10.4, 'W'),
  },
  {
    id: 'hospital',
    name: 'Vice General Hospital',
    kind: 'hospital',
    district: 'downtown',
    style: 'hospital',
    interiorId: null,
    ...corner([-110, 200], 'SE', 60, 40, 17, 'W'),
  },
  {
    id: 'police_downtown',
    name: 'VCPD Downtown',
    kind: 'police',
    district: 'downtown',
    style: 'police',
    interiorId: null,
    ...corner([250, -200], 'SW', 45, 35, 13.4, 'E'),
  },
  {
    id: 'police_gellhorn',
    name: 'VCPD Port Gellhorn',
    kind: 'police',
    district: 'portGellhorn',
    style: 'police',
    interiorId: null,
    ...corner([-400, -530], 'NE', 40, 30, 7.2, 'S'),
  },
  {
    id: 'prison',
    name: 'Leonida Penitentiary',
    kind: 'prison',
    district: 'portGellhorn',
    style: 'prison',
    interiorId: null,
    position: { x: -200, y: 0, z: -870 },
    footprint: { w: 240, d: 60, h: 12 },
    door: { x: -200, y: 0, z: -838 },
    doorHeading: 0,
  },
  {
    id: 'casino',
    name: 'Leonida Palms',
    kind: 'casino',
    district: 'downtown',
    style: 'casino',
    interiorId: 'casino',
    ...corner([250, 100], 'SW', 60, 56, 30, 'E'),
  },
  {
    id: 'gunstore',
    name: 'Bullet Barn',
    kind: 'gunstore',
    district: 'littleHabana',
    style: 'gunstore',
    interiorId: 'gunstore',
    ...corner([-500, -100], 'NE', 30, 20, 7.2, 'S'),
  },
  {
    id: 'store_habana',
    name: '24/7 Little Habana',
    kind: 'store',
    district: 'littleHabana',
    style: 'store',
    interiorId: 'store_habana',
    ...corner([-300, -200], 'SW', 20, 15, 4.4, 'E'),
  },
  {
    id: 'store_gellhorn',
    name: '24/7 Port Gellhorn',
    kind: 'store',
    district: 'portGellhorn',
    style: 'store',
    interiorId: 'store_gellhorn',
    ...corner([-600, -530], 'SE', 20, 15, 4.4, 'W'),
  },
  {
    id: 'store_downtown',
    name: '24/7 Downtown',
    kind: 'store',
    district: 'downtown',
    style: 'store',
    interiorId: 'store_downtown',
    ...corner([70, 300], 'SE', 20, 15, 4.4, 'W'),
  },
  {
    id: 'store_beach',
    name: '24/7 Vice Beach',
    kind: 'store',
    district: 'viceBeach',
    style: 'store',
    interiorId: 'store_beach',
    ...corner([760, -300], 'NE', 20, 15, 4.4, 'S'),
  },
  {
    id: 'gas_station',
    name: 'Gellhorn Gas',
    kind: 'gas',
    district: 'portGellhorn',
    style: 'gasStation',
    interiorId: 'gas_station',
    ...corner([160, -660], 'SW', 30, 15, 4.4, 'N'),
  },
  {
    id: 'waffle_hut',
    name: 'Waffle Hut',
    kind: 'food',
    district: 'portGellhorn',
    style: 'diner',
    interiorId: 'waffle_hut',
    ...corner([-200, -400], 'NW', 25, 15, 4.8, 'S'),
  },
  {
    id: 'jewelry',
    name: 'Vice Rocks Jewelers',
    kind: 'jewelry',
    district: 'downtown',
    style: 'jewelry',
    interiorId: 'jewelry',
    ...corner([70, -100], 'NW', 25, 20, 7.6, 'S'),
  },
  {
    id: 'bank',
    name: 'Leonida Trust Bank',
    kind: 'bank',
    district: 'downtown',
    style: 'bank',
    interiorId: 'bank',
    ...corner([-110, -100], 'NE', 40, 30, 16, 'S'),
  },
  {
    id: 'respray',
    name: 'Vice Sprays',
    kind: 'respray',
    district: 'littleHabana',
    style: 'respray',
    interiorId: null,
    ...corner([-600, 200], 'NE', 30, 20, 6, 'S'),
  },
  {
    id: 'chop_shop',
    name: 'Gellhorn Salvage (Chop Shop)',
    kind: 'chopshop',
    district: 'docks',
    style: 'chopShop',
    interiorId: null,
    ...corner([-400, 600], 'SE', 40, 30, 9, 'W'),
  },
  {
    id: 'marina',
    name: 'Vice Marina',
    kind: 'marina',
    district: 'docks',
    style: 'marina',
    interiorId: null,
    position: { x: -75, y: 0, z: 876 },
    footprint: { w: 30, d: 16, h: 5 },
    door: { x: -91.4, y: 0, z: 869.8 },
    doorHeading: -Math.PI / 2,
  },
  {
    id: 'mud_club',
    name: 'Gator Pit Mud Club',
    kind: 'club',
    district: 'grassrivers',
    style: 'club',
    interiorId: 'mud_club',
    position: { x: -925, y: 0, z: -625 },
    footprint: { w: 40, d: 30, h: 6 },
    door: { x: -908, y: 0, z: -608.6 },
    doorHeading: 0,
  },
];

export const LANDMARK_BY_ID: Record<string, LandmarkDef> = Object.fromEntries(LANDMARKS.map((l) => [l.id, l]));
