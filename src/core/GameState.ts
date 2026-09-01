/**
 * Shared read model. UI reads, systems write. Keep fields plain data so the UI
 * can render without touching systems.
 */
import type { V3 } from './entities';

export type WantedState = 'none' | 'responding' | 'searching' | 'active';

export type BlipKind =
  | 'safehouse'
  | 'hospital'
  | 'police'
  | 'prison'
  | 'casino'
  | 'gunstore'
  | 'store'
  | 'gas'
  | 'food'
  | 'jewelry'
  | 'bank'
  | 'respray'
  | 'chopshop'
  | 'marina'
  | 'club'
  | 'mission'
  | 'contact'
  | 'objective'
  | 'waypoint'
  | 'vehicle'
  | 'cop'
  | 'hotScene';

export interface Blip {
  id: string;
  kind: BlipKind;
  position: V3;
  label?: string;
}

export interface GameState {
  cash: number;
  cashDelta: number;
  health: number;
  armor: number;
  wanted: {
    level: number;
    state: WantedState;
    searchCenter: V3 | null;
    searchRadius: number;
    hotScenes: { position: V3; radius: number }[];
  };
  weapon: { id: string; ammoInMag: number; ammoReserve: number; isReloading: boolean } | null;
  vehicle: { id: string; speedKmh: number; health: number } | null;
  objective: string | null;
  prompts: { key: string; text: string }[];
  timeOfDay: number;
  day: number;
  playerPos: V3;
  playerHeading: number;
  blips: Blip[];
  waypoint: V3 | null;
  route: V3[];
  interior: string | null;
  chips: number;
  activeCharacter: 'lucia' | 'jason';
  abilityCharge: number;
  stats: Record<string, number>;
}

export function createGameState(): GameState {
  return {
    cash: 0,
    cashDelta: 0,
    health: 100,
    armor: 0,
    wanted: { level: 0, state: 'none', searchCenter: null, searchRadius: 0, hotScenes: [] },
    weapon: null,
    vehicle: null,
    objective: null,
    prompts: [],
    timeOfDay: 18.5,
    day: 1,
    playerPos: { x: 0, y: 0, z: 0 },
    playerHeading: 0,
    blips: [],
    waypoint: null,
    route: [],
    interior: null,
    chips: 0,
    activeCharacter: 'lucia',
    abilityCharge: 1,
    stats: {},
  };
}
