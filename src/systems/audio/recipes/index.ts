import type { AudioBus } from '../api';
import type { SoundId } from '../sounds';
import { AMBIENT_RECIPES } from './ambient';
import { CASINO_RECIPES } from './casino';
import { ECONOMY_RECIPES } from './economy';
import { EXPLOSION_RECIPES } from './explosions';
import { FOOTSTEP_RECIPES } from './footsteps';
import { IMPACT_RECIPES } from './impacts';
import { PED_RECIPES } from './peds';
import { RADIO_RECIPES } from './radio';
import type { Recipe } from './types';
import { UI_RECIPES } from './ui';
import { VEHICLE_RECIPES } from './vehicles';
import { WANTED_RECIPES } from './wanted';
import { WEAPON_RECIPES } from './weapons';
import { WORLD_RECIPES } from './world';

/** Every SoundId → synthesis recipe. The type makes a missing id a compile error. */
export const RECIPES: Readonly<Record<SoundId, Recipe>> = {
  ...WEAPON_RECIPES,
  ...IMPACT_RECIPES,
  ...FOOTSTEP_RECIPES,
  ...VEHICLE_RECIPES,
  ...EXPLOSION_RECIPES,
  ...PED_RECIPES,
  ...ECONOMY_RECIPES,
  ...UI_RECIPES,
  ...WANTED_RECIPES,
  ...CASINO_RECIPES,
  ...AMBIENT_RECIPES,
  ...WORLD_RECIPES,
  ...RADIO_RECIPES,
};

export function getRecipe(id: SoundId): Recipe {
  return RECIPES[id];
}

/** Default bus by id family; `PlayOptions.bus` overrides. */
export function defaultBus(id: SoundId): AudioBus {
  if (id.startsWith('ui.')) return 'ui';
  if (id.startsWith('ambient.') || id === 'casino.ambient.loop') return 'ambient';
  if (id.startsWith('vehicle.engine.') && id.endsWith('.loop')) return 'engine';
  if (id.startsWith('radio.')) return 'music';
  return 'sfx';
}

export function isLoopId(id: SoundId): boolean {
  return id.endsWith('.loop');
}

export type { Recipe } from './types';
