import type { CharacterId } from '../../core/peers';

/** Dual-protagonist tunables (SYSTEM_PROMPT §15). */
export const CharactersConfig = {
  wheel: {
    key: 'KeyQ',
    timeScale: 0.2,
    /** Horizontal mouse travel that flips the selection (px). */
    selectThresholdPx: 40,
    blocker: 'characterWheel',
  },
  switch: {
    /** Camera rises this high above the character before the cut. */
    skyHeight: 120,
    upSeconds: 0.6,
    downSeconds: 0.6,
    blocker: 'characterSwitch',
    /** Where a never-used character waits: in front of the safehouse door. */
    defaultLandmark: 'safehouse',
    defaultDoorOffset: 5,
  },
  ability: {
    key: 'CapsLock',
    timeScale: 0.3,
    /** Real-time seconds the slow-mo lasts. */
    seconds: 4,
    rechargeSeconds: 60,
    /** Jason: peds this close get a head marker. */
    markerRange: 40,
    maxMarkers: 20,
    /** Lucia: damage multiplier until her next shot lands. */
    luciaDamage: 2,
    headHeight: 1.7,
  },
  stats: {
    lucia: { sprintMultiplier: 1.08, recoilMultiplier: 1, damageMultiplier: 1 },
    jason: { sprintMultiplier: 1, recoilMultiplier: 0.8, damageMultiplier: 1 },
  } satisfies Record<CharacterId, { sprintMultiplier: number; recoilMultiplier: number; damageMultiplier: number }>,
  names: { lucia: 'Lucia', jason: 'Jason' } satisfies Record<CharacterId, string>,
  traits: { lucia: 'Faster · Focus: 2x damage', jason: 'Steadier aim · Focus: marks heads' } satisfies Record<CharacterId, string>,
} as const;

export const CHARACTER_IDS: readonly CharacterId[] = ['lucia', 'jason'];
