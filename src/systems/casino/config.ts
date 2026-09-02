import type { V3 } from '../../core/entities';

/** Every casino tunable. Units: meters, seconds, chips (1 chip = $1). */
export const CasinoConfig = {
  interiorId: 'casino',
  landmarkId: 'casino',
  name: 'Leonida Palms',

  /** Interior definition (local coordinates; heading 0 faces +Z). */
  spawnPoint: { x: 0, y: 0.05, z: 6 } as V3,
  spawnHeading: Math.PI,
  exitTrigger: { x: 0, y: 0, z: 7.5 } as V3,

  room: {
    width: 40,
    depth: 30,
    height: 6,
    /** Floor centre so the entrance wall sits at z = +8 (spawn/exit are near +6..+7.5). */
    center: { x: 0, y: 0, z: -7 } as V3,
    wallThickness: 0.4,
  },

  /** Interaction stations: local position, prompt radius and prompt text. */
  stations: {
    cashier: { x: -15.2, y: 0, z: 1, radius: 1.9 },
    blackjackLow: { x: -7, y: 0, z: -2.5, radius: 1.9 },
    blackjackHigh: { x: -7, y: 0, z: -11.5, radius: 1.9 },
    roulette: { x: 7, y: 0, z: -7, radius: 2.4 },
    slots: { x: 15.5, y: 0, z: -9, radius: 4.2 },
    wheel: { x: 0, y: 0, z: -17.5, radius: 2.2 },
  },

  tables: {
    low: { min: 50, max: 2000 },
    high: { min: 500, max: 10000 },
  },

  cashier: {
    presets: [500, 1000, 5000, 10000] as readonly number[],
  },

  roulette: {
    chipValues: [5, 25, 100, 500] as readonly number[],
    maxBetsOnTable: 24,
    spinSeconds: 5,
    /** Extra full turns before the wheel settles on the outcome. */
    wheelTurns: 4,
  },

  slots: {
    machineCount: 6,
    /** Seconds after the lever pull at which each reel stops. */
    reelStops: [0.8, 1.3, 1.8] as readonly number[],
    /** Symbol changes per second while a reel spins. */
    reelRate: 14,
  },

  wheel: {
    spinSeconds: 6,
    turns: 5,
    /** Sports car appears this far outside the door along the door heading. */
    vehicleOffset: 6,
    vehicleFallbackCash: 10000,
  },

  guards: {
    typeId: 'guard',
    /** Local positions flanking the entrance. */
    posts: [
      { x: -3, y: 0, z: 5, heading: Math.PI * 0.75 },
      { x: 3, y: 0, z: 5, heading: -Math.PI * 0.75 },
    ] as readonly { x: number; y: number; z: number; heading: number }[],
    weaponId: 'pistol',
    fireInterval: 0.8,
    accuracy: 0.5,
    eyeHeight: 1.6,
  },

  /** Wins at or above this post a notification and play the big-win sound. */
  bigWin: 1000,
  /** Days a weapon incident bans the player (applies from the next day). */
  banDays: 1,
} as const;
