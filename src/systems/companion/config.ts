import type { IPed, V3 } from '../../core/entities';

/** `IPed` has no teleport in the contract; the Ped class does. Used for catch-up only. */
export function teleportPed(ped: IPed, pos: V3, heading?: number): void {
  (ped as IPed & { teleport?: (p: V3, h?: number) => void }).teleport?.(pos, heading);
}

/** Companion tunables. Units: meters, seconds, m/s. */
export const CompanionConfig = {
  follow: {
    /** Preferred trailing distance behind the player. */
    distance: 3,
    /** Closer than this: stand still. */
    stopDistance: 2.5,
    /** Farther than this: run instead of walk. */
    runDistance: 6,
    /** Farther than this: teleport behind the player (fell off a bridge, left behind). */
    teleportDistance: 60,
  },
  vehicle: {
    /** Distance to the door at which the companion hops in. */
    enterDistance: 3,
    /** Beyond this the player clearly drove off: teleport to the door and get in. */
    catchUpDistance: 40,
    preferredSeat: 1,
  },
  combat: {
    range: 30,
    /** Cops this close while riding along get shot at from the passenger seat. */
    carRange: 40,
    shotsPerSecond: 2,
    accuracy: 0.6,
    weapon: 'pistol' as const,
    /** Peds that attacked the player stay hostile this long after their last attack. */
    hostileMemory: 15,
    /** Ped types treated as hostile once they are fighting. */
    gangTypes: ['gang', 'dealer', 'guard'] as readonly string[],
    eyeHeight: 1.6,
    chestHeight: 1.2,
  },
  /** Health pool managed by the brain; the ped's own health is pinned so it cannot die. */
  hitPoints: 250,
  pinnedHealth: 100000,
  downedSeconds: 10,
  /** If something still kills the ped outright, a new one appears after this long. */
  respawnDelay: 3,
  spawnOffset: 2.5,
} as const;
