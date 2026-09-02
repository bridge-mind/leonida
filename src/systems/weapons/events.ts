import type { Entity } from '../../core/entities';

/** Events private to the weapons system (declaration merging; see src/core/Events.ts). */
declare module '../../core/Events' {
  interface GameEvents {
    /** Weapon wheel opened/closed; the crosshair hides while it is open. */
    'weapon:wheelToggled': { open: boolean };
    /** A melee swing resolved (hit may be null); combo is 1..3. */
    'weapon:meleeSwung': { combo: number; hit: Entity | null };
  }
}

export {};
