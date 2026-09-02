/**
 * Private events for the shops / robbery / chop-shop cluster, merged into the
 * global map so `game.events` stays typed. Cross-system events (`shop:purchased`,
 * `robbery:*`, `chopshop:sold`) are already declared in core/Events.ts.
 */
declare module '../../core/Events' {
  interface GameEvents {
    /** A store locked its doors (after a hold-up) until the given in-game day. */
    'shops:closed': { storeId: string; untilDay: number };
    /** A clerk or guard went hostile inside a store. */
    'shops:hostile': { storeId: string; role: 'clerk' | 'guard' };
    'phone:opened': Record<string, never>;
    'phone:closed': Record<string, never>;
    /** Vice Sprays finished a job. */
    'respray:done': { cost: number; cleared: boolean };
  }
}

export {};
