/**
 * Events private to the casino (cross-system ones live in src/core/Events.ts).
 */
declare module '../../core/Events' {
  interface GameEvents {
    'casino:chipsChanged': { chips: number; delta: number };
    'casino:locked': { untilDay: number };
    'casino:guardsAlerted': { count: number };
  }
}

export {};
