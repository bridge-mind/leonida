import type { V3 } from '../../core/entities';

/**
 * Events private to the wanted/police pair. The police system reports sightings
 * made by units the wanted system cannot see through `peds.query` (helicopters,
 * cops doing their own aim raycasts) so both feed one `playerSeen` clock.
 */
declare module '../../core/Events' {
  interface GameEvents {
    /** A police unit has line of sight to the player right now. */
    'wanted:sighting': { position: V3; by: 'cop' | 'cruiser' | 'helicopter' };
  }
}

export {};
