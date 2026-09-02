import { CivilianBrain } from './CivilianBrain';

/**
 * Docks gang member: same sidewalk life as a civilian, but any damage starts a fight
 * (fists, or pistol fire through `weapons.npcFire` when armed) and nearby gunfire is
 * ignored instead of triggering flight.
 */
export class GangBrain extends CivilianBrain {
  constructor() {
    super(true);
  }
}
