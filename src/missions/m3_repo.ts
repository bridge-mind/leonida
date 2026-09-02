import type { IVehicle } from '../core/entities';
import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { IdleDriver } from './drivers/RouteDriver';
import { contactMarker, doorFront, parkingNear, sceneTalk, spawnVehicle } from './helpers';
import { action, deliver, enterVehicle } from './objectives';
import { cutscene, escapeWanted } from './objectivesWorld';
import type { MissionDef } from './types';

const story = STORY_BY_ID.get('m3')!;

/** Vice Beach hotel lot (no landmark of its own): the block north of the 24/7 Vice Beach. */
const HOTEL_LOT = { x: 735, y: 0, z: -272 };

/** 3 — Repo: steal a red coupe from a Vice Beach hotel lot and bring it to Manny's chop shop. */
export const m3: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m3'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build() {
    let coupe: IVehicle | null = null;
    const dropoff = doorFront('chop_shop', 10);
    return [
      cutscene(sceneTalk('m3_start', dialogue('m3_start'))),
      action((ctx) => {
        const spot = parkingNear(ctx.game, HOTEL_LOT, 80);
        // An idle driver makes the F prompt read "Steal": the vehicle system then emits the
        // carjack crime and spawns the fleeing owner, exactly like grabbing a traffic car.
        coupe = spawnVehicle(ctx, 'sports', spot.pos, spot.heading, { color: 0xd41c2c, driver: new IdleDriver() });
        if (coupe) (coupe as { driverTypeId?: string | null }).driverTypeId = 'business';
      }),
      enterVehicle(() => coupe, 'Steal the ~y~red coupe~y~ from the Vice Beach hotel lot'),
      escapeWanted('Lose the ~y~cops~y~ before Manny will see you'),
      deliver(() => coupe, dropoff, 6, 'Deliver the coupe to ~y~Gellhorn Salvage~y~ at the docks'),
      action((ctx) => {
        if (ctx.game.systems.has('player') && ctx.game.player.isMounted) ctx.game.vehicles?.exitCurrent();
        if (coupe) ctx.game.vehicles?.despawn(coupe);
        coupe = null;
      }),
      cutscene(sceneTalk('m3_done', dialogue('m3_done'))),
    ];
  },
};
