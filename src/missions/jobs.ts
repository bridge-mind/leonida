import type { IVehicle } from '../core/entities';
import type { IChopShopPeer, VehicleArchetypeId } from '../core/peers';
import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { contactMarker, doorFront, forwardOf, parkingNear, sceneTalk, spawnVehicle } from './helpers';
import { action, custom, forceMount, goto, playerVehicle } from './objectives';
import { cutscene, escapeWanted, robStore } from './objectivesWorld';
import type { MissionDef } from './types';

const chop = STORY_BY_ID.get('job_chopshop')!;
const rob = STORY_BY_ID.get('job_robbery')!;

const NAMES: Record<VehicleArchetypeId, string> = {
  sedan: 'sedan',
  sports: 'sports coupe',
  muscle: 'muscle car',
  suv: 'SUV',
  pickup: 'pickup',
  van: 'van',
  taxi: 'taxi',
  police: 'cruiser',
  swat: 'SWAT truck',
};

/** Repeatable after the finale: bring Manny the chop shop's wanted archetype. */
export const jobChopShop: MissionDef = {
  id: chop.id,
  name: chop.name,
  contact: chop.contact,
  position: contactMarker('job_chopshop'),
  unlockedAfter: chop.unlockedAfter,
  reward: chop.reward,
  repeatable: true,
  build(ctx) {
    const peer = ctx.game.systems.get('chopshop') as unknown as IChopShopPeer | undefined;
    const wanted: VehicleArchetypeId = peer?.wantedArchetype ?? 'sports';
    let ride: IVehicle | null = null;
    const dropoff = doorFront('chop_shop', 10);
    return [
      cutscene(sceneTalk('job_chopshop', dialogue('job_chopshop'))),
      custom({
        text: `Steal a ~y~${NAMES[wanted]}~y~`,
        blipKind: 'vehicle',
        update(c) {
          const v = playerVehicle(c);
          if (v && v.archetypeId === wanted) {
            ride = v;
            return 'done';
          }
          return 'running';
        },
        skip(c) {
          if (!c.game.systems.has('player')) return;
          const spot = parkingNear(c.game, forwardOf(c.game.player.position, c.game.player.heading, 8), 80);
          const v = spawnVehicle(c, wanted, spot.pos, spot.heading);
          if (v) forceMount(c, v);
        },
      }),
      goto(dropoff, 6, `Deliver the ~y~${NAMES[wanted]}~y~ to Gellhorn Salvage`, { vehicle: () => ride, blip: 'chopshop' }),
      action((c) => {
        if (ride) c.reward = Math.max(chop.reward, Math.round(peer?.valueOf(ride) ?? chop.reward));
        if (c.game.systems.has('player') && c.game.player.isMounted) c.game.vehicles?.exitCurrent();
        if (ride) c.game.vehicles?.despawn(ride);
        ride = null;
      }),
    ];
  },
};

/** Repeatable after the finale: rob any store and get away clean. */
export const jobRobbery: MissionDef = {
  id: rob.id,
  name: rob.name,
  contact: rob.contact,
  position: contactMarker('job_robbery'),
  unlockedAfter: rob.unlockedAfter,
  reward: rob.reward,
  repeatable: true,
  build() {
    return [
      cutscene(sceneTalk('job_robbery', dialogue('job_robbery'))),
      robStore(null, null, 'Rob ~y~any store~y~ — aim at the clerk behind the register'),
      escapeWanted(),
      action((c) => c.game.ui?.notify('Ray: "Clean. Come back tomorrow."', 'info')),
    ];
  },
};
