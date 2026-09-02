import type { IVehicle } from '../core/entities';
import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { contactMarker, doorFront, forwardOf, landmarkDoor, parkingNear, sceneOrbit, spawnVehicle, withCompanion } from './helpers';
import { action, enterVehicle, goto } from './objectives';
import { cutscene } from './objectivesWorld';
import type { MissionDef } from './types';

const story = STORY_BY_ID.get('m1')!;

/** 1 — Release: Lucia walks out of Leonida Penitentiary; Jason drives her to the safehouse. */
export const m1: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m1'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build() {
    let car: IVehicle | null = null;
    const gate = landmarkDoor('prison');
    const safehouse = doorFront('safehouse', 8);
    return [
      action((ctx) => {
        if (ctx.game.systems.has('player')) ctx.game.player.teleport(forwardOf(gate.pos, gate.heading, 3), gate.heading);
        const spot = parkingNear(ctx.game, forwardOf(gate.pos, gate.heading, 16, 8), 40);
        // Jason's car stays with the player after the intro, so it is not cleaned up.
        car = spawnVehicle(ctx, 'muscle', spot.pos, spot.heading, { color: 0x1f2a44, persistent: true });
      }),
      withCompanion(),
      cutscene(sceneOrbit('m1_gate', forwardOf(gate.pos, gate.heading, 3), dialogue('m1_gate'))),
      enterVehicle(() => car, "Get in ~y~Jason's car~y~"),
      goto(safehouse, 7, 'Drive to the ~y~safehouse~y~ in Little Habana', { vehicle: () => car, blip: 'safehouse' }),
      action((ctx) => {
        if (ctx.game.systems.has('player') && ctx.game.player.isMounted) ctx.game.vehicles?.exitCurrent();
      }),
      cutscene(sceneOrbit('m1_safehouse', safehouse, dialogue('m1_safehouse'))),
      action((ctx) => ctx.game.ui?.notify('Safehouse unlocked — sleep here to save', 'info')),
    ];
  },
};
