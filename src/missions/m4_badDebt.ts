import type { IPed, IVehicle, V3 } from '../core/entities';
import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { DebtorBrain } from './brains/DebtorBrain';
import { contactMarker, forwardOf, parkingNear, sceneTalk, sidewalkNear, spawnPed, spawnVehicle } from './helpers';
import { action, custom, horizontal, playerPos } from './objectives';
import { cutscene, intimidateOrKill } from './objectivesWorld';
import type { MissionDef } from './types';

const story = STORY_BY_ID.get('m4')!;
const DEBTOR_AREA = { x: -362, y: 0, z: 136 };
/** Where the getaway sedan tries to reach (Port Gellhorn 24/7): long enough for a real chase. */
const FLEE_TO = { x: -600, y: 0, z: -520 };
const REWARD_INTIMIDATED = 800;
const REWARD_KILLED = 1200;

/** 4 — Bad Debt: scare (aim 2 s) or kill a debtor who runs, then drives off. */
export const m4: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m4'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build() {
    let debtor: IPed | null = null;
    let sedan: IVehicle | null = null;
    const beat = intimidateOrKill(() => debtor, 2, 15, 'Aim at the ~y~debtor~y~ for 2 seconds, or kill him');
    return [
      cutscene(sceneTalk('m4_start', dialogue('m4_start'))),
      action((ctx) => {
        const game = ctx.game;
        const spot = sidewalkNear(game, DEBTOR_AREA);
        const park = parkingNear(game, forwardOf(spot, 0, 14), 60);
        let route: V3[] = game.systems.has('world') ? game.world.roadGraph.routeBetween(park.pos, FLEE_TO) : [];
        if (route.length < 2) route = [park.pos, forwardOf(park.pos, park.heading, 120)];
        sedan = spawnVehicle(ctx, 'sedan', park.pos, park.heading, { color: 0x8a8f99 });
        debtor = spawnPed(ctx, 'business', spot, { brain: new DebtorBrain(() => sedan, route) });
      }),
      custom({
        text: 'Find the ~y~debtor~y~ in Little Habana',
        target: () => debtor?.position ?? null,
        update(ctx) {
          const p = playerPos(ctx);
          if (!debtor || !p) return 'running';
          if (!debtor.isAlive) return 'done';
          return horizontal(p, debtor.position) < 22 ? 'done' : 'running';
        },
        skip(ctx) {
          if (debtor && ctx.game.systems.has('player')) ctx.game.player.teleport(forwardOf(debtor.position, 0, -6));
        },
      }),
      beat,
      action((ctx) => {
        const killed = beat.result() === 'killed';
        ctx.reward = killed ? REWARD_KILLED : REWARD_INTIMIDATED;
        if (!killed && debtor) ctx.game.ui?.notify('The debtor paid up', 'money');
        // He drove off: the sedan must not end up as a phantom parked car with a scripted driver.
        if (sedan && sedan.isAlive) sedan.setDriver(null);
      }),
      cutscene(() => sceneTalk('m4_end', dialogue(beat.result() === 'killed' ? 'm4_killed' : 'm4_intimidated'))),
    ];
  },
};
