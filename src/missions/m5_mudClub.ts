import type { IPed, IVehicle, V3 } from '../core/entities';
import type { WeaponId } from '../core/peers';
import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { HostileBrain } from './brains/HostileBrain';
import { contactMarker, doorFront, forwardOf, parkingNear, roadPointNear, sceneTalk, sidewalkNear, spawnPed, spawnVehicle } from './helpers';
import { action, goto, survive } from './objectives';
import { cutscene } from './objectivesWorld';
import type { MissionContext, MissionDef } from './types';

const story = STORY_BY_ID.get('m5')!;
const PACKAGE_AREA = { x: 58, y: 0, z: 332 };
const ESCAPE_AREA = { x: -770, y: 0, z: -598 };
const GANG_COUNT = 6;
const AMBUSH_RADIUS = 14;
const SURVIVE_SECONDS = 90;

/** 5 — Mud Club: courier run to the Gator Pit that turns into an ambush; leave in the pickup. */
export const m5: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m5'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build(ctx) {
    const gang: IPed[] = [];
    let truck: IVehicle | null = null;
    const club = doorFront('mud_club', 8);
    const pkg = sidewalkNear(ctx.game, PACKAGE_AREA);
    const escape = roadPointNear(ctx.game, ESCAPE_AREA).pos;
    return [
      cutscene(sceneTalk('m5_start', dialogue('m5_start'))),
      goto(pkg, 2.5, 'Pick up the ~y~package~y~ Downtown'),
      action((c) => c.game.ui?.notify('Package collected', 'info')),
      goto(club, 8, 'Deliver the package to the ~y~Gator Pit~y~ in Grassrivers', { blip: 'club' }),
      cutscene({
        id: 'm5_ambush',
        letterbox: false,
        steps: [
          { dialogue: dialogue('m5_ambush') },
          // Spawns live inside the scene so the rifle and the truck exist the moment control returns.
          { run: () => spawnAmbush(ctx, club, gang, (v) => (truck = v)) },
        ],
      }),
      survive(SURVIVE_SECONDS, 'Survive the ~y~ambush~y~ — the rifle is on the ground', { orKill: () => gang, showTimer: true }),
      goto(escape, 8, 'Take the ~y~pickup~y~ and get out of the swamp', { vehicle: () => truck, blip: 'vehicle' }),
      cutscene(sceneTalk('m5_done', dialogue('m5_done'))),
    ];
  },
};

function spawnAmbush(ctx: MissionContext, club: V3, gang: IPed[], setTruck: (v: IVehicle | null) => void): void {
  const game = ctx.game;
  const center = game.systems.has('player') ? game.player.position : club;
  for (let i = 0; i < GANG_COUNT; i++) {
    const a = (i / GANG_COUNT) * Math.PI * 2;
    const pos = { x: center.x + Math.sin(a) * AMBUSH_RADIUS, y: center.y, z: center.z + Math.cos(a) * AMBUSH_RADIUS };
    const weapon: WeaponId = i % 3 === 0 ? 'smg' : 'pistol';
    const ped = spawnPed(ctx, 'gang', pos, { brain: new HostileBrain({ weapon, fireRate: weapon === 'smg' ? 3 : 1.2, accuracy: 0.4 }), weaponId: weapon });
    if (ped) gang.push(ped);
  }
  game.pickups?.spawn('weapon', { x: center.x, y: center.y + 0.3, z: center.z + 2.5 }, 90, 'rifle');
  const spot = parkingNear(game, forwardOf(club, 0, 18, 10), 60);
  setTruck(spawnVehicle(ctx, 'pickup', spot.pos, spot.heading, { color: 0x3b2f2a }));
}
