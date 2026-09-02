import type { IPed, IVehicle, V3 } from '../core/entities';
import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { MissionsConfig } from '../systems/missions/config';
import { HostileBrain } from './brains/HostileBrain';
import { contactMarker, doorFront, forwardOf, parkingNear, sceneOrbit, spawnPed, spawnVehicle, withCompanion } from './helpers';
import { action, custom, goto, timed } from './objectives';
import { cutscene, lootGrab } from './objectivesWorld';
import type { MissionContext, MissionDef } from './types';

const story = STORY_BY_ID.get('m8')!;
const INTERIOR_ID = 'jewelry';
/** Counter sits this far ahead of the interior spawn point (see GuardedInterior jewelry layout). */
const COUNTER_AHEAD = 5.5;
const LOOT_SECONDS = 20;
const LOOT_RADIUS = 6;
const GETAWAY_MINUTES = 5;
const HEAT_LEVEL = 5;

/** 8 — The Score: Vice Rocks Jewelers with Jason, five stars, and a boat at the marina. */
export const m8: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m8'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build() {
    let counter: V3 | null = null;
    let getaway: IVehicle | null = null;
    const guards: IPed[] = [];
    const safehouse = doorFront('safehouse', 6);
    const store = doorFront('jewelry', 5);
    const marina = doorFront('marina', 10);
    return [
      withCompanion(),
      cutscene(sceneOrbit('m8_start', safehouse, dialogue('m8_start'))),
      goto(store, 6, 'Go to ~y~Vice Rocks Jewelers~y~ Downtown with Jason', { blip: 'jewelry' }),
      enterStore(() => counter !== null, (ctx) => {
        counter = counterFrom(ctx);
        ensureGuards(ctx, counter, guards);
      }),
      lootGrab(() => counter, LOOT_SECONDS, LOOT_RADIUS, 'Fill the bag at the ~y~counter~y~'),
      leaveStore(),
      action((ctx) => {
        ctx.game.wanted?.setLevel(HEAT_LEVEL);
        const spot = parkingNear(ctx.game, marina, 60);
        getaway = spawnVehicle(ctx, 'sports', spot.pos, spot.heading, { color: 0xf2f2f2 });
        ctx.game.ui?.notify('Dominic: "Boat leaves the marina in five minutes. Move."', 'warning');
      }),
      timed(goto(marina, 8, 'Get to the ~y~marina~y~ before the boat leaves', { blip: 'marina' }), GETAWAY_MINUTES * 60, 'the boat left without you'),
      action((ctx) => {
        ctx.game.wanted?.clear('mission');
        if (ctx.game.systems.has('player') && ctx.game.player.isMounted) ctx.game.vehicles?.exitCurrent();
        if (getaway) ctx.game.vehicles?.despawn(getaway);
        getaway = null;
      }),
      cutscene(sceneOrbit('m8_end', marina, dialogue('m8_end'))),
      action((ctx) => ctx.game.ui?.card('custom', 'THE END — LEONIDA', ctx.fast ? MissionsConfig.cards.fastSeconds : 6)),
    ];
  },
};

/** Counter position: a few meters ahead of where the interior placed the player. */
function counterFrom(ctx: MissionContext): V3 {
  if (!ctx.game.systems.has('player')) return { x: 0, y: 0, z: 0 };
  return forwardOf(ctx.game.player.position, ctx.game.player.heading, COUNTER_AHEAD);
}

/** Uses the interior's own guards when present; otherwise two of ours flank the counter. */
function ensureGuards(ctx: MissionContext, counter: V3, guards: IPed[]): void {
  const peds = ctx.game.peds;
  if (!peds) return;
  const existing: IPed[] = [];
  peds.query(counter, 15, existing, (p) => p.typeId === 'guard' && p.isAlive);
  if (existing.length > 0) return;
  for (const side of [-3, 3]) {
    const g = spawnPed(ctx, 'guard', { x: counter.x + side, y: counter.y, z: counter.z + 1.5 }, { brain: new HostileBrain({ weapon: 'pistol', minRange: 2, maxRange: 10, engageRange: 30 }), weaponId: 'pistol' });
    if (g) guards.push(g);
  }
}

function enterStore(entered: () => boolean, onEnter: (ctx: MissionContext) => void) {
  let off: (() => void) | null = null;
  return custom({
    text: 'Walk into the ~y~jewelry store~y~',
    blipKind: 'jewelry',
    start(ctx) {
      if (ctx.game.state.interior === INTERIOR_ID) onEnter(ctx);
      off = ctx.game.events.on('interior:entered', (e) => {
        if (e.id === INTERIOR_ID && !entered()) onEnter(ctx);
      });
    },
    update: () => (entered() ? 'done' : 'running'),
    skip(ctx) {
      // Without walking through the door the counter is simply where the player stands.
      if (!entered()) onEnter(ctx);
    },
    stop() {
      off?.();
      off = null;
    },
  });
}

function leaveStore() {
  let left = false;
  return custom({
    text: 'Get out of the ~y~store~y~',
    update(ctx) {
      if (ctx.game.state.interior !== INTERIOR_ID) left = true;
      return left ? 'done' : 'running';
    },
    skip(ctx) {
      left = true;
      if (ctx.game.interiors?.isInside) void ctx.game.interiors.exit();
    },
  });
}
