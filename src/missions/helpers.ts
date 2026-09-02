import type { IPed, IVehicle, PedBrain, V3, VehicleDriver } from '../core/entities';
import type { Game } from '../core/Game';
import type { DialogueLine, VehicleArchetypeId, WeaponId } from '../core/peers';
import { LANDMARK_BY_ID } from '../data/landmarks';
import { CONTACTS, STORY_BY_ID, type ContactId } from '../data/story';
import type { CutsceneDef, CutsceneStep } from '../systems/cutscene';
import type { CompanionSystem } from '../systems/companion';
import { action } from './objectives';
import type { MissionContext, Objective } from './types';

/** Shared spawn/position helpers for mission scripts. Everything spawned is cleaned up with the run. */

export function forwardOf(pos: V3, heading: number, meters: number, side = 0): V3 {
  return {
    x: pos.x + Math.sin(heading) * meters + Math.cos(heading) * side,
    y: pos.y,
    z: pos.z + Math.cos(heading) * meters - Math.sin(heading) * side,
  };
}

export function landmarkDoor(id: string): { pos: V3; heading: number } {
  const lm = LANDMARK_BY_ID[id];
  if (!lm) return { pos: { x: 0, y: 0, z: 0 }, heading: 0 };
  return { pos: { x: lm.door.x, y: lm.door.y, z: lm.door.z }, heading: lm.doorHeading };
}

/** Point `meters` out from a landmark door along the door normal. */
export function doorFront(id: string, meters: number, side = 0): V3 {
  const d = landmarkDoor(id);
  return forwardOf(d.pos, d.heading, meters, side);
}

/** Contact marker position from story data (`markerLandmarkId` overrides the contact's hangout). */
export function contactMarker(missionId: string): V3 {
  const story = STORY_BY_ID.get(missionId);
  if (!story) return { x: 0, y: 0, z: 0 };
  const contact = CONTACTS[story.contact];
  return doorFront(story.markerLandmarkId ?? contact.landmarkId, story.markerOffset ?? contact.doorOffset);
}

export function contactName(id: ContactId): string {
  return CONTACTS[id].name;
}

/** Nearest road lane point (for cars) or the input when the world is not loaded. */
export function roadPointNear(game: Game, pos: V3): { pos: V3; heading: number } {
  if (!game.systems.has('world')) return { pos, heading: 0 };
  const lane = game.world.roadGraph.nearestLanePoint(pos);
  if (!lane) return { pos, heading: 0 };
  return { pos: { x: lane.point.x, y: lane.point.y, z: lane.point.z }, heading: Math.atan2(lane.direction.x, lane.direction.z) };
}

/** Nearest curb/lot parking spot within `maxDist`, else the nearest lane point. */
export function parkingNear(game: Game, pos: V3, maxDist = 60): { pos: V3; heading: number } {
  if (game.systems.has('world')) {
    let best: { pos: V3; heading: number } | null = null;
    let bestD = maxDist * maxDist;
    for (const spot of game.world.roadGraph.parkingSpots) {
      const dx = spot.position.x - pos.x;
      const dz = spot.position.z - pos.z;
      const d = dx * dx + dz * dz;
      if (d < bestD) {
        bestD = d;
        best = { pos: { x: spot.position.x, y: spot.position.y, z: spot.position.z }, heading: spot.heading };
      }
    }
    if (best) return best;
  }
  return roadPointNear(game, pos);
}

export function sidewalkNear(game: Game, pos: V3): V3 {
  if (!game.systems.has('world')) return pos;
  const sw = game.world.roadGraph.nearestSidewalkPoint(pos);
  return sw ? { x: sw.point.x, y: sw.point.y, z: sw.point.z } : pos;
}

export function playerHere(ctx: MissionContext): V3 {
  if (!ctx.game.systems.has('player')) return { x: 0, y: 0, z: 0 };
  const p = ctx.game.player.position;
  return { x: p.x, y: p.y, z: p.z };
}

// --- Spawning with cleanup --------------------------------------------------------------

export function spawnVehicle(
  ctx: MissionContext,
  archetype: VehicleArchetypeId,
  pos: V3,
  heading: number,
  opts: { color?: number; driver?: VehicleDriver | null; parked?: boolean; persistent?: boolean } = {},
): IVehicle | null {
  const vehicles = ctx.game.vehicles;
  if (!vehicles) return null;
  const ground = ctx.game.physics.groundHeightAt(pos.x, pos.z);
  const { persistent, ...spawnOpts } = opts;
  const v = vehicles.spawn(archetype, { x: pos.x, y: (ground ?? pos.y) + 0.3, z: pos.z }, heading, { parked: true, ...spawnOpts });
  if (persistent) return v;
  ctx.onCleanup(() => {
    // Leave the player's current ride alone; everything else goes.
    if (v.isAlive && ctx.game.vehicles?.playerVehicle !== v) ctx.game.vehicles?.despawn(v);
  });
  return v;
}

export function spawnPed(ctx: MissionContext, typeId: string, pos: V3, opts: { brain?: PedBrain; heading?: number; weaponId?: WeaponId } = {}): IPed | null {
  const peds = ctx.game.peds;
  if (!peds) return null;
  const ped = peds.spawn(typeId, pos, opts);
  ped.state = 'scripted';
  ctx.onCleanup(() => {
    if (ped.isAlive) ctx.game.peds?.despawn(ped);
  });
  return ped;
}

// --- Companion ---------------------------------------------------------------------------

export function companionOf(game: Game): CompanionSystem | undefined {
  return game.systems.get<CompanionSystem>('companion');
}

/** Summons the other protagonist for the mission and dismisses them when it ends. */
export function withCompanion(): Objective {
  return action((c) => {
    const comp = companionOf(c.game);
    if (!comp) return;
    const other = c.game.characters?.active === 'jason' ? 'lucia' : 'jason';
    comp.summon(other);
    c.onCleanup(() => comp.dismiss());
  });
}

// --- Cutscene builders ---------------------------------------------------------------------

/** Orbit around `center` while the lines play, then hand back to gameplay. */
export function sceneOrbit(id: string, center: V3, lines: DialogueLine[], extra: CutsceneStep[] = []): CutsceneDef {
  const c = { x: center.x, y: center.y + 1.2, z: center.z };
  return {
    id,
    steps: [...extra, { orbit: { center: c, seconds: 0.8 } }, { dialogue: lines }],
  };
}

/** Radio-style chatter without bars or camera changes. */
export function sceneTalk(id: string, lines: DialogueLine[]): CutsceneDef {
  return { id, letterbox: false, steps: [{ dialogue: lines }] };
}

export function despawnAll(ctx: MissionContext, list: (IPed | null)[]): void {
  for (const p of list) if (p && p.isAlive) ctx.game.peds?.despawn(p);
}
