import { Vector3 } from 'three';
import type { IPed, PedContext } from '../../../core/entities';
import type { WeaponId } from '../../../core/peers';
import { ShopsConfig } from '../config';

const origin = new Vector3();
const dir = new Vector3();

/** Fires the ped's weapon at the player's chest through the weapons system (no-op without it). */
export function shootPlayer(ped: IPed, ctx: PedContext, weaponId: WeaponId, accuracy: number): boolean {
  const weapons = ctx.game.systems.has('weapons') ? ctx.game.weapons : undefined;
  if (!weapons) return false;
  const p = ped.position;
  const t = ctx.playerPosition;
  origin.set(p.x, p.y + ShopsConfig.chestHeight, p.z);
  dir.set(t.x - origin.x, t.y + ShopsConfig.playerChestHeight - origin.y, t.z - origin.z);
  if (dir.lengthSq() < 1e-4) return false;
  dir.normalize();
  weapons.npcFire(ped, weaponId, origin, dir, accuracy);
  return true;
}

/** Heading (0 faces +Z) from the ped to a point. */
export function headingTo(ped: IPed, target: { x: number; z: number }): number {
  return Math.atan2(target.x - ped.position.x, target.z - ped.position.z);
}
