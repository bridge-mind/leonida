import type { BodyPart, DamageInfo, DamageType, Entity, PedEvent, V3 } from '../../core/entities';
import type { PedSystem } from './index';

/**
 * Runtime shape of ped brain events. `PedEventMap.damaged` is declared as a bare
 * `DamageInfo`, whose `type: DamageType` collides with the `type: 'damaged'`
 * discriminator (intersection = never), so the damage kind travels as `damageType`
 * and the full info as `info`. Brains cast `PedEvent` to this union (`asBrainEvent`).
 */
export type BrainEvent =
  | { type: 'aimedAt'; seconds: number }
  | { type: 'heardGunshot'; position: V3; distance: number }
  | { type: 'hitByVehicle'; impulse: number }
  | { type: 'damaged'; info: DamageInfo; damageType: DamageType; amount: number; point?: V3; direction?: V3; part?: BodyPart; source?: Entity | 'player' }
  | { type: 'explosion'; position: V3; distance: number }
  | { type: 'horn'; position: V3 }
  | { type: 'crimeSeen'; position: V3; distance: number; loud: boolean }
  | { type: 'carjacked'; threat: V3 };

export function toPedEvent(e: BrainEvent): PedEvent {
  return e as unknown as PedEvent;
}

export function asBrainEvent(e: PedEvent): BrainEvent {
  return e as unknown as BrainEvent;
}

export function damagedEvent(info: DamageInfo): PedEvent {
  return toPedEvent({
    type: 'damaged',
    info,
    damageType: info.type,
    amount: info.amount,
    point: info.point,
    direction: info.direction,
    part: info.part,
    source: info.source,
  });
}

/**
 * Ped-system additions to the shared contracts via declaration merging:
 * extra brain events (explosions, horns, crimes seen) and the `peds` handle on
 * `PedContext` so brains written by other agents can query neighbours.
 */
declare module '../../core/entities' {
  interface PedEventMap {
    /** A vehicle blew up within hearing range. */
    explosion: { position: V3; distance: number };
    /** A vehicle honked right behind this ped. */
    horn: { position: V3 };
    /** The ped saw or heard a crime (`loud` crimes carry further). */
    crimeSeen: { position: V3; distance: number; loud: boolean };
    /** Spawned as a carjacked driver: start fleeing from `threat`. */
    carjacked: { threat: V3 };
  }

  interface PedContext {
    peds: PedSystem;
    /** Random source for brains (unseeded gameplay RNG). */
    rng: import('../../core/Random').Random;
    /** True when the day/night clock says night. */
    isNight: boolean;
  }
}