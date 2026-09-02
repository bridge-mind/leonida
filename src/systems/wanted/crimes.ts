import type { Entity, IPed, IVehicle, V3 } from '../../core/entities';
import type { CrimeType } from '../../core/Events';
import type { Game } from '../../core/Game';
import { WantedConfig } from './config';
import type { WitnessInfo } from './machine';

const W = WantedConfig.witness;

export interface CrimeSink {
  crime(type: CrimeType, position: V3, loud: boolean): void;
  phoneReport(position: V3): void;
}

/**
 * Turns engine events into crimes. Most systems emit `crime:committed` directly; the
 * rest (ramming cruisers, hitting cops, taking an empty cruiser) are derived here from
 * collision / damage / vehicle events so those systems need no knowledge of police.
 */
export function subscribeCrimes(game: Game, sink: CrimeSink): () => void {
  const ev = game.events;
  const offs = [
    ev.on('crime:committed', (e) => {
      if (e.actor === 'player' || e.actor === 'companion') sink.crime(e.type, e.position, e.loud);
    }),
    ev.on('ped:reportedCrime', (e) => sink.phoneReport(e.position)),
    ev.on('vehicle:collided', (e) => {
      if (e.impulse < WantedConfig.collision.copVehicleImpulse) return;
      const mine = game.vehicles?.playerVehicle ?? null;
      if (!mine) return;
      if (e.vehicle === mine && e.other) {
        if (isVehicle(e.other) && e.other.isPolice) sink.crime('hit_cop_vehicle', e.vehicle.position, false);
        else if (isPed(e.other) && e.other.isCop) sink.crime('hit_cop', e.other.position, false);
      } else if (e.vehicle.isPolice && e.other === mine) {
        sink.crime('hit_cop_vehicle', mine.position, false);
      }
    }),
    ev.on('damage:applied', (e) => {
      // A lethal hit is reported as kill_cop by the damage system; do not stack hit_cop on top.
      if (e.info.source !== 'player' || !isPed(e.target) || !e.target.isCop || !e.target.isAlive || e.target.health <= 0) return;
      if (e.info.type === 'bullet' || e.info.type === 'melee') sink.crime('hit_cop', e.target.position, e.info.type === 'bullet');
    }),
    ev.on('vehicle:entered', (e) => {
      if (e.vehicle.isPolice && e.who.kind === 'player') sink.crime('steal_police_vehicle', e.vehicle.position, false);
    }),
  ];
  return () => offs.forEach((off) => off());
}

/** Who saw the crime: a cop nearby counts as direct sight, a civilian as a phone-call witness. */
export function witnessesOf(game: Game, position: V3, loud: boolean, scratch: IPed[], out: WitnessInfo): WitnessInfo {
  out.copSaw = false;
  out.civilianSaw = false;
  const peds = game.peds;
  if (!peds) return out;
  peds.query(position, loud ? W.copLoudRadius : W.copRadius, scratch, isLiveCop);
  out.copSaw = anyOnSameLevel(scratch, position.y);
  if (!out.copSaw) {
    peds.witnessesNear(position, loud ? W.civilianLoudRadius : W.civilianRadius, scratch);
    out.civilianSaw = anyOnSameLevel(scratch, position.y);
  }
  return out;
}

/** The ped spatial hash is 2-D: a shot fired inside a store must not be "seen" by peds on the street below it. */
function anyOnSameLevel(peds: IPed[], y: number): boolean {
  for (let i = 0; i < peds.length; i++) if (Math.abs(peds[i].position.y - y) <= W.maxVerticalGap) return true;
  return false;
}

function isLiveCop(p: IPed): boolean {
  return p.isCop && p.isAlive && p.health > 0;
}

function isVehicle(e: Entity): e is IVehicle {
  return e.kind === 'vehicle';
}

function isPed(e: Entity): e is IPed {
  return e.kind === 'ped';
}
