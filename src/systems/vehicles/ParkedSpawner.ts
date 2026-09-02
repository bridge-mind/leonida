import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { VehicleArchetypeId } from '../../core/peers';
import type { Random } from '../../core/Random';
import { DISTRICT_VEHICLE_MIX, VEHICLES } from '../../data/vehicles';
import type { IVehiclesPeer } from '../../core/peers';
import { VehicleConfig } from './config';

const P = VehicleConfig.parked;

/** Weighted archetype pick for a district (`DISTRICT_VEHICLE_MIX`), optionally excluding taxis. */
export function pickArchetypeFor(game: Game, pos: V3, rng: Random, allowTaxi: boolean): VehicleArchetypeId {
  const district = game.systems.has('world') ? game.world.districtAt(pos) : null;
  const mix = DISTRICT_VEHICLE_MIX[district?.id ?? 'default'] ?? DISTRICT_VEHICLE_MIX.default;
  let total = 0;
  for (const id in mix) {
    if (!allowTaxi && id === 'taxi') continue;
    total += mix[id as VehicleArchetypeId] ?? 0;
  }
  let r = rng.next() * total;
  for (const id in mix) {
    if (!allowTaxi && id === 'taxi') continue;
    r -= mix[id as VehicleArchetypeId] ?? 0;
    if (r <= 0) return id as VehicleArchetypeId;
  }
  return 'sedan';
}

export function pickColor(id: VehicleArchetypeId, rng: Random): number {
  return rng.pick(VEHICLES[id].palette);
}

/**
 * Fills curb and lot parking spots with sleeping cars at init: spots are shuffled
 * (seeded) and accepted when nothing else is parked within `minSpacing`, which thins
 * the 6.5 m curb grid to roughly one car per 40 m of street, capped at `cap`.
 */
export function spawnParkedCars(game: Game, vehicles: IVehiclesPeer, rng: Random): number {
  if (!game.systems.has('world')) return 0;
  const world = game.world;
  const spots = world.roadGraph.parkingSpots;
  const order: number[] = [];
  for (let i = 0; i < spots.length; i++) order.push(i);
  rng.shuffle(order);

  const placedX: number[] = [];
  const placedZ: number[] = [];
  const min2 = P.minSpacing * P.minSpacing;
  let count = 0;
  for (let k = 0; k < order.length && count < P.cap; k++) {
    const spot = spots[order[k]];
    if (!rng.chance(spot.kind === 'curb' ? P.curbChance : P.lotChance)) continue;
    const p = spot.position;
    if (world.isInWater(p)) continue;
    let tooClose = false;
    for (let i = 0; i < placedX.length; i++) {
      const dx = placedX[i] - p.x;
      const dz = placedZ[i] - p.z;
      if (dx * dx + dz * dz < min2) {
        tooClose = true;
        break;
      }
    }
    if (tooClose) continue;
    const id = pickArchetypeFor(game, p, rng, false);
    vehicles.spawn(id, { x: p.x, y: p.y + 0.02, z: p.z }, spot.heading, { color: pickColor(id, rng), parked: true });
    placedX.push(p.x);
    placedZ.push(p.z);
    count++;
  }
  return count;
}
