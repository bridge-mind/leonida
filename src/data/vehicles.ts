import type { VehicleArchetypeId } from '../core/peers';
import type { SoundId } from '../systems/audio/sounds';

/**
 * Vehicle archetypes (SYSTEM_PROMPT §5). Every archetype is the same set of unit
 * parts (see gfx/CarBuilder.ts) arranged by these dimensions, so adding one is a
 * data change. Handling numbers feed the arcade model in systems/vehicles/handling.ts.
 * Units: meters, seconds, kg, radians.
 */
export interface VehicleDims {
  length: number;
  width: number;
  /** Roof height above ground. */
  height: number;
  /** Bottom of the body slab above ground. */
  clearance: number;
  /** Top of the body slab (hood / trunk line) above ground. */
  bodyTop: number;
  /** Greenhouse length and its center offset along the forward axis (+ = toward the nose). */
  cabinLength: number;
  cabinOffset: number;
  wheelRadius: number;
  wheelWidth: number;
  wheelbase: number;
  track: number;
  /** Open rear bed (pickup) instead of a trunk lid. */
  bed: boolean;
}

export interface VehicleArchetype {
  id: VehicleArchetypeId;
  name: string;
  dims: VehicleDims;
  mass: number;
  /** Peak forward acceleration at low speed (m/s²). */
  engineAccel: number;
  /** Hard top speed (m/s). */
  topSpeed: number;
  /** Braking deceleration (m/s²). */
  brake: number;
  reverseSpeed: number;
  /** Peak yaw rate at full lock (rad/s) before speed falloff. */
  maxSteerRate: number;
  /** Fraction of lateral velocity removed per 60 Hz step; lower = looser rear end. */
  grip: number;
  handbrakeGrip: number;
  health: number;
  palette: readonly number[];
  seats: number;
  chopValue: number;
  hornSound: SoundId;
}

const sedanDims: VehicleDims = {
  length: 4.5,
  width: 1.85,
  height: 1.45,
  clearance: 0.32,
  bodyTop: 0.9,
  cabinLength: 2.0,
  cabinOffset: -0.25,
  wheelRadius: 0.34,
  wheelWidth: 0.24,
  wheelbase: 2.7,
  track: 1.62,
  bed: false,
};

const vanDims: VehicleDims = {
  length: 5.2,
  width: 2.0,
  height: 2.25,
  clearance: 0.36,
  bodyTop: 1.05,
  cabinLength: 4.3,
  cabinOffset: -0.35,
  wheelRadius: 0.38,
  wheelWidth: 0.26,
  wheelbase: 3.2,
  track: 1.74,
  bed: false,
};

const CIVILIAN_PALETTE = [0xd8d8dc, 0x1f1f24, 0x8c1c2c, 0x243a7a, 0x4a4f57, 0xe8e2d0, 0x2f6b4f, 0x7a2f8f, 0xc7621e, 0x9aa5b1];

export const VEHICLES: Record<VehicleArchetypeId, VehicleArchetype> = {
  sedan: {
    id: 'sedan',
    name: 'Sedan',
    dims: sedanDims,
    mass: 1200,
    engineAccel: 6.5,
    topSpeed: 38,
    brake: 11,
    reverseSpeed: 8,
    maxSteerRate: 2.3,
    grip: 0.9,
    handbrakeGrip: 0.12,
    health: 1000,
    palette: CIVILIAN_PALETTE,
    seats: 4,
    chopValue: 800,
    hornSound: 'vehicle.horn',
  },
  sports: {
    id: 'sports',
    name: 'Sports Coupe',
    dims: {
      length: 4.4,
      width: 1.95,
      height: 1.2,
      clearance: 0.22,
      bodyTop: 0.72,
      cabinLength: 1.6,
      cabinOffset: -0.45,
      wheelRadius: 0.35,
      wheelWidth: 0.3,
      wheelbase: 2.6,
      track: 1.72,
      bed: false,
    },
    mass: 1100,
    engineAccel: 10,
    topSpeed: 55,
    brake: 13,
    reverseSpeed: 10,
    maxSteerRate: 2.6,
    grip: 0.93,
    handbrakeGrip: 0.1,
    health: 1000,
    palette: [0xff3b3b, 0xffd23f, 0x35e6ff, 0xf7f7f7, 0x141414, 0xff6fa3, 0x7cff5a, 0xff8a1f],
    seats: 2,
    chopValue: 4000,
    hornSound: 'vehicle.horn.short',
  },
  muscle: {
    id: 'muscle',
    name: 'Muscle Car',
    dims: {
      length: 4.9,
      width: 1.95,
      height: 1.35,
      clearance: 0.28,
      bodyTop: 0.86,
      cabinLength: 1.7,
      cabinOffset: -0.55,
      wheelRadius: 0.36,
      wheelWidth: 0.3,
      wheelbase: 2.9,
      track: 1.7,
      bed: false,
    },
    mass: 1500,
    engineAccel: 9,
    topSpeed: 48,
    brake: 10.5,
    reverseSpeed: 9,
    maxSteerRate: 2.3,
    grip: 0.8,
    handbrakeGrip: 0.06,
    health: 1000,
    palette: [0x1a1a1a, 0xb81d1d, 0xf28c28, 0x2b5cbf, 0x4f7a3a, 0xe6e6e6, 0x5a2d82],
    seats: 2,
    chopValue: 2500,
    hornSound: 'vehicle.horn.long',
  },
  suv: {
    id: 'suv',
    name: 'SUV',
    dims: {
      length: 4.8,
      width: 2.0,
      height: 1.85,
      clearance: 0.4,
      bodyTop: 1.05,
      cabinLength: 3.0,
      cabinOffset: -0.35,
      wheelRadius: 0.4,
      wheelWidth: 0.28,
      wheelbase: 2.85,
      track: 1.74,
      bed: false,
    },
    mass: 2000,
    engineAccel: 5.5,
    topSpeed: 36,
    brake: 10,
    reverseSpeed: 7,
    maxSteerRate: 2.0,
    grip: 0.88,
    handbrakeGrip: 0.14,
    health: 1000,
    palette: [0x111111, 0xe9e9e9, 0x3b3f46, 0x6b2a2a, 0x2a3f6b, 0x8a7a5a, 0xd9d2c0],
    seats: 4,
    chopValue: 1500,
    hornSound: 'vehicle.horn',
  },
  pickup: {
    id: 'pickup',
    name: 'Pickup',
    dims: {
      length: 5.4,
      width: 2.0,
      height: 1.8,
      clearance: 0.42,
      bodyTop: 1.02,
      cabinLength: 1.6,
      cabinOffset: 0.55,
      wheelRadius: 0.4,
      wheelWidth: 0.28,
      wheelbase: 3.3,
      track: 1.74,
      bed: true,
    },
    mass: 1900,
    engineAccel: 5.8,
    topSpeed: 37,
    brake: 10,
    reverseSpeed: 7,
    maxSteerRate: 2.0,
    grip: 0.86,
    handbrakeGrip: 0.12,
    health: 1000,
    palette: [0xc9302c, 0x2b4c7e, 0xf2f2f2, 0x4a4a4a, 0x8b5a2b, 0x3f6b3f, 0xd9a441],
    seats: 2,
    chopValue: 1000,
    hornSound: 'vehicle.horn.long',
  },
  van: {
    id: 'van',
    name: 'Van',
    dims: vanDims,
    mass: 2200,
    engineAccel: 4.8,
    topSpeed: 33,
    brake: 9.5,
    reverseSpeed: 6,
    maxSteerRate: 1.9,
    grip: 0.9,
    handbrakeGrip: 0.16,
    health: 1000,
    palette: [0xf0f0f0, 0xb8b8b8, 0x2f4f7f, 0x7a2f2f, 0x3a3a3a, 0xd9c79a],
    seats: 2,
    chopValue: 900,
    hornSound: 'vehicle.horn.long',
  },
  taxi: {
    id: 'taxi',
    name: 'Taxi',
    dims: sedanDims,
    mass: 1250,
    engineAccel: 6.2,
    topSpeed: 37,
    brake: 11,
    reverseSpeed: 8,
    maxSteerRate: 2.3,
    grip: 0.9,
    handbrakeGrip: 0.12,
    health: 1000,
    palette: [0xffc61a],
    seats: 4,
    chopValue: 600,
    hornSound: 'vehicle.horn.short',
  },
  police: {
    id: 'police',
    name: 'Police Cruiser',
    dims: { ...sedanDims, length: 4.7, wheelbase: 2.8 },
    mass: 1300,
    engineAccel: 8,
    topSpeed: 46,
    brake: 12,
    reverseSpeed: 9,
    maxSteerRate: 2.4,
    grip: 0.9,
    handbrakeGrip: 0.1,
    health: 1000,
    palette: [0xf4f4f4],
    seats: 4,
    chopValue: 1200,
    hornSound: 'vehicle.horn',
  },
  swat: {
    id: 'swat',
    name: 'SWAT Van',
    dims: { ...vanDims, length: 5.6, height: 2.4, wheelbase: 3.4, clearance: 0.42 },
    mass: 2600,
    engineAccel: 5,
    topSpeed: 34,
    brake: 10,
    reverseSpeed: 6,
    maxSteerRate: 1.8,
    grip: 0.92,
    handbrakeGrip: 0.2,
    health: 1000,
    palette: [0x1c1f26],
    seats: 6,
    chopValue: 3000,
    hornSound: 'vehicle.horn.long',
  },
};

export const VEHICLE_IDS: readonly VehicleArchetypeId[] = Object.keys(VEHICLES) as VehicleArchetypeId[];

export function isVehicleArchetypeId(id: string): id is VehicleArchetypeId {
  return Object.prototype.hasOwnProperty.call(VEHICLES, id);
}

/** Relative archetype weights for parked and traffic cars per district. */
export const DISTRICT_VEHICLE_MIX: Record<string, Partial<Record<VehicleArchetypeId, number>>> = {
  viceBeach: { sports: 0.32, muscle: 0.16, sedan: 0.24, suv: 0.1, taxi: 0.18 },
  downtown: { sedan: 0.38, taxi: 0.26, suv: 0.14, sports: 0.1, van: 0.12 },
  littleHabana: { sedan: 0.38, pickup: 0.2, van: 0.14, muscle: 0.12, taxi: 0.16 },
  portGellhorn: { pickup: 0.36, van: 0.2, sedan: 0.26, suv: 0.18 },
  docks: { van: 0.36, pickup: 0.34, sedan: 0.2, suv: 0.1 },
  grassrivers: { pickup: 0.6, van: 0.2, sedan: 0.2 },
  default: { sedan: 0.45, suv: 0.15, pickup: 0.15, van: 0.1, taxi: 0.15 },
};

/** Ped type ids used as invisible traffic drivers (spawned as fleeing peds on carjack). */
export const TRAFFIC_DRIVER_TYPES: readonly string[] = ['business', 'worker', 'tourist', 'beachgoer'];
