import type { IVehicle } from '../../core/entities';

/** Events private to the vehicle/traffic systems (declaration-merged into GameEvents). */
declare module '../../core/Events' {
  interface GameEvents {
    /** A vehicle was recycled into the pool; holders of the reference must drop it. */
    'vehicle:despawned': { vehicle: IVehicle };
    /** A vehicle was spawned (any kind: parked, traffic, player). */
    'vehicle:spawned': { vehicle: IVehicle };
  }
}

export {};
