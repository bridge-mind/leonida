import {
  BoxGeometry,
  CylinderGeometry,
  MeshBasicMaterial,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
  type BufferGeometry,
  type Material,
} from 'three';
import type { VehicleArchetypeId } from '../core/peers';
import { VEHICLES, type VehicleArchetype } from '../data/vehicles';

/**
 * Box-built cars for the shared instanced renderer. Every archetype is the same set
 * of unit parts (BoxGeometry / CylinderGeometry of size 1) placed and scaled by its
 * dimensions, so one InstancedMesh per part kind draws every car in the world.
 * Local frame: origin on the ground under the car's center, +Z forward, +X right.
 */
export type CarPartKind =
  | 'body'
  | 'roof'
  | 'glass'
  | 'hood'
  | 'trunk'
  | 'bumper'
  | 'wheel'
  | 'headlight'
  | 'taillight'
  | 'lightbar'
  | 'taxiSign';

export type CarColorRole = 'paint' | 'glass' | 'dark' | 'rubber' | 'headlight' | 'taillight' | 'lightbar' | 'taxi';

export const CAR_PART_KINDS: readonly CarPartKind[] = [
  'body',
  'roof',
  'glass',
  'hood',
  'trunk',
  'bumper',
  'wheel',
  'headlight',
  'taillight',
  'lightbar',
  'taxiSign',
];

/** Instances of each part kind per vehicle (wheel 4, bumpers/lights 2, the rest 1). */
export const CAR_PART_INSTANCES: Record<CarPartKind, number> = {
  body: 1,
  roof: 1,
  glass: 1,
  hood: 1,
  trunk: 1,
  bumper: 2,
  wheel: 4,
  headlight: 2,
  taillight: 2,
  lightbar: 1,
  taxiSign: 1,
};

export interface CarPart {
  kind: CarPartKind;
  /** Instance index within the kind (0..CAR_PART_INSTANCES[kind]-1). */
  index: number;
  offset: Vector3;
  size: Vector3;
  /** Pitch applied to lids so hoods slope toward the nose. */
  rotX: number;
  role: CarColorRole;
  /** Wheel slot: 0 FL, 1 FR, 2 RL, 3 RR; -1 for non-wheels. */
  wheel: number;
}

export interface CarLayout {
  archetype: VehicleArchetype;
  parts: CarPart[];
  /** Where smoke/fire spawn (top of the hood). */
  hoodPoint: Vector3;
  /** Collider half extents (x, y, z) and its center height. */
  colliderHalf: Vector3;
  colliderCenterY: number;
}

const layouts = new Map<VehicleArchetypeId, CarLayout>();

export function carLayout(id: VehicleArchetypeId): CarLayout {
  let layout = layouts.get(id);
  if (!layout) {
    layout = buildLayout(VEHICLES[id]);
    layouts.set(id, layout);
  }
  return layout;
}

function part(kind: CarPartKind, index: number, role: CarColorRole, x: number, y: number, z: number, sx: number, sy: number, sz: number, rotX = 0, wheel = -1): CarPart {
  return { kind, index, offset: new Vector3(x, y, z), size: new Vector3(sx, sy, sz), rotX, role, wheel };
}

function buildLayout(a: VehicleArchetype): CarLayout {
  const d = a.dims;
  const parts: CarPart[] = [];
  const halfL = d.length / 2;
  const bodyH = d.bodyTop - d.clearance;
  const cabinH = d.height - d.bodyTop;
  const cabinFront = d.cabinOffset + d.cabinLength / 2;
  const cabinRear = d.cabinOffset - d.cabinLength / 2;
  const isPolice = a.id === 'police' || a.id === 'swat';
  const isSports = a.id === 'sports';

  parts.push(part('body', 0, 'paint', 0, d.clearance + bodyH / 2, 0, d.width, bodyH, d.length * 0.98));

  // Greenhouse: dark glass box with a paint-coloured roof slab; vans/SUVs are nearly full-length.
  const glassInset = 0.12;
  parts.push(part('glass', 0, 'glass', 0, d.bodyTop + (cabinH - 0.08) / 2, d.cabinOffset, d.width - glassInset * 2, cabinH - 0.08, d.cabinLength));
  parts.push(part('roof', 0, 'paint', 0, d.height - 0.04, d.cabinOffset, d.width - glassInset * 2 + 0.04, 0.08, d.cabinLength * 0.96));

  // Hood: thin lid from the windshield base to the nose, sloping down toward the front.
  const hoodLen = Math.max(0.3, halfL - cabinFront - 0.15);
  const hoodRot = isSports ? -0.14 : -0.06;
  parts.push(part('hood', 0, 'paint', 0, d.bodyTop + 0.03 - (Math.sin(-hoodRot) * hoodLen) / 2, cabinFront + hoodLen / 2, d.width * 0.94, 0.08, hoodLen, hoodRot));

  // Trunk lid or open bed behind the cabin.
  const trunkLen = Math.max(0.3, halfL + cabinRear - 0.15);
  if (d.bed) {
    parts.push(part('trunk', 0, 'dark', 0, d.bodyTop - 0.2, cabinRear - trunkLen / 2, d.width * 0.86, 0.06, trunkLen - 0.2));
  } else {
    parts.push(part('trunk', 0, 'paint', 0, d.bodyTop + 0.03, cabinRear - trunkLen / 2, d.width * 0.94, 0.08, trunkLen, 0.03));
  }

  const bumperY = d.clearance + 0.18;
  parts.push(part('bumper', 0, 'dark', 0, bumperY, halfL - 0.1, d.width * 0.98, 0.24, 0.26));
  parts.push(part('bumper', 1, 'dark', 0, bumperY, -halfL + 0.1, d.width * 0.98, 0.24, 0.26));

  const wr = d.wheelRadius;
  const wheelPositions: [number, number][] = [
    [-d.track / 2, d.wheelbase / 2],
    [d.track / 2, d.wheelbase / 2],
    [-d.track / 2, -d.wheelbase / 2],
    [d.track / 2, -d.wheelbase / 2],
  ];
  wheelPositions.forEach(([x, z], i) => {
    parts.push(part('wheel', i, 'rubber', x, wr, z, d.wheelWidth, wr * 2, wr * 2, 0, i));
  });

  const lightY = d.bodyTop - 0.14;
  const lightX = d.width / 2 - 0.36;
  parts.push(part('headlight', 0, 'headlight', -lightX, lightY, halfL - 0.03, 0.34, 0.14, 0.1));
  parts.push(part('headlight', 1, 'headlight', lightX, lightY, halfL - 0.03, 0.34, 0.14, 0.1));
  parts.push(part('taillight', 0, 'taillight', -lightX, lightY, -halfL + 0.03, 0.32, 0.12, 0.1));
  parts.push(part('taillight', 1, 'taillight', lightX, lightY, -halfL + 0.03, 0.32, 0.12, 0.1));

  if (isPolice) parts.push(part('lightbar', 0, 'lightbar', 0, d.height + 0.1, d.cabinOffset + 0.1, d.width * 0.62, 0.16, 0.32));
  if (a.id === 'taxi') parts.push(part('taxiSign', 0, 'taxi', 0, d.height + 0.14, d.cabinOffset + 0.1, 0.72, 0.26, 0.22));

  const colliderH = d.height * 0.9;
  return {
    archetype: a,
    parts,
    hoodPoint: new Vector3(0, d.bodyTop + 0.1, cabinFront + hoodLen / 2),
    colliderHalf: new Vector3(d.width / 2, colliderH / 2, d.length / 2),
    colliderCenterY: colliderH / 2,
  };
}

let geometries: Record<CarPartKind, BufferGeometry> | null = null;

/** Unit geometries shared by every car; scaled per instance. */
export function carPartGeometries(): Record<CarPartKind, BufferGeometry> {
  if (geometries) return geometries;
  const box = new BoxGeometry(1, 1, 1);
  // Wheel: unit cylinder whose axis is X (the axle) so scale = (width, diameter, diameter).
  const wheel = new CylinderGeometry(0.5, 0.5, 1, 14);
  wheel.rotateZ(Math.PI / 2);
  geometries = {
    body: box,
    roof: box,
    glass: box,
    hood: box,
    trunk: box,
    bumper: box,
    wheel,
    headlight: box,
    taillight: box,
    lightbar: box,
    taxiSign: box,
  };
  return geometries;
}

export interface CarMaterials {
  paint: MeshStandardMaterial;
  glass: MeshStandardMaterial;
  dark: MeshStandardMaterial;
  rubber: MeshStandardMaterial;
  /** Unlit so per-instance colour reads as emitted light day and night. */
  light: MeshBasicMaterial;
}

export function createCarMaterials(): CarMaterials {
  return {
    paint: new MeshStandardMaterial({ roughness: 0.32, metalness: 0.45 }),
    glass: new MeshStandardMaterial({ roughness: 0.12, metalness: 0.7 }),
    dark: new MeshStandardMaterial({ roughness: 0.8, metalness: 0.2 }),
    rubber: new MeshStandardMaterial({ roughness: 0.95, metalness: 0.0 }),
    light: new MeshBasicMaterial({ toneMapped: false }),
  };
}

export function materialForPart(kind: CarPartKind, m: CarMaterials): Material {
  switch (kind) {
    case 'glass':
      return m.glass;
    case 'bumper':
      return m.dark;
    case 'wheel':
      return m.rubber;
    case 'headlight':
    case 'taillight':
    case 'lightbar':
    case 'taxiSign':
      return m.light;
    default:
      return m.paint;
  }
}

/** Local door position for a seat: 0 driver (left front), 1 passenger, 2/3 rear. */
export function seatDoorLocal(a: VehicleArchetype, seat: number, outset: number, out: Vector3): Vector3 {
  const side = seat % 2 === 0 ? -1 : 1;
  const z = seat < 2 ? 0.3 : -1.0;
  return out.set(side * (a.dims.width / 2 + outset), 0, z);
}

export const CAR_UP = new Vector3(0, 1, 0);
export const CAR_RIGHT_AXIS = new Vector3(1, 0, 0);

/** Yaw-only quaternion for a heading (0 faces +Z). */
export function headingQuaternion(heading: number, out: Quaternion): Quaternion {
  return out.setFromAxisAngle(CAR_UP, heading);
}
