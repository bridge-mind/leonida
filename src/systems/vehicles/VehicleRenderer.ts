import { Color, DynamicDrawUsage, InstancedMesh, Matrix4, Quaternion, Vector3, type Group } from 'three';
import type { Vehicle } from '../../entities/Vehicle';
import {
  CAR_PART_INSTANCES,
  CAR_PART_KINDS,
  CAR_RIGHT_AXIS,
  CAR_UP,
  carPartGeometries,
  createCarMaterials,
  materialForPart,
  type CarMaterials,
  type CarPart,
  type CarPartKind,
} from '../../gfx/CarBuilder';
import { VehicleConfig } from './config';
import { damageTier } from './VehicleFx';

const ZERO_MATRIX = new Matrix4().makeScale(0, 0, 0);
const GLASS = new Color(0x1c2733);
const GLASS_WRECK = new Color(0x0d0f12);
const DARK = new Color(0x2a2a2e);
const DARK_WRECK = new Color(0x0c0c0c);
const RUBBER = new Color(0x141414);
const HEAD_OFF = new Color(0.5, 0.5, 0.45);
const HEAD_ON = new Color(1.0, 0.96, 0.8);
const HEAD_WRECK = new Color(0.12, 0.12, 0.1);
const TAIL_OFF = new Color(0.32, 0.05, 0.05);
const TAIL_ON = new Color(0.72, 0.08, 0.08);
const TAIL_BRAKE = new Color(1.0, 0.14, 0.12);
const TAIL_WRECK = new Color(0.08, 0.02, 0.02);
const BAR_OFF = new Color(0x1a1a28);
const BAR_RED = new Color(1.0, 0.1, 0.1);
const BAR_BLUE = new Color(0.2, 0.35, 1.0);
const TAXI_SIGN = new Color(1.0, 0.82, 0.25);
const TAXI_SIGN_DAY = new Color(0.75, 0.62, 0.2);

/**
 * One InstancedMesh per car part for every vehicle in the world (11 draw calls).
 * Each vehicle owns one slot; parts of kind K use instances slot·n(K) .. slot·n(K)+n(K)-1.
 */
export class VehicleRenderer {
  private readonly meshes = new Map<CarPartKind, InstancedMesh>();
  private readonly freeSlots: number[] = [];
  private readonly materials: CarMaterials;
  private readonly mat = new Matrix4();
  private readonly pos = new Vector3();
  private readonly scale = new Vector3();
  private readonly quat = new Quaternion();
  private readonly steerQuat = new Quaternion();
  private readonly spinQuat = new Quaternion();
  private readonly color = new Color();
  private readonly base = new Color();
  /** Slots whose matrices are currently non-zero (so hiding writes once). */
  private readonly shown: boolean[];

  constructor(parent: Group, readonly capacity: number) {
    this.materials = createCarMaterials();
    const geometries = carPartGeometries();
    for (const kind of CAR_PART_KINDS) {
      const count = capacity * CAR_PART_INSTANCES[kind];
      const mesh = new InstancedMesh(geometries[kind], materialForPart(kind, this.materials), count);
      mesh.name = `vehicles:${kind}`;
      mesh.frustumCulled = false;
      mesh.castShadow = kind === 'body' || kind === 'glass' || kind === 'roof';
      mesh.receiveShadow = false;
      mesh.instanceMatrix.setUsage(DynamicDrawUsage);
      for (let i = 0; i < count; i++) {
        mesh.setMatrixAt(i, ZERO_MATRIX);
        mesh.setColorAt(i, DARK);
      }
      if (mesh.instanceColor) mesh.instanceColor.setUsage(DynamicDrawUsage);
      parent.add(mesh);
      this.meshes.set(kind, mesh);
    }
    this.shown = new Array<boolean>(capacity).fill(false);
    for (let i = capacity - 1; i >= 0; i--) this.freeSlots.push(i);
  }

  get freeCount(): number {
    return this.freeSlots.length;
  }

  alloc(): number {
    return this.freeSlots.length ? (this.freeSlots.pop() as number) : -1;
  }

  free(slot: number): void {
    if (slot < 0) return;
    this.hide(slot);
    this.freeSlots.push(slot);
  }

  hide(slot: number): void {
    if (!this.shown[slot]) return;
    this.shown[slot] = false;
    for (const kind of CAR_PART_KINDS) {
      const mesh = this.meshes.get(kind) as InstancedMesh;
      const n = CAR_PART_INSTANCES[kind];
      for (let k = 0; k < n; k++) mesh.setMatrixAt(slot * n + k, ZERO_MATRIX);
    }
  }

  /** Writes matrices and colours for one visible vehicle. */
  write(v: Vehicle, night01: number, timeSec: number): void {
    const slot = v.slot;
    if (slot < 0) return;
    this.shown[slot] = true;
    const layout = v.layout;
    const parts = layout.parts;
    const tier = damageTier(v);
    this.base.set(v.color).multiplyScalar(VehicleConfig.damage.tierDarkening[tier]);
    const wrecked = v.state.wrecked;
    const lightsOn = v.state.lightsOn && !wrecked;
    const braking = !wrecked && v.hasController && v.controls.brake > 0.1 && v.state.forwardSpeed > 0.3;
    const sirenRed = v.state.sirenOn && (timeSec % VehicleConfig.lights.sirenFlashPeriod) * 2 < VehicleConfig.lights.sirenFlashPeriod;

    // Parts absent from this archetype (light bar, taxi sign) collapse to zero scale.
    let presentMask = 0;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      this.writePart(v, slot, p, night01, lightsOn, braking, sirenRed, wrecked);
      if (p.kind === 'lightbar') presentMask |= 1;
      if (p.kind === 'taxiSign') presentMask |= 2;
    }
    if (!(presentMask & 1)) (this.meshes.get('lightbar') as InstancedMesh).setMatrixAt(slot, ZERO_MATRIX);
    if (!(presentMask & 2)) (this.meshes.get('taxiSign') as InstancedMesh).setMatrixAt(slot, ZERO_MATRIX);
  }

  private writePart(v: Vehicle, slot: number, p: CarPart, night01: number, lightsOn: boolean, braking: boolean, sirenRed: boolean, wrecked: boolean): void {
    const mesh = this.meshes.get(p.kind) as InstancedMesh;
    const index = slot * CAR_PART_INSTANCES[p.kind] + p.index;
    this.pos.copy(p.offset).applyQuaternion(v.quaternion).add(v.position);
    this.quat.copy(v.quaternion);
    if (p.wheel >= 0) {
      if (p.wheel < 2) this.quat.multiply(this.steerQuat.setFromAxisAngle(CAR_UP, -v.steerVisual));
      this.quat.multiply(this.spinQuat.setFromAxisAngle(CAR_RIGHT_AXIS, v.wheelSpin));
    } else if (p.rotX !== 0) {
      this.quat.multiply(this.steerQuat.setFromAxisAngle(CAR_RIGHT_AXIS, p.rotX));
    }
    this.scale.copy(p.size);
    this.mat.compose(this.pos, this.quat, this.scale);
    mesh.setMatrixAt(index, this.mat);

    const c = this.color;
    switch (p.role) {
      case 'paint':
        c.copy(this.base);
        break;
      case 'glass':
        c.copy(wrecked ? GLASS_WRECK : GLASS);
        break;
      case 'dark':
        c.copy(wrecked ? DARK_WRECK : DARK);
        break;
      case 'rubber':
        c.copy(RUBBER);
        break;
      case 'headlight':
        c.copy(wrecked ? HEAD_WRECK : lightsOn ? HEAD_ON : HEAD_OFF);
        break;
      case 'taillight':
        c.copy(wrecked ? TAIL_WRECK : braking ? TAIL_BRAKE : lightsOn ? TAIL_ON : TAIL_OFF);
        break;
      case 'lightbar':
        c.copy(wrecked ? DARK_WRECK : v.state.sirenOn ? (sirenRed ? BAR_RED : BAR_BLUE) : BAR_OFF);
        break;
      case 'taxi':
        c.copy(wrecked ? DARK_WRECK : night01 > 0.5 ? TAXI_SIGN : TAXI_SIGN_DAY);
        break;
    }
    mesh.setColorAt(index, c);
  }

  /** Upload the changed buffers once per frame. */
  commit(): void {
    for (const mesh of this.meshes.values()) {
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  dispose(parent: Group): void {
    for (const mesh of this.meshes.values()) {
      parent.remove(mesh);
      mesh.dispose();
    }
    for (const m of Object.values(this.materials)) m.dispose();
    this.meshes.clear();
  }
}
