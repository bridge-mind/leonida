import {
  BoxGeometry,
  CanvasTexture,
  DynamicDrawUsage,
  InstancedMesh,
  Matrix4,
  MeshStandardMaterial,
  Quaternion,
  Vector3,
  type BufferGeometry,
  type Group,
} from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import type { PickupKind } from '../../core/peers';
import { PickupConfig } from './config';

export type PickupMeshKind = 'cash' | 'weapon' | 'armor' | 'health';

const ZERO_MATRIX = new Matrix4().makeScale(0, 0, 0);
const Y_AXIS = new Vector3(0, 1, 0);
const ONE = new Vector3(1, 1, 1);

/** Ammo boxes share the weapon mesh; everything else has its own. */
export function meshKindOf(kind: PickupKind): PickupMeshKind {
  return kind === 'ammo' ? 'weapon' : kind;
}

/**
 * Four InstancedMeshes (cash stack, gun-shaped box, armor box, health box with a
 * canvas red cross). Slots are handed out per kind; matrices are rewritten every frame.
 */
export class PickupRenderer {
  private readonly meshes: Record<PickupMeshKind, InstancedMesh>;
  private readonly free: Record<PickupMeshKind, number[]>;
  private readonly geometries: BufferGeometry[] = [];
  private readonly materials: MeshStandardMaterial[] = [];
  private readonly matrix = new Matrix4();
  private readonly quat = new Quaternion();
  private readonly pos = new Vector3();
  private readonly scale = new Vector3();
  private texture: CanvasTexture | null = null;

  constructor(private readonly root: Group) {
    const cap = PickupConfig.capacityPerKind;
    const make = (geometry: BufferGeometry, material: MeshStandardMaterial, name: string): InstancedMesh => {
      this.geometries.push(geometry);
      this.materials.push(material);
      const mesh = new InstancedMesh(geometry, material, cap);
      mesh.instanceMatrix.setUsage(DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.castShadow = true;
      mesh.name = name;
      for (let i = 0; i < cap; i++) mesh.setMatrixAt(i, ZERO_MATRIX);
      mesh.count = cap;
      root.add(mesh);
      return mesh;
    };
    this.meshes = {
      cash: make(cashGeometry(), new MeshStandardMaterial({ color: 0x3fbf5a, roughness: 0.7, emissive: 0x0f4f1f, emissiveIntensity: 0.6 }), 'pickups-cash'),
      weapon: make(weaponGeometry(), new MeshStandardMaterial({ color: 0x2a2a30, roughness: 0.45, metalness: 0.7 }), 'pickups-weapon'),
      armor: make(new BoxGeometry(0.42, 0.3, 0.2), new MeshStandardMaterial({ color: 0x2f6fff, roughness: 0.5, emissive: 0x10305f, emissiveIntensity: 0.5 }), 'pickups-armor'),
      health: make(new BoxGeometry(0.36, 0.28, 0.24), this.healthMaterial(), 'pickups-health'),
    };
    this.free = { cash: [], weapon: [], armor: [], health: [] };
    for (const k of Object.keys(this.free) as PickupMeshKind[]) for (let i = cap - 1; i >= 0; i--) this.free[k].push(i);
  }

  private healthMaterial(): MeshStandardMaterial {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#f4f4f4';
      ctx.fillRect(0, 0, 64, 64);
      ctx.fillStyle = '#e0202a';
      ctx.fillRect(26, 10, 12, 44);
      ctx.fillRect(10, 26, 44, 12);
    }
    this.texture = new CanvasTexture(canvas);
    return new MeshStandardMaterial({ map: this.texture, roughness: 0.6, emissive: 0x402020, emissiveIntensity: 0.4 });
  }

  acquire(kind: PickupMeshKind): number {
    const slot = this.free[kind].pop();
    return slot === undefined ? -1 : slot;
  }

  release(kind: PickupMeshKind, slot: number): void {
    if (slot < 0) return;
    this.meshes[kind].setMatrixAt(slot, ZERO_MATRIX);
    this.free[kind].push(slot);
  }

  write(kind: PickupMeshKind, slot: number, x: number, y: number, z: number, yaw: number, scale: number): void {
    if (slot < 0) return;
    this.pos.set(x, y, z);
    this.quat.setFromAxisAngle(Y_AXIS, yaw);
    this.scale.copy(ONE).multiplyScalar(scale);
    this.matrix.compose(this.pos, this.quat, this.scale);
    this.meshes[kind].setMatrixAt(slot, this.matrix);
  }

  commit(): void {
    for (const k of Object.keys(this.meshes) as PickupMeshKind[]) this.meshes[k].instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    for (const k of Object.keys(this.meshes) as PickupMeshKind[]) {
      this.root.remove(this.meshes[k]);
      this.meshes[k].dispose();
    }
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    this.texture?.dispose();
  }
}

/** Three slightly offset bill stacks. */
function cashGeometry(): BufferGeometry {
  const parts: BufferGeometry[] = [];
  for (let i = 0; i < 3; i++) {
    const g = new BoxGeometry(0.3, 0.06, 0.14);
    g.translate((i - 1) * 0.03, i * 0.065, (i % 2) * 0.03 - 0.015);
    g.rotateY((i - 1) * 0.18);
    parts.push(g);
  }
  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  return merged ?? new BoxGeometry(0.3, 0.18, 0.14);
}

/** Barrel + grip: reads as a gun silhouette while spinning. */
function weaponGeometry(): BufferGeometry {
  const barrel = new BoxGeometry(0.5, 0.1, 0.07);
  barrel.translate(0.05, 0.08, 0);
  const grip = new BoxGeometry(0.1, 0.22, 0.07);
  grip.translate(-0.12, -0.06, 0);
  grip.rotateZ(0.2);
  const merged = mergeGeometries([barrel, grip], false);
  barrel.dispose();
  grip.dispose();
  return merged ?? new BoxGeometry(0.5, 0.2, 0.07);
}
