import { BoxGeometry, Color, DynamicDrawUsage, InstancedMesh, Matrix4, MeshStandardMaterial, type BufferGeometry, type Group } from 'three';
import type { Ped } from '../../entities/Ped';
import { computePedPartMatrices, PED_HAIR, PED_PART_COUNT, pedPartGeometries, type PedPalette } from '../../gfx/PedBuilder';
import { PedConfig } from './config';

const ZERO_MATRIX = new Matrix4().makeScale(0, 0, 0);

/**
 * Seven InstancedMeshes (head, hair, torso, armL, armR, legL, legR) shared by every
 * ped. Each ped owns one instance index; colors are written once at spawn, matrices
 * every frame from the ped's pose. Hidden peds (pooled, in vehicles) collapse to scale 0.
 */
export class PedRenderer {
  readonly capacity = PedConfig.sim.capacity;
  private readonly meshes: InstancedMesh[] = [];
  private readonly material: MeshStandardMaterial;
  private readonly hairGeometry: BufferGeometry;
  private readonly freeSlots: number[] = [];
  private readonly matrices: Matrix4[] = [];
  private readonly color = new Color();
  private readonly colorScratch = [0, 0, 0, 0, 0, 0, 0];

  constructor(private readonly root: Group) {
    const g = pedPartGeometries();
    this.hairGeometry = new BoxGeometry(PED_HAIR.w, PED_HAIR.h, PED_HAIR.d);
    this.material = new MeshStandardMaterial({ color: 0xffffff, roughness: 0.85, metalness: 0 });
    const parts: BufferGeometry[] = [g.head, this.hairGeometry, g.torso, g.arm, g.arm, g.leg, g.leg];
    for (let i = 0; i < PED_PART_COUNT; i++) {
      const mesh = new InstancedMesh(parts[i], this.material, this.capacity);
      mesh.instanceMatrix.setUsage(DynamicDrawUsage);
      mesh.frustumCulled = false;
      mesh.castShadow = true;
      mesh.receiveShadow = false;
      mesh.name = `peds-part-${i}`;
      for (let k = 0; k < this.capacity; k++) {
        mesh.setMatrixAt(k, ZERO_MATRIX);
        mesh.setColorAt(k, this.color.setHex(0xffffff));
      }
      mesh.count = this.capacity;
      this.meshes.push(mesh);
      root.add(mesh);
      this.matrices.push(new Matrix4());
    }
    for (let k = this.capacity - 1; k >= 0; k--) this.freeSlots.push(k);
  }

  /** Reserves an instance slot and writes the palette; returns -1 when full. */
  acquire(palette: PedPalette): number {
    const slot = this.freeSlots.pop();
    if (slot === undefined) return -1;
    this.setPalette(slot, palette);
    return slot;
  }

  release(slot: number): void {
    if (slot < 0) return;
    for (let i = 0; i < PED_PART_COUNT; i++) {
      this.meshes[i].setMatrixAt(slot, ZERO_MATRIX);
      this.meshes[i].instanceMatrix.needsUpdate = true;
    }
    this.freeSlots.push(slot);
  }

  setPalette(slot: number, palette: PedPalette): void {
    const c = this.colorScratch;
    c[0] = palette.skin;
    c[1] = palette.hair;
    c[2] = c[3] = c[4] = palette.shirt;
    c[5] = c[6] = palette.pants;
    for (let i = 0; i < PED_PART_COUNT; i++) {
      const mesh = this.meshes[i];
      mesh.setColorAt(slot, this.color.setHex(c[i]));
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    }
  }

  /** Writes the seven matrices for one ped (call once per frame per live ped). */
  write(ped: Ped): void {
    const slot = ped.instanceIndex;
    if (slot < 0) return;
    if (ped.vehicle || !ped.active) {
      for (let i = 0; i < PED_PART_COUNT; i++) this.meshes[i].setMatrixAt(slot, ZERO_MATRIX);
      return;
    }
    computePedPartMatrices(ped.pose, ped.heading, ped.position, this.matrices, ped.ragdolled ? ped.bodyQuat : undefined);
    for (let i = 0; i < PED_PART_COUNT; i++) this.meshes[i].setMatrixAt(slot, this.matrices[i]);
  }

  /** Flags instance buffers for upload after a batch of `write` calls. */
  commit(): void {
    for (let i = 0; i < PED_PART_COUNT; i++) this.meshes[i].instanceMatrix.needsUpdate = true;
  }

  dispose(): void {
    for (const m of this.meshes) {
      this.root.remove(m);
      m.dispose();
    }
    this.hairGeometry.dispose();
    this.material.dispose();
  }
}
