import { Group, Mesh, type Material } from 'three';
import { Config } from '../../core/Config';
import { GeometryBatch } from '../../gfx/GeometryBatch';
import type { CityMaterials } from '../../gfx/Materials';

export type BatchLayer = 'walls' | 'deco' | 'glass' | 'flat' | 'concrete' | 'neon' | 'planks';

export interface ChunkBatch {
  cx: number;
  cz: number;
  layers: Record<BatchLayer, GeometryBatch>;
}

const LAYERS: BatchLayer[] = ['walls', 'deco', 'glass', 'flat', 'concrete', 'neon', 'planks'];

/**
 * Routes static geometry into per-200 m chunk batches keyed by material layer, then
 * builds one Mesh per non-empty (chunk, layer) so frustum culling works per chunk and
 * the whole static city stays well under the draw-call budget.
 */
export class ChunkBatches {
  private readonly chunks = new Map<number, ChunkBatch>();
  readonly chunkSize = Config.world.chunkSize;

  private key(cx: number, cz: number): number {
    return (cx + 512) * 1024 + (cz + 512);
  }

  /** Batch for the chunk containing world position (x, z). */
  at(x: number, z: number): ChunkBatch {
    const cx = Math.floor(x / this.chunkSize);
    const cz = Math.floor(z / this.chunkSize);
    const k = this.key(cx, cz);
    let chunk = this.chunks.get(k);
    if (!chunk) {
      const layers = {} as Record<BatchLayer, GeometryBatch>;
      for (const l of LAYERS) layers[l] = new GeometryBatch();
      chunk = { cx, cz, layers };
      this.chunks.set(k, chunk);
    }
    return chunk;
  }

  layer(x: number, z: number, layer: BatchLayer): GeometryBatch {
    return this.at(x, z).layers[layer];
  }

  /** Builds all meshes into `parent`; returns the draw-call count. */
  build(parent: Group, materials: CityMaterials, castShadow = true): { meshes: Mesh[]; triangles: number } {
    const meshes: Mesh[] = [];
    let triangles = 0;
    const materialFor: Record<BatchLayer, Material> = {
      walls: materials.walls,
      deco: materials.deco,
      glass: materials.glass,
      flat: materials.flat,
      concrete: materials.concrete,
      neon: materials.neon,
      planks: materials.planks,
    };
    for (const chunk of this.chunks.values()) {
      for (const layer of LAYERS) {
        const batch = chunk.layers[layer];
        if (batch.isEmpty) continue;
        const geom = batch.build();
        const mesh = new Mesh(geom, materialFor[layer]);
        mesh.name = `chunk:${chunk.cx},${chunk.cz}:${layer}`;
        mesh.castShadow = castShadow && layer !== 'neon' && layer !== 'flat';
        mesh.receiveShadow = layer !== 'neon';
        mesh.matrixAutoUpdate = false;
        mesh.updateMatrix();
        parent.add(mesh);
        meshes.push(mesh);
        const idx = geom.getIndex();
        triangles += idx ? idx.count / 3 : 0;
        batch.clear();
      }
    }
    return { meshes, triangles };
  }

  get chunkCount(): number {
    return this.chunks.size;
  }
}
