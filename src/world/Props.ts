import {
  BoxGeometry,
  Color,
  CylinderGeometry,
  DynamicDrawUsage,
  Group,
  InstancedMesh,
  Matrix4,
  Mesh,
  Quaternion,
  Vector3,
  type BufferGeometry,
  type Material,
} from 'three';
import type { Random } from '../core/Random';
import { GeometryBatch } from '../gfx/GeometryBatch';
import type { CityMaterials } from '../gfx/Materials';
import { Groups, type Physics } from '../physics/Physics';
import type { StaticEntities, StaticEntity } from './city/StaticEntities';

export type PropKind =
  | 'lampPost'
  | 'lampGlow'
  | 'trafficPole'
  | 'signalLamp'
  | 'palm'
  | 'bench'
  | 'hydrant'
  | 'dumpster'
  | 'bollard'
  | 'reeds'
  | 'container'
  | 'umbrella'
  | 'cypress'
  | 'pump'
  | 'lifeguard';

interface PropDef {
  geometry: BufferGeometry;
  material: Material;
  /** Collider half extents and center height; undefined = no collision. */
  collider?: { hx: number; hy: number; hz: number; cy: number };
  entity: StaticEntity;
  castShadow: boolean;
}

interface Instance {
  x: number;
  y: number;
  z: number;
  rotY: number;
  scale: number;
  color: number;
}

const UP = new Vector3(0, 1, 0);

/**
 * All small repeated street furniture as InstancedMesh per kind (one draw call each).
 * Traffic signal lamps expose `setSignal(nodeId, nsGreen)` for the traffic system.
 */
export class Props {
  private readonly defs = new Map<PropKind, PropDef>();
  private readonly instances = new Map<PropKind, Instance[]>();
  private readonly meshes = new Map<PropKind, InstancedMesh>();
  /** nodeId -> signal lamp instance indices with their axis. */
  private readonly signals = new Map<number, { index: number; axis: 'ns' | 'ew' }[]>();
  private readonly tmpMatrix = new Matrix4();
  private readonly tmpQuat = new Quaternion();
  private readonly tmpPos = new Vector3();
  private readonly tmpScale = new Vector3();
  private readonly tmpColor = new Color();

  constructor(
    private readonly materials: CityMaterials,
    private readonly physics: Physics,
    private readonly entities: StaticEntities,
    rng: Random,
  ) {
    this.defineAll(rng);
  }

  private define(kind: PropKind, def: PropDef): void {
    this.defs.set(kind, def);
    this.instances.set(kind, []);
  }

  private defineAll(rng: Random): void {
    const flat = this.materials.flat;
    const E = this.entities;

    // Lamp post: pole + arm + head; the glowing head is a separate kind.
    const lamp = new GeometryBatch();
    lamp.addGeometry(new CylinderGeometry(0.08, 0.12, 7, 6), new Matrix4().makeTranslation(0, 3.5, 0), 0x3a3f46);
    lamp.addBox(0.9, 6.9, 0, 1.8, 0.1, 0.1, 0x3a3f46);
    lamp.addBox(1.7, 6.8, 0, 0.7, 0.18, 0.35, 0x2a2e33);
    this.define('lampPost', { geometry: lamp.build(), material: flat, collider: { hx: 0.14, hy: 3.5, hz: 0.14, cy: 3.5 }, entity: E.metalProp, castShadow: true });
    this.define('lampGlow', { geometry: new BoxGeometry(0.6, 0.08, 0.3), material: this.materials.lampGlow, entity: E.metalProp, castShadow: false });

    // Traffic light: pole, mast arm and a housing; the lit lamp is instanced separately.
    const tl = new GeometryBatch();
    tl.addGeometry(new CylinderGeometry(0.09, 0.11, 5.2, 6), new Matrix4().makeTranslation(0, 2.6, 0), 0x2f3338);
    tl.addBox(-2.2, 5.1, 0, 4.6, 0.12, 0.12, 0x2f3338);
    tl.addBox(-4.2, 4.5, 0, 0.36, 1.05, 0.36, 0x1d1f22);
    tl.addBox(-4.2, 4.85, 0.19, 0.2, 0.2, 0.02, 0x3a1010);
    tl.addBox(-4.2, 4.15, 0.19, 0.2, 0.2, 0.02, 0x0f2a12);
    this.define('trafficPole', { geometry: tl.build(), material: flat, collider: { hx: 0.12, hy: 2.6, hz: 0.12, cy: 2.6 }, entity: E.metalProp, castShadow: true });
    this.define('signalLamp', { geometry: new BoxGeometry(0.22, 0.22, 0.06), material: this.materials.signal, entity: E.metalProp, castShadow: false });

    // Palm: tapered trunk with a slight lean and six fronds.
    const palm = new GeometryBatch();
    palm.addGeometry(new CylinderGeometry(0.16, 0.3, 7.5, 6), new Matrix4().makeTranslation(0, 3.75, 0), 0x8a6a45);
    const frond = new BoxGeometry(3.0, 0.08, 0.7);
    for (let i = 0; i < 7; i++) {
      const a = (i / 7) * Math.PI * 2 + rng.range(-0.2, 0.2);
      const m = new Matrix4().compose(
        new Vector3(Math.cos(a) * 1.3, 7.4 - 0.25 * (i % 2), Math.sin(a) * 1.3),
        new Quaternion().setFromAxisAngle(UP, -a).multiply(new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), -0.45 - 0.15 * (i % 3))),
        new Vector3(1, 1, 1),
      );
      palm.addGeometry(frond, m, i % 2 ? 0x3f8f3a : 0x4fa845);
    }
    palm.addGeometry(new CylinderGeometry(0.25, 0.25, 0.5, 6), new Matrix4().makeTranslation(0, 7.35, 0), 0x5a4a2a);
    this.define('palm', { geometry: palm.build(), material: flat, collider: { hx: 0.28, hy: 3.6, hz: 0.28, cy: 3.6 }, entity: E.wood, castShadow: true });

    const bench = new GeometryBatch();
    bench.addBox(0, 0.45, 0, 1.8, 0.08, 0.5, 0x7a5a3a);
    bench.addBox(0, 0.8, -0.24, 1.8, 0.5, 0.06, 0x7a5a3a);
    bench.addBox(-0.75, 0.22, 0, 0.08, 0.44, 0.5, 0x2c2c2c);
    bench.addBox(0.75, 0.22, 0, 0.08, 0.44, 0.5, 0x2c2c2c);
    this.define('bench', { geometry: bench.build(), material: flat, collider: { hx: 0.9, hy: 0.45, hz: 0.3, cy: 0.45 }, entity: E.wood, castShadow: true });

    const hydrant = new GeometryBatch();
    hydrant.addGeometry(new CylinderGeometry(0.14, 0.16, 0.7, 8), new Matrix4().makeTranslation(0, 0.35, 0), 0xc93a2e);
    hydrant.addBox(0, 0.5, 0, 0.5, 0.12, 0.14, 0xc93a2e);
    hydrant.addGeometry(new CylinderGeometry(0.1, 0.14, 0.15, 8), new Matrix4().makeTranslation(0, 0.77, 0), 0xa02a22);
    this.define('hydrant', { geometry: hydrant.build(), material: flat, collider: { hx: 0.18, hy: 0.4, hz: 0.18, cy: 0.4 }, entity: E.metalProp, castShadow: true });

    const dumpster = new GeometryBatch();
    dumpster.addBox(0, 0.62, 0, 1.9, 1.1, 1.2, 0x2f5e3a);
    dumpster.addBox(0, 1.2, 0, 1.94, 0.08, 1.24, 0x244a2c);
    dumpster.addBox(-0.8, 0.08, 0, 0.14, 0.16, 1.1, 0x1a1a1a);
    dumpster.addBox(0.8, 0.08, 0, 0.14, 0.16, 1.1, 0x1a1a1a);
    this.define('dumpster', { geometry: dumpster.build(), material: flat, collider: { hx: 0.97, hy: 0.62, hz: 0.62, cy: 0.62 }, entity: E.metalProp, castShadow: true });

    this.define('bollard', {
      geometry: new CylinderGeometry(0.12, 0.14, 0.9, 8).translate(0, 0.45, 0),
      material: flat,
      collider: { hx: 0.14, hy: 0.45, hz: 0.14, cy: 0.45 },
      entity: E.metalProp,
      castShadow: true,
    });

    const reeds = new GeometryBatch();
    for (let i = 0; i < 9; i++) {
      const a = rng.range(0, Math.PI * 2);
      const r = rng.range(0, 0.6);
      const h = rng.range(1.2, 2.2);
      reeds.addGeometry(
        new CylinderGeometry(0.01, 0.05, h, 4),
        new Matrix4().compose(new Vector3(Math.cos(a) * r, h / 2, Math.sin(a) * r), new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), rng.range(-0.15, 0.15)), new Vector3(1, 1, 1)),
        i % 3 === 0 ? 0x8a8a3a : 0x5f7a2f,
      );
    }
    this.define('reeds', { geometry: reeds.build(), material: flat, entity: E.dirt, castShadow: false });

    const container = new GeometryBatch();
    container.addBox(0, 1.3, 0, 12.2, 2.6, 2.44, 0xffffff);
    container.addBox(0, 1.3, 0, 12.3, 2.3, 2.5, 0xe0e0e0);
    this.define('container', { geometry: container.build(), material: flat, collider: { hx: 6.1, hy: 1.3, hz: 1.22, cy: 1.3 }, entity: E.metalProp, castShadow: true });

    const umbrella = new GeometryBatch();
    umbrella.addGeometry(new CylinderGeometry(0.03, 0.03, 2.2, 5), new Matrix4().makeTranslation(0, 1.1, 0), 0xdddddd);
    umbrella.addGeometry(new CylinderGeometry(0.05, 1.4, 0.5, 8), new Matrix4().makeTranslation(0, 2.1, 0), 0xffffff);
    this.define('umbrella', { geometry: umbrella.build(), material: flat, entity: E.wood, castShadow: true });

    const cypress = new GeometryBatch();
    cypress.addGeometry(new CylinderGeometry(0.25, 0.55, 9, 6), new Matrix4().makeTranslation(0, 4.5, 0), 0x5a4633);
    cypress.addGeometry(new CylinderGeometry(0.2, 2.6, 6, 7), new Matrix4().makeTranslation(0, 9.5, 0), 0x3e5f2e);
    cypress.addGeometry(new CylinderGeometry(0.1, 1.6, 4, 7), new Matrix4().makeTranslation(0, 13, 0), 0x4a6f34);
    this.define('cypress', { geometry: cypress.build(), material: flat, collider: { hx: 0.5, hy: 4.5, hz: 0.5, cy: 4.5 }, entity: E.wood, castShadow: true });

    const pump = new GeometryBatch();
    pump.addBox(0, 0.9, 0, 0.9, 1.8, 0.5, 0xe8e8e8);
    pump.addBox(0, 1.35, 0.26, 0.7, 0.5, 0.04, 0x1a1a2a);
    pump.addBox(0, 0.1, 0, 1.2, 0.2, 0.8, 0x777777);
    this.define('pump', { geometry: pump.build(), material: flat, collider: { hx: 0.5, hy: 0.9, hz: 0.3, cy: 0.9 }, entity: E.metalProp, castShadow: true });

    const lifeguard = new GeometryBatch();
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) lifeguard.addBox(sx * 1.1, 1.2, sz * 1.1, 0.16, 2.4, 0.16, 0xd8c8a8);
    lifeguard.addBox(0, 2.5, 0, 2.8, 0.15, 2.8, 0xd8c8a8);
    lifeguard.addBox(0, 3.5, 0, 2.6, 1.9, 2.6, 0xf7a8c4);
    lifeguard.addGableRoof(0, 4.45, 0, 3.0, 3.0, 0.8, 0x35e6ff);
    this.define('lifeguard', { geometry: lifeguard.build(), material: flat, collider: { hx: 1.3, hy: 2.2, hz: 1.3, cy: 2.2 }, entity: E.wood, castShadow: true });
  }

  add(kind: PropKind, x: number, y: number, z: number, rotY = 0, scale = 1, color = 0xffffff): number {
    const list = this.instances.get(kind);
    if (!list) throw new Error(`unknown prop kind ${kind}`);
    list.push({ x, y, z, rotY, scale, color });
    const def = this.defs.get(kind) as PropDef;
    if (def.collider) {
      const c = def.collider;
      this.physics.addStaticCuboid({ x, y: y + c.cy * scale, z }, { x: c.hx * scale, y: c.hy * scale, z: c.hz * scale }, { entity: def.entity, groups: Groups.STATIC, rotationY: rotY });
    }
    return list.length - 1;
  }

  /** Lamp post whose arm points along `rotY` (toward the road); adds the glow head too. */
  addLamp(x: number, y: number, z: number, rotY: number): void {
    this.add('lampPost', x, y, z, rotY);
    this.add('lampGlow', x + Math.cos(rotY) * 1.7, y + 6.72, z - Math.sin(rotY) * 1.7, rotY);
  }

  /** Traffic light for one approach at a node; the mast arm extends along `rotY` toward the road. */
  addTrafficLight(nodeId: number, axis: 'ns' | 'ew', x: number, y: number, z: number, rotY: number, green: boolean): void {
    this.add('trafficPole', x, y, z, rotY);
    const lampIndex = this.add('signalLamp', x + Math.cos(rotY) * 4.2, y + 4.5, z - Math.sin(rotY) * 4.2, rotY, 1, green ? 0x2eff4a : 0xff2a2a);
    let list = this.signals.get(nodeId);
    if (!list) {
      list = [];
      this.signals.set(nodeId, list);
    }
    list.push({ index: lampIndex, axis });
  }

  /** Sets the signal colors at an intersection: NS approaches green when `nsGreen`, or all red when null. */
  setSignal(nodeId: number, nsGreen: boolean | null, yellowAxis: 'ns' | 'ew' | null = null): void {
    const list = this.signals.get(nodeId);
    const mesh = this.meshes.get('signalLamp');
    if (!list || !mesh) return;
    for (const s of list) {
      let color = 0xff2a2a;
      if (yellowAxis === s.axis) color = 0xffc23a;
      else if (nsGreen !== null && (s.axis === 'ns') === nsGreen) color = 0x2eff4a;
      mesh.setColorAt(s.index, this.tmpColor.set(color));
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  get signalNodeIds(): number[] {
    return [...this.signals.keys()];
  }

  count(kind: PropKind): number {
    return this.instances.get(kind)?.length ?? 0;
  }

  build(parent: Group): Mesh[] {
    const out: Mesh[] = [];
    for (const [kind, def] of this.defs) {
      const list = this.instances.get(kind) as Instance[];
      if (list.length === 0) continue;
      const mesh = new InstancedMesh(def.geometry, def.material, list.length);
      mesh.name = `props:${kind}`;
      mesh.castShadow = def.castShadow;
      mesh.receiveShadow = def.castShadow;
      list.forEach((inst, i) => {
        this.tmpQuat.setFromAxisAngle(UP, inst.rotY);
        this.tmpPos.set(inst.x, inst.y, inst.z);
        this.tmpScale.setScalar(inst.scale);
        this.tmpMatrix.compose(this.tmpPos, this.tmpQuat, this.tmpScale);
        mesh.setMatrixAt(i, this.tmpMatrix);
        mesh.setColorAt(i, this.tmpColor.set(inst.color));
      });
      mesh.instanceMatrix.needsUpdate = true;
      if (mesh.instanceColor) {
        mesh.instanceColor.needsUpdate = true;
        if (kind === 'signalLamp') mesh.instanceColor.setUsage(DynamicDrawUsage);
      }
      mesh.computeBoundingSphere();
      parent.add(mesh);
      this.meshes.set(kind, mesh);
      out.push(mesh);
    }
    return out;
  }

  dispose(): void {
    for (const mesh of this.meshes.values()) {
      mesh.parent?.remove(mesh);
      mesh.dispose();
    }
    for (const def of this.defs.values()) def.geometry.dispose();
  }
}
