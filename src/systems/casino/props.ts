import {
  BoxGeometry,
  CylinderGeometry,
  DoubleSide,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  SphereGeometry,
  type BufferGeometry,
  type Material,
  type Object3D,
  type Texture,
} from 'three';
import type { InteriorBuildContext } from '../../world/interiors/Interiors';
import { Groups } from '../../physics/Physics';
import { SlotScreen, limitSignTexture, signTexture } from './textures';

/**
 * Primitive-built furniture for the casino. Every builder adds meshes to a parent
 * group in local coordinates and registers world-space colliders through the context.
 */
export class PropKit {
  private readonly geometries: BufferGeometry[] = [];
  private readonly materials: Material[] = [];
  private readonly textures: Texture[] = [];
  readonly mat = {
    darkWood: this.std(0x3b2414, 0.7),
    gold: this.std(0xc9a13a, 0.35, 0.8),
    plum: this.std(0x2a1430, 0.9),
    black: this.std(0x141418, 0.6),
    chrome: this.std(0x9aa3ad, 0.3, 0.9),
    redLeather: this.std(0x7a1524, 0.6),
    cream: this.std(0xe8dcc0, 0.8),
    skin: this.std(0xc98a5f, 0.8),
    vest: this.std(0x1a1a22, 0.8),
    shirt: this.std(0xf2f2f2, 0.8),
    marble: this.std(0xd9d2c5, 0.35),
  };

  constructor(private readonly ctx: InteriorBuildContext) {}

  std(color: number, roughness: number, metalness = 0, map?: Texture): MeshStandardMaterial {
    const m = new MeshStandardMaterial({ color, roughness, metalness, map });
    this.materials.push(m);
    if (map) this.textures.push(map);
    return m;
  }

  emissive(color: number, intensity = 1, map?: Texture): MeshStandardMaterial {
    const m = new MeshStandardMaterial({ color: 0x000000, emissive: color, emissiveIntensity: intensity, emissiveMap: map, roughness: 1 });
    this.materials.push(m);
    if (map) this.textures.push(map);
    return m;
  }

  unlit(map: Texture): MeshBasicMaterial {
    const m = new MeshBasicMaterial({ map, toneMapped: false });
    this.materials.push(m);
    this.textures.push(map);
    return m;
  }

  geo<T extends BufferGeometry>(g: T): T {
    this.geometries.push(g);
    return g;
  }

  /** Axis-aligned box mesh centred at (x, y, z); optionally a matching static collider. */
  box(parent: Object3D, w: number, h: number, d: number, material: Material, x: number, y: number, z: number, collide = false): Mesh {
    const mesh = new Mesh(this.geo(new BoxGeometry(w, h, d)), material);
    mesh.position.set(x, y, z);
    parent.add(mesh);
    if (collide) this.collider(parent, x, y, z, w / 2, h / 2, d / 2);
    return mesh;
  }

  /**
   * Static cuboid collider from a centre expressed in `parent`'s frame. Walks the
   * translation chain up to the interior root; furniture groups are never rotated.
   */
  collider(parent: Object3D, x: number, y: number, z: number, hx: number, hy: number, hz: number): void {
    let px = x;
    let py = y;
    let pz = z;
    let node: Object3D | null = parent;
    while (node && node !== this.ctx.root) {
      px += node.position.x;
      py += node.position.y;
      pz += node.position.z;
      node = node.parent;
    }
    this.ctx.physics.addStaticCuboid(this.ctx.toWorld({ x: px, y: py, z: pz }), { x: hx, y: hy, z: hz }, { groups: Groups.STATIC });
  }

  cylinder(parent: Object3D, rTop: number, rBottom: number, h: number, material: Material, x: number, y: number, z: number, segments = 16): Mesh {
    const mesh = new Mesh(this.geo(new CylinderGeometry(rTop, rBottom, h, segments)), material);
    mesh.position.set(x, y, z);
    parent.add(mesh);
    return mesh;
  }

  sign(parent: Object3D, map: Texture, w: number, h: number, x: number, y: number, z: number, rotY = 0): Mesh {
    const mesh = new Mesh(this.geo(new PlaneGeometry(w, h)), this.unlit(map));
    mesh.position.set(x, y, z);
    mesh.rotation.y = rotY;
    parent.add(mesh);
    return mesh;
  }

  dispose(): void {
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials) m.dispose();
    for (const t of this.textures) t.dispose();
  }

  // --- Furniture -------------------------------------------------------------

  /** Counter with brass bars and a CASHIER sign; the opening faces +X (into the room). */
  cashierCage(parent: Object3D): void {
    const m = this.mat;
    this.box(parent, 1, 1.1, 5, m.darkWood, 0, 0.55, 0, true);
    this.box(parent, 1.1, 0.08, 5.2, m.marble, 0, 1.14, 0);
    for (let i = 0; i < 11; i++) this.cylinder(parent, 0.03, 0.03, 1.6, m.gold, 0.45, 1.98, -2.4 + i * 0.48, 8);
    this.box(parent, 1.1, 0.12, 5.2, m.gold, 0, 2.84, 0);
    this.box(parent, 0.4, 2.2, 5.4, m.plum, -1.4, 1.1, 0, true);
    this.box(parent, 0.3, 0.05, 4.6, m.marble, -1.3, 1.3, 0);
    this.box(parent, 0.3, 0.05, 4.6, m.marble, -1.3, 1.9, 0);
    this.sign(parent, signTexture('CASHIER', 'BUY - CASH OUT'), 3.2, 1.2, 0.6, 3.6, 0, Math.PI / 2);
  }

  /** Semi-circular-ish blackjack table: felt top, padded rail, legs, three stools, dealer dummy and a limit sign. */
  blackjackTable(parent: Object3D, felt: Texture, limits: { min: number; max: number }): void {
    const m = this.mat;
    const top = this.box(parent, 3.2, 0.1, 1.9, this.std(0xffffff, 0.95, 0, felt), 0, 0.92, 0, true);
    top.position.y = 0.92;
    this.box(parent, 3.4, 0.14, 0.18, m.redLeather, 0, 0.98, 1.0);
    this.box(parent, 0.18, 0.14, 2.1, m.redLeather, -1.7, 0.98, 0.05);
    this.box(parent, 0.18, 0.14, 2.1, m.redLeather, 1.7, 0.98, 0.05);
    this.box(parent, 2.8, 0.85, 1.4, m.darkWood, 0, 0.45, 0);
    this.box(parent, 0.5, 0.12, 0.3, m.chrome, 0, 0.99, -0.75);
    for (let i = -1; i <= 1; i++) {
      this.cylinder(parent, 0.22, 0.22, 0.06, m.redLeather, i * 1.0, 0.72, 1.55, 12);
      this.cylinder(parent, 0.04, 0.04, 0.7, m.chrome, i * 1.0, 0.35, 1.55, 8);
      this.collider(parent, i * 1.0, 0.4, 1.55, 0.22, 0.4, 0.22);
    }
    this.dealer(parent, 0, -1.2, 0);
    this.cylinder(parent, 0.03, 0.03, 2.2, m.gold, -1.9, 1.1, -1.0, 8);
    this.sign(parent, limitSignTexture(limits.min, limits.max), 1.4, 0.52, -1.9, 2.4, -1.0);
  }

  /** Long roulette table with the layout felt and the wheel at the -X end. Returns the spinning wheel mesh and its ball. */
  rouletteTable(parent: Object3D, layout: Texture, wheelFace: Texture): { wheel: Mesh; ball: Mesh } {
    const m = this.mat;
    this.box(parent, 5.2, 0.1, 2.2, this.std(0xffffff, 0.95, 0, layout), 0.6, 0.92, 0, true);
    this.box(parent, 4.6, 0.85, 1.7, m.darkWood, 0.6, 0.45, 0);
    this.box(parent, 5.4, 0.14, 0.18, m.redLeather, 0.6, 0.98, 1.15);
    this.box(parent, 5.4, 0.14, 0.18, m.redLeather, 0.6, 0.98, -1.15);
    this.box(parent, 0.18, 0.14, 2.5, m.redLeather, 3.3, 0.98, 0);
    this.cylinder(parent, 1.15, 1.15, 0.3, m.darkWood, -2.7, 0.85, 0, 32);
    this.collider(parent, -2.7, 0.5, 0, 1.15, 0.5, 1.15);
    const wheelGeo = this.geo(new CylinderGeometry(1.0, 1.0, 0.12, 48));
    const rim = this.std(0x5b3a1e, 0.6);
    const wheel = new Mesh(wheelGeo, [rim, this.std(0xffffff, 0.6, 0.1, wheelFace), rim]);
    wheel.position.set(-2.7, 1.06, 0);
    parent.add(wheel);
    const ball = new Mesh(this.geo(new SphereGeometry(0.05, 12, 8)), this.std(0xf5f5f5, 0.2, 0.3));
    ball.position.set(0, 0.1, 0.86);
    wheel.add(ball);
    // Pointer on the +Z rim: the wheel is rotated so the winning pocket ends up here.
    this.box(parent, 0.08, 0.06, 0.25, m.gold, -2.7, 1.14, 1.2);
    this.sign(parent, signTexture('ROULETTE', 'AMERICAN 0 - 00'), 1.6, 0.6, 0.6, 2.2, -1.5);
    return { wheel, ball };
  }

  /**
   * One slot machine: cabinet, emissive screen, lever, top light. `rotY` turns the
   * machine (screen faces +Z at 0); the collider is centred on the cabinet so only
   * its half extents need swapping for quarter turns. Returns the redrawable screen.
   */
  slotMachine(parent: Object3D, x: number, z: number, rotY = 0): SlotScreen {
    const m = this.mat;
    const g = new Group();
    g.position.set(x, 0, z);
    g.rotation.y = rotY;
    parent.add(g);
    this.box(g, 0.9, 1.9, 0.75, m.black, 0, 0.95, 0);
    const quarter = Math.abs(Math.abs(rotY) - Math.PI / 2) < 1e-3;
    this.collider(g, 0, 0.95, 0, quarter ? 0.375 : 0.45, 0.95, quarter ? 0.45 : 0.375);
    this.box(g, 0.92, 0.06, 0.78, m.gold, 0, 1.93, 0);
    this.box(g, 0.92, 0.06, 0.78, m.gold, 0, 0.6, 0);
    const screen = new SlotScreen();
    const screenMesh = new Mesh(this.geo(new PlaneGeometry(0.76, 0.48)), this.emissive(0xffffff, 1.6, screen.texture));
    screenMesh.position.set(0, 1.45, 0.38);
    g.add(screenMesh);
    this.box(g, 0.7, 0.3, 0.04, m.redLeather, 0, 0.9, 0.38);
    this.cylinder(g, 0.025, 0.025, 0.5, m.chrome, 0.52, 1.35, 0.1, 8);
    this.cylinder(g, 0.07, 0.07, 0.07, m.redLeather, 0.52, 1.62, 0.1, 10);
    this.cylinder(g, 0.1, 0.12, 0.22, this.emissive(0xffd23f, 1.2), 0, 2.07, 0, 12);
    return screen;
  }

  /** Vertical Lucky Wheel on a stand; the disc faces +Z. Returns the spinning disc mesh. */
  luckyWheel(parent: Object3D, face: Texture): Mesh {
    const m = this.mat;
    this.box(parent, 1.2, 0.3, 1.2, m.darkWood, 0, 0.15, 0, true);
    this.cylinder(parent, 0.12, 0.16, 1.6, m.gold, 0, 1.1, 0, 12);
    this.collider(parent, 0, 1.9, 0, 0.3, 1.9, 0.3);
    const tilt = new Group();
    tilt.position.set(0, 2.55, 0.2);
    tilt.rotation.x = Math.PI / 2;
    parent.add(tilt);
    const rim = this.std(0x2a1430, 0.7);
    const disc = new Mesh(this.geo(new CylinderGeometry(1.7, 1.7, 0.1, 48)), [rim, this.std(0xffffff, 0.8, 0, face), rim]);
    tilt.add(disc);
    const pegs = this.std(0xffd23f, 0.4, 0.6);
    for (let i = 0; i < 20; i++) {
      const a = (i / 20) * Math.PI * 2;
      const peg = new Mesh(this.geo(new CylinderGeometry(0.03, 0.03, 0.12, 6)), pegs);
      peg.position.set(Math.sin(a) * 1.62, 0.1, Math.cos(a) * 1.62);
      disc.add(peg);
    }
    // Pointer: small wedge above the disc pointing down.
    const pointer = new Mesh(this.geo(new CylinderGeometry(0, 0.12, 0.35, 4)), this.emissive(0xff4fa3, 1.5));
    pointer.position.set(0, 4.45, 0.3);
    pointer.rotation.x = Math.PI;
    parent.add(pointer);
    this.sign(parent, signTexture('LUCKY WHEEL', 'ONE SPIN A DAY', '#ff4fa3'), 3.2, 1.2, 0, 5.0, -0.2);
    return disc;
  }

  /** Bar along X with the back shelf on +Z (against the entrance wall) and stools on -Z. */
  bar(parent: Object3D): void {
    const m = this.mat;
    this.box(parent, 6, 1.1, 0.8, m.darkWood, 0, 0.55, 0, true);
    this.box(parent, 6.2, 0.08, 0.9, m.marble, 0, 1.14, 0);
    this.box(parent, 6.4, 3, 0.4, m.plum, 0, 1.5, 1.6, true);
    this.box(parent, 6, 0.05, 0.35, m.marble, 0, 1.4, 1.35);
    this.box(parent, 6, 0.05, 0.35, m.marble, 0, 2.0, 1.35);
    const colors = [0x35e6ff, 0xffd23f, 0xff4fa3, 0x4ee06a, 0xa35bff, 0xff8a1f];
    for (let i = 0; i < 12; i++) {
      const bottle = this.std(colors[i % colors.length], 0.2, 0.1);
      this.cylinder(parent, 0.05, 0.06, 0.32, bottle, -2.6 + i * 0.47, 1.58 + (i % 2) * 0.6, 1.35, 8);
    }
    for (let i = -2; i <= 2; i++) {
      this.cylinder(parent, 0.2, 0.2, 0.06, m.redLeather, i * 1.1, 0.75, -0.85, 12);
      this.cylinder(parent, 0.04, 0.04, 0.72, m.chrome, i * 1.1, 0.36, -0.85, 8);
    }
    this.sign(parent, signTexture('PALMS BAR', '', '#35e6ff'), 2.4, 0.9, 0, 3.4, 1.38, Math.PI);
  }

  /** Box-built dealer standing at (x, z), facing +Z. */
  dealer(parent: Object3D, x: number, z: number, rotY: number): void {
    const m = this.mat;
    const g = new Group();
    g.position.set(x, 0, z);
    g.rotation.y = rotY;
    parent.add(g);
    this.box(g, 0.34, 0.85, 0.24, m.black, 0, 0.42, 0);
    this.box(g, 0.5, 0.62, 0.3, m.vest, 0, 1.15, 0);
    this.box(g, 0.14, 0.2, 0.3, m.shirt, 0, 1.36, 0.01);
    this.box(g, 0.14, 0.58, 0.14, m.shirt, -0.33, 1.15, 0);
    this.box(g, 0.14, 0.58, 0.14, m.shirt, 0.33, 1.15, 0);
    this.box(g, 0.26, 0.28, 0.26, m.skin, 0, 1.64, 0);
    this.box(g, 0.28, 0.08, 0.28, m.black, 0, 1.8, 0);
  }

  /** Ceiling emissive strip. */
  lightStrip(parent: Object3D, w: number, d: number, x: number, y: number, z: number): void {
    const strip = new Mesh(this.geo(new PlaneGeometry(w, d)), this.emissive(0xfff1c0, 2.2));
    strip.rotation.x = Math.PI / 2;
    strip.position.set(x, y, z);
    (strip.material as MeshStandardMaterial).side = DoubleSide;
    parent.add(strip);
  }
}
