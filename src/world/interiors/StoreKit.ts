import {
  BoxGeometry,
  CanvasTexture,
  Color,
  DoubleSide,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PlaneGeometry,
  PointLight,
  SRGBColorSpace,
  Vector3,
  type BufferGeometry,
  type Material,
} from 'three';
import type { V3 } from '../../core/entities';
import type { InteriorBuildContext } from './Interiors';

/**
 * Shared primitives for the store interiors. Every store is built inside the
 * framework's single interior root, so each one is shifted along +X by its own
 * offset; all points in a StoreLayout are already in root-local space (offset
 * included) and the shop systems add the framework origin to get world space.
 */

export interface Post {
  pos: V3;
  heading: number;
}

export interface StoreLayout {
  /** Where the player stands to use the counter (food / gun store UI). */
  counter: V3;
  /** Where the clerk is posted; null for unstaffed interiors. */
  clerk: Post | null;
  /** Where robbery cash appears (on the register). */
  register: V3;
  guards: Post[];
  /** Save-point bed (safehouse only). */
  bed: V3 | null;
  interactRadius: number;
}

export interface RoomSpec {
  w: number;
  d: number;
  h: number;
  floor: number;
  wall: number;
  ceiling: number;
  /** Trim colour used for the door frame and skirting. */
  accent: number;
}

/** Spacing between store interiors along +X so their colliders never overlap (casino sits at 0). */
export const STORE_SLOT_SPACING = 120;

export function storeOffset(slot: number): number {
  return STORE_SLOT_SPACING * slot;
}

export class StoreKit {
  private readonly materials = new Map<string, Material>();
  private readonly geometries = new Map<string, BufferGeometry>();
  private readonly textures: CanvasTexture[] = [];
  private readonly lights: PointLight[] = [];
  private readonly tmp = new Vector3();

  constructor(
    private readonly ctx: InteriorBuildContext,
    readonly ox: number,
  ) {}

  /** Root-local point for this store: offsets x, passes y/z through. */
  p(x: number, y: number, z: number): V3 {
    return { x: x + this.ox, y, z };
  }

  post(x: number, z: number, heading: number): Post {
    return { pos: this.p(x, 0, z), heading };
  }

  std(color: number, roughness = 0.85, metalness = 0): MeshStandardMaterial {
    const key = `s${color}|${roughness}|${metalness}`;
    let m = this.materials.get(key) as MeshStandardMaterial | undefined;
    if (!m) {
      m = new MeshStandardMaterial({ color, roughness, metalness });
      this.materials.set(key, m);
    }
    return m;
  }

  glow(color: number, intensity = 1): MeshStandardMaterial {
    const key = `g${color}|${intensity}`;
    let m = this.materials.get(key) as MeshStandardMaterial | undefined;
    if (!m) {
      m = new MeshStandardMaterial({
        color: 0x111111,
        emissive: new Color(color),
        emissiveIntensity: intensity,
        roughness: 0.4,
      });
      this.materials.set(key, m);
    }
    return m;
  }

  private boxGeometry(w: number, h: number, d: number): BufferGeometry {
    const key = `${w}|${h}|${d}`;
    let g = this.geometries.get(key);
    if (!g) {
      g = new BoxGeometry(w, h, d);
      this.geometries.set(key, g);
    }
    return g;
  }

  /**
   * Box with its base at `y` (not its centre) placed at store-local x/z.
   * `collide` adds a matching static collider in world space.
   */
  box(
    parent: Group,
    w: number,
    h: number,
    d: number,
    material: Material,
    x: number,
    y: number,
    z: number,
    opts: { collide?: boolean; rotY?: number; shadow?: boolean } = {},
  ): Mesh {
    const mesh = new Mesh(this.boxGeometry(w, h, d), material);
    mesh.position.set(x + this.ox, y + h / 2, z);
    if (opts.rotY) mesh.rotation.y = opts.rotY;
    mesh.castShadow = opts.shadow ?? true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    if (opts.collide) this.collider(w, h, d, x, y, z, opts.rotY ?? 0);
    return mesh;
  }

  /** Static cuboid collider with its base at `y`; rotation is snapped to the nearest quarter turn (all our furniture is axis-aligned). */
  collider(w: number, h: number, d: number, x: number, y: number, z: number, rotY = 0): void {
    const quarter = Math.round(rotY / (Math.PI / 2)) % 2 !== 0;
    const hw = (quarter ? d : w) / 2;
    const hd = (quarter ? w : d) / 2;
    const o = this.ctx.origin;
    this.tmp.set(o.x + x + this.ox, o.y + y + h / 2, o.z + z);
    this.ctx.physics.addStaticCuboid({ x: hw, y: h / 2, z: hd }, this.tmp);
  }

  light(parent: Group, color: number, intensity: number, distance: number, x: number, y: number, z: number): PointLight {
    const l = new PointLight(color, intensity, distance, 1.6);
    l.position.set(x + this.ox, y, z);
    parent.add(l);
    this.lights.push(l);
    return l;
  }

  /** Canvas-textured sign (title + optional subtitle) facing `rotY` (0 = faces +Z). */
  sign(parent: Group, title: string, sub: string, color: string, w: number, h: number, x: number, y: number, z: number, rotY = 0): Mesh {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = Math.max(64, Math.round((512 * h) / w));
    const c = canvas.getContext('2d');
    if (c) {
      c.fillStyle = '#0b0b12';
      c.fillRect(0, 0, canvas.width, canvas.height);
      c.strokeStyle = color;
      c.lineWidth = 10;
      c.strokeRect(8, 8, canvas.width - 16, canvas.height - 16);
      c.fillStyle = color;
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      const titleSize = Math.min(canvas.height * 0.5, 110);
      c.font = `900 ${titleSize}px system-ui, sans-serif`;
      c.fillText(title, canvas.width / 2, sub ? canvas.height * 0.4 : canvas.height / 2);
      if (sub) {
        c.fillStyle = '#f5f5ff';
        c.font = `600 ${Math.min(canvas.height * 0.22, 44)}px system-ui, sans-serif`;
        c.fillText(sub, canvas.width / 2, canvas.height * 0.78);
      }
    }
    const tex = new CanvasTexture(canvas);
    tex.colorSpace = SRGBColorSpace;
    this.textures.push(tex);
    const mat = new MeshBasicMaterial({ map: tex, side: DoubleSide });
    this.materials.set(`sign${this.textures.length}`, mat);
    const mesh = new Mesh(new PlaneGeometry(w, h), mat);
    this.geometries.set(`plane${this.textures.length}`, mesh.geometry);
    mesh.position.set(x + this.ox, y, z);
    mesh.rotation.y = rotY;
    parent.add(mesh);
    return mesh;
  }

  /** Floor, ceiling and four walls with a door gap centred on the +Z wall. */
  room(parent: Group, spec: RoomSpec): void {
    const { w, d, h } = spec;
    const t = 0.3;
    this.box(parent, w, 0.1, d, this.std(spec.floor, 0.9), 0, -0.1, 0, {
      collide: true,
      shadow: false,
    });
    this.box(parent, w, 0.1, d, this.std(spec.ceiling, 1), 0, h, 0, {
      shadow: false,
    });
    const wall = this.std(spec.wall);
    // Back wall (-Z) and side walls.
    this.box(parent, w + t * 2, h, t, wall, 0, 0, -d / 2 - t / 2, {
      collide: true,
    });
    this.box(parent, t, h, d, wall, -w / 2 - t / 2, 0, 0, { collide: true });
    this.box(parent, t, h, d, wall, w / 2 + t / 2, 0, 0, { collide: true });
    // Front wall (+Z) split around a 2.4 m door.
    const door = 2.4;
    const side = (w - door) / 2;
    this.box(parent, side, h, t, wall, -(door / 2 + side / 2), 0, d / 2 + t / 2, { collide: true });
    this.box(parent, side, h, t, wall, door / 2 + side / 2, 0, d / 2 + t / 2, {
      collide: true,
    });
    this.box(parent, door + 0.4, h - 2.6, t, wall, 0, 2.6, d / 2 + t / 2, {
      collide: true,
    });
    // Door frame + exit mat so the player can see where the trigger is.
    const accent = this.std(spec.accent, 0.5, 0.3);
    this.box(parent, 0.12, 2.6, t + 0.05, accent, -door / 2, 0, d / 2 + t / 2);
    this.box(parent, 0.12, 2.6, t + 0.05, accent, door / 2, 0, d / 2 + t / 2);
    this.box(parent, door, 0.02, 1.2, this.std(0x3a2a22), 0, 0, d / 2 - 0.6, {
      shadow: false,
    });
    // Skirting.
    this.box(parent, w, 0.12, 0.04, accent, 0, 0, -d / 2 + 0.02, {
      shadow: false,
    });
  }

  dispose(): void {
    for (const m of this.materials.values()) m.dispose();
    for (const g of this.geometries.values()) g.dispose();
    for (const t of this.textures) t.dispose();
    for (const l of this.lights) l.dispose();
    this.materials.clear();
    this.geometries.clear();
    this.textures.length = 0;
    this.lights.length = 0;
  }
}

/** Standard entrance geometry shared by every store: spawn just inside the door, exit trigger on the mat. */
export function entrance(
  ox: number,
  depth: number,
): {
  spawnPoint: V3;
  spawnHeading: number;
  exitTrigger: V3;
  exitRadius: number;
} {
  return {
    spawnPoint: { x: ox, y: 0, z: depth / 2 - 2.4 },
    spawnHeading: Math.PI,
    exitTrigger: { x: ox, y: 0, z: depth / 2 - 0.4 },
    exitRadius: 1.1,
  };
}

/** Rows of shelving: a spine with three loaded shelves, goods as coloured blocks. */
export function shelfRun(kit: StoreKit, parent: Group, x: number, z: number, length: number, rotY: number, goods: number[]): void {
  const spine = kit.std(0xd6d6dc, 0.6, 0.2);
  kit.box(parent, length, 1.8, 0.5, spine, x, 0, z, { collide: true, rotY });
  const along = Math.cos(rotY);
  const across = Math.sin(rotY);
  for (let level = 0; level < 3; level++) {
    const y = 0.35 + level * 0.55;
    const count = Math.max(2, Math.floor(length / 0.45));
    for (let i = 0; i < count; i++) {
      const t = -length / 2 + 0.25 + (i * (length - 0.5)) / Math.max(1, count - 1);
      const color = goods[(i + level) % goods.length];
      for (const side of [-1, 1]) {
        const dx = t * along + side * 0.3 * across;
        const dz = -t * across + side * 0.3 * along;
        kit.box(parent, 0.3, 0.3, 0.16, kit.std(color, 0.7), x + dx, y, z + dz, { rotY, shadow: false });
      }
    }
  }
}

/** Sales counter with a register block; returns the register top point for cash pickups. */
export function counter(kit: StoreKit, parent: Group, x: number, z: number, length: number, rotY: number, color: number): V3 {
  kit.box(parent, length, 1.05, 0.8, kit.std(color, 0.6), x, 0, z, {
    collide: true,
    rotY,
  });
  kit.box(parent, length + 0.1, 0.06, 0.9, kit.std(0x2b2b33, 0.3, 0.4), x, 1.05, z, { rotY, shadow: false });
  const rx = x + Math.cos(rotY) * (length / 2 - 0.7);
  const rz = z - Math.sin(rotY) * (length / 2 - 0.7);
  kit.box(parent, 0.45, 0.32, 0.4, kit.std(0x1a1a22, 0.4, 0.5), rx, 1.11, rz, {
    rotY,
  });
  kit.box(parent, 0.3, 0.2, 0.05, kit.glow(0x63f5b2, 0.8), rx, 1.25, rz + 0.22 * Math.cos(rotY), { rotY, shadow: false });
  return kit.p(rx, 1.1, rz);
}
