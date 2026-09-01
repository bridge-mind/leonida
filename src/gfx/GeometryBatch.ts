import { BufferAttribute, BufferGeometry, Color, Matrix4, Vector3, type BufferGeometry as BG } from 'three';

/**
 * Accumulates boxes/quads/arbitrary geometry into typed arrays and emits one
 * BufferGeometry with positions, normals, uvs, vertex colors and indices.
 * Used to merge a whole 200 m city chunk into a handful of draw calls.
 */
export class GeometryBatch {
  private positions: number[] = [];
  private normals: number[] = [];
  private uvs: number[] = [];
  private colors: number[] = [];
  private indices: number[] = [];
  private readonly tmpColor = new Color();
  private readonly v = new Vector3();
  private readonly n = new Vector3();

  get vertexCount(): number {
    return this.positions.length / 3;
  }

  get isEmpty(): boolean {
    return this.indices.length === 0;
  }

  /**
   * Axis-aligned box (optionally rotated around Y) with per-face UVs in meters
   * scaled by `uvScale` (1 uv unit = uvScale meters), so tiling textures repeat correctly.
   */
  addBox(
    cx: number,
    cy: number,
    cz: number,
    sx: number,
    sy: number,
    sz: number,
    color: number | Color,
    opts?: { rotationY?: number; uvScale?: number; uvOffset?: number; faces?: BoxFaces; topColor?: number | Color },
  ): void {
    const hx = sx / 2;
    const hy = sy / 2;
    const hz = sz / 2;
    const rot = opts?.rotationY ?? 0;
    const cosR = Math.cos(rot);
    const sinR = Math.sin(rot);
    const uvScale = opts?.uvScale ?? 1;
    const uvOffset = opts?.uvOffset ?? 0;
    const faces = opts?.faces ?? ALL_FACES;
    const c = this.tmpColor.set(color);
    const r = c.r;
    const g = c.g;
    const b = c.b;
    let tr = r;
    let tg = g;
    let tb = b;
    if (opts?.topColor !== undefined) {
      const tc = this.tmpColor.set(opts.topColor);
      tr = tc.r;
      tg = tc.g;
      tb = tc.b;
    }

    const emit = (
      corners: [number, number, number][],
      nx: number,
      ny: number,
      nz: number,
      uSize: number,
      vSize: number,
      cr: number,
      cg: number,
      cb: number,
    ): void => {
      const base = this.vertexCount;
      const rnx = nx * cosR + nz * sinR;
      const rnz = -nx * sinR + nz * cosR;
      const uvs: [number, number][] = [
        [uvOffset, 0],
        [uvOffset + uSize / uvScale, 0],
        [uvOffset + uSize / uvScale, vSize / uvScale],
        [uvOffset, vSize / uvScale],
      ];
      for (let i = 0; i < 4; i++) {
        const [lx, ly, lz] = corners[i];
        const wx = lx * cosR + lz * sinR;
        const wz = -lx * sinR + lz * cosR;
        this.positions.push(cx + wx, cy + ly, cz + wz);
        this.normals.push(rnx, ny, rnz);
        this.uvs.push(uvs[i][0], uvs[i][1]);
        this.colors.push(cr, cg, cb);
      }
      this.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    };

    // +Z (front), -Z (back), +X (right), -X (left), +Y (top), -Y (bottom)
    if (faces.front) emit([[-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]], 0, 0, 1, sx, sy, r, g, b);
    if (faces.back) emit([[hx, -hy, -hz], [-hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz]], 0, 0, -1, sx, sy, r, g, b);
    if (faces.right) emit([[hx, -hy, hz], [hx, -hy, -hz], [hx, hy, -hz], [hx, hy, hz]], 1, 0, 0, sz, sy, r, g, b);
    if (faces.left) emit([[-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, hz], [-hx, hy, -hz]], -1, 0, 0, sz, sy, r, g, b);
    if (faces.top) emit([[-hx, hy, hz], [hx, hy, hz], [hx, hy, -hz], [-hx, hy, -hz]], 0, 1, 0, sx, sz, tr, tg, tb);
    if (faces.bottom) emit([[-hx, -hy, -hz], [hx, -hy, -hz], [hx, -hy, hz], [-hx, -hy, hz]], 0, -1, 0, sx, sz, r, g, b);
  }

  /** Horizontal quad (facing +Y) from (x0,z0) to (x1,z1) at height y. */
  addQuad(x0: number, z0: number, x1: number, z1: number, y: number, color: number | Color, uvScale = 1): void {
    const c = this.tmpColor.set(color);
    const base = this.vertexCount;
    const w = x1 - x0;
    const d = z1 - z0;
    const pts: [number, number][] = [
      [x0, z1],
      [x1, z1],
      [x1, z0],
      [x0, z0],
    ];
    const uv: [number, number][] = [
      [0, 0],
      [w / uvScale, 0],
      [w / uvScale, d / uvScale],
      [0, d / uvScale],
    ];
    for (let i = 0; i < 4; i++) {
      this.positions.push(pts[i][0], y, pts[i][1]);
      this.normals.push(0, 1, 0);
      this.uvs.push(uv[i][0], uv[i][1]);
      this.colors.push(c.r, c.g, c.b);
    }
    this.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  /**
   * Arbitrary quad from four corners (counter-clockwise seen from the front) with explicit UVs.
   */
  addQuadCorners(corners: [Vector3, Vector3, Vector3, Vector3], color: number | Color, uvs?: [number, number][]): void {
    const c = this.tmpColor.set(color);
    const base = this.vertexCount;
    const e1 = this.v.subVectors(corners[1], corners[0]);
    const e2 = this.n.subVectors(corners[3], corners[0]);
    const normal = new Vector3().crossVectors(e1, e2).normalize();
    const defaultUv: [number, number][] = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
    ];
    for (let i = 0; i < 4; i++) {
      this.positions.push(corners[i].x, corners[i].y, corners[i].z);
      this.normals.push(normal.x, normal.y, normal.z);
      const uv = (uvs ?? defaultUv)[i];
      this.uvs.push(uv[0], uv[1]);
      this.colors.push(c.r, c.g, c.b);
    }
    this.indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
  }

  /** Triangular prism roof (gable) along the x axis: ridge at height `cy + ridge`. */
  addGableRoof(cx: number, cy: number, cz: number, sx: number, sz: number, ridge: number, color: number | Color, alongZ = false): void {
    const c = this.tmpColor.set(color);
    const hx = alongZ ? sz / 2 : sx / 2;
    const hz = alongZ ? sx / 2 : sz / 2;
    const rot = alongZ ? Math.PI / 2 : 0;
    const cosR = Math.cos(rot);
    const sinR = Math.sin(rot);
    const local: [number, number, number][] = [
      [-hx, 0, -hz],
      [hx, 0, -hz],
      [hx, 0, hz],
      [-hx, 0, hz],
      [-hx, ridge, 0],
      [hx, ridge, 0],
    ];
    const world = local.map(([lx, ly, lz]) => new Vector3(cx + lx * cosR + lz * sinR, cy + ly, cz - lx * sinR + lz * cosR));
    // Two slopes + two gable triangles (as degenerate quads).
    this.addQuadCorners([world[3], world[2], world[5], world[4]], c);
    this.addQuadCorners([world[1], world[0], world[4], world[5]], c);
    this.addTriangle(world[0], world[3], world[4], c);
    this.addTriangle(world[2], world[1], world[5], c);
  }

  addTriangle(a: Vector3, b: Vector3, c: Vector3, color: Color | number): void {
    const col = this.tmpColor.set(color);
    const base = this.vertexCount;
    const normal = new Vector3().crossVectors(this.v.subVectors(b, a), this.n.subVectors(c, a)).normalize();
    for (const p of [a, b, c]) {
      this.positions.push(p.x, p.y, p.z);
      this.normals.push(normal.x, normal.y, normal.z);
      this.uvs.push(0, 0);
      this.colors.push(col.r, col.g, col.b);
    }
    this.indices.push(base, base + 1, base + 2);
  }

  /** Appends an existing indexed geometry transformed by `matrix` with a flat color. */
  addGeometry(geom: BG, matrix: Matrix4, color: number | Color): void {
    const c = this.tmpColor.set(color);
    const pos = geom.getAttribute('position');
    const nor = geom.getAttribute('normal');
    const uv = geom.getAttribute('uv');
    const base = this.vertexCount;
    const normalMatrix = new Matrix4().copy(matrix);
    normalMatrix.setPosition(0, 0, 0);
    for (let i = 0; i < pos.count; i++) {
      this.v.fromBufferAttribute(pos, i).applyMatrix4(matrix);
      this.positions.push(this.v.x, this.v.y, this.v.z);
      if (nor) {
        this.n.fromBufferAttribute(nor, i).applyMatrix4(normalMatrix).normalize();
        this.normals.push(this.n.x, this.n.y, this.n.z);
      } else this.normals.push(0, 1, 0);
      if (uv) this.uvs.push(uv.getX(i), uv.getY(i));
      else this.uvs.push(0, 0);
      this.colors.push(c.r, c.g, c.b);
    }
    const idx = geom.getIndex();
    if (idx) for (let i = 0; i < idx.count; i++) this.indices.push(base + idx.getX(i));
    else for (let i = 0; i < pos.count; i++) this.indices.push(base + i);
  }

  build(): BufferGeometry {
    const geom = new BufferGeometry();
    geom.setAttribute('position', new BufferAttribute(new Float32Array(this.positions), 3));
    geom.setAttribute('normal', new BufferAttribute(new Float32Array(this.normals), 3));
    geom.setAttribute('uv', new BufferAttribute(new Float32Array(this.uvs), 2));
    geom.setAttribute('color', new BufferAttribute(new Float32Array(this.colors), 3));
    const IndexArray = this.vertexCount > 65535 ? Uint32Array : Uint16Array;
    geom.setIndex(new BufferAttribute(new IndexArray(this.indices), 1));
    geom.computeBoundingSphere();
    geom.computeBoundingBox();
    return geom;
  }

  clear(): void {
    this.positions = [];
    this.normals = [];
    this.uvs = [];
    this.colors = [];
    this.indices = [];
  }
}

export interface BoxFaces {
  front: boolean;
  back: boolean;
  left: boolean;
  right: boolean;
  top: boolean;
  bottom: boolean;
}

export const ALL_FACES: BoxFaces = { front: true, back: true, left: true, right: true, top: true, bottom: true };
export const NO_BOTTOM: BoxFaces = { front: true, back: true, left: true, right: true, top: true, bottom: false };
export const SIDES_ONLY: BoxFaces = { front: true, back: true, left: true, right: true, top: false, bottom: false };
export const TOP_ONLY: BoxFaces = { front: false, back: false, left: false, right: false, top: true, bottom: false };
