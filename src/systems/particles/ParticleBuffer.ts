import { BufferAttribute, BufferGeometry, DynamicDrawUsage, Points, type ShaderMaterial } from 'three';

export interface SpawnParams {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  life: number;
  size0: number;
  size1: number;
  /** Start colour (linear-ish rgb 0..1). */
  r0: number;
  g0: number;
  b0: number;
  r1: number;
  g1: number;
  b1: number;
  gravity: number;
  drag: number;
  /** 0 = soft disc, 1 = square. */
  shape: number;
  /** Alpha at birth; fades to 0 over the life. */
  alpha0: number;
}

/**
 * Fixed-capacity particle pool rendered as one `THREE.Points`. Alive particles are
 * kept packed in [0, count) by swap-remove so the draw range is contiguous and no
 * per-frame allocation happens after construction.
 */
export class ParticleBuffer {
  readonly points: Points;
  private count = 0;

  private readonly posAttr: BufferAttribute;
  private readonly sizeAttr: BufferAttribute;
  private readonly colorAttr: BufferAttribute;
  private readonly alphaAttr: BufferAttribute;
  private readonly shapeAttr: BufferAttribute;

  private readonly vel: Float32Array;
  private readonly life: Float32Array;
  private readonly maxLife: Float32Array;
  private readonly size0: Float32Array;
  private readonly size1: Float32Array;
  private readonly col0: Float32Array;
  private readonly col1: Float32Array;
  private readonly gravity: Float32Array;
  private readonly drag: Float32Array;
  private readonly alpha0: Float32Array;

  constructor(
    readonly capacity: number,
    material: ShaderMaterial,
    private readonly floorY: number,
  ) {
    const geo = new BufferGeometry();
    this.posAttr = new BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(DynamicDrawUsage);
    this.sizeAttr = new BufferAttribute(new Float32Array(capacity), 1).setUsage(DynamicDrawUsage);
    this.colorAttr = new BufferAttribute(new Float32Array(capacity * 3), 3).setUsage(DynamicDrawUsage);
    this.alphaAttr = new BufferAttribute(new Float32Array(capacity), 1).setUsage(DynamicDrawUsage);
    this.shapeAttr = new BufferAttribute(new Float32Array(capacity), 1).setUsage(DynamicDrawUsage);
    geo.setAttribute('position', this.posAttr);
    geo.setAttribute('size', this.sizeAttr);
    geo.setAttribute('color', this.colorAttr);
    geo.setAttribute('alpha', this.alphaAttr);
    geo.setAttribute('shape', this.shapeAttr);
    geo.setDrawRange(0, 0);
    this.points = new Points(geo, material);
    // Bounds are never recomputed; particles can be anywhere near the player.
    this.points.frustumCulled = false;
    this.points.renderOrder = 10;

    this.vel = new Float32Array(capacity * 3);
    this.life = new Float32Array(capacity);
    this.maxLife = new Float32Array(capacity);
    this.size0 = new Float32Array(capacity);
    this.size1 = new Float32Array(capacity);
    this.col0 = new Float32Array(capacity * 3);
    this.col1 = new Float32Array(capacity * 3);
    this.gravity = new Float32Array(capacity);
    this.drag = new Float32Array(capacity);
    this.alpha0 = new Float32Array(capacity);
  }

  get activeCount(): number {
    return this.count;
  }

  /** Spawns one particle; when full, the oldest slot (index 0) is recycled. */
  spawn(p: SpawnParams): void {
    let i = this.count;
    if (i >= this.capacity) {
      i = 0;
    } else {
      this.count++;
    }
    const pos = this.posAttr.array as Float32Array;
    pos[i * 3] = p.x;
    pos[i * 3 + 1] = p.y;
    pos[i * 3 + 2] = p.z;
    this.vel[i * 3] = p.vx;
    this.vel[i * 3 + 1] = p.vy;
    this.vel[i * 3 + 2] = p.vz;
    this.life[i] = p.life;
    this.maxLife[i] = p.life;
    this.size0[i] = p.size0;
    this.size1[i] = p.size1;
    this.col0[i * 3] = p.r0;
    this.col0[i * 3 + 1] = p.g0;
    this.col0[i * 3 + 2] = p.b0;
    this.col1[i * 3] = p.r1;
    this.col1[i * 3 + 1] = p.g1;
    this.col1[i * 3 + 2] = p.b1;
    this.gravity[i] = p.gravity;
    this.drag[i] = p.drag;
    this.alpha0[i] = p.alpha0;
    (this.shapeAttr.array as Float32Array)[i] = p.shape;
    (this.sizeAttr.array as Float32Array)[i] = p.size0;
    const col = this.colorAttr.array as Float32Array;
    col[i * 3] = p.r0;
    col[i * 3 + 1] = p.g0;
    col[i * 3 + 2] = p.b0;
    (this.alphaAttr.array as Float32Array)[i] = p.alpha0;
  }

  update(dt: number): void {
    if (this.count === 0) return;
    const pos = this.posAttr.array as Float32Array;
    const size = this.sizeAttr.array as Float32Array;
    const col = this.colorAttr.array as Float32Array;
    const alpha = this.alphaAttr.array as Float32Array;
    let i = 0;
    while (i < this.count) {
      const life = this.life[i] - dt;
      const i3 = i * 3;
      if (life <= 0 || pos[i3 + 1] < this.floorY) {
        this.swapRemove(i);
        continue;
      }
      this.life[i] = life;
      const t = 1 - life / this.maxLife[i];
      const k = Math.max(0, 1 - this.drag[i] * dt);
      this.vel[i3 + 1] += this.gravity[i] * dt;
      this.vel[i3] *= k;
      this.vel[i3 + 1] *= k;
      this.vel[i3 + 2] *= k;
      pos[i3] += this.vel[i3] * dt;
      pos[i3 + 1] += this.vel[i3 + 1] * dt;
      pos[i3 + 2] += this.vel[i3 + 2] * dt;
      size[i] = this.size0[i] + (this.size1[i] - this.size0[i]) * t;
      col[i3] = this.col0[i3] + (this.col1[i3] - this.col0[i3]) * t;
      col[i3 + 1] = this.col0[i3 + 1] + (this.col1[i3 + 1] - this.col0[i3 + 1]) * t;
      col[i3 + 2] = this.col0[i3 + 2] + (this.col1[i3 + 2] - this.col0[i3 + 2]) * t;
      // Quick fade-in over the first 10 % keeps spawns from popping, then ease out.
      const fadeIn = Math.min(1, t * 10);
      alpha[i] = this.alpha0[i] * fadeIn * (1 - t) * (1 - t * 0.5);
      i++;
    }
    this.posAttr.needsUpdate = true;
    this.sizeAttr.needsUpdate = true;
    this.colorAttr.needsUpdate = true;
    this.alphaAttr.needsUpdate = true;
    this.shapeAttr.needsUpdate = true;
    this.points.geometry.setDrawRange(0, this.count);
  }

  private swapRemove(i: number): void {
    const last = this.count - 1;
    this.count = last;
    if (i === last) return;
    const pos = this.posAttr.array as Float32Array;
    const shape = this.shapeAttr.array as Float32Array;
    const i3 = i * 3;
    const l3 = last * 3;
    for (let k = 0; k < 3; k++) {
      pos[i3 + k] = pos[l3 + k];
      this.vel[i3 + k] = this.vel[l3 + k];
      this.col0[i3 + k] = this.col0[l3 + k];
      this.col1[i3 + k] = this.col1[l3 + k];
    }
    this.life[i] = this.life[last];
    this.maxLife[i] = this.maxLife[last];
    this.size0[i] = this.size0[last];
    this.size1[i] = this.size1[last];
    this.gravity[i] = this.gravity[last];
    this.drag[i] = this.drag[last];
    this.alpha0[i] = this.alpha0[last];
    shape[i] = shape[last];
  }

  clear(): void {
    this.count = 0;
    this.points.geometry.setDrawRange(0, 0);
  }

  dispose(): void {
    this.points.geometry.dispose();
  }
}
