import { AdditiveBlending, BufferAttribute, BufferGeometry, Color, DynamicDrawUsage, LineBasicMaterial, LineSegments } from 'three';
import type { V3 } from '../../core/entities';

/**
 * Pooled bullet tracers: one `LineSegments` with two vertices per tracer. Additive
 * blending lets the fade be done by darkening the vertex colour toward black.
 */
export class TracerBuffer {
  readonly lines: LineSegments;
  private count = 0;
  private readonly posAttr: BufferAttribute;
  private readonly colAttr: BufferAttribute;
  private readonly life: Float32Array;
  private readonly maxLife: Float32Array;
  private readonly base: Float32Array;
  private readonly tmpColor = new Color();

  constructor(
    readonly capacity: number,
    private readonly defaultLife: number,
  ) {
    const geo = new BufferGeometry();
    this.posAttr = new BufferAttribute(new Float32Array(capacity * 6), 3).setUsage(DynamicDrawUsage);
    this.colAttr = new BufferAttribute(new Float32Array(capacity * 6), 3).setUsage(DynamicDrawUsage);
    geo.setAttribute('position', this.posAttr);
    geo.setAttribute('color', this.colAttr);
    geo.setDrawRange(0, 0);
    const mat = new LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
      toneMapped: false,
    });
    this.lines = new LineSegments(geo, mat);
    this.lines.frustumCulled = false;
    this.lines.renderOrder = 11;
    this.life = new Float32Array(capacity);
    this.maxLife = new Float32Array(capacity);
    this.base = new Float32Array(capacity * 3);
  }

  get activeCount(): number {
    return this.count;
  }

  add(from: V3, to: V3, color: number, life = this.defaultLife): void {
    let i = this.count;
    if (i >= this.capacity) i = 0;
    else this.count++;
    const pos = this.posAttr.array as Float32Array;
    const i6 = i * 6;
    pos[i6] = from.x;
    pos[i6 + 1] = from.y;
    pos[i6 + 2] = from.z;
    pos[i6 + 3] = to.x;
    pos[i6 + 4] = to.y;
    pos[i6 + 5] = to.z;
    this.tmpColor.setHex(color);
    this.base[i * 3] = this.tmpColor.r;
    this.base[i * 3 + 1] = this.tmpColor.g;
    this.base[i * 3 + 2] = this.tmpColor.b;
    this.life[i] = life;
    this.maxLife[i] = life;
  }

  update(dt: number): void {
    if (this.count === 0) return;
    const col = this.colAttr.array as Float32Array;
    let i = 0;
    while (i < this.count) {
      const life = this.life[i] - dt;
      if (life <= 0) {
        this.swapRemove(i);
        continue;
      }
      this.life[i] = life;
      const k = life / this.maxLife[i];
      const i3 = i * 3;
      const i6 = i * 6;
      // Head of the tracer stays brighter than the tail so it reads as motion.
      col[i6] = this.base[i3] * k * 0.6;
      col[i6 + 1] = this.base[i3 + 1] * k * 0.6;
      col[i6 + 2] = this.base[i3 + 2] * k * 0.6;
      col[i6 + 3] = this.base[i3] * k;
      col[i6 + 4] = this.base[i3 + 1] * k;
      col[i6 + 5] = this.base[i3 + 2] * k;
      i++;
    }
    this.posAttr.needsUpdate = true;
    this.colAttr.needsUpdate = true;
    this.lines.geometry.setDrawRange(0, this.count * 2);
  }

  private swapRemove(i: number): void {
    const last = this.count - 1;
    this.count = last;
    if (i === last) return;
    const pos = this.posAttr.array as Float32Array;
    for (let k = 0; k < 6; k++) pos[i * 6 + k] = pos[last * 6 + k];
    for (let k = 0; k < 3; k++) this.base[i * 3 + k] = this.base[last * 3 + k];
    this.life[i] = this.life[last];
    this.maxLife[i] = this.maxLife[last];
  }

  dispose(): void {
    this.lines.geometry.dispose();
    (this.lines.material as LineBasicMaterial).dispose();
  }
}
