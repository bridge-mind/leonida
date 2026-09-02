import { Color, type ShaderMaterial } from 'three';
import type { V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IParticlesPeer, ParticleEffectId } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { MuzzleFlash, particlePointMaterial, particleSpriteTexture } from '../../gfx/Effects';
import { ParticlesConfig as C } from './config';
import { EFFECTS, type BurstDef } from './effects';
import { ParticleBuffer, type SpawnParams } from './ParticleBuffer';
import { TracerBuffer } from './TracerBuffer';

export interface EmitOptions {
  direction?: V3;
  end?: V3;
  scale?: number;
  color?: number;
}

interface CompiledBurst {
  def: BurstDef;
  c0: [number, number, number];
  c1: [number, number, number];
}

const UP: V3 = { x: 0, y: 1, z: 0 };

/**
 * Pooled, allocation-free particle effects: two point-sprite layers (additive and
 * normal blending), a tracer line pool and the muzzle flash light.
 */
export class ParticleSystem implements GameSystem, IParticlesPeer {
  private additive!: ParticleBuffer;
  private normal!: ParticleBuffer;
  private tracers!: TracerBuffer;
  private flash!: MuzzleFlash;
  private readonly materials: ShaderMaterial[] = [];
  private readonly compiled = new Map<string, CompiledBurst[]>();
  private readonly spawn: SpawnParams = {
    x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, life: 1, size0: 0.1, size1: 0.1,
    r0: 1, g0: 1, b0: 1, r1: 1, g1: 1, b1: 1, gravity: 0, drag: 0, shape: 0, alpha0: 1,
  };
  private readonly dir: V3 = { x: 0, y: 1, z: 0 };
  private readonly tint = new Color();

  constructor(private readonly game: Game) {}

  init(): void {
    const map = particleSpriteTexture();
    const addMat = particlePointMaterial(map, true);
    const normMat = particlePointMaterial(map, false);
    this.materials.push(addMat, normMat);
    this.additive = new ParticleBuffer(C.additiveCapacity, addMat, C.floorY);
    this.normal = new ParticleBuffer(C.normalCapacity, normMat, C.floorY);
    this.tracers = new TracerBuffer(C.tracerCapacity, C.tracerLife);
    this.flash = new MuzzleFlash(this.game.scene, C.muzzleFlash);
    this.game.scene.add(this.additive.points, this.normal.points, this.tracers.lines);
    for (const [id, bursts] of Object.entries(EFFECTS)) {
      this.compiled.set(
        id,
        bursts.map((def) => {
          const a = new Color(def.color0);
          const b = new Color(def.color1);
          return { def, c0: [a.r, a.g, a.b], c1: [b.r, b.g, b.b] };
        }),
      );
    }
  }

  // --- IParticlesPeer ------------------------------------------------------------

  emit(id: ParticleEffectId, position: V3, opts?: EmitOptions): void {
    if (id === 'tracer') {
      if (opts?.end) this.tracers.add(position, opts.end, opts.color ?? C.tracerColor);
      return;
    }
    if (id === 'muzzleFlash') this.flash.flash(position, opts?.direction ?? UP);
    const bursts = this.compiled.get(id);
    if (!bursts) return;
    const scale = opts?.scale ?? 1;
    const d = opts?.direction;
    if (d) {
      const len = Math.hypot(d.x, d.y, d.z) || 1;
      this.dir.x = d.x / len;
      this.dir.y = d.y / len;
      this.dir.z = d.z / len;
    } else {
      this.dir.x = 0;
      this.dir.y = 1;
      this.dir.z = 0;
    }
    if (opts?.color !== undefined) this.tint.setHex(opts.color);
    for (let b = 0; b < bursts.length; b++) this.emitBurst(bursts[b], position, scale, opts?.color !== undefined);
  }

  private emitBurst(cb: CompiledBurst, p: V3, scale: number, tinted: boolean): void {
    const def = cb.def;
    const buffer = def.layer === 'additive' ? this.additive : this.normal;
    const s = this.spawn;
    for (let i = 0; i < def.count; i++) {
      const speed = lerp(def.speed[0], def.speed[1], Math.random()) * scale;
      if (def.emit === 'sphere') randomUnit(s);
      else randomCone(this.dir, def.cone, s);
      s.vx *= speed;
      s.vy = s.vy * speed + def.upBias * scale;
      s.vz *= speed;
      s.x = p.x;
      s.y = p.y;
      s.z = p.z;
      s.life = lerp(def.life[0], def.life[1], Math.random());
      s.size0 = def.size[0] * scale;
      s.size1 = def.size[1] * scale;
      const j = 1 - def.jitter * Math.random();
      if (tinted) {
        s.r0 = this.tint.r * j;
        s.g0 = this.tint.g * j;
        s.b0 = this.tint.b * j;
        s.r1 = this.tint.r * j * 0.5;
        s.g1 = this.tint.g * j * 0.5;
        s.b1 = this.tint.b * j * 0.5;
      } else {
        s.r0 = cb.c0[0] * j;
        s.g0 = cb.c0[1] * j;
        s.b0 = cb.c0[2] * j;
        s.r1 = cb.c1[0] * j;
        s.g1 = cb.c1[1] * j;
        s.b1 = cb.c1[2] * j;
      }
      s.gravity = def.gravity;
      s.drag = def.drag;
      s.shape = def.shape;
      s.alpha0 = def.alpha;
      buffer.spawn(s);
    }
  }

  /** Live particles across both layers (tracers excluded). */
  get count(): number {
    return this.additive.activeCount + this.normal.activeCount;
  }

  get tracerCount(): number {
    return this.tracers.activeCount;
  }

  // --- Loop ----------------------------------------------------------------------

  update(dt: number): void {
    const h = this.game.renderer.domElement.height || window.innerHeight;
    for (const m of this.materials) m.uniforms.uHeight.value = h;
    this.additive.update(dt);
    this.normal.update(dt);
    this.tracers.update(dt);
    this.flash.update(dt);
    this.game.perf.set('particles', this.count);
  }

  dispose(): void {
    this.game.scene.remove(this.additive.points, this.normal.points, this.tracers.lines);
    this.additive.dispose();
    this.normal.dispose();
    this.tracers.dispose();
    this.flash.dispose(this.game.scene);
    for (const m of this.materials) {
      (m.uniforms.uMap.value as { dispose(): void } | null)?.dispose();
      m.dispose();
    }
  }

  debugActions = {
    /** `particles.emit:<id>:<x>:<z>` emits at ground level + 1 m. */
    emit: (id: unknown, x: unknown, z: unknown): number => {
      const px = Number(x);
      const pz = Number(z);
      const y = (this.game.physics.groundHeightAt(px, pz) ?? 0) + 1;
      this.emit(String(id) as ParticleEffectId, { x: px, y, z: pz });
      return this.count;
    },
    count: (): { particles: number; tracers: number } => ({ particles: this.count, tracers: this.tracerCount }),
    clear: (): void => {
      this.additive.clear();
      this.normal.clear();
    },
  };
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Uniform random unit vector written to the velocity fields of `s`. */
function randomUnit(s: SpawnParams): void {
  const z = Math.random() * 2 - 1;
  const phi = Math.random() * Math.PI * 2;
  const r = Math.sqrt(1 - z * z);
  s.vx = r * Math.cos(phi);
  s.vy = z;
  s.vz = r * Math.sin(phi);
}

/** Random unit vector within `half` radians of `dir`, written to the velocity fields of `s`. */
function randomCone(dir: V3, half: number, s: SpawnParams): void {
  const cosMax = Math.cos(half);
  const cosT = cosMax + (1 - cosMax) * Math.random();
  const sinT = Math.sqrt(Math.max(0, 1 - cosT * cosT));
  const phi = Math.random() * Math.PI * 2;
  const ox = sinT * Math.cos(phi);
  const oy = sinT * Math.sin(phi);
  let ux: number;
  let uy: number;
  let uz: number;
  if (Math.abs(dir.y) < 0.99) {
    ux = dir.z;
    uy = 0;
    uz = -dir.x;
  } else {
    ux = 0;
    uy = dir.z;
    uz = -dir.y;
  }
  const ul = Math.hypot(ux, uy, uz) || 1;
  ux /= ul;
  uy /= ul;
  uz /= ul;
  const vx = dir.y * uz - dir.z * uy;
  const vy = dir.z * ux - dir.x * uz;
  const vz = dir.x * uy - dir.y * ux;
  s.vx = dir.x * cosT + ux * ox + vx * oy;
  s.vy = dir.y * cosT + uy * ox + vy * oy;
  s.vz = dir.z * cosT + uz * ox + vz * oy;
}

const module: SystemModule = {
  id: 'particles',
  priority: PRIORITY.particles,
  create: (game) => new ParticleSystem(game),
};

export default module;
