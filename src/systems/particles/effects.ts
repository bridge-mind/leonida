import type { ParticleEffectId } from '../../core/peers';

export interface BurstDef {
  layer: 'additive' | 'normal';
  count: number;
  /** [min, max] seconds. */
  life: [number, number];
  /** [min, max] m/s. */
  speed: [number, number];
  /** 'cone' around the emit direction (or +Y when none is given), 'sphere' all around. */
  emit: 'cone' | 'sphere';
  /** Cone half-angle in radians (ignored for 'sphere'). */
  cone: number;
  /** Extra upward velocity in m/s (buoyancy for smoke, splash for water). */
  upBias: number;
  /** [start, end] diameter in meters. */
  size: [number, number];
  color0: number;
  color1: number;
  gravity: number;
  drag: number;
  /** 0 soft disc, 1 square. */
  shape: 0 | 1;
  alpha: number;
  /** Random per-particle brightness jitter 0..1. */
  jitter: number;
}

const DEG = Math.PI / 180;

function burst(partial: Partial<BurstDef> & Pick<BurstDef, 'layer' | 'count' | 'color0' | 'color1'>): BurstDef {
  return {
    life: [0.3, 0.6],
    speed: [1, 3],
    emit: 'cone',
    cone: 60 * DEG,
    upBias: 0,
    size: [0.1, 0.2],
    gravity: 0,
    drag: 0,
    shape: 0,
    alpha: 1,
    jitter: 0.15,
    ...partial,
  };
}

const SPARKS = burst({
  layer: 'additive',
  count: 10,
  color0: 0xffe6a0,
  color1: 0xff6a10,
  life: [0.15, 0.4],
  speed: [4, 9],
  cone: 70 * DEG,
  size: [0.06, 0.02],
  gravity: -9.8,
  drag: 0.5,
});

const CONCRETE_DUST = burst({
  layer: 'normal',
  count: 8,
  color0: 0xa4a29a,
  color1: 0x6f6d66,
  life: [0.45, 0.8],
  speed: [1, 3],
  cone: 75 * DEG,
  size: [0.12, 0.45],
  drag: 2.5,
  alpha: 0.75,
});

/** Every `ParticleEffectId` except 'tracer' (which draws a line, see index.ts). */
export const EFFECTS: Record<Exclude<ParticleEffectId, 'tracer'>, BurstDef[]> = {
  muzzleFlash: [
    burst({
      layer: 'additive',
      count: 5,
      color0: 0xfff6c0,
      color1: 0xff9a20,
      life: [0.03, 0.05],
      speed: [2, 5],
      cone: 30 * DEG,
      size: [0.22, 0.05],
    }),
  ],
  impactConcrete: [CONCRETE_DUST, { ...SPARKS, count: 3 }],
  impactMetal: [{ ...SPARKS, count: 12, speed: [5, 11] }, { ...CONCRETE_DUST, count: 2, color0: 0x8a8a8a, color1: 0x555555 }],
  impactDirt: [
    burst({
      layer: 'normal',
      count: 10,
      color0: 0x8a6a45,
      color1: 0x5a4430,
      life: [0.5, 0.9],
      speed: [1.5, 3.5],
      cone: 70 * DEG,
      size: [0.12, 0.5],
      gravity: -2,
      drag: 2,
      alpha: 0.8,
    }),
  ],
  impactWater: [
    burst({
      layer: 'normal',
      count: 12,
      color0: 0xd6f0ff,
      color1: 0x9cc8e6,
      life: [0.4, 0.7],
      speed: [2, 4.5],
      cone: 35 * DEG,
      upBias: 1.5,
      size: [0.08, 0.16],
      gravity: -9.8,
      alpha: 0.85,
    }),
  ],
  impactGlass: [
    burst({
      layer: 'normal',
      count: 10,
      color0: 0xffffff,
      color1: 0xcfe4ff,
      life: [0.5, 0.8],
      speed: [2, 5],
      cone: 80 * DEG,
      size: [0.06, 0.05],
      gravity: -9.8,
      shape: 1,
      alpha: 0.9,
    }),
  ],
  blood: [
    burst({
      layer: 'normal',
      count: 8,
      color0: 0x8a1016,
      color1: 0x4a0508,
      life: [0.3, 0.5],
      speed: [1, 3],
      cone: 60 * DEG,
      size: [0.1, 0.28],
      gravity: -3,
      drag: 1.5,
      alpha: 0.9,
    }),
  ],
  sparks: [SPARKS],
  dust: [
    burst({
      layer: 'normal',
      count: 10,
      color0: 0xbfb5a0,
      color1: 0x8b8474,
      life: [0.8, 1.2],
      speed: [0.5, 1.5],
      emit: 'sphere',
      upBias: 0.6,
      size: [0.3, 0.9],
      drag: 1.5,
      alpha: 0.6,
    }),
  ],
  smoke: [
    burst({
      layer: 'normal',
      count: 6,
      color0: 0x7a7a7a,
      color1: 0x333333,
      life: [1.5, 1.5],
      speed: [0.3, 0.8],
      cone: 25 * DEG,
      upBias: 1.5,
      size: [0.4, 1.4],
      drag: 1.5,
      alpha: 0.55,
    }),
  ],
  fire: [
    burst({
      layer: 'additive',
      count: 8,
      color0: 0xffa030,
      color1: 0xff2a00,
      life: [0.4, 0.7],
      speed: [0.5, 1.5],
      cone: 20 * DEG,
      upBias: 2,
      size: [0.35, 0.1],
      drag: 1,
      alpha: 0.9,
    }),
  ],
  explosion: [
    burst({
      layer: 'additive',
      count: 30,
      color0: 0xffd890,
      color1: 0xff3000,
      life: [0.4, 0.8],
      speed: [6, 14],
      emit: 'sphere',
      upBias: 2,
      size: [0.8, 1.6],
      drag: 2.5,
    }),
    burst({
      layer: 'normal',
      count: 20,
      color0: 0x555555,
      color1: 0x1e1e1e,
      life: [1.2, 2.0],
      speed: [3, 7],
      emit: 'sphere',
      upBias: 3,
      size: [1.0, 2.6],
      drag: 1.5,
      alpha: 0.7,
    }),
  ],
  cashBurst: [
    burst({
      layer: 'normal',
      count: 16,
      color0: 0x4ee06a,
      color1: 0x2f9a45,
      life: [0.8, 1.2],
      speed: [2, 4],
      cone: 35 * DEG,
      size: [0.09, 0.09],
      gravity: -9.8,
      drag: 0.8,
      shape: 1,
      jitter: 0.25,
    }),
  ],
  tireSmoke: [
    burst({
      layer: 'normal',
      count: 4,
      color0: 0xe0e0e0,
      color1: 0xb0b0b0,
      life: [0.5, 0.8],
      speed: [0.5, 1],
      emit: 'sphere',
      upBias: 0.4,
      size: [0.3, 0.8],
      drag: 2,
      alpha: 0.45,
    }),
  ],
};
