import { describe, expect, it } from 'vitest';
import { VEHICLES } from '../../src/data/vehicles';
import { rpmFromSpeed, stepHandling, wrapAngle, type HandlingState } from '../../src/systems/vehicles/handling';
import { signalAt } from '../../src/systems/traffic/TrafficLights';
import { TrafficConfig } from '../../src/systems/traffic/config';

const DT = 1 / 60;

function fresh(): HandlingState {
  return { vf: 0, vl: 0, yawRate: 0 };
}

/** Vehicle.fixedUpdate re-projects the world velocity into the new body frame each step; mirror that. */
function rotateFrame(s: HandlingState, dt: number): void {
  const dh = s.yawRate * dt;
  const cos = Math.cos(dh);
  const sin = Math.sin(dh);
  const vf = s.vf * cos + s.vl * sin;
  const vl = -s.vf * sin + s.vl * cos;
  s.vf = vf;
  s.vl = vl;
}

function controls(over: Partial<{ throttle: number; brake: number; steer: number; handbrake: boolean }>) {
  return { throttle: 0, brake: 0, steer: 0, handbrake: false, horn: false, ...over };
}

/** Integrates the body frame into world space so we can measure turning radii. */
function drive(id: keyof typeof VEHICLES, seconds: number, c: ReturnType<typeof controls>, s = fresh(), start = { x: 0, z: 0, h: 0 }) {
  const a = VEHICLES[id];
  let { x, z, h } = start;
  const steps = Math.round(seconds / DT);
  for (let i = 0; i < steps; i++) {
    stepHandling(s, a, c, DT, true);
    rotateFrame(s, DT);
    h = wrapAngle(h + s.yawRate * DT);
    const fx = Math.sin(h);
    const fz = Math.cos(h);
    x += (fx * s.vf + fz * s.vl) * DT;
    z += (fz * s.vf - fx * s.vl) * DT;
  }
  return { s, x, z, h };
}

describe('arcade handling', () => {
  it('a sedan reaches > 12 m/s after 4 s of full throttle', () => {
    const { s } = drive('sedan', 4, controls({ throttle: 1 }));
    expect(s.vf).toBeGreaterThan(12);
  });

  it('every archetype settles within 5% of its top speed', () => {
    for (const a of Object.values(VEHICLES)) {
      const { s } = drive(a.id, 60, controls({ throttle: 1 }));
      expect(s.vf, a.id).toBeGreaterThan(a.topSpeed * 0.95);
      expect(s.vf, a.id).toBeLessThanOrEqual(a.topSpeed + 0.01);
    }
  });

  it('the sports car is the fastest and tops out near 55 m/s', () => {
    const { s } = drive('sports', 60, controls({ throttle: 1 }));
    expect(s.vf).toBeGreaterThan(52);
    for (const a of Object.values(VEHICLES)) expect(a.topSpeed).toBeLessThanOrEqual(VEHICLES.sports.topSpeed);
  });

  it('reverse is limited and brakes stop the car', () => {
    const { s } = drive('sedan', 10, controls({ brake: 1 }));
    expect(s.vf).toBeLessThan(0);
    expect(Math.abs(s.vf)).toBeLessThanOrEqual(VEHICLES.sedan.reverseSpeed + 0.01);
    const moving = drive('sedan', 3, controls({ throttle: 1 })).s;
    const stopped = drive('sedan', 5, controls({ brake: 1 }), moving).s;
    expect(Math.abs(stopped.vf)).toBeLessThan(VEHICLES.sedan.reverseSpeed + 0.01);
  });

  it('a sedan turns a 90° corner at ~15 m/s within a city block radius', () => {
    const s: HandlingState = { vf: 15, vl: 0, yawRate: 0 };
    let t = 0;
    let x = 0;
    let z = 0;
    let h = 0;
    const c = controls({ steer: -1, throttle: 0.4 });
    while (Math.abs(h) < Math.PI / 2 && t < 10) {
      stepHandling(s, VEHICLES.sedan, c, DT, true);
      rotateFrame(s, DT);
      h += s.yawRate * DT;
      x += (Math.sin(h) * s.vf + Math.cos(h) * s.vl) * DT;
      z += (Math.cos(h) * s.vf - Math.sin(h) * s.vl) * DT;
      t += DT;
    }
    expect(t).toBeLessThan(2.5);
    // Corner "radius" from the chord of a quarter turn: keep it under two lane widths of drift.
    const radius = Math.hypot(x, z) / Math.SQRT2;
    expect(radius).toBeGreaterThan(5);
    expect(radius).toBeLessThan(18);
    expect(s.vf).toBeGreaterThan(9);
  });

  it('handbrake + steer at speed spins faster than 0.8 rad/s and slides', () => {
    const s: HandlingState = { vf: 18, vl: 0, yawRate: 0 };
    const c = controls({ steer: 1, handbrake: true });
    let peak = 0;
    let slide = 0;
    for (let i = 0; i < 60; i++) {
      stepHandling(s, VEHICLES.sedan, c, DT, true);
      rotateFrame(s, DT);
      peak = Math.max(peak, Math.abs(s.yawRate));
      slide = Math.max(slide, Math.abs(s.vl));
    }
    expect(peak).toBeGreaterThan(0.8);
    expect(slide).toBeGreaterThan(2);
  });

  it('muscle keeps more sideways slip than the sedan (tail-happy)', () => {
    const run = (id: 'sedan' | 'muscle') => {
      const s: HandlingState = { vf: 20, vl: 0, yawRate: 0 };
      const c = controls({ steer: 1, throttle: 1 });
      let slide = 0;
      for (let i = 0; i < 90; i++) {
        stepHandling(s, VEHICLES[id], c, DT, true);
        rotateFrame(s, DT);
        slide = Math.max(slide, Math.abs(s.vl));
      }
      return slide;
    };
    expect(run('muscle')).toBeGreaterThan(run('sedan'));
  });

  it('an uncontrolled car coasts to a stop and airborne cars keep their vector', () => {
    const s: HandlingState = { vf: 10, vl: 1, yawRate: 0.5 };
    for (let i = 0; i < 1800; i++) stepHandling(s, VEHICLES.van, controls({}), DT, true);
    expect(s.vf).toBeLessThan(0.1);
    expect(Math.abs(s.vl)).toBeLessThan(0.1);
    expect(Math.abs(s.yawRate)).toBeLessThan(0.01);
    const air: HandlingState = { vf: 20, vl: 3, yawRate: 0 };
    stepHandling(air, VEHICLES.van, controls({ steer: 1, throttle: 1 }), DT, false);
    expect(air.vf).toBeGreaterThan(19.5);
    expect(air.vl).toBe(3);
    expect(air.yawRate).toBe(0);
  });

  it('rpm rises through the gears and stays in [0, 1]', () => {
    let last = -1;
    let drops = 0;
    for (let v = 0; v <= VEHICLES.sedan.topSpeed; v += 0.5) {
      const r = rpmFromSpeed(v, VEHICLES.sedan.topSpeed, 1);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(1);
      if (r < last - 0.05) drops++;
      last = r;
    }
    expect(drops).toBeGreaterThanOrEqual(2);
  });
});

describe('traffic light clock', () => {
  const L = TrafficConfig.lights;
  const out = { green: null as 'ns' | 'ew' | null, yellow: false, timeToRed: Infinity };

  it('alternates axes with an all-red gap and staggers by phase', () => {
    signalAt(1, 0, out);
    expect(out.green).toBe('ns');
    signalAt(1, 1, out);
    expect(out.green).toBe('ew');
    signalAt(L.greenSeconds + 0.5, 0, out);
    expect(out.green).toBeNull();
    signalAt(L.greenSeconds + L.allRedSeconds + 1, 0, out);
    expect(out.green).toBe('ew');
    signalAt(2 * (L.greenSeconds + L.allRedSeconds) + 1, 0, out);
    expect(out.green).toBe('ns');
  });

  it('flags yellow just before red', () => {
    signalAt(L.greenSeconds - L.yellowSeconds / 2, 0, out);
    expect(out.green).toBe('ns');
    expect(out.yellow).toBe(true);
    signalAt(1, 0, out);
    expect(out.yellow).toBe(false);
  });
});
