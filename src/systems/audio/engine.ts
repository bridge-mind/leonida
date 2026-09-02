import { AudioConfig } from './config';
import type { Synth } from './synth';

const E = AudioConfig.engine;

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Gear 1..5 from ground speed. */
export function gearFor(speedMs: number): number {
  const s = Math.max(0, speedMs);
  let gear = 1;
  for (let i = 1; i < E.gearSpeeds.length; i++) if (s >= E.gearSpeeds[i]) gear = i + 1;
  return gear;
}

/** 0..1 position within the current gear's speed band (top gear saturates). */
export function gearFraction(speedMs: number): number {
  const gear = gearFor(speedMs);
  const lo = E.gearSpeeds[gear - 1];
  const hi = gear < E.gearSpeeds.length ? E.gearSpeeds[gear] : lo * 1.6;
  return clamp01((Math.max(0, speedMs) - lo) / Math.max(0.01, hi - lo));
}

/** Oscillator frequency for an rpm fraction: idle at 0, idle × (1 + rpmRange) at 1. Monotonic. */
export function engineHz(baseHz: number, rpm01: number): number {
  return baseHz * (1 + E.rpmRange * clamp01(rpm01));
}

/**
 * Turns the caller's rpm01/speed into an "effective" rpm with a gearbox feel:
 * the in-gear speed fraction is blended in (so rpm climbs and drops with each
 * gear) and every gear change dips the rpm briefly.
 */
export class EngineModel {
  gear = 1;
  private shiftUntil = -1;

  effectiveRpm(rpm01: number, speedMs: number, now: number): number {
    const gear = gearFor(speedMs);
    if (gear !== this.gear) {
      this.gear = gear;
      this.shiftUntil = now + E.shiftSeconds;
    }
    const rolling = speedMs > 1 ? gearFraction(speedMs) : rpm01;
    let rpm = clamp01(rpm01) * (1 - E.gearBlend) + rolling * E.gearBlend;
    if (now < this.shiftUntil) rpm *= E.shiftDip;
    return clamp01(rpm);
  }
}

export interface EngineVariant {
  baseHz: number;
  wave: OscillatorType;
  wave2: OscillatorType;
  /** Second oscillator ratio (slight detune gives the beating rumble). */
  detuneRatio: number;
  subRatio: number;
  subGain: number;
  noise: 'white' | 'pink' | 'brown';
  noiseFreq: number;
  noiseGain: number;
  cutoffIdle: number;
  cutoffRange: number;
  drive: number;
  gainIdle: number;
}

export const ENGINE_VARIANTS = {
  car: { baseHz: 55, wave: 'sawtooth', wave2: 'sawtooth', detuneRatio: 1.012, subRatio: 0.5, subGain: 0.5, noise: 'brown', noiseFreq: 320, noiseGain: 0.18, cutoffIdle: 320, cutoffRange: 2200, drive: 1.8, gainIdle: 0.45 },
  motorcycle: { baseHz: 95, wave: 'square', wave2: 'sawtooth', detuneRatio: 1.008, subRatio: 0.5, subGain: 0.3, noise: 'white', noiseFreq: 900, noiseGain: 0.12, cutoffIdle: 600, cutoffRange: 3200, drive: 2.4, gainIdle: 0.4 },
  boat: { baseHz: 38, wave: 'sawtooth', wave2: 'triangle', detuneRatio: 1.02, subRatio: 0.5, subGain: 0.7, noise: 'brown', noiseFreq: 220, noiseGain: 0.3, cutoffIdle: 240, cutoffRange: 900, drive: 1.6, gainIdle: 0.5 },
  airboat: { baseHz: 60, wave: 'sawtooth', wave2: 'square', detuneRatio: 1.015, subRatio: 0.5, subGain: 0.4, noise: 'white', noiseFreq: 1400, noiseGain: 0.45, cutoffIdle: 800, cutoffRange: 2600, drive: 2, gainIdle: 0.5 },
} as const satisfies Record<string, EngineVariant>;

/** Two detuned oscillators + sub + intake noise → lowpass → saturation, driven by `s.onEngine`. */
export function buildEngine(s: Synth, v: EngineVariant): void {
  const model = new EngineModel();
  const level = s.gain(v.gainIdle);
  const drive = s.shaper(v.drive, level);
  const lp = s.filter('lowpass', v.cutoffIdle, 0.9, drive);
  const o1 = s.osc(v.wave, v.baseHz, lp);
  const o2 = s.osc(v.wave2, v.baseHz * v.detuneRatio, lp);
  const subGain = s.gain(v.subGain, lp);
  const sub = s.osc('triangle', v.baseHz * v.subRatio, subGain);
  const intake = s.gain(v.noiseGain * 0.4, lp);
  const intakeBp = s.filter('bandpass', v.noiseFreq, 0.8, intake);
  s.noiseSrc(v.noise, intakeBp);
  // Idle lope: slow amplitude wobble that the rpm ramp smooths away.
  s.lfo(7, 0.04, level.gain);

  s.onEngine = (rpm01, throttle01, speedMs, at) => {
    const rpm = model.effectiveRpm(rpm01, speedMs, at);
    const hz = engineHz(v.baseHz, rpm) * s.currentPitch;
    const r = E.ramp;
    o1.frequency.setTargetAtTime(hz, at, r);
    o2.frequency.setTargetAtTime(hz * v.detuneRatio, at, r);
    sub.frequency.setTargetAtTime(hz * v.subRatio, at, r);
    lp.frequency.setTargetAtTime(v.cutoffIdle + rpm * v.cutoffRange + throttle01 * 600, at, r);
    intake.gain.setTargetAtTime(v.noiseGain * (0.4 + throttle01 * 0.6 + rpm * 0.3), at, r);
    level.gain.setTargetAtTime(v.gainIdle + throttle01 * 0.25 + rpm * 0.3, at, r);
  };
}

/** Rotor: pulsed brown noise (12 Hz LFO) + thin turbine whine; the LFO rate follows rpm. */
export function buildHelicopter(s: Synth): void {
  const model = new EngineModel();
  const level = s.gain(0.6);
  const lp = s.filter('lowpass', 520, 0.7, level);
  const rotorGain = s.gain(0.5, lp);
  s.noiseSrc('brown', rotorGain);
  const rotorLfo = s.lfo(12, 0.5, rotorGain.gain, 'square');
  const whineGain = s.gain(0.06, level);
  const whineBp = s.filter('bandpass', 1400, 4, whineGain);
  const whine = s.osc('sawtooth', 140, whineBp);
  const chop = s.gain(0.2, level);
  const chopBp = s.filter('bandpass', 90, 2, chop);
  s.noiseSrc('white', chopBp);
  s.lfo(12, 0.2, chop.gain, 'square');

  s.onEngine = (rpm01, throttle01, speedMs, at) => {
    const rpm = model.effectiveRpm(rpm01, speedMs, at);
    const r = E.ramp;
    rotorLfo.frequency.setTargetAtTime((9 + rpm * 6) * s.currentPitch, at, r);
    whine.frequency.setTargetAtTime((120 + rpm * 90) * s.currentPitch, at, r);
    lp.frequency.setTargetAtTime(420 + rpm * 700 + throttle01 * 300, at, r);
    level.gain.setTargetAtTime(0.5 + rpm * 0.3 + throttle01 * 0.2, at, r);
  };
}
