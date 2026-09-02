import { buildEngine, buildHelicopter, ENGINE_VARIANTS } from '../engine';
import type { Synth } from '../synth';
import { buildFire, explosion } from './explosions';
import type { Recipe, RecipeTable } from './types';

/** Two-tone square horn (A4 + C#5) through a lowpass, fast attack. */
function horn(dur: number): Recipe {
  return {
    build: (s) => {
      const lp = s.filter('lowpass', 2400, 1);
      s.tone({ freq: 440, dur, vol: 0.35, type: 'square', shape: 'hold', attack: 0.01, release: 0.06, dest: lp });
      s.tone({ freq: 554, dur, vol: 0.3, type: 'square', shape: 'hold', attack: 0.01, release: 0.06, dest: lp });
    },
  };
}

function crash(dur: number, thumpHz: number, vol: number, metal: boolean): Recipe {
  return {
    reverb: 0.2,
    build: (s) => {
      s.noiseBurst({ dur, vol, filter: 'lowpass', freq: 3200, freqTo: 250, q: 0.8, attack: 0.002, sweepTime: dur });
      s.thump(s.t, thumpHz, thumpHz * 0.5, dur * 0.8, vol * 0.9);
      if (metal) {
        s.ping(s.t + 0.01, s.rand(1200, 1800), 0.3, vol * 0.5, 14);
        s.ping(s.t + 0.05, s.rand(2000, 3000), 0.2, vol * 0.35, 18);
      }
    },
  };
}

/** Wail 600→1200 Hz on a 0.8 Hz LFO; `onSiren` fades the whole thing. */
function buildSiren(s: Synth, at = s.t, dur = 0): void {
  const level = s.gain(0.5);
  const bp = s.filter('bandpass', 1400, 0.6, level);
  const saw = s.osc('sawtooth', 900, bp, 0, at);
  const sqGain = s.gain(0.4, bp);
  const sq = s.osc('square', 900, sqGain, 0, at);
  s.lfo(0.8, 300, saw.frequency, 'triangle', at);
  s.lfo(0.8, 300, sq.frequency, 'triangle', at);
  s.onSiren = (on, when) => level.gain.setTargetAtTime(on ? 0.5 : 0.0001, when, 0.15);
  if (dur > 0) {
    level.gain.setValueAtTime(0.5, at + dur - 0.15);
    level.gain.exponentialRampToValueAtTime(0.0001, at + dur);
    s.stopAt(at + dur + 0.02);
  }
}

export const VEHICLE_RECIPES: RecipeTable<'vehicle.'> = {
  'vehicle.engine.loop': { build: (s) => buildEngine(s, ENGINE_VARIANTS.car) },
  'vehicle.engine.motorcycle.loop': { build: (s) => buildEngine(s, ENGINE_VARIANTS.motorcycle) },
  'vehicle.engine.boat.loop': { build: (s) => buildEngine(s, ENGINE_VARIANTS.boat) },
  'vehicle.engine.airboat.loop': {
    build: (s) => {
      buildEngine(s, ENGINE_VARIANTS.airboat);
      // Prop wash: amplitude pulsing on the whole voice.
      s.lfo(38, 0.15, s.out.gain, 'sine');
    },
  },
  'vehicle.engine.helicopter.loop': { build: (s) => buildHelicopter(s) },
  'vehicle.engine.start': {
    build: (s) => {
      // Starter motor chatter then the engine catching.
      const lp = s.filter('lowpass', 700, 1);
      const starter = s.tone({ freq: 26, glideTo: 48, dur: 0.6, vol: 0.5, type: 'sawtooth', shape: 'hold', attack: 0.02, release: 0.05, dest: lp });
      s.lfo(30, 0.5, starter.detune, 'square');
      s.noiseBurst({ dur: 0.6, vol: 0.15, filter: 'bandpass', freq: 400, q: 1, shape: 'hold' });
      s.thump(s.t + 0.62, 90, 55, 0.25, 0.6);
      s.tone({ at: s.t + 0.65, freq: 70, glideTo: 55, dur: 0.5, vol: 0.4, type: 'sawtooth', dest: lp });
    },
  },
  'vehicle.engine.stop': {
    build: (s) => {
      const lp = s.filter('lowpass', 500, 1);
      s.tone({ freq: 60, glideTo: 18, dur: 0.65, vol: 0.45, type: 'sawtooth', dest: lp });
      s.click(s.t + 0.55, 0.3, 900, 3, 0.03);
    },
  },
  'vehicle.skid.loop': {
    build: (s) => {
      const g = s.gain(0.5);
      const bp = s.filter('bandpass', 1800, 2.2, g);
      s.noiseSrc('white', bp);
      s.lfo(3.1, 220, bp.frequency, 'sine');
      s.lfo(0.7, 0.1, g.gain, 'sine');
    },
  },
  'vehicle.brake.squeal': {
    build: (s) => {
      s.noiseBurst({ dur: 0.5, vol: 0.35, filter: 'bandpass', freq: 3500, q: 14, attack: 0.03, shape: 'hold', release: 0.08 });
      s.tone({ freq: 3500, glideTo: 3100, dur: 0.5, vol: 0.12, shape: 'hold', attack: 0.04, release: 0.1 });
    },
  },
  'vehicle.horn': horn(0.6),
  'vehicle.horn.short': horn(0.22),
  'vehicle.horn.long': horn(1.6),
  'vehicle.siren.loop': { build: (s) => buildSiren(s) },
  'vehicle.siren.burst': { build: (s) => buildSiren(s, s.t, 1.3) },
  'vehicle.crash.light': crash(0.15, 90, 0.5, false),
  'vehicle.crash.medium': crash(0.3, 65, 0.8, true),
  'vehicle.crash.heavy': crash(0.55, 45, 1, true),
  'vehicle.scrape.loop': {
    build: (s) => {
      const g = s.gain(0.4);
      const bp = s.filter('bandpass', 2500, 3, g);
      s.noiseSrc('white', bp);
      s.lfo(7, 0.25, g.gain, 'sawtooth');
      const ring = s.gain(0.15, g);
      const ringBp = s.filter('bandpass', 1200, 25, ring);
      s.noiseSrc('white', ringBp);
    },
  },
  'vehicle.door.open': {
    build: (s) => {
      s.click(s.t, 0.5, 600, 4, 0.03);
      s.noiseBurst({ at: s.t + 0.03, dur: 0.12, vol: 0.2, filter: 'lowpass', freq: 1200, attack: 0.02 });
    },
  },
  'vehicle.door.close': {
    build: (s) => {
      s.thump(s.t, 150, 80, 0.09, 0.6);
      s.click(s.t + 0.01, 0.4, 900, 3, 0.02);
      s.noiseBurst({ dur: 0.06, vol: 0.3, filter: 'lowpass', freq: 800 });
    },
  },
  'vehicle.window.smash': {
    build: (s) => {
      s.noiseBurst({ dur: 0.08, vol: 0.5, filter: 'highpass', freq: 3000, attack: 0.001 });
      for (let i = 0; i < 6; i++) s.ping(s.t + s.rand(0, 0.25), s.rand(3000, 7000), 0.12, 0.3, 30);
      s.thump(s.t, 200, 120, 0.08, 0.3);
    },
  },
  'vehicle.tire.pop': {
    build: (s) => {
      s.thump(s.t, 90, 50, 0.08, 0.9);
      s.noiseBurst({ dur: 0.05, vol: 0.7, filter: 'lowpass', freq: 3000, attack: 0.001 });
      s.noiseBurst({ at: s.t + 0.05, dur: 0.45, vol: 0.25, filter: 'bandpass', freq: 4000, q: 1, attack: 0.01 });
    },
  },
  'vehicle.tire.flat.loop': {
    build: (s) => {
      s.periodic(
        () => 0.34 / s.currentPitch,
        (at) => {
          s.thump(at, 70, 45, 0.09, 0.5);
          s.noiseBurst({ at, dur: 0.04, vol: 0.25, filter: 'lowpass', freq: 600 });
        },
      );
    },
  },
  'vehicle.alarm.loop': {
    build: (s) => {
      let hi = false;
      const lp = s.filter('lowpass', 3000, 1);
      s.periodic(
        () => 0.4,
        (at) => {
          hi = !hi;
          s.tone({ at, freq: hi ? 1000 : 800, dur: 0.36, vol: 0.35, type: 'square', shape: 'hold', release: 0.03, dest: lp });
        },
      );
    },
  },
  'vehicle.fire.loop': { build: (s) => buildFire(s, 1.1) },
  'vehicle.smoke.hiss': {
    build: (s) => s.noiseBurst({ kind: 'pink', dur: 0.8, vol: 0.3, filter: 'bandpass', freq: 3000, q: 0.7, attack: 0.05, shape: 'hold', release: 0.3 }),
  },
  'vehicle.explosion': {
    reverb: 0.5,
    build: (s) => {
      explosion(s, { dur: 1.7, lpFrom: 3200, lpTo: 150, boomHz: 48, boomDur: 1.2, vol: 1, crack: true, rumbleDelay: 0.3, rumbleDur: 2 });
      for (let i = 0; i < 5; i++) s.ping(s.t + s.rand(0.2, 1), s.rand(900, 2200), 0.2, 0.25, 15);
    },
  },
  'vehicle.reverse.beep': { build: (s) => s.tone({ freq: 1000, dur: 0.22, vol: 0.3, shape: 'hold', attack: 0.005, release: 0.02 }) },
  'vehicle.gear.shift': {
    build: (s) => {
      s.click(s.t, 0.3, 1200, 3, 0.02);
      s.tone({ freq: 220, glideTo: 120, dur: 0.08, vol: 0.25, type: 'triangle' });
    },
  },
  'vehicle.wheel.hydroplane': {
    build: (s) => s.noiseBurst({ dur: 0.45, vol: 0.4, filter: 'lowpass', freq: 1500, freqTo: 3800, q: 0.8, attack: 0.03 }),
  },
};
