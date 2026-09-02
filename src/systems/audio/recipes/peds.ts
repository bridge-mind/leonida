import type { SoundId } from '../sounds';
import { MIN_GAIN, type Synth } from '../synth';
import type { Recipe } from './types';

export interface VoiceLine {
  /** Pitch at start, middle and end (Hz). */
  f0: number;
  f1: number;
  f2: number;
  dur: number;
  /** Formant bandpass center. */
  formant: number;
  vol: number;
  type?: OscillatorType;
  q?: number;
  vibrato?: number;
  drive?: number;
  dest?: AudioNode;
}

/**
 * Text-free "voice": a sawtooth/pulse with a three-point pitch envelope through a
 * formant-ish bandpass. Screams sweep up then down, yelps are short, grunts fall.
 */
export function voiceLine(s: Synth, at: number, v: VoiceLine): void {
  const g = s.gain(MIN_GAIN, v.dest ?? s.out);
  const dest: AudioNode = v.drive ? s.shaper(v.drive, g) : g;
  const bp = s.filter('bandpass', v.formant, v.q ?? 2.5, dest);
  const o = s.ctx.createOscillator();
  o.type = v.type ?? 'sawtooth';
  const p = s.pitch;
  o.frequency.setValueAtTime(v.f0 * p, at);
  o.frequency.exponentialRampToValueAtTime(v.f1 * p, at + v.dur * 0.45);
  o.frequency.exponentialRampToValueAtTime(v.f2 * p, at + v.dur);
  o.connect(bp);
  if (v.vibrato) {
    const lfo = s.ctx.createOscillator();
    lfo.frequency.value = 6.5;
    const depth = s.ctx.createGain();
    depth.gain.value = v.vibrato;
    lfo.connect(depth);
    depth.connect(o.frequency);
    lfo.start(at);
    lfo.stop(at + v.dur + 0.1);
    s.track(lfo, at + v.dur + 0.1);
  }
  g.gain.setValueAtTime(MIN_GAIN, at);
  g.gain.linearRampToValueAtTime(v.vol, at + 0.03);
  g.gain.setValueAtTime(v.vol, at + v.dur * 0.7);
  g.gain.exponentialRampToValueAtTime(MIN_GAIN, at + v.dur + 0.08);
  o.start(at);
  o.stop(at + v.dur + 0.1);
  s.track(o, at + v.dur + 0.1);
}

/** Radio squelch: bandpassed static burst that ends with a short squelch tone. */
export function radioSquelch(s: Synth, at: number, dur: number, vol: number): void {
  s.noiseBurst({ at, dur, vol, filter: 'bandpass', freq: 1800, q: 1.5, attack: 0.005, shape: 'hold', release: 0.02 });
  s.tone({ at: at + dur, freq: 1400, dur: 0.05, vol: vol * 0.7, type: 'square', attack: 0.002 });
  s.click(at + dur + 0.02, vol * 0.5, 2500, 3);
}

function shout(f0: number, f1: number, f2: number, dur: number, formant: number, vol: number, drive = 1): Recipe {
  return { build: (s) => voiceLine(s, s.t, { f0, f1, f2, dur, formant, vol, drive: drive > 1 ? drive : undefined }) };
}

type PedId = Extract<SoundId, `ped.${string}` | `cop.${string}` | `gang.${string}` | `clerk.${string}` | `guard.${string}`>;

export const PED_RECIPES: Record<PedId, Recipe> = {
  'ped.yelp': shout(300, 520, 380, 0.25, 900, 0.45),
  'ped.scream': { build: (s) => voiceLine(s, s.t, { f0: 380, f1: 700, f2: 500, dur: 0.9, formant: 1100, vol: 0.5, vibrato: 25 }) },
  'ped.scream.female': { build: (s) => voiceLine(s, s.t, { f0: 550, f1: 950, f2: 700, dur: 0.8, formant: 1500, vol: 0.5, vibrato: 35 }) },
  'ped.grunt': shout(140, 125, 100, 0.18, 500, 0.45),
  'ped.hit': {
    build: (s) => {
      voiceLine(s, s.t, { f0: 170, f1: 210, f2: 120, dur: 0.2, formant: 650, vol: 0.45 });
      s.thump(s.t, 120, 60, 0.1, 0.4);
      s.noiseBurst({ dur: 0.05, vol: 0.3, filter: 'lowpass', freq: 1000 });
    },
  },
  'ped.death': {
    build: (s) => {
      voiceLine(s, s.t, { f0: 220, f1: 160, f2: 90, dur: 0.6, formant: 600, vol: 0.45 });
      s.thump(s.t + 0.45, 65, 40, 0.2, 0.4);
    },
  },
  'ped.knockdown': {
    build: (s) => {
      s.noiseBurst({ kind: 'brown', dur: 0.18, vol: 0.5, filter: 'lowpass', freq: 500, attack: 0.002 });
      s.thump(s.t, 75, 40, 0.2, 0.5);
      voiceLine(s, s.t + 0.03, { f0: 160, f1: 130, f2: 100, dur: 0.16, formant: 500, vol: 0.3 });
    },
  },
  'ped.getup': {
    build: (s) => {
      s.noiseBurst({ kind: 'pink', dur: 0.15, vol: 0.2, filter: 'bandpass', freq: 1500, q: 0.8, attack: 0.03 });
      s.noiseBurst({ kind: 'pink', at: s.t + 0.25, dur: 0.15, vol: 0.2, filter: 'bandpass', freq: 1800, q: 0.8, attack: 0.03 });
      voiceLine(s, s.t + 0.1, { f0: 120, f1: 140, f2: 110, dur: 0.25, formant: 450, vol: 0.25 });
    },
  },
  'ped.punch': {
    build: (s) => {
      s.whoosh(s.t, 400, 1800, 0.1, 0.25);
      s.thump(s.t + 0.06, 140, 70, 0.1, 0.5);
      s.noiseBurst({ at: s.t + 0.06, dur: 0.06, vol: 0.4, filter: 'lowpass', freq: 1000 });
    },
  },
  'ped.cower': {
    build: (s) => {
      voiceLine(s, s.t, { f0: 400, f1: 350, f2: 420, dur: 0.35, formant: 1000, vol: 0.25, vibrato: 30 });
    },
  },
  'ped.phone.call': {
    build: (s) => {
      for (let i = 0; i < 3; i++) s.tone({ at: s.t + i * 0.12, freq: 1200, dur: 0.06, vol: 0.2, shape: 'hold', release: 0.02 });
      const muffled = s.filter('bandpass', 900, 3);
      voiceLine(s, s.t + 0.5, { f0: 200, f1: 260, f2: 190, dur: 0.5, formant: 800, vol: 0.2, dest: muffled });
    },
  },
  'ped.chatter.loop': {
    build: (s) => {
      s.periodic(
        () => s.rand(0.15, 0.4),
        (at) => {
          const f = s.rand(150, 330);
          voiceLine(s, at, { f0: f, f1: f * s.rand(0.85, 1.25), f2: f * s.rand(0.7, 1.1), dur: s.rand(0.08, 0.2), formant: s.rand(600, 1200), vol: 0.14 });
        },
      );
    },
  },
  'ped.body.fall': {
    build: (s) => {
      s.noiseBurst({ kind: 'brown', dur: 0.22, vol: 0.5, filter: 'lowpass', freq: 500, attack: 0.002 });
      s.thump(s.t, 60, 35, 0.22, 0.5);
      s.noiseBurst({ at: s.t + 0.08, dur: 0.08, vol: 0.2, filter: 'bandpass', freq: 1200, q: 0.8 });
    },
  },
  'cop.shout': shout(200, 260, 180, 0.4, 800, 0.55, 1.8),
  'cop.radio': {
    build: (s) => {
      radioSquelch(s, s.t, 0.5, 0.35);
      const bp = s.filter('bandpass', 1500, 2);
      voiceLine(s, s.t + 0.06, { f0: 180, f1: 230, f2: 170, dur: 0.35, formant: 900, vol: 0.2, dest: bp });
    },
  },
  'cop.megaphone': {
    reverb: 0.25,
    build: (s) => {
      const horn = s.filter('bandpass', 1200, 2.5);
      const drive = s.shaper(3, horn);
      voiceLine(s, s.t, { f0: 190, f1: 240, f2: 200, dur: 0.28, formant: 900, vol: 0.5, dest: drive });
      voiceLine(s, s.t + 0.32, { f0: 210, f1: 260, f2: 180, dur: 0.24, formant: 1000, vol: 0.5, dest: drive });
      voiceLine(s, s.t + 0.6, { f0: 180, f1: 220, f2: 150, dur: 0.35, formant: 850, vol: 0.5, dest: drive });
    },
  },
  'cop.handcuffs': {
    build: (s) => {
      for (let i = 0; i < 6; i++) s.click(s.t + i * 0.025, 0.3, 3200, 6, 0.008);
      s.ping(s.t + 0.17, 2800, 0.12, 0.35, 25);
      s.click(s.t + 0.2, 0.4, 2400, 5);
    },
  },
  'gang.taunt': {
    build: (s) => {
      voiceLine(s, s.t, { f0: 180, f1: 240, f2: 170, dur: 0.22, formant: 700, vol: 0.45 });
      voiceLine(s, s.t + 0.26, { f0: 170, f1: 210, f2: 140, dur: 0.3, formant: 750, vol: 0.45 });
    },
  },
  'clerk.alarm': {
    build: (s) => {
      for (let i = 0; i < 5; i++) {
        s.tone({ at: s.t + i * 0.3, freq: i % 2 ? 1200 : 900, dur: 0.22, vol: 0.35, type: 'square', shape: 'hold', release: 0.03 });
      }
    },
  },
  'guard.alert': {
    build: (s) => {
      voiceLine(s, s.t, { f0: 220, f1: 300, f2: 210, dur: 0.3, formant: 850, vol: 0.5, drive: 1.5 });
      s.tone({ at: s.t + 0.35, freq: 2200, dur: 0.3, vol: 0.25, shape: 'hold', release: 0.05 });
    },
  },
};
