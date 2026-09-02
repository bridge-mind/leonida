import type { NoiseKind } from '../noise';
import type { Synth } from '../synth';
import { voiceLine } from './peds';
import type { RecipeTable } from './types';

interface Bed {
  kind: NoiseKind;
  filter: BiquadFilterType;
  freq: number;
  q?: number;
  vol: number;
  /** Slow gain LFO (swell). */
  lfoHz?: number;
  lfoDepth?: number;
  /** Optional filter-frequency LFO. */
  filterLfoHz?: number;
  filterLfoDepth?: number;
}

/** Filtered noise with slow LFOs — the base of every ambient bed. Returns the bed gain for extras. */
function bed(s: Synth, b: Bed): GainNode {
  const g = s.gain(b.vol);
  const f = s.filter(b.filter, b.freq, b.q ?? 0.7, g);
  s.noiseSrc(b.kind, f);
  if (b.lfoHz) s.lfo(b.lfoHz, (b.lfoDepth ?? 0.3) * b.vol, g.gain);
  if (b.filterLfoHz) s.lfo(b.filterLfoHz, b.filterLfoDepth ?? b.freq * 0.4, f.frequency, 'triangle');
  return g;
}

function hum(s: Synth, freq: number, vol: number, cutoff: number): void {
  const lp = s.filter('lowpass', cutoff, 1);
  const g = s.gain(vol, lp);
  s.osc('sawtooth', freq, g);
  s.osc('sine', freq * 2, g);
}

function birdChirp(s: Synth, at: number, vol: number): void {
  const f = s.rand(2200, 3600);
  s.tone({ at, freq: f, glideTo: f * 1.35, dur: 0.07, vol, attack: 0.01 });
  s.tone({ at: at + 0.09, freq: f * 1.1, glideTo: f * 0.8, dur: 0.08, vol: vol * 0.8, attack: 0.01 });
}

function seagull(s: Synth, at: number, vol: number): void {
  voiceLine(s, at, { f0: 1200, f1: 1650, f2: 1050, dur: 0.35, formant: 1800, vol, q: 3 });
}

function distantHonk(s: Synth, at: number, vol: number): void {
  const lp = s.filter('lowpass', 800, 1);
  s.tone({ at, freq: 440, dur: 0.3, vol, type: 'square', shape: 'hold', attack: 0.02, dest: lp });
  s.tone({ at, freq: 554, dur: 0.3, vol: vol * 0.8, type: 'square', shape: 'hold', attack: 0.02, dest: lp });
}

const HABANA_SCALE = [261.6, 293.7, 329.6, 392, 440, 523.3];

export const AMBIENT_RECIPES: RecipeTable<'ambient.'> = {
  'ambient.beach.loop': {
    build: (s) => {
      // Waves: pink noise swelling at 0.1 Hz with a brighter hiss out of phase.
      bed(s, { kind: 'pink', filter: 'lowpass', freq: 900, vol: 0.5, lfoHz: 0.1, lfoDepth: 0.5 });
      bed(s, { kind: 'white', filter: 'bandpass', freq: 3000, q: 0.6, vol: 0.12, lfoHz: 0.13, lfoDepth: 0.7 });
      s.periodic(
        () => s.rand(6, 15),
        (at) => {
          for (let i = 0; i < 3; i++) seagull(s, at + i * s.rand(0.35, 0.5), 0.06);
        },
      );
    },
  },
  'ambient.city.loop': {
    build: (s) => {
      bed(s, { kind: 'brown', filter: 'lowpass', freq: 260, vol: 0.5, lfoHz: 0.07, lfoDepth: 0.2 });
      hum(s, 60, 0.05, 200);
      s.periodic(
        () => s.rand(4, 12),
        (at) => distantHonk(s, at, s.rand(0.04, 0.09)),
      );
    },
  },
  'ambient.habana.loop': {
    build: (s) => {
      bed(s, { kind: 'brown', filter: 'lowpass', freq: 360, vol: 0.45, lfoHz: 0.06, lfoDepth: 0.2 });
      hum(s, 55, 0.04, 260);
      const music = s.filter('lowpass', 900, 1);
      s.periodic(
        () => 0.5,
        (at) => s.tone({ at, freq: HABANA_SCALE[Math.floor(Math.random() * HABANA_SCALE.length)], dur: 0.35, vol: 0.045, type: 'triangle', dest: music }),
      );
    },
  },
  'ambient.suburb.loop': {
    build: (s) => {
      bed(s, { kind: 'pink', filter: 'lowpass', freq: 600, vol: 0.28, lfoHz: 0.09, lfoDepth: 0.25 });
      s.periodic(
        () => s.rand(0.4, 2.2),
        (at) => birdChirp(s, at, s.rand(0.04, 0.09)),
      );
    },
  },
  'ambient.docks.loop': {
    build: (s) => {
      bed(s, { kind: 'brown', filter: 'lowpass', freq: 200, vol: 0.55, lfoHz: 0.05, lfoDepth: 0.2 });
      hum(s, 50, 0.07, 180);
      s.periodic(
        () => s.rand(3, 9),
        (at) => {
          s.ping(at, s.rand(700, 1100), 0.4, 0.14, 15);
          s.thump(at, 120, 70, 0.15, 0.15);
        },
      );
    },
  },
  'ambient.swamp.loop': {
    build: (s) => {
      // Insects: high resonant noise with a fast tremolo; frogs: low square pulses.
      const insects = bed(s, { kind: 'white', filter: 'bandpass', freq: 5200, q: 8, vol: 0.12 });
      s.lfo(25, 0.08, insects.gain, 'sine');
      bed(s, { kind: 'brown', filter: 'lowpass', freq: 240, vol: 0.3, lfoHz: 0.08, lfoDepth: 0.3 });
      const frogLp = s.filter('lowpass', 600, 1);
      s.periodic(
        () => s.rand(0.8, 2.5),
        (at) => s.tone({ at, freq: 180, glideTo: 120, dur: 0.12, vol: 0.08, type: 'square', dest: frogLp }),
      );
    },
  },
  'ambient.night.crickets.loop': {
    build: (s) => {
      const g = s.gain(0.35);
      const bp = s.filter('bandpass', 4500, 10, g);
      s.noiseSrc('white', bp);
      const chirpGain = s.gain(0.001, g);
      s.osc('sine', 4200, chirpGain);
      s.lfo(30, 0.3, g.gain, 'square');
      s.periodic(
        () => s.rand(0.3, 0.6),
        (at) => {
          chirpGain.gain.setValueAtTime(0.001, at);
          chirpGain.gain.linearRampToValueAtTime(0.5, at + 0.05);
          chirpGain.gain.exponentialRampToValueAtTime(0.001, at + 0.22);
        },
      );
    },
  },
  'ambient.club.loop': {
    build: (s) => {
      // Behind a wall: everything through a 200 Hz lowpass, kick at 120 BPM.
      const wall = s.filter('lowpass', 200, 1);
      const bassGain = s.gain(0.25, wall);
      s.osc('sawtooth', 55, bassGain);
      bed(s, { kind: 'brown', filter: 'lowpass', freq: 150, vol: 0.2 });
      s.periodic(
        () => 0.5,
        (at) => s.tone({ at, freq: 65, glideTo: 40, dur: 0.22, vol: 0.9, attack: 0.005, dest: wall }),
      );
    },
  },
  'ambient.interior.loop': {
    build: (s) => {
      bed(s, { kind: 'brown', filter: 'lowpass', freq: 300, vol: 0.25, lfoHz: 0.05, lfoDepth: 0.15 });
      hum(s, 60, 0.03, 150);
    },
  },
  'ambient.prison.loop': {
    build: (s) => {
      bed(s, { kind: 'brown', filter: 'lowpass', freq: 400, vol: 0.3, lfoHz: 0.06, lfoDepth: 0.2 });
      hum(s, 60, 0.04, 200);
      s.periodic(
        () => s.rand(5, 12),
        (at) => {
          s.ping(at, s.rand(600, 900), 0.6, 0.12, 12);
          s.thump(at, 100, 60, 0.2, 0.12);
        },
      );
    },
  },
  'ambient.traffic.loop': {
    build: (s) => {
      bed(s, { kind: 'pink', filter: 'lowpass', freq: 500, vol: 0.4, lfoHz: 0.2, lfoDepth: 0.3 });
      s.periodic(
        () => s.rand(2, 5),
        (at) => s.noiseBurst({ kind: 'pink', at, dur: 1, vol: 0.2, filter: 'lowpass', freq: 400, freqTo: 1200, q: 0.8, attack: 0.5, sweepTime: 0.5 }),
      );
    },
  },
  'ambient.crowd.loop': {
    build: (s) => {
      bed(s, { kind: 'pink', filter: 'lowpass', freq: 1200, vol: 0.18 });
      const murmur = s.filter('bandpass', 800, 1.5);
      s.periodic(
        () => s.rand(0.08, 0.2),
        (at) => {
          const f = s.rand(150, 350);
          voiceLine(s, at, { f0: f, f1: f * s.rand(0.85, 1.25), f2: f * s.rand(0.7, 1.1), dur: s.rand(0.08, 0.2), formant: s.rand(600, 1200), vol: 0.1, dest: murmur });
        },
      );
    },
  },
  'ambient.rain.loop': {
    build: (s) => {
      bed(s, { kind: 'white', filter: 'lowpass', freq: 3500, vol: 0.3, lfoHz: 0.12, lfoDepth: 0.2 });
      bed(s, { kind: 'pink', filter: 'lowpass', freq: 900, vol: 0.25 });
      s.periodic(
        () => s.rand(0.1, 0.5),
        (at) => s.tone({ at, freq: s.rand(1800, 4200), glideTo: 1200, dur: 0.04, vol: 0.06 }),
      );
    },
  },
  'ambient.wind.loop': {
    build: (s) => {
      bed(s, { kind: 'pink', filter: 'bandpass', freq: 500, q: 0.5, vol: 0.4, lfoHz: 0.15, lfoDepth: 0.6, filterLfoHz: 0.11, filterLfoDepth: 300 });
    },
  },
  'ambient.thunder': {
    reverb: 0.5,
    build: (s) => {
      s.noiseBurst({ kind: 'brown', dur: 3, vol: 0.8, filter: 'lowpass', freq: 400, freqTo: 80, q: 0.7, attack: 0.05, sweepTime: 3 });
      s.tone({ freq: 32, glideTo: 22, dur: 2, vol: 0.6, attack: 0.02 });
      s.noiseBurst({ kind: 'brown', at: s.t + 0.6, dur: 2.5, vol: 0.4, filter: 'lowpass', freq: 200, freqTo: 60, attack: 0.3 });
    },
  },
  'ambient.seagulls': {
    build: (s) => {
      for (let i = 0; i < 3; i++) seagull(s, s.t + i * s.rand(0.35, 0.5), 0.3);
    },
  },
  'ambient.gator.growl': {
    build: (s) => {
      const lp = s.filter('lowpass', 300, 1);
      const growl = s.tone({ freq: 60, glideTo: 48, dur: 1.2, vol: 0.5, type: 'sawtooth', shape: 'hold', attack: 0.1, release: 0.3, dest: lp });
      s.lfo(25, 0.5, growl.detune, 'square');
      s.noiseBurst({ kind: 'brown', dur: 1.2, vol: 0.3, filter: 'lowpass', freq: 400, attack: 0.1, shape: 'hold', release: 0.3 });
    },
  },
  'ambient.boat.horn': {
    reverb: 0.5,
    build: (s) => {
      const lp = s.filter('lowpass', 500, 1);
      s.tone({ freq: 110, dur: 1.5, vol: 0.35, type: 'sawtooth', shape: 'hold', attack: 0.15, release: 0.4, dest: lp });
      s.tone({ freq: 165, dur: 1.5, vol: 0.25, type: 'sawtooth', shape: 'hold', attack: 0.15, release: 0.4, dest: lp });
    },
  },
};
