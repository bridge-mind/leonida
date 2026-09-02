import { AudioConfig } from '../config';
import type { Synth } from '../synth';
import type { RecipeTable } from './types';

/** Four-bar progression (Am – F – C – G) as [bass root, chord tones]. */
const BARS: readonly { root: number; chord: readonly number[] }[] = [
  { root: 55, chord: [220, 261.6, 329.6] },
  { root: 43.65, chord: [174.6, 220, 261.6] },
  { root: 65.4, chord: [261.6, 329.6, 392] },
  { root: 49, chord: [196, 246.9, 293.7] },
];
const BASS_STEPS = [0, 2, 3, 6, 8, 10, 11, 14];
const KICK_STEPS = [0, 4, 8, 12];
const SNARE_STEPS = [4, 12];
const HAT_STEPS = [2, 6, 10, 14];
const ARP_STEPS = [0, 3, 6, 9, 12, 15];

/** A small 16-step sequencer: bass saw, detuned pad, kick, snare, hats and a delayed arpeggio. */
function buildStation(s: Synth): void {
  const R = AudioConfig.radio;
  const stepSeconds = 60 / R.bpm / 4;
  const master = s.gain(R.volume);
  const bassLp = s.filter('lowpass', 650, 1.5, master);
  const padLp = s.filter('lowpass', 1300, 0.8, master);
  const arpDelay = s.delay(stepSeconds * 3, 0.35, master);
  const arpLp = s.filter('lowpass', 2500, 1, arpDelay);
  const drums = s.gain(0.9, master);
  let step = 0;
  let bar = 0;

  s.periodic(
    () => stepSeconds,
    (at) => {
      const b = BARS[bar];
      if (step === 0) {
        for (const f of b.chord) {
          s.tone({ at, freq: f, dur: stepSeconds * 15, vol: 0.06, type: 'sawtooth', shape: 'hold', attack: 0.25, release: 0.4, detune: -8, dest: padLp });
          s.tone({ at, freq: f, dur: stepSeconds * 15, vol: 0.06, type: 'sawtooth', shape: 'hold', attack: 0.25, release: 0.4, detune: 8, dest: padLp });
        }
      }
      if (BASS_STEPS.includes(step)) {
        const octave = step % 8 === 3 || step % 8 === 6 ? 2 : 1;
        s.tone({ at, freq: b.root * octave, dur: stepSeconds * 1.6, vol: 0.32, type: 'sawtooth', attack: 0.005, dest: bassLp });
      }
      if (KICK_STEPS.includes(step)) s.tone({ at, freq: 150, glideTo: 45, glideTime: 0.08, dur: 0.25, vol: 0.7, attack: 0.002, dest: drums });
      if (SNARE_STEPS.includes(step)) {
        s.noiseBurst({ at, dur: 0.12, vol: 0.3, filter: 'bandpass', freq: 1800, q: 0.8, attack: 0.002, dest: drums });
        s.tone({ at, freq: 220, glideTo: 160, dur: 0.08, vol: 0.2, type: 'triangle', dest: drums });
      }
      if (HAT_STEPS.includes(step)) s.noiseBurst({ at, dur: 0.04, vol: 0.14, filter: 'highpass', freq: 7000, attack: 0.001, dest: drums });
      if (bar % 2 === 1 && ARP_STEPS.includes(step)) {
        const note = b.chord[(step / 3) % b.chord.length] * 2;
        s.tone({ at, freq: note, dur: stepSeconds * 0.9, vol: 0.1, type: 'triangle', attack: 0.003, dest: arpLp });
      }
      step = (step + 1) % R.steps;
      if (step === 0) bar = (bar + 1) % BARS.length;
    },
  );
}

export const RADIO_RECIPES: RecipeTable<'radio.'> = {
  'radio.synthwave.loop': { build: buildStation },
  'radio.static': {
    build: (s) => s.noiseBurst({ dur: 0.4, vol: 0.3, filter: 'bandpass', freq: 2000, q: 0.5, attack: 0.01, shape: 'hold', release: 0.05 }),
  },
  'radio.switch': {
    build: (s) => {
      s.click(s.t, 0.35, 1800, 4, 0.012);
      s.noiseBurst({ at: s.t + 0.01, dur: 0.12, vol: 0.25, filter: 'bandpass', freq: 2200, q: 0.6, attack: 0.005, shape: 'hold', release: 0.03 });
    },
  },
  'radio.off': {
    build: (s) => {
      s.click(s.t, 0.35, 1500, 4, 0.012);
      s.tone({ at: s.t + 0.01, freq: 800, glideTo: 200, dur: 0.15, vol: 0.2 });
    },
  },
};
