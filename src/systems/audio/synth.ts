import { AudioConfig } from './config';
import type { NoiseBank, NoiseKind } from './noise';

/** Smallest gain used by exponential ramps (they cannot reach 0). */
export const MIN_GAIN = 0.0001;

export type EngineFn = (rpm01: number, throttle01: number, speedMs: number, at: number) => void;

interface Pitched {
  param: AudioParam;
  base: number;
}

interface Transient {
  src: AudioScheduledSourceNode;
  end: number;
}

export interface ToneOpts {
  freq: number;
  dur: number;
  type?: OscillatorType;
  at?: number;
  vol?: number;
  attack?: number;
  /** Release after `dur` when `shape` is 'hold'. */
  release?: number;
  /** 'pluck' decays exponentially over `dur`; 'hold' sustains then releases. */
  shape?: 'pluck' | 'hold';
  glideTo?: number;
  glideTime?: number;
  detune?: number;
  dest?: AudioNode;
}

export interface NoiseOpts {
  dur: number;
  kind?: NoiseKind;
  at?: number;
  vol?: number;
  attack?: number;
  release?: number;
  shape?: 'pluck' | 'hold';
  filter?: BiquadFilterType;
  freq?: number;
  freqTo?: number;
  sweepTime?: number;
  q?: number;
  rate?: number;
  dest?: AudioNode;
}

/**
 * Per-voice synthesis toolkit. A recipe receives one of these with `out` already
 * routed to the voice gain → (panner) → bus. Every frequency is multiplied by
 * `pitch`; oscillators and noise sources created through `osc`/`noiseSrc` follow
 * later `setPitch` calls (loops).
 */
export class Synth {
  /** Persistent sources (stopped when the voice is killed). */
  readonly sources: AudioScheduledSourceNode[] = [];
  private readonly transients: Transient[] = [];
  private readonly pitched: Pitched[] = [];
  endTime: number;
  /** Absolute pitch multiplier after the latest `setPitch` (starts at `pitch`). */
  currentPitch: number;
  onUpdate: ((now: number) => void) | null = null;
  onEngine: EngineFn | null = null;
  onSiren: ((on: boolean, at: number) => void) | null = null;

  constructor(
    readonly ctx: BaseAudioContext,
    readonly noise: NoiseBank,
    readonly out: GainNode,
    private readonly reverbIn: AudioNode | null,
    readonly t: number,
    readonly pitch: number,
  ) {
    this.endTime = t;
    this.currentPitch = pitch;
  }

  // --- Randomness -----------------------------------------------------------

  rand(lo: number, hi: number): number {
    return lo + Math.random() * (hi - lo);
  }

  jitter(value: number, fraction: number): number {
    return value * (1 + (Math.random() * 2 - 1) * fraction);
  }

  // --- Nodes ------------------------------------------------------------------

  gain(value: number, dest: AudioNode = this.out): GainNode {
    const g = this.ctx.createGain();
    g.gain.value = value;
    g.connect(dest);
    return g;
  }

  filter(type: BiquadFilterType, freq: number, q = 1, dest: AudioNode = this.out, gainDb = 0): BiquadFilterNode {
    const f = this.ctx.createBiquadFilter();
    f.type = type;
    f.frequency.value = freq;
    f.Q.value = q;
    f.gain.value = gainDb;
    f.connect(dest);
    return f;
  }

  /** Soft saturation (tanh) with `drive` ≥ 1. */
  shaper(drive: number, dest: AudioNode = this.out): WaveShaperNode {
    const w = this.ctx.createWaveShaper();
    const curve = new Float32Array(257);
    for (let i = 0; i < curve.length; i++) {
      const x = (i / 128 - 1) * drive;
      curve[i] = Math.tanh(x);
    }
    w.curve = curve;
    w.connect(dest);
    return w;
  }

  delay(seconds: number, feedback: number, dest: AudioNode = this.out): DelayNode {
    const d = this.ctx.createDelay(Math.max(1, seconds));
    d.delayTime.value = seconds;
    const fb = this.ctx.createGain();
    fb.gain.value = feedback;
    d.connect(fb);
    fb.connect(d);
    d.connect(dest);
    return d;
  }

  /** Persistent oscillator following `setPitch`. */
  osc(type: OscillatorType, freq: number, dest: AudioNode = this.out, detune = 0, at = this.t): OscillatorNode {
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.value = freq * this.pitch;
    o.detune.value = detune;
    o.connect(dest);
    o.start(at);
    this.sources.push(o);
    this.pitched.push({ param: o.frequency, base: freq });
    return o;
  }

  /** Persistent looping noise source following `setPitch`. */
  noiseSrc(kind: NoiseKind, dest: AudioNode = this.out, rate = 1, at = this.t): AudioBufferSourceNode {
    const s = this.ctx.createBufferSource();
    s.buffer = this.noise.get(kind);
    s.loop = true;
    s.playbackRate.value = rate * this.pitch;
    s.connect(dest);
    s.start(at, Math.random() * AudioConfig.noise.seconds * 0.5);
    this.sources.push(s);
    this.pitched.push({ param: s.playbackRate, base: rate });
    return s;
  }

  /** Modulator: oscillator → depth gain → `target` param. Not pitched. */
  lfo(freq: number, depth: number, target: AudioParam, type: OscillatorType = 'sine', at = this.t): OscillatorNode {
    const o = this.ctx.createOscillator();
    o.type = type;
    o.frequency.value = freq;
    const g = this.ctx.createGain();
    g.gain.value = depth;
    o.connect(g);
    g.connect(target);
    o.start(at);
    this.sources.push(o);
    return o;
  }

  /** Routes `node` into the shared reverb at `amount`. */
  send(node: AudioNode, amount: number): void {
    if (!this.reverbIn) return;
    const g = this.ctx.createGain();
    g.gain.value = amount;
    node.connect(g);
    g.connect(this.reverbIn);
  }

  // --- Envelopes -----------------------------------------------------------------

  /** Attack to `peak`, then exponential decay to `floor` over `decay`. */
  ad(param: AudioParam, peak: number, attack: number, decay: number, at = this.t, floor = MIN_GAIN): void {
    param.setValueAtTime(MIN_GAIN, at);
    param.linearRampToValueAtTime(Math.max(peak, MIN_GAIN), at + attack);
    param.exponentialRampToValueAtTime(Math.max(floor, MIN_GAIN), at + attack + decay);
  }

  /** Exponential sweep of a frequency-like param (values clamped ≥ 1). */
  sweep(param: AudioParam, from: number, to: number, at: number, dur: number): void {
    param.setValueAtTime(Math.max(1, from), at);
    param.exponentialRampToValueAtTime(Math.max(1, to), at + dur);
  }

  hold(until: number): void {
    if (until > this.endTime) this.endTime = until;
  }

  /** Stops every persistent source at `at` (one-shots call this last). */
  stopAt(at: number): void {
    for (const s of this.sources) {
      try {
        s.stop(at);
      } catch {
        // Already stopped.
      }
    }
    this.hold(at);
  }

  /** Multiplies every registered frequency/playbackRate by `mult` with a short ramp. */
  setPitch(mult: number, at: number, ramp = AudioConfig.voices.paramRamp): void {
    this.currentPitch = mult;
    for (const p of this.pitched) p.param.setTargetAtTime(p.base * mult, at, ramp);
  }

  /** Immediate stop of everything (voice kill). */
  killAll(): void {
    for (const s of this.sources) stopNow(s);
    for (const t of this.transients) stopNow(t.src);
    this.sources.length = 0;
    this.transients.length = 0;
  }

  // --- Scheduled sub-sounds ---------------------------------------------------------

  /** Registers a started-and-stopping source so the voice ends after it and kills it early if needed. */
  track(src: AudioScheduledSourceNode, end: number): void {
    this.transients.push({ src, end });
    this.hold(end);
  }

  /** Drops finished transients (called by periodic loops before scheduling more). */
  pruneTransients(now: number): void {
    for (let i = this.transients.length - 1; i >= 0; i--) {
      if (this.transients[i].end < now) this.transients.splice(i, 1);
    }
  }

  /**
   * Schedules `fn(at)` repeatedly for the lifetime of a loop. `interval()` is
   * re-evaluated per event so gaps can be random.
   */
  periodic(interval: () => number, fn: (at: number) => void): void {
    let next = this.t;
    const lookahead = AudioConfig.voices.lookahead;
    this.onUpdate = (now) => {
      this.pruneTransients(now);
      if (next < now - 1) next = now;
      let guard = 0;
      while (next < now + lookahead && guard++ < 32) {
        fn(next);
        next += Math.max(0.01, interval());
      }
    };
  }

  tone(o: ToneOpts): OscillatorNode {
    const at = o.at ?? this.t;
    const attack = o.attack ?? 0.005;
    const release = o.release ?? 0.05;
    const vol = o.vol ?? 0.3;
    const g = this.gain(MIN_GAIN, o.dest ?? this.out);
    const osc = this.ctx.createOscillator();
    osc.type = o.type ?? 'sine';
    osc.frequency.value = o.freq * this.pitch;
    if (o.detune) osc.detune.value = o.detune;
    if (o.glideTo !== undefined) this.sweep(osc.frequency, o.freq * this.pitch, o.glideTo * this.pitch, at, o.glideTime ?? o.dur);
    osc.connect(g);
    const end = this.envelope(g.gain, at, attack, o.dur, release, vol, o.shape ?? 'pluck');
    osc.start(at);
    osc.stop(end + 0.01);
    this.track(osc, end + 0.01);
    return osc;
  }

  noiseBurst(o: NoiseOpts): AudioBufferSourceNode {
    const at = o.at ?? this.t;
    const attack = o.attack ?? 0.002;
    const release = o.release ?? 0.03;
    const vol = o.vol ?? 0.4;
    const g = this.gain(MIN_GAIN, o.dest ?? this.out);
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise.get(o.kind ?? 'white');
    src.loop = true;
    src.playbackRate.value = (o.rate ?? 1) * this.pitch;
    if (o.filter) {
      const freq = (o.freq ?? 1000) * this.pitch;
      const f = this.filter(o.filter, freq, o.q ?? 1, g);
      if (o.freqTo !== undefined) this.sweep(f.frequency, freq, o.freqTo * this.pitch, at, o.sweepTime ?? o.dur);
      src.connect(f);
    } else {
      src.connect(g);
    }
    const end = this.envelope(g.gain, at, attack, o.dur, release, vol, o.shape ?? 'pluck');
    src.start(at, Math.random() * AudioConfig.noise.seconds * 0.5);
    src.stop(end + 0.01);
    this.track(src, end + 0.01);
    return src;
  }

  private envelope(param: AudioParam, at: number, attack: number, dur: number, release: number, vol: number, shape: 'pluck' | 'hold'): number {
    const peak = Math.max(vol, MIN_GAIN);
    param.setValueAtTime(MIN_GAIN, at);
    param.linearRampToValueAtTime(peak, at + attack);
    if (shape === 'pluck') {
      param.exponentialRampToValueAtTime(MIN_GAIN, at + attack + dur);
      return at + attack + dur;
    }
    param.setValueAtTime(peak, at + dur);
    param.exponentialRampToValueAtTime(MIN_GAIN, at + dur + release);
    return at + dur + release;
  }

  // --- Presets -----------------------------------------------------------------------

  /** Short filtered noise tick. */
  click(at: number, vol: number, freq: number, q = 3, dur = 0.015): void {
    this.noiseBurst({ at, dur, vol, filter: 'bandpass', freq, q, attack: 0.001, release: 0.01 });
  }

  /** Low sine with a downward pitch glide. */
  thump(at: number, freq: number, dropTo: number, dur: number, vol: number, type: OscillatorType = 'sine'): void {
    this.tone({ at, type, freq, glideTo: dropTo, dur, vol, attack: 0.003 });
  }

  /** Metallic ring: high-Q noise band plus a sine at the same frequency. */
  ping(at: number, freq: number, dur: number, vol: number, q = 30): void {
    this.noiseBurst({ at, dur, vol, filter: 'bandpass', freq, q, attack: 0.001 });
    this.tone({ at, type: 'sine', freq, dur: dur * 0.8, vol: vol * 0.5, attack: 0.001 });
  }

  /** Notes played `step` seconds apart. */
  arp(at: number, freqs: readonly number[], step: number, dur: number, vol: number, type: OscillatorType = 'sine'): void {
    for (let i = 0; i < freqs.length; i++) this.tone({ at: at + i * step, freq: freqs[i], dur, vol, type });
  }

  chord(at: number, freqs: readonly number[], dur: number, vol: number, type: OscillatorType = 'sine', shape: 'pluck' | 'hold' = 'hold', dest?: AudioNode): void {
    for (const f of freqs) this.tone({ at, freq: f, dur, vol, type, shape, attack: 0.02, release: 0.3, dest });
  }

  /** Bandpassed noise sweep (swings, throws, whooshes). */
  whoosh(at: number, from: number, to: number, dur: number, vol: number, q = 1.2): void {
    this.noiseBurst({ at, dur, vol, filter: 'bandpass', freq: from, freqTo: to, q, attack: dur * 0.35, shape: 'pluck' });
  }
}

function stopNow(s: AudioScheduledSourceNode): void {
  try {
    s.onended = null;
    s.stop();
  } catch {
    // Not started or already stopped.
  }
  try {
    s.disconnect();
  } catch {
    // Already disconnected.
  }
}
