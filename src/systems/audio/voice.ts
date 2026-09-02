import type { V3 } from '../../core/entities';
import type { LoopHandle } from './api';
import { AudioConfig } from './config';
import type { AudioGraph } from './graph';
import type { Recipe } from './recipes/types';
import type { SoundId } from './sounds';
import { MIN_GAIN, Synth } from './synth';
import type { PoolVoice } from './voicePool';

/**
 * One playing sound: recipe nodes → `input` gain → optional panner → bus.
 * Created per play (Web Audio nodes are single-use) and freed by the pool.
 */
export class Voice implements PoolVoice {
  readonly input: GainNode;
  readonly panner: PannerNode | null;
  readonly synth: Synth;
  readonly startedAt: number;
  /** Wall-clock deadline used when the context is suspended (its clock does not advance). */
  readonly wallEnd: number;
  volume: number;
  ended = false;
  onEnd: ((voice: Voice) => void) | null = null;
  /** Set for voices started through `loop()`. */
  handle: LoopHandle | null = null;
  private userVolume: number;
  private readonly recipeGain: number;
  private lastPos: V3 | null = null;
  private lastPosTime = 0;
  private doppler = 1;
  private userPitch: number;
  private stopTime = Infinity;
  private wallStop = Infinity;

  constructor(
    readonly id: number,
    readonly sound: SoundId,
    readonly loop: boolean,
    private readonly graph: AudioGraph,
    bus: GainNode,
    recipe: Recipe,
    volume: number,
    pitch: number,
    position: V3 | undefined,
    cullDistance: number,
  ) {
    const ctx = graph.ctx;
    const now = ctx.currentTime;
    this.startedAt = now;
    this.userVolume = volume;
    this.userPitch = pitch;
    this.recipeGain = recipe.gain ?? 1;
    this.input = ctx.createGain();
    this.input.gain.value = Math.max(MIN_GAIN, volume * this.recipeGain);
    if (position) {
      this.panner = graph.createPanner();
      this.panner.positionX.value = position.x;
      this.panner.positionY.value = position.y;
      this.panner.positionZ.value = position.z;
      this.input.connect(this.panner);
      this.panner.connect(bus);
      this.lastPos = { x: position.x, y: position.y, z: position.z };
      this.lastPosTime = now;
      this.volume = volume * this.recipeGain * attenuation(graph.distanceTo(position), cullDistance);
    } else {
      this.panner = null;
      this.input.connect(bus);
      this.volume = volume * this.recipeGain;
    }
    this.synth = new Synth(ctx, graph.noise, this.input, graph.reverbIn, now, pitch);
    if (recipe.reverb) this.synth.send(this.input, recipe.reverb);
    recipe.build(this.synth);
    const duration = Math.max(0.05, this.synth.endTime - now);
    this.wallEnd = performance.now() + duration * 1000 + 250;
  }

  get endTime(): number {
    return this.synth.endTime;
  }

  /** True once a one-shot has finished or a stopped loop has faded (either clock). */
  expired(now: number, wallNow: number): boolean {
    if (this.loop) return now > this.stopTime || wallNow > this.wallStop;
    return now > this.synth.endTime + 0.05 || wallNow > this.wallEnd;
  }

  setVolume(volume: number, at: number, timeConstant = AudioConfig.voices.paramRamp): void {
    this.userVolume = volume;
    this.input.gain.setTargetAtTime(Math.max(MIN_GAIN, volume * this.recipeGain), at, timeConstant);
    if (!this.panner) this.volume = volume * this.recipeGain;
  }

  setPitch(pitch: number, at: number): void {
    this.userPitch = pitch;
    this.synth.setPitch(pitch * this.doppler, at);
  }

  setPosition(p: V3, at: number): void {
    if (!this.panner) return;
    const ramp = AudioConfig.voices.paramRamp;
    this.panner.positionX.setTargetAtTime(p.x, at, ramp);
    this.panner.positionY.setTargetAtTime(p.y, at, ramp);
    this.panner.positionZ.setTargetAtTime(p.z, at, ramp);
    const dist = this.graph.distanceTo(p);
    this.volume = this.userVolume * this.recipeGain * attenuation(dist, AudioConfig.panner.maxDistance);
    if (this.loop && this.lastPos) {
      const dt = at - this.lastPosTime;
      if (dt > 0.02) {
        // Radial speed toward the listener → cheap doppler on the loop's pitch.
        const prevDist = this.graph.distanceTo(this.lastPos);
        const radial = (prevDist - dist) / dt;
        const D = AudioConfig.doppler;
        const target = clamp(1 + radial / D.speedOfSound, 1 - D.maxShift, 1 + D.maxShift);
        this.doppler += (target - this.doppler) * 0.5;
        this.synth.setPitch(this.userPitch * this.doppler, at, D.smoothing);
        this.lastPos.x = p.x;
        this.lastPos.y = p.y;
        this.lastPos.z = p.z;
        this.lastPosTime = at;
      }
    }
  }

  /** Fades the voice out and marks it to expire; the pool prunes it afterwards. */
  fadeOut(at: number, seconds: number): void {
    this.input.gain.setTargetAtTime(MIN_GAIN, at, seconds / 3);
    this.synth.stopAt(at + seconds);
    const end = at + seconds;
    this.stopTime = Math.min(this.stopTime, end);
    this.wallStop = Math.min(this.wallStop, performance.now() + Math.max(0, end - this.graph.now) * 1000 + 250);
  }

  get stopping(): boolean {
    return this.stopTime !== Infinity;
  }

  kill(): void {
    if (this.ended) return;
    this.ended = true;
    this.synth.killAll();
    try {
      this.input.disconnect();
      this.panner?.disconnect();
    } catch {
      // Already disconnected.
    }
    this.onEnd?.(this);
  }
}

function attenuation(distance: number, maxDistance: number): number {
  if (distance >= maxDistance) return 0;
  return 1 / (1 + Math.max(0, distance - AudioConfig.panner.refDistance) / AudioConfig.panner.refDistance);
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
