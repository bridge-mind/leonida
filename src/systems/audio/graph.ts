import { Quaternion, Vector3 } from 'three';
import type { V3 } from '../../core/entities';
import type { AudioBus } from './api';
import { AudioConfig } from './config';
import { NoiseBank } from './noise';

export const BUSES: readonly AudioBus[] = ['master', 'sfx', 'engine', 'ambient', 'ui', 'music'];

/**
 * The static part of the Web Audio graph:
 * buses → master gain → compressor → destination, plus a shared convolver
 * (generated impulse) fed by a `reverbIn` send that every voice may use.
 */
export class AudioGraph {
  readonly master: GainNode;
  readonly compressor: DynamicsCompressorNode;
  readonly buses: Record<AudioBus, GainNode>;
  readonly reverbIn: GainNode;
  readonly noise: NoiseBank;
  private readonly hasListenerParams: boolean;
  private readonly fwd = new Vector3();
  private readonly up = new Vector3();
  private readonly quat = new Quaternion();
  readonly listenerPos = new Vector3();

  private constructor(readonly ctx: AudioContext, volumes: Record<AudioBus, number>) {
    const c = AudioConfig.compressor;
    this.compressor = ctx.createDynamicsCompressor();
    this.compressor.threshold.value = c.threshold;
    this.compressor.knee.value = c.knee;
    this.compressor.ratio.value = c.ratio;
    this.compressor.attack.value = c.attack;
    this.compressor.release.value = c.release;
    this.compressor.connect(ctx.destination);

    this.master = ctx.createGain();
    this.master.gain.value = volumes.master;
    this.master.connect(this.compressor);

    const mk = (bus: AudioBus): GainNode => {
      const g = ctx.createGain();
      g.gain.value = volumes[bus];
      g.connect(this.master);
      return g;
    };
    this.buses = { master: this.master, sfx: mk('sfx'), engine: mk('engine'), ambient: mk('ambient'), ui: mk('ui'), music: mk('music') };

    this.noise = new NoiseBank(ctx);
    const convolver = ctx.createConvolver();
    convolver.buffer = this.noise.impulse;
    const wet = ctx.createGain();
    wet.gain.value = AudioConfig.reverb.wet;
    convolver.connect(wet);
    wet.connect(this.master);
    this.reverbIn = ctx.createGain();
    this.reverbIn.gain.value = 1;
    this.reverbIn.connect(convolver);

    const l = ctx.listener as AudioListener & { positionX?: AudioParam };
    this.hasListenerParams = l.positionX !== undefined;
  }

  /** Returns null when Web Audio is unavailable or construction throws (headless runs). */
  static tryCreate(volumes: Record<AudioBus, number>): AudioGraph | null {
    if (typeof window === 'undefined' || typeof AudioContext === 'undefined') return null;
    try {
      const ctx = new AudioContext({ latencyHint: 'interactive' });
      return new AudioGraph(ctx, volumes);
    } catch (err) {
      console.warn('[audio] AudioContext unavailable, running silent', err);
      return null;
    }
  }

  get now(): number {
    return this.ctx.currentTime;
  }

  get running(): boolean {
    return this.ctx.state === 'running';
  }

  resume(): void {
    if (this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => undefined);
    }
  }

  setBusVolume(bus: AudioBus, volume: number): void {
    this.buses[bus].gain.setTargetAtTime(volume, this.ctx.currentTime, 0.03);
  }

  setListener(position: V3, quaternion: Quaternion, _velocity: V3): void {
    this.listenerPos.set(position.x, position.y, position.z);
    this.quat.copy(quaternion);
    this.fwd.set(0, 0, -1).applyQuaternion(this.quat);
    this.up.set(0, 1, 0).applyQuaternion(this.quat);
    const l = this.ctx.listener;
    if (this.hasListenerParams) {
      l.positionX.value = position.x;
      l.positionY.value = position.y;
      l.positionZ.value = position.z;
      l.forwardX.value = this.fwd.x;
      l.forwardY.value = this.fwd.y;
      l.forwardZ.value = this.fwd.z;
      l.upX.value = this.up.x;
      l.upY.value = this.up.y;
      l.upZ.value = this.up.z;
    } else {
      l.setPosition(position.x, position.y, position.z);
      l.setOrientation(this.fwd.x, this.fwd.y, this.fwd.z, this.up.x, this.up.y, this.up.z);
    }
  }

  distanceTo(p: V3): number {
    const dx = p.x - this.listenerPos.x;
    const dy = p.y - this.listenerPos.y;
    const dz = p.z - this.listenerPos.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  createPanner(): PannerNode {
    const p = this.ctx.createPanner();
    const c = AudioConfig.panner;
    p.panningModel = c.model;
    p.distanceModel = c.distanceModel;
    p.refDistance = c.refDistance;
    p.maxDistance = c.maxDistance;
    p.rolloffFactor = c.rolloff;
    return p;
  }

  dispose(): void {
    this.ctx.close().catch(() => undefined);
  }
}
