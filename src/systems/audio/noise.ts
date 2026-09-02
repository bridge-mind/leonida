import { AudioConfig } from './config';

export type NoiseKind = 'white' | 'pink' | 'brown';

/**
 * Generated noise buffers shared by every voice (AudioBufferSourceNodes are
 * single-use but the buffers are not) plus the reverb impulse response.
 */
export class NoiseBank {
  private readonly buffers: Record<NoiseKind, AudioBuffer>;
  readonly impulse: AudioBuffer;

  constructor(ctx: BaseAudioContext) {
    const n = Math.floor(ctx.sampleRate * AudioConfig.noise.seconds);
    this.buffers = {
      white: fill(ctx, n, white),
      pink: fill(ctx, n, pink),
      brown: fill(ctx, n, brown),
    };
    this.impulse = impulse(ctx);
  }

  get(kind: NoiseKind): AudioBuffer {
    return this.buffers[kind];
  }
}

function fill(ctx: BaseAudioContext, n: number, gen: (out: Float32Array) => void): AudioBuffer {
  const buffer = ctx.createBuffer(1, n, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  gen(data);
  return buffer;
}

function white(out: Float32Array): void {
  for (let i = 0; i < out.length; i++) out[i] = Math.random() * 2 - 1;
}

/** Paul Kellet's economy pink noise filter. */
function pink(out: Float32Array): void {
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  let b3 = 0;
  let b4 = 0;
  let b5 = 0;
  let b6 = 0;
  for (let i = 0; i < out.length; i++) {
    const w = Math.random() * 2 - 1;
    b0 = 0.99886 * b0 + w * 0.0555179;
    b1 = 0.99332 * b1 + w * 0.0750759;
    b2 = 0.969 * b2 + w * 0.153852;
    b3 = 0.8665 * b3 + w * 0.3104856;
    b4 = 0.55 * b4 + w * 0.5329522;
    b5 = -0.7616 * b5 - w * 0.016898;
    out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362) * 0.11;
    b6 = w * 0.115926;
  }
}

function brown(out: Float32Array): void {
  let last = 0;
  for (let i = 0; i < out.length; i++) {
    const w = Math.random() * 2 - 1;
    last = (last + 0.02 * w) / 1.02;
    out[i] = last * 3.5;
  }
}

/** Stereo exponentially decaying noise: a small, slightly bright room. */
function impulse(ctx: BaseAudioContext): AudioBuffer {
  const n = Math.floor(ctx.sampleRate * AudioConfig.reverb.seconds);
  const buffer = ctx.createBuffer(2, n, ctx.sampleRate);
  for (let c = 0; c < 2; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < n; i++) {
      const t = i / n;
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - t, AudioConfig.reverb.decay);
    }
  }
  return buffer;
}
