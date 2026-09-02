import {
  CanvasTexture,
  ClampToEdgeWrapping,
  LinearFilter,
  LinearMipmapLinearFilter,
  NearestFilter,
  RepeatWrapping,
  SRGBColorSpace,
  type Texture,
} from 'three';
import { Random } from '../core/Random';
import type { RoadKind } from '../world/roadNetwork';

/** All textures are generated from <canvas>; nothing is downloaded. */

function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  return { canvas, ctx };
}

function toTexture(canvas: HTMLCanvasElement, repeat = true, srgb = true): CanvasTexture {
  const tex = new CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = repeat ? RepeatWrapping : ClampToEdgeWrapping;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.magFilter = LinearFilter;
  tex.anisotropy = 4;
  if (srgb) tex.colorSpace = SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

function noise(ctx: CanvasRenderingContext2D, w: number, h: number, rng: Random, alpha: number, cell = 4): void {
  for (let y = 0; y < h; y += cell) {
    for (let x = 0; x < w; x += cell) {
      const v = Math.floor(rng.range(0, 255));
      ctx.fillStyle = `rgba(${v},${v},${v},${alpha})`;
      ctx.fillRect(x, y, cell, cell);
    }
  }
}

export interface WindowTextures {
  map: Texture;
  emissiveMap: Texture;
  /** Meters covered by one UV unit (4 floors × 4 window columns). */
  uvScale: number;
}

/**
 * 4×4 window grid per texture repeat. Walls are white so vertex colors tint them;
 * windows are dark. The emissive map lights a random subset for the night look.
 */
export function windowTextures(seed: string, style: 'concrete' | 'glass' | 'deco'): WindowTextures {
  const size = 512;
  const rng = Random.fromString(`windows:${seed}:${style}`);
  const { canvas, ctx } = makeCanvas(size, size);
  const { canvas: eCanvas, ctx: ectx } = makeCanvas(size, size);
  const cell = size / 4;

  ctx.fillStyle = style === 'glass' ? '#c8d2e0' : '#ffffff';
  ctx.fillRect(0, 0, size, size);
  if (style !== 'glass') noise(ctx, size, size, rng, 0.06, 8);
  ectx.fillStyle = '#000000';
  ectx.fillRect(0, 0, size, size);

  const litChance = style === 'glass' ? 0.35 : style === 'deco' ? 0.5 : 0.45;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const x = c * cell;
      const y = r * cell;
      // Floor band line at the bottom of each cell.
      ctx.fillStyle = style === 'glass' ? 'rgba(20,28,40,0.9)' : 'rgba(0,0,0,0.16)';
      ctx.fillRect(x, y + cell - 6, cell, 6);
      if (style === 'glass') {
        ctx.fillStyle = '#5b6f8c';
        ctx.fillRect(x + 4, y + 4, cell - 8, cell - 14);
        ctx.fillStyle = 'rgba(200,220,245,0.45)';
        ctx.fillRect(x + 4, y + 4, cell - 8, 14);
        ctx.fillStyle = 'rgba(30,40,55,0.5)';
        ctx.fillRect(x + cell / 2 - 2, y + 4, 4, cell - 14);
      } else {
        const wx = x + cell * 0.3;
        const wy = y + cell * 0.22;
        const ww = cell * 0.4;
        const wh = cell * 0.46;
        ctx.fillStyle = '#26303f';
        ctx.fillRect(wx, wy, ww, wh);
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.fillRect(wx, wy, ww, wh * 0.2);
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        ctx.fillRect(wx - 4, wy + wh, ww + 8, 5);
        if (style === 'deco') {
          ctx.fillStyle = 'rgba(0,0,0,0.12)';
          ctx.fillRect(x + cell * 0.1, y + cell * 0.08, cell * 0.05, cell * 0.85);
          ctx.fillRect(x + cell * 0.85, y + cell * 0.08, cell * 0.05, cell * 0.85);
        }
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        ctx.fillRect(wx, wy + wh - 4, ww, 4);
        if (rng.chance(litChance)) {
          const warm = rng.chance(0.75);
          ectx.fillStyle = warm ? `rgb(${255},${200 + rng.int(0, 40)},${120 + rng.int(0, 60)})` : 'rgb(190,225,255)';
          ectx.fillRect(wx, wy, ww, wh);
        }
        continue;
      }
      if (rng.chance(litChance)) {
        ectx.fillStyle = rng.chance(0.5) ? 'rgb(255,220,160)' : 'rgb(170,210,255)';
        ectx.fillRect(x + 4, y + 4, cell - 8, cell - 14);
      }
    }
  }
  return { map: toTexture(canvas), emissiveMap: toTexture(eCanvas), uvScale: 12.8 };
}

/** Road surface textures. `u` spans the road width, `v` repeats every 12 m along the road. */
export function roadTexture(kind: RoadKind | 'asphalt'): Texture {
  const w = 256;
  const h = 256;
  const rng = Random.fromString(`road:${kind}`);
  const { canvas, ctx } = makeCanvas(w, h);
  const isDirt = kind === 'dirt';
  ctx.fillStyle = isDirt ? '#6f5a3c' : '#3a3a3d';
  ctx.fillRect(0, 0, w, h);
  noise(ctx, w, h, rng, isDirt ? 0.14 : 0.09, 4);
  if (isDirt) {
    ctx.fillStyle = 'rgba(40,30,18,0.28)';
    ctx.fillRect(w * 0.28, 0, w * 0.1, h);
    ctx.fillRect(w * 0.62, 0, w * 0.1, h);
    return toTexture(canvas);
  }
  if (kind === 'asphalt') return toTexture(canvas);

  const width = kind === 'street' ? 12 : 20;
  const px = (meters: number): number => (meters / width) * w;
  const line = (atM: number, thickM: number, color: string, dashed: boolean): void => {
    ctx.fillStyle = color;
    const x = px(atM) - px(thickM) / 2;
    if (!dashed) ctx.fillRect(x, 0, Math.max(2, px(thickM)), h);
    else for (let y = 0; y < h; y += h / 2) ctx.fillRect(x, y, Math.max(2, px(thickM)), h / 4);
  };
  const yellow = '#d9b43a';
  const white = '#d8d8d0';
  if (kind === 'street') {
    line(width / 2 - 0.16, 0.12, yellow, false);
    line(width / 2 + 0.16, 0.12, yellow, false);
    line(2.5, 0.12, white, false);
    line(9.5, 0.12, white, false);
  } else {
    // avenue / highway / bridge: 4 lanes with 3 m shoulders (highway) or parking (avenue)
    line(width / 2 - 0.18, 0.14, yellow, false);
    line(width / 2 + 0.18, 0.14, yellow, false);
    line(6.5, 0.12, white, true);
    line(13.5, 0.12, white, true);
    line(3, 0.14, white, false);
    line(17, 0.14, white, false);
  }
  return toTexture(canvas);
}

export function sandTexture(): Texture {
  const rng = Random.fromString('sand');
  const { canvas, ctx } = makeCanvas(256, 256);
  ctx.fillStyle = '#e6d3a3';
  ctx.fillRect(0, 0, 256, 256);
  noise(ctx, 256, 256, rng, 0.08, 3);
  return toTexture(canvas);
}

export function concreteTexture(): Texture {
  const rng = Random.fromString('concrete');
  const { canvas, ctx } = makeCanvas(256, 256);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 256, 256);
  noise(ctx, 256, 256, rng, 0.07, 4);
  // Faint slab joints every 128 px (= 4 m at uvScale 8).
  ctx.fillStyle = 'rgba(0,0,0,0.12)';
  ctx.fillRect(0, 126, 256, 3);
  ctx.fillRect(126, 0, 3, 256);
  return toTexture(canvas);
}

export function grassTexture(): Texture {
  const rng = Random.fromString('grass');
  const { canvas, ctx } = makeCanvas(256, 256);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 256, 256);
  noise(ctx, 256, 256, rng, 0.12, 6);
  return toTexture(canvas);
}

export function plankTexture(): Texture {
  const rng = Random.fromString('planks');
  const { canvas, ctx } = makeCanvas(256, 256);
  ctx.fillStyle = '#a57a4e';
  ctx.fillRect(0, 0, 256, 256);
  noise(ctx, 256, 256, rng, 0.1, 4);
  ctx.fillStyle = 'rgba(40,25,10,0.5)';
  for (let y = 0; y < 256; y += 32) ctx.fillRect(0, y, 256, 2);
  return toTexture(canvas);
}

export interface SignAtlas {
  texture: Texture;
  /** UV rect [u0, v0, u1, v1] per sign id. */
  rects: Map<string, [number, number, number, number]>;
}

/** One atlas of illuminated store signs (4 columns × 8 rows of 512×128 px tiles). */
export function signAtlas(entries: { id: string; text: string; color: number; bg?: number }[]): SignAtlas {
  const cols = 4;
  const rows = 8;
  const tw = 512;
  const th = 128;
  const { canvas, ctx } = makeCanvas(cols * tw, rows * th);
  ctx.fillStyle = '#101010';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const rects = new Map<string, [number, number, number, number]>();
  entries.slice(0, cols * rows).forEach((e, i) => {
    const cx = (i % cols) * tw;
    const cy = Math.floor(i / cols) * th;
    const color = `#${e.color.toString(16).padStart(6, '0')}`;
    ctx.fillStyle = e.bg !== undefined ? `#${e.bg.toString(16).padStart(6, '0')}` : '#15151a';
    ctx.fillRect(cx + 2, cy + 2, tw - 4, th - 4);
    ctx.strokeStyle = color;
    ctx.lineWidth = 6;
    ctx.strokeRect(cx + 10, cy + 10, tw - 20, th - 20);
    ctx.font = `bold ${e.text.length > 14 ? 52 : 68}px Impact, "Arial Narrow", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = color;
    ctx.shadowBlur = 24;
    ctx.fillStyle = color;
    ctx.fillText(e.text.toUpperCase(), cx + tw / 2, cy + th / 2 + 4, tw - 40);
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffffff';
    ctx.globalAlpha = 0.55;
    ctx.fillText(e.text.toUpperCase(), cx + tw / 2, cy + th / 2 + 4, tw - 40);
    ctx.globalAlpha = 1;
    // Flip v so that v0 is the bottom of the tile in GL coordinates.
    const v1 = 1 - cy / canvas.height;
    const v0 = 1 - (cy + th) / canvas.height;
    rects.set(e.id, [cx / canvas.width, v0, (cx + tw) / canvas.width, v1]);
  });
  const texture = toTexture(canvas, false);
  texture.minFilter = LinearFilter;
  texture.magFilter = LinearFilter;
  texture.generateMipmaps = false;
  return { texture, rects };
}

/** Small 4×4 white noise texture used to break up flat water/ground shading. */
export function tinyNoiseTexture(): Texture {
  const rng = Random.fromString('tiny');
  const { canvas, ctx } = makeCanvas(64, 64);
  noise(ctx, 64, 64, rng, 1, 2);
  const tex = toTexture(canvas, true, false);
  tex.magFilter = NearestFilter;
  return tex;
}
