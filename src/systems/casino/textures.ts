import { CanvasTexture, ClampToEdgeWrapping, LinearFilter, LinearMipmapLinearFilter, RepeatWrapping, SRGBColorSpace } from 'three';
import { WHEEL_SEGMENTS, WHEEL_SEGMENT_COUNT } from '../../casino/LuckyWheel';
import { WHEEL_ORDER, pocketColor, pocketLabel } from '../../casino/Roulette';
import type { SlotSymbol } from '../../casino/Slots';
import { Random } from '../../core/Random';

/** Canvas-generated textures for the Leonida Palms interior. Nothing is downloaded. */

const FONT = 'Impact, "Arial Narrow", "Helvetica Neue", Arial, sans-serif';

function makeCanvas(w: number, h: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  return { canvas, ctx };
}

function toTexture(canvas: HTMLCanvasElement, repeat: boolean): CanvasTexture {
  const tex = new CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = repeat ? RepeatWrapping : ClampToEdgeWrapping;
  tex.minFilter = LinearMipmapLinearFilter;
  tex.magFilter = LinearFilter;
  tex.anisotropy = 4;
  tex.colorSpace = SRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

/** Deep red carpet with a gold diamond lattice; one repeat covers 4 m. */
export function carpetTexture(): CanvasTexture {
  const size = 512;
  const { canvas, ctx } = makeCanvas(size, size);
  const rng = Random.fromString('casino-carpet');
  ctx.fillStyle = '#5a0f1c';
  ctx.fillRect(0, 0, size, size);
  for (let y = 0; y < size; y += 4) {
    for (let x = 0; x < size; x += 4) {
      ctx.fillStyle = `rgba(0,0,0,${rng.range(0, 0.18).toFixed(3)})`;
      ctx.fillRect(x, y, 4, 4);
    }
  }
  ctx.strokeStyle = 'rgba(214,170,60,0.85)';
  ctx.lineWidth = 3;
  const cell = size / 4;
  for (let i = -4; i <= 8; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cell, 0);
    ctx.lineTo(i * cell + size, size);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(i * cell, size);
    ctx.lineTo(i * cell + size, 0);
    ctx.stroke();
  }
  ctx.fillStyle = 'rgba(255,210,63,0.9)';
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      ctx.beginPath();
      ctx.arc(c * cell + cell / 2, r * cell + cell / 2, 9, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  return toTexture(canvas, true);
}

/** Blackjack felt: green with the payout arc and the table limits. */
export function blackjackFeltTexture(limits: { min: number; max: number }): CanvasTexture {
  const w = 1024;
  const h = 512;
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.fillStyle = '#0f5f37';
  ctx.fillRect(0, 0, w, h);
  const rng = Random.fromString('felt');
  for (let i = 0; i < 6000; i++) {
    ctx.fillStyle = `rgba(0,0,0,${rng.range(0.02, 0.1).toFixed(3)})`;
    ctx.fillRect(rng.range(0, w), rng.range(0, h), 2, 2);
  }
  ctx.strokeStyle = '#f2e6b3';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.arc(w / 2, h * 0.05, h * 0.78, Math.PI * 0.18, Math.PI * 0.82);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(w / 2, h * 0.05, h * 0.6, Math.PI * 0.2, Math.PI * 0.8);
  ctx.stroke();
  ctx.fillStyle = '#f2e6b3';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `bold 46px ${FONT}`;
  ctx.fillText('BLACKJACK PAYS 3 TO 2', w / 2, h * 0.5);
  ctx.font = `bold 32px ${FONT}`;
  ctx.fillText('DEALER STANDS ON SOFT 17', w / 2, h * 0.6);
  ctx.font = `bold 30px ${FONT}`;
  ctx.fillText(`$${limits.min.toLocaleString('en-US')} MIN  -  $${limits.max.toLocaleString('en-US')} MAX`, w / 2, h * 0.72);
  // Betting circles.
  ctx.lineWidth = 4;
  for (let i = 0; i < 5; i++) {
    const a = Math.PI * 0.25 + (i / 4) * Math.PI * 0.5;
    ctx.beginPath();
    ctx.arc(w / 2 + Math.cos(a) * h * 0.95, h * 0.02 + Math.sin(a) * h * 0.95, 28, 0, Math.PI * 2);
    ctx.stroke();
  }
  return toTexture(canvas, false);
}

/** Roulette betting layout (numbers grid with 0/00 and outside bets). */
export function rouletteLayoutTexture(): CanvasTexture {
  const w = 1024;
  const h = 512;
  const { canvas, ctx } = makeCanvas(w, h);
  ctx.fillStyle = '#0f5f37';
  ctx.fillRect(0, 0, w, h);
  const gridX = 120;
  const gridY = 40;
  const cellW = 64;
  const cellH = 96;
  ctx.strokeStyle = '#f2e6b3';
  ctx.lineWidth = 3;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `bold 34px ${FONT}`;
  for (let n = 1; n <= 36; n++) {
    const col = Math.floor((n - 1) / 3);
    const row = 2 - ((n - 1) % 3);
    const x = gridX + col * cellW;
    const y = gridY + row * cellH;
    ctx.fillStyle = pocketColor(n) === 'red' ? '#b3202a' : '#141414';
    ctx.fillRect(x, y, cellW, cellH);
    ctx.strokeRect(x, y, cellW, cellH);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(String(n), x + cellW / 2, y + cellH / 2);
  }
  ctx.fillStyle = '#1f7a3a';
  ctx.fillRect(gridX - 80, gridY, 80, cellH * 1.5);
  ctx.fillRect(gridX - 80, gridY + cellH * 1.5, 80, cellH * 1.5);
  ctx.strokeRect(gridX - 80, gridY, 80, cellH * 1.5);
  ctx.strokeRect(gridX - 80, gridY + cellH * 1.5, 80, cellH * 1.5);
  ctx.fillStyle = '#ffffff';
  ctx.fillText('00', gridX - 40, gridY + cellH * 0.75);
  ctx.fillText('0', gridX - 40, gridY + cellH * 2.25);
  ctx.font = `bold 24px ${FONT}`;
  for (let r = 0; r < 3; r++) {
    ctx.strokeRect(gridX + 12 * cellW, gridY + r * cellH, 72, cellH);
    ctx.fillText('2 TO 1', gridX + 12 * cellW + 36, gridY + r * cellH + cellH / 2);
  }
  const outsideY = gridY + 3 * cellH;
  const dozens = ['1ST 12', '2ND 12', '3RD 12'];
  for (let d = 0; d < 3; d++) {
    ctx.strokeRect(gridX + d * 4 * cellW, outsideY, 4 * cellW, 60);
    ctx.fillText(dozens[d], gridX + d * 4 * cellW + 2 * cellW, outsideY + 30);
  }
  const even = ['1 TO 18', 'EVEN', 'RED', 'BLACK', 'ODD', '19 TO 36'];
  for (let e = 0; e < 6; e++) {
    const x = gridX + e * 2 * cellW;
    if (even[e] === 'RED') ctx.fillStyle = '#b3202a';
    else if (even[e] === 'BLACK') ctx.fillStyle = '#141414';
    else ctx.fillStyle = '#0f5f37';
    ctx.fillRect(x, outsideY + 60, 2 * cellW, 60);
    ctx.strokeRect(x, outsideY + 60, 2 * cellW, 60);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(even[e], x + cellW, outsideY + 90);
  }
  return toTexture(canvas, false);
}

/** Top view of the American wheel; pocket 0 is at angle 0 (the +X axis of the texture). */
export function rouletteWheelTexture(): CanvasTexture {
  const size = 1024;
  const { canvas, ctx } = makeCanvas(size, size);
  const cx = size / 2;
  const cy = size / 2;
  const outer = size * 0.49;
  const inner = size * 0.36;
  ctx.fillStyle = '#5b3a1e';
  ctx.beginPath();
  ctx.arc(cx, cy, outer, 0, Math.PI * 2);
  ctx.fill();
  const step = (Math.PI * 2) / WHEEL_ORDER.length;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < WHEEL_ORDER.length; i++) {
    const pocket = WHEEL_ORDER[i];
    const a0 = i * step - step / 2;
    const a1 = a0 + step;
    const color = pocketColor(pocket);
    ctx.fillStyle = color === 'red' ? '#c0262f' : color === 'black' ? '#151515' : '#1d8a3f';
    ctx.beginPath();
    ctx.arc(cx, cy, outer * 0.96, a0, a1);
    ctx.arc(cx, cy, inner, a1, a0, true);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#d6aa3c';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(i * step);
    ctx.translate(outer * 0.86, 0);
    ctx.rotate(Math.PI / 2);
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold 40px ${FONT}`;
    ctx.fillText(pocketLabel(pocket), 0, 0);
    ctx.restore();
  }
  ctx.fillStyle = '#3b2a18';
  ctx.beginPath();
  ctx.arc(cx, cy, inner, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#d6aa3c';
  ctx.beginPath();
  ctx.arc(cx, cy, inner * 0.3, 0, Math.PI * 2);
  ctx.fill();
  return toTexture(canvas, false);
}

/** Lucky Wheel face; segment 0 is centred on angle 0 (+X of the texture). */
export function luckyWheelTexture(): CanvasTexture {
  const size = 1024;
  const { canvas, ctx } = makeCanvas(size, size);
  const cx = size / 2;
  const cy = size / 2;
  const outer = size * 0.49;
  const step = (Math.PI * 2) / WHEEL_SEGMENT_COUNT;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < WHEEL_SEGMENT_COUNT; i++) {
    const seg = WHEEL_SEGMENTS[i];
    const a0 = i * step - step / 2;
    ctx.fillStyle = seg.color;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, outer, a0, a0 + step);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#fff6d5';
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(i * step);
    ctx.translate(outer * 0.62, 0);
    ctx.fillStyle = '#08090d';
    ctx.font = `bold ${seg.label.length > 8 ? 34 : 44}px ${FONT}`;
    ctx.fillText(seg.label, 0, 0);
    ctx.restore();
  }
  ctx.fillStyle = '#08090d';
  ctx.beginPath();
  ctx.arc(cx, cy, outer * 0.16, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ffd23f';
  ctx.font = `bold 48px ${FONT}`;
  ctx.fillText('LUCKY', cx, cy - 22);
  ctx.fillText('WHEEL', cx, cy + 26);
  return toTexture(canvas, false);
}

/** Two-line illuminated sign (title in gold, subtitle in white). */
export function signTexture(title: string, subtitle = '', accent = '#ffd23f'): CanvasTexture {
  const { canvas, ctx } = makeCanvas(512, 192);
  ctx.fillStyle = '#12121a';
  ctx.fillRect(0, 0, 512, 192);
  ctx.strokeStyle = accent;
  ctx.lineWidth = 8;
  ctx.strokeRect(8, 8, 496, 176);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = accent;
  ctx.shadowColor = accent;
  ctx.shadowBlur = 18;
  if (subtitle) {
    ctx.font = `bold 44px ${FONT}`;
    ctx.fillText(title, 256, 60, 470);
    ctx.shadowBlur = 0;
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold 52px ${FONT}`;
    ctx.fillText(subtitle, 256, 128, 470);
  } else {
    ctx.font = `bold 72px ${FONT}`;
    ctx.fillText(title, 256, 96, 470);
  }
  ctx.shadowBlur = 0;
  return toTexture(canvas, false);
}

export function limitSignTexture(min: number, max: number): CanvasTexture {
  return signTexture('BLACKJACK', `$${min.toLocaleString('en-US')} - $${max.toLocaleString('en-US')}`);
}

const SYMBOL_COLORS: Record<SlotSymbol, string> = {
  cherry: '#ff3b5c',
  bar: '#ffd23f',
  seven: '#ff8a1f',
  diamond: '#35e6ff',
  gator: '#4ee06a',
};

export const SYMBOL_GLYPH: Record<SlotSymbol, string> = {
  cherry: '\u25CF\u25CF',
  bar: 'BAR',
  seven: '7',
  diamond: '\u25C6',
  gator: 'GATOR',
};

/** Emissive screen of one slot machine; redraw when the reels stop. */
export class SlotScreen {
  readonly texture: CanvasTexture;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly w = 256;
  private readonly h = 160;

  constructor() {
    const { canvas, ctx } = makeCanvas(this.w, this.h);
    this.ctx = ctx;
    this.texture = toTexture(canvas, false);
    this.texture.generateMipmaps = false;
    this.texture.minFilter = LinearFilter;
    this.draw(null, false);
  }

  draw(reels: readonly SlotSymbol[] | null, spinning: boolean): void {
    const ctx = this.ctx;
    ctx.fillStyle = '#080a12';
    ctx.fillRect(0, 0, this.w, this.h);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const cellW = this.w / 3;
    for (let i = 0; i < 3; i++) {
      ctx.fillStyle = '#f5f0e0';
      ctx.fillRect(i * cellW + 8, 30, cellW - 16, this.h - 60);
      const s = reels?.[i];
      if (spinning) {
        ctx.fillStyle = '#8a8a99';
        ctx.font = `bold 40px ${FONT}`;
        ctx.fillText('?', i * cellW + cellW / 2, this.h / 2);
      } else if (s) {
        ctx.fillStyle = SYMBOL_COLORS[s];
        ctx.font = `bold ${s === 'gator' ? 22 : 36}px ${FONT}`;
        ctx.fillText(SYMBOL_GLYPH[s], i * cellW + cellW / 2, this.h / 2);
      }
    }
    ctx.fillStyle = '#ffd23f';
    ctx.font = `bold 20px ${FONT}`;
    ctx.fillText(spinning ? 'GOOD LUCK' : 'LEONIDA PALMS SLOTS', this.w / 2, 14);
    ctx.fillStyle = '#35e6ff';
    ctx.font = `16px ${FONT}`;
    ctx.fillText('JACKPOT 500x  -  BET 10 / 50 / 100', this.w / 2, this.h - 14);
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.texture.dispose();
  }
}
