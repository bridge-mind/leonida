import type { IPed, IVehicle, V3 } from '../core/entities';
import type { Game } from '../core/Game';
import type { Blip, GameState } from '../core/GameState';
import { BLIP_STYLES } from './blipStyles';
import type { MapLayer } from './MapLayer';

/** World→screen mapping shared by the minimap (rotated) and the full map (north up). */
export interface MapView {
  /** Screen center in canvas pixels. */
  cx: number;
  cy: number;
  /** World point under the screen center. */
  wx: number;
  wz: number;
  /** Pixels per meter. */
  zoom: number;
  /** Rotation applied after scaling (radians; 0 = north up). */
  cos: number;
  sin: number;
}

export interface ScreenRect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const pt = { x: 0, y: 0 };

export function setViewRotation(view: MapView, angle: number): void {
  view.cos = Math.cos(angle);
  view.sin = Math.sin(angle);
}

export function toScreen(view: MapView, wx: number, wz: number, out = pt): { x: number; y: number } {
  const dx = (wx - view.wx) * view.zoom;
  const dz = (wz - view.wz) * view.zoom;
  out.x = view.cx + dx * view.cos - dz * view.sin;
  out.y = view.cy + dx * view.sin + dz * view.cos;
  return out;
}

export function toWorld(view: MapView, sx: number, sy: number, out: { x: number; z: number }): { x: number; z: number } {
  const dx = sx - view.cx;
  const dy = sy - view.cy;
  // Inverse rotation, then unscale.
  const rx = dx * view.cos + dy * view.sin;
  const rz = -dx * view.sin + dy * view.cos;
  out.x = view.wx + rx / view.zoom;
  out.z = view.wz + rz / view.zoom;
  return out;
}

/** Blits the static layer under the current view. Call inside the caller's clip. */
export function drawStaticLayer(ctx: CanvasRenderingContext2D, view: MapView, layer: MapLayer): void {
  const scale = view.zoom / layer.pxPerMeter;
  ctx.save();
  ctx.translate(view.cx, view.cy);
  ctx.rotate(Math.atan2(view.sin, view.cos));
  ctx.scale(scale, scale);
  ctx.translate(-layer.toLayer(view.wx), -layer.toLayer(view.wz));
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(layer.canvas, 0, 0);
  ctx.restore();
}

export function drawRoute(ctx: CanvasRenderingContext2D, view: MapView, route: readonly V3[], width: number, color: string): void {
  if (route.length < 2) return;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'rgba(0,0,0,0.55)';
  ctx.lineWidth = width + 2;
  strokePolyline(ctx, view, route);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  strokePolyline(ctx, view, route);
}

function strokePolyline(ctx: CanvasRenderingContext2D, view: MapView, route: readonly V3[]): void {
  ctx.beginPath();
  for (let i = 0; i < route.length; i++) {
    const p = toScreen(view, route[i].x, route[i].z);
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
}

/** Search circle (flashing while searching) and hot scenes. */
export function drawWantedAreas(ctx: CanvasRenderingContext2D, view: MapView, wanted: GameState['wanted'], seconds: number): void {
  for (const h of wanted.hotScenes) {
    const p = toScreen(view, h.position.x, h.position.z);
    ctx.fillStyle = 'rgba(255, 59, 59, 0.22)';
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(3, h.radius * view.zoom), 0, Math.PI * 2);
    ctx.fill();
  }
  if (wanted.state === 'searching' && wanted.searchCenter) {
    const p = toScreen(view, wanted.searchCenter.x, wanted.searchCenter.z);
    const r = Math.max(4, wanted.searchRadius * view.zoom);
    const pulse = 0.5 + 0.5 * Math.sin(seconds * 6);
    ctx.fillStyle = `rgba(255, 59, 59, ${0.12 + 0.16 * pulse})`;
    ctx.strokeStyle = `rgba(255, 90, 90, ${0.45 + 0.5 * pulse})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}

export function drawBlips(
  ctx: CanvasRenderingContext2D,
  view: MapView,
  blips: readonly Blip[],
  size: number,
  rim: ScreenRect | null,
  rimMargin: number,
): void {
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `bold ${Math.round(size * 0.72)}px Impact, "Arial Narrow", Arial, sans-serif`;
  for (const b of blips) {
    const style = BLIP_STYLES[b.kind];
    if (!style) continue;
    const p = toScreen(view, b.position.x, b.position.z);
    let x = p.x;
    let y = p.y;
    if (rim) {
      const inside = x >= rim.x0 && x <= rim.x1 && y >= rim.y0 && y <= rim.y1;
      if (!inside) {
        if (!style.pinToRim) continue;
        x = Math.min(rim.x1 - rimMargin, Math.max(rim.x0 + rimMargin, x));
        y = Math.min(rim.y1 - rimMargin, Math.max(rim.y0 + rimMargin, y));
      }
    }
    switch (style.shape) {
      case 'area':
        ctx.fillStyle = style.color;
        ctx.beginPath();
        ctx.arc(x, y, Math.max(4, 30 * view.zoom), 0, Math.PI * 2);
        ctx.fill();
        break;
      case 'dot':
        drawDot(ctx, x, y, size * 0.45, style.color);
        break;
      case 'pin':
        drawPin(ctx, x, y, size, style.color);
        break;
      case 'square': {
        const half = size / 2;
        ctx.fillStyle = 'rgba(0,0,0,0.75)';
        ctx.fillRect(x - half - 1, y - half - 1, size + 2, size + 2);
        ctx.fillStyle = style.color;
        ctx.fillRect(x - half, y - half, size, size);
        const glyph = style.glyph === 'label' ? (b.label ? b.label.charAt(0).toUpperCase() : '!') : style.glyph;
        if (glyph) {
          ctx.fillStyle = style.glyphColor;
          ctx.fillText(glyph, x, y + 0.5);
        }
        break;
      }
    }
  }
}

export function drawDot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, color: string): void {
  ctx.fillStyle = 'rgba(0,0,0,0.8)';
  ctx.beginPath();
  ctx.arc(x, y, r + 1.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}

/** GTA-style waypoint: a filled pin whose tip sits on the target. */
function drawPin(ctx: CanvasRenderingContext2D, x: number, y: number, size: number, color: string): void {
  const r = size * 0.42;
  const top = y - size * 0.95;
  ctx.fillStyle = color;
  ctx.strokeStyle = 'rgba(0,0,0,0.8)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - r, top + r * 0.3);
  ctx.arc(x, top, r, Math.PI * 0.85, Math.PI * 2.15);
  ctx.lineTo(x, y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.arc(x, top, r * 0.38, 0, Math.PI * 2);
  ctx.fill();
}

/** White arrow at (x, y); `angle` is the on-screen facing direction in radians (0 = right). */
export function drawPlayerArrow(ctx: CanvasRenderingContext2D, x: number, y: number, angle: number, size: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(angle + Math.PI / 2);
  ctx.beginPath();
  ctx.moveTo(0, -size);
  ctx.lineTo(size * 0.7, size * 0.8);
  ctx.lineTo(0, size * 0.35);
  ctx.lineTo(-size * 0.7, size * 0.8);
  ctx.closePath();
  ctx.fillStyle = '#ffffff';
  ctx.strokeStyle = 'rgba(0,0,0,0.85)';
  ctx.lineWidth = 1.5;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

/** Scratch buffers so unit queries never allocate per frame. */
export class UnitDots {
  private readonly vehicles: IVehicle[] = [];
  private readonly cops: IPed[] = [];

  constructor(private readonly game: Game) {}

  /**
   * Vehicles show as white dots only while wanted (GTA V behaviour); cops are blue,
   * red once the police have eyes on the player.
   */
  draw(ctx: CanvasRenderingContext2D, view: MapView, radius: number, dotRadius: number, max: number): void {
    const game = this.game;
    const state = game.state;
    const center = state.playerPos;
    const copColor = game.wanted?.state === 'active' ? '#ff3b3b' : '#3a7bff';

    const vehicles = game.vehicles;
    if (vehicles) {
      vehicles.query(center, radius, this.vehicles);
      const n = Math.min(max, this.vehicles.length);
      for (let i = 0; i < n; i++) {
        const v = this.vehicles[i];
        if (!v.isAlive) continue;
        if (!v.isPolice && state.wanted.level === 0) continue;
        const p = toScreen(view, v.position.x, v.position.z);
        drawDot(ctx, p.x, p.y, dotRadius, v.isPolice ? copColor : '#f2f2f2');
      }
      this.vehicles.length = 0;
    }

    const police = game.police;
    if (police) {
      police.copsNear(center, radius, this.cops);
      const n = Math.min(max, this.cops.length);
      for (let i = 0; i < n; i++) {
        const c = this.cops[i];
        if (!c.isAlive || c.vehicle) continue;
        const p = toScreen(view, c.position.x, c.position.z);
        drawDot(ctx, p.x, p.y, dotRadius * 0.9, copColor);
      }
      this.cops.length = 0;
    }
  }
}
