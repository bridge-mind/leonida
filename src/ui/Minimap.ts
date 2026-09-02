import { Config } from '../core/Config';
import type { Game } from '../core/Game';
import type { GameState } from '../core/GameState';
import type { NavigationSystem } from '../systems/navigation';
import { UiConfig } from '../systems/ui/config';
import { el, setText, toggleClass, type UiComponent } from './dom';
import { formatDistance } from './format';
import {
  drawBlips,
  drawPlayerArrow,
  drawRoute,
  drawStaticLayer,
  drawWantedAreas,
  setViewRotation,
  toScreen,
  UnitDots,
  type MapView,
  type ScreenRect,
} from './MapDraw';
import type { MapLayer } from './MapLayer';
import './minimap.css';

const M = UiConfig.minimap;

/**
 * Bottom-left rotating minimap with GTA V health/armor bars underneath. The static
 * city is a pre-rendered layer; everything drawn per frame is allocation-free.
 */
export class Minimap implements UiComponent {
  readonly el: HTMLElement;
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly healthBar: HTMLElement;
  private readonly healthFill: HTMLElement;
  private readonly armorFill: HTMLElement;
  private readonly distanceEl: HTMLElement;
  private readonly units: UnitDots;
  private readonly view: MapView = { cx: 0, cy: 0, wx: 0, wz: 0, zoom: M.zoomOnFoot, cos: 1, sin: 0 };
  private readonly rim: ScreenRect = { x0: 0, y0: 0, x1: 0, y1: 0 };
  private dpr = 1;
  private zoom = M.zoomOnFoot;
  private lastHealth = -1;
  private lastArmor = -1;
  private lastDistance = -1;

  constructor(
    private readonly game: Game,
    private readonly layer: MapLayer,
  ) {
    this.el = el('div', 'minimap');
    this.canvas = el('canvas');
    this.el.appendChild(this.canvas);
    const bars = el('div', 'minimap-bars');
    this.healthBar = el('div', 'minimap-bar minimap-bar-health');
    this.healthFill = el('div', 'minimap-bar-fill');
    this.healthBar.appendChild(this.healthFill);
    const armorBar = el('div', 'minimap-bar minimap-bar-armor');
    this.armorFill = el('div', 'minimap-bar-fill');
    armorBar.appendChild(this.armorFill);
    bars.append(this.healthBar, armorBar);
    this.distanceEl = el('div', 'minimap-distance');
    this.el.append(bars, this.distanceEl);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2d canvas unavailable for the minimap');
    this.ctx = ctx;
    this.units = new UnitDots(game);
    this.resize();
    window.addEventListener('resize', this.resize);
  }

  private readonly resize = (): void => {
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(M.width * this.dpr);
    this.canvas.height = Math.round(M.height * this.dpr);
  };

  update(state: GameState, realDt: number): void {
    this.updateBars(state);
    this.updateDistance(state);
    this.draw(state, realDt);
  }

  private updateBars(state: GameState): void {
    const h = Math.round(Math.max(0, Math.min(1, state.health / Config.player.maxHealth)) * 100);
    if (h !== this.lastHealth) {
      this.lastHealth = h;
      this.healthFill.style.transform = `scaleX(${h / 100})`;
      toggleClass(this.healthBar, 'low', h > 0 && h <= 25);
    }
    const a = Math.round(Math.max(0, Math.min(1, state.armor / Config.player.maxArmor)) * 100);
    if (a !== this.lastArmor) {
      this.lastArmor = a;
      this.armorFill.style.transform = `scaleX(${a / 100})`;
    }
  }

  private updateDistance(state: GameState): void {
    if (!state.waypoint) {
      if (this.lastDistance !== -1) {
        this.lastDistance = -1;
        setText(this.distanceEl, '');
      }
      return;
    }
    const nav = this.game.systems.get<NavigationSystem>('navigation');
    const d = nav ? nav.distanceToWaypoint : Math.hypot(state.waypoint.x - state.playerPos.x, state.waypoint.z - state.playerPos.z);
    const rounded = Math.round(d / 5) * 5;
    if (rounded !== this.lastDistance) {
      this.lastDistance = rounded;
      setText(this.distanceEl, formatDistance(rounded));
    }
  }

  private draw(state: GameState, realDt: number): void {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const dpr = this.dpr;

    // Zoom out with vehicle speed; the blend is smoothed so gear changes do not pop.
    const speed = state.vehicle ? Math.min(1, Math.abs(state.vehicle.speedKmh) / M.zoomSpeedKmh) : 0;
    const targetZoom = M.zoomOnFoot + (M.zoomAtSpeed - M.zoomOnFoot) * speed;
    this.zoom += (targetZoom - this.zoom) * Math.min(1, realDt * M.zoomSmoothing);

    const view = this.view;
    view.cx = w / 2;
    view.cy = h * 0.58;
    view.wx = state.playerPos.x;
    view.wz = state.playerPos.z;
    view.zoom = this.zoom * dpr;
    // Rig heading 0 faces +Z (south, screen-down when north is up); rotating by heading - π
    // brings the player's forward vector to screen-up, so heading π (north) leaves north up.
    setViewRotation(view, state.playerHeading - Math.PI);

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.beginPath();
    roundedRect(ctx, 0, 0, w, h, M.cornerRadius * dpr);
    ctx.clip();

    drawStaticLayer(ctx, view, this.layer);
    drawWantedAreas(ctx, view, state.wanted, this.game.time.realElapsed);
    drawRoute(ctx, view, state.route, M.routeWidth * dpr, UiConfig.map.colors.route);

    this.rim.x0 = 0;
    this.rim.y0 = 0;
    this.rim.x1 = w;
    this.rim.y1 = h;
    drawBlips(ctx, view, state.blips, M.blipSize * dpr, this.rim, M.rimMargin * dpr);
    this.units.draw(ctx, view, M.unitQueryRadius, (M.dotSize * dpr) / 2, M.maxUnits);
    drawPlayerArrow(ctx, view.cx, view.cy, -Math.PI / 2, 7 * dpr);
    this.drawNorth(ctx, w, h, dpr);
    ctx.restore();
  }

  /** 'N' marker on the rim in the direction of world north (-Z). */
  private drawNorth(ctx: CanvasRenderingContext2D, w: number, h: number, dpr: number): void {
    const view = this.view;
    const p = toScreen(view, view.wx, view.wz - 1);
    let dx = p.x - view.cx;
    let dy = p.y - view.cy;
    const len = Math.hypot(dx, dy) || 1;
    dx /= len;
    dy /= len;
    const margin = 11 * dpr;
    const tx = dx !== 0 ? ((dx > 0 ? w - view.cx : view.cx) - margin) / Math.abs(dx) : Infinity;
    const ty = dy !== 0 ? ((dy > 0 ? h - view.cy : view.cy) - margin) / Math.abs(dy) : Infinity;
    const t = Math.min(tx, ty);
    const x = view.cx + dx * t;
    const y = view.cy + dy * t;
    ctx.fillStyle = 'rgba(0,0,0,0.7)';
    ctx.beginPath();
    ctx.arc(x, y, 7.5 * dpr, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#ffffff';
    ctx.font = `bold ${Math.round(10 * dpr)}px Impact, "Arial Narrow", Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('N', x, y + 0.5 * dpr);
  }

  dispose(): void {
    window.removeEventListener('resize', this.resize);
    this.el.remove();
  }
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
