import type { Game } from '../core/Game';
import type { GameState } from '../core/GameState';
import { UiConfig } from '../systems/ui/config';
import { BLIP_STYLES, LEGEND_KINDS } from './blipStyles';
import { el, setText, toggleClass, type UiComponent } from './dom';
import './fullmap.css';
import {
  drawBlips,
  drawPlayerArrow,
  drawRoute,
  drawStaticLayer,
  drawWantedAreas,
  toScreen,
  toWorld,
  UnitDots,
  type MapView,
} from './MapDraw';
import type { MapLayer } from './MapLayer';

const F = UiConfig.map;

/**
 * Full-screen pannable map (M). Left click places a waypoint, clicking the existing
 * waypoint clears it. Input blocking and pointer lock are handled by the UI system.
 */
export class FullMap implements UiComponent {
  readonly el: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly legend: HTMLElement;
  private readonly coords: HTMLElement;
  private readonly units: UnitDots;
  private readonly view: MapView = { cx: 0, cy: 0, wx: 0, wz: 0, zoom: F.startZoom, cos: 1, sin: 0 };
  private readonly world = { x: 0, z: 0 };
  private dpr = 1;
  private dragging = false;
  private dragMoved = 0;
  private lastMouseX = 0;
  private lastMouseY = 0;
  private legendCount = -1;
  private lastCoords = '';
  visible = false;
  /** Set by the UI system so Esc/M and the close path share one implementation. */
  onClose: (() => void) | null = null;

  constructor(
    private readonly game: Game,
    private readonly layer: MapLayer,
  ) {
    this.el = el('div', 'fullmap');
    this.el.hidden = true;
    this.canvas = el('canvas');
    const side = el('div', 'fullmap-side');
    side.append(el('div', 'fullmap-title', 'Vice City'), el('div', 'fullmap-sub', 'Leonida'));
    this.legend = el('div', 'fullmap-legend');
    side.appendChild(this.legend);
    const hints = el('div', 'fullmap-hints');
    hints.innerHTML = '<b>LMB</b> Set / clear waypoint<br><b>Drag</b> Pan<br><b>Wheel</b> Zoom<br><b>M</b> / <b>Esc</b> Close';
    side.appendChild(hints);
    this.coords = el('div', 'fullmap-coords');
    this.el.append(this.canvas, side, this.coords);
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2d canvas unavailable for the full map');
    this.ctx = ctx;
    this.units = new UnitDots(game);
    this.bind();
    window.addEventListener('resize', this.resize);
  }

  private bind(): void {
    const c = this.canvas;
    c.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      this.dragging = true;
      this.dragMoved = 0;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
      toggleClass(this.el, 'dragging', true);
    });
    window.addEventListener('mousemove', (e) => {
      if (!this.visible) return;
      this.updateCoords(e.clientX, e.clientY);
      if (!this.dragging) return;
      const dx = e.clientX - this.lastMouseX;
      const dy = e.clientY - this.lastMouseY;
      this.lastMouseX = e.clientX;
      this.lastMouseY = e.clientY;
      this.dragMoved += Math.abs(dx) + Math.abs(dy);
      this.view.wx -= (dx * this.dpr) / this.view.zoom;
      this.view.wz -= (dy * this.dpr) / this.view.zoom;
    });
    window.addEventListener('mouseup', (e) => {
      if (!this.dragging || e.button !== 0) return;
      this.dragging = false;
      toggleClass(this.el, 'dragging', false);
      if (this.dragMoved <= F.clickSlop) this.click(e.clientX, e.clientY);
    });
    c.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1 / F.wheelZoomFactor : F.wheelZoomFactor;
        this.zoomAt(e.clientX, e.clientY, factor);
      },
      { passive: false },
    );
    c.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  private readonly resize = (): void => {
    if (!this.visible) return;
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.round(rect.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(rect.height * this.dpr));
  };

  open(): void {
    if (this.visible) return;
    this.visible = true;
    this.el.hidden = false;
    this.resize();
    const p = this.game.state.playerPos;
    this.view.wx = p.x;
    this.view.wz = p.z;
    this.view.zoom = F.startZoom * this.dpr;
    this.legendCount = -1;
    this.game.audio.play('ui.map.open');
  }

  close(): void {
    if (!this.visible) return;
    this.visible = false;
    this.el.hidden = true;
    this.dragging = false;
    this.game.audio.play('ui.close');
    this.onClose?.();
  }

  toggle(): void {
    if (this.visible) this.close();
    else this.open();
  }

  private zoomAt(clientX: number, clientY: number, factor: number): void {
    const rect = this.canvas.getBoundingClientRect();
    const sx = (clientX - rect.left) * this.dpr;
    const sy = (clientY - rect.top) * this.dpr;
    const before = toWorld(this.view, sx, sy, this.world);
    const bx = before.x;
    const bz = before.z;
    const min = F.minZoom * this.dpr;
    const max = F.maxZoom * this.dpr;
    this.view.zoom = Math.min(max, Math.max(min, this.view.zoom * factor));
    // Keep the world point under the cursor fixed while zooming.
    const after = toWorld(this.view, sx, sy, this.world);
    this.view.wx += bx - after.x;
    this.view.wz += bz - after.z;
  }

  private click(clientX: number, clientY: number): void {
    const nav = this.game.navigation;
    if (!nav) return;
    const rect = this.canvas.getBoundingClientRect();
    const w = toWorld(this.view, (clientX - rect.left) * this.dpr, (clientY - rect.top) * this.dpr, this.world);
    const current = nav.waypoint;
    if (current && Math.hypot(current.x - w.x, current.z - w.z) < F.clearRadius) {
      nav.setWaypoint(null);
      this.game.audio.play('ui.back');
      return;
    }
    nav.setWaypoint({ x: w.x, y: 0, z: w.z });
    this.game.audio.play('ui.map.pin');
  }

  private updateCoords(clientX: number, clientY: number): void {
    const rect = this.canvas.getBoundingClientRect();
    const w = toWorld(this.view, (clientX - rect.left) * this.dpr, (clientY - rect.top) * this.dpr, this.world);
    const district = this.game.systems.has('world') ? this.game.world.districtAt({ x: w.x, y: 0, z: w.z }) : null;
    const text = `${Math.round(w.x)}, ${Math.round(w.z)}  ${district ? district.name : 'Water'}`;
    if (text !== this.lastCoords) {
      this.lastCoords = text;
      setText(this.coords, text);
    }
  }

  update(state: GameState): void {
    if (!this.visible) return;
    if (state.blips.length !== this.legendCount) this.rebuildLegend(state);
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const view = this.view;
    view.cx = w / 2;
    view.cy = h / 2;
    view.cos = 1;
    view.sin = 0;
    ctx.fillStyle = F.colors.water;
    ctx.fillRect(0, 0, w, h);
    drawStaticLayer(ctx, view, this.layer);
    drawWantedAreas(ctx, view, state.wanted, this.game.time.realElapsed);
    drawRoute(ctx, view, state.route, F.routeWidth * this.dpr, F.colors.route);
    drawBlips(ctx, view, state.blips, F.blipSize * this.dpr, null, 0);
    this.units.draw(ctx, view, 4000, 3 * this.dpr, 200);
    const p = toScreen(view, state.playerPos.x, state.playerPos.z);
    // Heading 0 faces +Z (screen down): the screen angle of the forward vector (sin h, cos h).
    drawPlayerArrow(ctx, p.x, p.y, Math.atan2(Math.cos(state.playerHeading), Math.sin(state.playerHeading)), 9 * this.dpr);
  }

  private rebuildLegend(state: GameState): void {
    this.legendCount = state.blips.length;
    this.legend.replaceChildren();
    const present = new Set(state.blips.map((b) => b.kind));
    for (const kind of LEGEND_KINDS) {
      const style = BLIP_STYLES[kind];
      const row = el('div', 'fullmap-legend-row');
      if (!present.has(kind)) row.classList.add('dim');
      const glyph = el('span', `fullmap-legend-glyph${style.shape === 'dot' ? ' dot' : ''}`);
      glyph.style.background = style.color;
      glyph.style.color = style.glyphColor;
      if (style.shape === 'square') glyph.textContent = style.glyph === 'label' ? 'A' : style.glyph;
      if (style.shape === 'pin') glyph.textContent = '\u25BC';
      row.append(glyph, el('span', undefined, style.legend));
      this.legend.appendChild(row);
    }
    for (const b of state.blips) {
      if (b.kind !== 'mission' && b.kind !== 'contact') continue;
      const style = BLIP_STYLES[b.kind];
      const row = el('div', 'fullmap-legend-row');
      const glyph = el('span', 'fullmap-legend-glyph', b.label ? b.label.charAt(0).toUpperCase() : '!');
      glyph.style.background = style.color;
      row.append(glyph, el('span', undefined, b.label ?? b.id));
      this.legend.appendChild(row);
    }
  }

  dispose(): void {
    window.removeEventListener('resize', this.resize);
    this.el.remove();
  }
}
