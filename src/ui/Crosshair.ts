import type { Game } from '../core/Game';
import './crosshair.css';

export interface CrosshairView {
  visible: boolean;
  /** 0..1 while reloading, null otherwise. */
  reload01: number | null;
  scope: boolean;
}

const RELOAD_CIRCUMFERENCE = 2 * Math.PI * 16;

/**
 * Center crosshair shown while aiming (or driving with a window weapon), hit markers
 * (white hit / red kill) from `weapon:hit`, a reload arc and the sniper scope overlay.
 */
export class Crosshair {
  private readonly root: HTMLDivElement;
  private readonly marker: HTMLDivElement;
  private readonly reload: SVGSVGElement;
  private readonly reloadArc: SVGCircleElement;
  private readonly scope: HTMLDivElement;
  private readonly offs: (() => void)[] = [];
  private wheelOpen = false;
  private visible = false;
  private reloading = false;
  private scopeOn = false;
  private lastArc = -1;

  constructor(
    private readonly game: Game,
    parent: HTMLElement,
  ) {
    this.root = document.createElement('div');
    this.root.className = 'crosshair';
    this.root.innerHTML =
      '<div class="crosshair-dot"></div>' +
      '<div class="crosshair-line n"></div><div class="crosshair-line e"></div>' +
      '<div class="crosshair-line s"></div><div class="crosshair-line w"></div>' +
      '<div class="crosshair-marker"><span></span><span></span><span></span><span></span></div>';
    this.marker = this.root.querySelector('.crosshair-marker') as HTMLDivElement;

    const svgNs = 'http://www.w3.org/2000/svg';
    this.reload = document.createElementNS(svgNs, 'svg');
    this.reload.setAttribute('class', 'crosshair-reload');
    this.reload.setAttribute('viewBox', '0 0 40 40');
    const track = document.createElementNS(svgNs, 'circle');
    track.setAttribute('class', 'track');
    track.setAttribute('cx', '20');
    track.setAttribute('cy', '20');
    track.setAttribute('r', '16');
    this.reloadArc = document.createElementNS(svgNs, 'circle');
    this.reloadArc.setAttribute('cx', '20');
    this.reloadArc.setAttribute('cy', '20');
    this.reloadArc.setAttribute('r', '16');
    this.reload.append(track, this.reloadArc);
    this.root.append(this.reload);

    this.scope = document.createElement('div');
    this.scope.className = 'scope';
    this.scope.innerHTML = '<div class="scope-ring"></div>';

    parent.append(this.scope, this.root);
    this.offs.push(
      game.events.on('weapon:hit', ({ killed }) => this.flash(killed)),
      game.events.on('weapon:wheelToggled', ({ open }) => {
        this.wheelOpen = open;
      }),
    );
  }

  update(view: CrosshairView): void {
    const visible = view.visible && !this.wheelOpen;
    if (visible !== this.visible) {
      this.visible = visible;
      this.root.classList.toggle('visible', visible);
    }
    const reloading = view.reload01 !== null;
    if (reloading !== this.reloading) {
      this.reloading = reloading;
      this.reload.classList.toggle('active', reloading);
      this.lastArc = -1;
    }
    if (reloading) {
      const p = Math.min(1, Math.max(0, view.reload01 ?? 0));
      // Quantise so the DOM is only touched when the arc visibly moves.
      const q = Math.round(p * 60);
      if (q !== this.lastArc) {
        this.lastArc = q;
        this.reloadArc.setAttribute('stroke-dasharray', `${(RELOAD_CIRCUMFERENCE * q) / 60} ${RELOAD_CIRCUMFERENCE}`);
      }
    }
    const scope = view.scope && !this.wheelOpen;
    if (scope !== this.scopeOn) {
      this.scopeOn = scope;
      this.scope.classList.toggle('visible', scope);
    }
  }

  private flash(killed: boolean): void {
    const m = this.marker;
    m.classList.remove('flash');
    // Forcing a reflow restarts the CSS animation when hits arrive back to back.
    void m.offsetWidth;
    m.classList.toggle('kill', killed);
    m.classList.add('flash');
    this.game.audio.play(killed ? 'ui.hitmarker.kill' : 'ui.hitmarker');
  }

  get element(): HTMLElement {
    return this.root;
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.root.remove();
    this.scope.remove();
  }
}
