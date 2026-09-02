import { Vector3 } from 'three';
import type { IPed } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { CharacterId } from '../../core/peers';
import { CharactersConfig } from '../../systems/characters/config';
import { el } from '../dom';
import './missions.css';

const A = CharactersConfig.ability;
const tmp = new Vector3();

/**
 * Adrenaline visuals: a coloured vignette while the ability runs and, for Jason, red
 * markers projected over nearby ped heads (DOM, pooled, ≤ `maxMarkers`).
 */
export class AbilityFx {
  readonly el: HTMLElement;
  private readonly vignette: HTMLElement;
  private readonly markers: HTMLElement[] = [];
  private readonly buf: IPed[] = [];
  private markersOn = false;

  constructor(
    private readonly game: Game,
    parent: HTMLElement,
  ) {
    this.el = el('div', 'ability-fx');
    this.vignette = el('div', 'ability-vignette');
    this.el.append(this.vignette);
    for (let i = 0; i < A.maxMarkers; i++) {
      const m = el('div', 'ability-head-marker');
      m.hidden = true;
      this.el.append(m);
      this.markers.push(m);
    }
    parent.append(this.el);
  }

  start(character: CharacterId): void {
    this.vignette.className = `ability-vignette ${character} on`;
    this.markersOn = character === 'jason';
  }

  stop(): void {
    this.vignette.classList.remove('on');
    this.markersOn = false;
    for (const m of this.markers) m.hidden = true;
  }

  update(): void {
    if (!this.markersOn) return;
    const game = this.game;
    const peds = game.peds;
    if (!peds || !game.systems.has('player')) return;
    peds.query(game.player.position, A.markerRange, this.buf, (p) => p.isAlive && !p.vehicle);
    const cam = game.viewCamera;
    const w = window.innerWidth;
    const h = window.innerHeight;
    let used = 0;
    for (let i = 0; i < this.buf.length && used < this.markers.length; i++) {
      const p = this.buf[i];
      tmp.set(p.position.x, p.position.y + A.headHeight, p.position.z).project(cam);
      if (tmp.z > 1 || tmp.x < -1 || tmp.x > 1 || tmp.y < -1 || tmp.y > 1) continue;
      const m = this.markers[used++];
      m.hidden = false;
      m.style.transform = `translate(${((tmp.x + 1) * 0.5 * w).toFixed(1)}px, ${((1 - tmp.y) * 0.5 * h).toFixed(1)}px)`;
    }
    for (let i = used; i < this.markers.length; i++) this.markers[i].hidden = true;
  }

  dispose(): void {
    this.el.remove();
  }
}
