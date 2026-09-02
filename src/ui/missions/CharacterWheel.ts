import type { Game } from '../../core/Game';
import type { CharacterId } from '../../core/peers';
import { PED_PALETTES } from '../../gfx/PedBuilder';
import { CHARACTER_IDS, CharactersConfig } from '../../systems/characters/config';
import { el } from '../dom';
import './missions.css';

const W = CharactersConfig.wheel;

/**
 * Hold Q: slow-mo two-portrait picker (WeaponWheel pattern — own input blocker, own
 * slow-mo, mouse X selects, release confirms). Portraits are canvases painted from
 * the character palettes so no art assets are needed.
 */
export class CharacterWheel {
  readonly el: HTMLElement;
  private readonly slots = new Map<CharacterId, HTMLElement>();
  private open = false;
  private selected: CharacterId = 'lucia';
  private selX = 0;

  constructor(
    private readonly game: Game,
    parent: HTMLElement,
    private readonly current: () => CharacterId,
    private readonly canOpen: () => boolean,
    private readonly onPick: (id: CharacterId) => void,
  ) {
    this.el = el('div', 'character-wheel');
    this.el.hidden = true;
    const ring = el('div', 'character-wheel-ring');
    for (const id of CHARACTER_IDS) {
      const slot = el('div', 'character-slot');
      slot.dataset.character = id;
      slot.append(this.portrait(id), el('span', 'cw-name', CharactersConfig.names[id]), el('span', 'cw-trait', CharactersConfig.traits[id]));
      ring.append(slot);
      this.slots.set(id, slot);
    }
    this.el.append(ring, el('div', 'character-wheel-hint', 'move mouse to select · release Q to switch'));
    parent.append(this.el);
  }

  get isOpen(): boolean {
    return this.open;
  }

  private portrait(id: CharacterId): HTMLCanvasElement {
    const p = PED_PALETTES[id];
    const canvas = el('canvas', 'cw-portrait');
    canvas.width = 122;
    canvas.height = 122;
    const g = canvas.getContext('2d');
    if (g && p) {
      const hex = (c: number): string => `#${c.toString(16).padStart(6, '0')}`;
      g.fillStyle = id === 'lucia' ? 'rgba(255,79,163,0.25)' : 'rgba(53,230,255,0.25)';
      g.fillRect(0, 0, 122, 122);
      g.fillStyle = hex(p.shirt);
      g.fillRect(28, 78, 66, 44);
      g.fillStyle = hex(p.skin);
      g.fillRect(43, 30, 36, 44);
      g.fillStyle = hex(p.hair);
      g.fillRect(41, 22, 40, 16);
      if (id === 'lucia') g.fillRect(39, 30, 6, 40);
    }
    return canvas;
  }

  update(): void {
    const input = this.game.input;
    const held = input.isDown(W.key);
    if (!this.open) {
      if (held && this.canOpen()) this.show();
      return;
    }
    if (!held) {
      this.hide();
      return;
    }
    this.selX += input.mouseDeltaX;
    if (this.selX > 140) this.selX = 140;
    if (this.selX < -140) this.selX = -140;
    if (Math.abs(this.selX) > W.selectThresholdPx) {
      const pick: CharacterId = this.selX < 0 ? CHARACTER_IDS[0] : CHARACTER_IDS[1];
      if (pick !== this.selected) {
        this.selected = pick;
        this.game.audio.play('ui.wheel.tick');
        this.refresh();
      }
    }
  }

  /** Scripted selection for autotests. */
  select(id: CharacterId): void {
    this.selected = id;
    this.refresh();
  }

  private show(): void {
    this.open = true;
    this.selected = this.current();
    this.selX = 0;
    this.game.input.pushBlocker(W.blocker);
    this.game.time.scale = W.timeScale;
    this.el.hidden = false;
    this.refresh();
    this.game.audio.play('ui.wheel.open');
  }

  private hide(): void {
    this.open = false;
    this.game.input.popBlocker(W.blocker);
    if (this.game.time.scale === W.timeScale) this.game.time.scale = 1;
    this.el.hidden = true;
    if (this.selected !== this.current()) {
      this.game.audio.play('ui.wheel.select');
      this.onPick(this.selected);
    }
  }

  private refresh(): void {
    const cur = this.current();
    for (const [id, slot] of this.slots) {
      slot.classList.toggle('selected', id === this.selected);
      slot.classList.toggle('current', id === cur);
    }
  }

  dispose(): void {
    if (this.open) this.hide();
    this.el.remove();
  }
}
