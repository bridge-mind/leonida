import type { Game } from '../../core/Game';
import { el, setText, toggleClass } from '../dom';
import { formatCash } from '../format';
import { MenuNav } from '../MenuNav';
import './shops.css';

/** What the panel needs from the shop system; kept as an interface so the UI stays testable in isolation. */
export interface GunStoreCatalogue {
  rows(): CatalogueRow[];
  purchase(item: string): PurchaseResult;
}

export interface CatalogueRow {
  id: string;
  name: string;
  /** Second line: damage / magazine / current ammo. */
  detail: string;
  /** Button verb: "Buy", "Buy ammo", "Full". */
  action: string;
  price: number;
  affordable: boolean;
  disabled: boolean;
}

export interface PurchaseResult {
  ok: boolean;
  message: string;
}

/**
 * Bullet Barn shopfront. One row per weapon (buy, or buy ammo once owned) plus
 * armour. Keyboard (arrows/WASD + Enter, Backspace/E/Esc to leave) and mouse.
 */
export class GunStorePanel {
  readonly el: HTMLElement;
  private readonly list: HTMLElement;
  private readonly cashEl: HTMLElement;
  private readonly msgEl: HTMLElement;
  private readonly nav = new MenuNav();
  private opened = false;
  private offCash: (() => void) | null = null;
  private rowEls: HTMLButtonElement[] = [];

  constructor(
    private readonly game: Game,
    private readonly catalogue: GunStoreCatalogue,
  ) {
    this.el = el('div', 'shop-panel');
    const win = el('div', 'shop-window');
    const header = el('div', 'shop-header');
    header.append(el('div', 'shop-title', 'Bullet Barn'), (this.cashEl = el('div', 'shop-cash')));
    this.list = el('div', 'shop-list');
    const footer = el('div', 'shop-footer');
    this.msgEl = el('div', 'shop-msg');
    footer.append(this.msgEl, el('div', '', '↑↓ browse · Enter buy · Backspace leave'));
    win.append(header, this.list, footer);
    this.el.appendChild(win);
    this.nav.onMove = () => this.game.audio.play('ui.blip');
  }

  get isOpen(): boolean {
    return this.opened;
  }

  open(): void {
    if (this.opened) return;
    this.opened = true;
    (document.getElementById('ui') ?? document.body).appendChild(this.el);
    const input = this.game.input;
    input.pushBlocker('shop');
    input.exitPointerLock();
    this.offCash = this.game.events.on('economy:cashChanged', () => this.refresh());
    this.message('');
    this.refresh();
    this.game.audio.play('store.bell');
  }

  close(): void {
    if (!this.opened) return;
    this.opened = false;
    this.offCash?.();
    this.offCash = null;
    this.el.remove();
    this.game.input.popBlocker('shop');
    this.game.audio.play('ui.close');
    this.game.input.requestPointerLock();
  }

  /** Called every frame by the shop system while open. */
  update(): void {
    if (!this.opened) return;
    const input = this.game.input;
    if (input.wasPressed('Backspace') || input.wasPressed('KeyE') || input.wasPressed('Escape')) {
      // Same-frame consumers run after us: E would re-open the counter prompt, Escape would open the pause menu.
      input.consume('Backspace');
      input.consume('KeyE');
      input.consume('Escape');
      this.close();
      return;
    }
    this.nav.handleInput(input);
  }

  private refresh(): void {
    const economy = this.game.economy;
    setText(this.cashEl, economy ? formatCash(economy.cash) : '');
    const rows = this.catalogue.rows();
    if (rows.length !== this.rowEls.length) this.rebuild(rows);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const b = this.rowEls[i];
      b.disabled = r.disabled;
      toggleClass(b, 'unaffordable', !r.affordable && !r.disabled);
      setText(b.children[0].children[0] as HTMLElement, r.name);
      setText(b.children[0].children[1] as HTMLElement, r.detail);
      setText(b.children[1] as HTMLElement, r.action);
      setText(b.children[2] as HTMLElement, r.disabled ? '—' : formatCash(r.price));
    }
  }

  private rebuild(rows: CatalogueRow[]): void {
    this.list.replaceChildren();
    this.rowEls = rows.map((r) => {
      const b = el('button', 'shop-row');
      b.type = 'button';
      b.dataset.item = r.id;
      const name = el('div');
      name.append(el('span', 'shop-row-name', r.name), el('span', 'shop-row-sub', r.detail));
      b.append(name, el('span', 'shop-row-action', r.action), el('span', 'shop-row-price', formatCash(r.price)));
      b.addEventListener('click', () => this.buy(r.id));
      this.list.appendChild(b);
      return b;
    });
    this.nav.setItems(this.rowEls);
  }

  private buy(item: string): void {
    const result = this.catalogue.purchase(item);
    this.message(result.message, result.ok ? 'bought' : 'denied');
    this.refresh();
  }

  private message(text: string, kind: 'bought' | 'denied' | '' = ''): void {
    this.msgEl.textContent = text;
    this.msgEl.className = `shop-msg ${kind}`.trim();
  }

  dispose(): void {
    this.close();
  }
}
