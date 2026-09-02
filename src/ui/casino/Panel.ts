import type { CasinoSystem } from '../../systems/casino';
import './casino.css';

/**
 * Modal panel shared by every casino game and the cashier. Opening blocks game input
 * and releases the pointer; Esc or the Close button closes it. Subclasses build their
 * content into `body` and may refuse to close mid-animation via `canClose()`.
 */
export abstract class CasinoPanel {
  readonly el: HTMLElement;
  protected readonly body: HTMLElement;
  protected readonly msgEl: HTMLElement;
  private readonly chipsEl: HTMLElement;
  private readonly cashEl: HTMLElement;
  private offChips: (() => void) | null = null;
  private readonly onKey = (e: KeyboardEvent): void => {
    if (e.code === 'Escape') {
      e.preventDefault();
      // Eat the edge so the UI system does not also read this Esc as "pause" next frame.
      this.casino.game.input.consume('Escape');
      this.close();
    }
  };
  private readonly timers = new Set<number>();
  private opened = false;

  constructor(
    protected readonly casino: CasinoSystem,
    readonly id: string,
    title: string,
  ) {
    this.el = document.createElement('section');
    this.el.className = `casino-panel casino-${id}`;
    this.el.dataset.panel = id;
    const header = document.createElement('header');
    header.className = 'casino-header';
    const titleEl = document.createElement('div');
    titleEl.className = 'casino-title';
    titleEl.textContent = title;
    const right = document.createElement('div');
    right.className = 'casino-header-right';
    this.cashEl = document.createElement('div');
    this.cashEl.className = 'casino-cash';
    this.chipsEl = document.createElement('div');
    this.chipsEl.className = 'casino-chips';
    const closeBtn = this.button('Close', () => this.close(), 'secondary small');
    closeBtn.dataset.action = 'close';
    right.append(this.cashEl, this.chipsEl, closeBtn);
    header.append(titleEl, right);
    this.body = document.createElement('div');
    this.body.className = 'casino-body';
    this.msgEl = document.createElement('div');
    this.msgEl.className = 'casino-msg';
    this.el.append(header, this.body);
    this.refreshChips();
  }

  /** Subclasses fill `body` here (called once from `open`). */
  protected abstract build(): void;

  /** Return false while an animation must finish before closing. */
  protected canClose(): boolean {
    return true;
  }

  protected onClosed(): void {}

  open(): void {
    if (this.opened) return;
    this.opened = true;
    this.build();
    const host = document.getElementById('ui') ?? document.body;
    host.appendChild(this.el);
    const input = this.casino.game.input;
    input.pushBlocker('casino');
    input.exitPointerLock();
    window.addEventListener('keydown', this.onKey);
    this.offChips = this.casino.game.events.on('casino:chipsChanged', () => this.refreshChips());
    this.casino.game.audio.play('ui.open');
  }

  close(force = false): void {
    if (!this.opened) return;
    if (!force && !this.canClose()) {
      this.casino.game.audio.play('ui.error');
      return;
    }
    this.opened = false;
    for (const t of this.timers) window.clearTimeout(t);
    this.timers.clear();
    window.removeEventListener('keydown', this.onKey);
    this.offChips?.();
    this.offChips = null;
    this.el.remove();
    const game = this.casino.game;
    game.input.popBlocker('casino');
    if (game.ui?.isModal !== true && game.ui?.isPaused !== true) game.input.requestPointerLock();
    game.audio.play('ui.close');
    this.onClosed();
    this.casino.onPanelClosed(this);
  }

  get isOpen(): boolean {
    return this.opened;
  }

  // --- Helpers for subclasses ------------------------------------------------------

  protected refreshChips(): void {
    this.chipsEl.textContent = `${this.casino.chips.toLocaleString('en-US')} CHIPS`;
    const economy = this.casino.game.economy;
    this.cashEl.textContent = economy ? `$${economy.cash.toLocaleString('en-US')}` : '';
  }

  protected message(text: string, kind: 'win' | 'lose' | 'info' | '' = ''): void {
    this.msgEl.textContent = text;
    this.msgEl.className = `casino-msg ${kind}`.trim();
  }

  protected button(label: string, onClick: () => void, className = '', action?: string): HTMLButtonElement {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = `casino-btn ${className}`.trim();
    b.textContent = label;
    if (action) b.dataset.action = action;
    b.addEventListener('click', () => {
      this.casino.game.audio.play('ui.select');
      onClick();
    });
    return b;
  }

  protected row(...children: (HTMLElement | string)[]): HTMLElement {
    const r = document.createElement('div');
    r.className = 'casino-row';
    for (const c of children) {
      if (typeof c === 'string') {
        const span = document.createElement('span');
        span.className = 'casino-label';
        span.textContent = c;
        r.appendChild(span);
      } else r.appendChild(c);
    }
    return r;
  }

  protected div(className: string, text = ''): HTMLElement {
    const d = document.createElement('div');
    d.className = className;
    if (text) d.textContent = text;
    return d;
  }

  /** setTimeout that is cancelled automatically when the panel closes. */
  protected after(ms: number, fn: () => void): void {
    const id = window.setTimeout(() => {
      this.timers.delete(id);
      if (this.opened) fn();
    }, ms);
    this.timers.add(id);
  }

  protected fail(text: string): void {
    this.message(text, 'lose');
    this.casino.game.audio.play('ui.error');
  }
}
