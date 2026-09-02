import type { Game } from '../core/Game';
import { UiConfig } from '../systems/ui/config';
import { el, type UiComponent } from './dom';
import { formatCashDelta, humaniseReason } from './format';
import './notifications.css';

export type NotificationKind = 'info' | 'money' | 'warning';

interface Entry {
  el: HTMLElement;
  /** Seconds of life left; negative while fading out. */
  ttl: number;
  fading: boolean;
}

const N = UiConfig.notifications;
const FADE_SECONDS = 0.32;

/**
 * Top-left feed. Listens to the `notification` event and posts money entries for
 * significant cash changes on its own so every system gets feed entries for free.
 */
export class Notifications implements UiComponent {
  readonly el: HTMLElement;
  private readonly entries: Entry[] = [];
  private readonly offs: (() => void)[] = [];

  constructor(private readonly game: Game) {
    this.el = el('div', 'notifications');
    this.offs.push(
      game.events.on('notification', (e) => this.post(e.text, e.kind ?? 'info')),
      game.events.on('economy:cashChanged', (e) => {
        if (Math.abs(e.delta) < N.moneyThreshold) return;
        this.post(humaniseReason(e.reason), 'money', e.delta);
      }),
      game.events.on('save:saved', () => this.post('Game saved', 'info')),
    );
  }

  post(text: string, kind: NotificationKind = 'info', amount?: number): void {
    const node = el('div', `notification ${kind}`);
    if (amount !== undefined) {
      const amt = el('span', `notification-amount ${amount >= 0 ? 'gain' : 'loss'}`, formatCashDelta(amount));
      node.appendChild(amt);
    }
    node.appendChild(el('span', 'notification-text', text));
    this.el.appendChild(node);
    // Two frames so the initial transform is committed before the class flips (CSS transition).
    requestAnimationFrame(() => requestAnimationFrame(() => node.classList.add('in')));
    this.entries.push({ el: node, ttl: N.seconds, fading: false });
    while (this.entries.length > N.max) this.retire(this.entries[0]);
    this.game.audio.play(kind === 'money' ? 'ui.notification.money' : 'ui.notification');
  }

  private retire(entry: Entry): void {
    if (entry.fading) return;
    entry.fading = true;
    entry.ttl = FADE_SECONDS;
    entry.el.classList.remove('in');
    entry.el.classList.add('out');
  }

  update(_state: unknown, realDt: number): void {
    for (let i = this.entries.length - 1; i >= 0; i--) {
      const e = this.entries[i];
      e.ttl -= realDt;
      if (e.ttl > 0) continue;
      if (!e.fading) {
        this.retire(e);
        continue;
      }
      e.el.remove();
      this.entries.splice(i, 1);
    }
  }

  get count(): number {
    return this.entries.length;
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.el.remove();
  }
}
