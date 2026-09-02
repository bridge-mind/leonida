import type { Game } from '../../core/Game';
import type { MissionSummary } from '../../core/peers';
import { el, setText } from '../dom';
import { formatCash } from '../format';
import { MenuNav } from '../MenuNav';
import { Feed } from './Feed';
import './phone.css';
import { Snapmatic } from './Snapmatic';

type AppId = 'home' | 'contacts' | 'map' | 'snap' | 'feed';

interface AppDef {
  id: Exclude<AppId, 'home'>;
  label: string;
  glyph: string;
  color: string;
}

const APPS: readonly AppDef[] = [
  { id: 'contacts', label: 'Contacts', glyph: '☎', color: '#4ee06a' },
  { id: 'map', label: 'Map', glyph: '◎', color: '#35e6ff' },
  { id: 'snap', label: 'Snapmatic', glyph: '◉', color: '#ffd23f' },
  { id: 'feed', label: 'Feed', glyph: '#', color: '#ff4fa3' },
];

/**
 * The phone. Up arrow opens it; while open it behaves like the pause menu
 * (pointer released, gameplay input blocked, arrows/Enter navigate, Backspace
 * goes back, Escape or the home button closes). The clock keeps running.
 */
export class Phone {
  readonly el: HTMLElement;
  readonly feed: Feed;
  readonly snapmatic: Snapmatic;
  private readonly screenTitle: HTMLElement;
  private readonly body: HTMLElement;
  private readonly clock: HTMLElement;
  private readonly cash: HTMLElement;
  private readonly nav = new MenuNav();
  private app: AppId = 'home';
  private selectedContact: MissionSummary | null = null;
  private opened = false;

  constructor(private readonly game: Game) {
    this.feed = new Feed(game);
    this.snapmatic = new Snapmatic(game);
    this.el = el('div', 'phone-layer');
    const shell = el('div', 'phone');
    const screen = el('div', 'phone-screen');
    const status = el('div', 'phone-status');
    status.append((this.clock = el('span')), (this.cash = el('span')));
    this.screenTitle = el('div', 'phone-title');
    this.body = el('div', 'phone-body');
    screen.append(status, this.screenTitle, this.body);
    const home = el('div', 'phone-home');
    const homeBtn = el('button', 'phone-home-btn');
    homeBtn.type = 'button';
    homeBtn.title = 'Home / close';
    homeBtn.addEventListener('click', () => (this.app === 'home' ? this.close() : this.openApp('home')));
    home.appendChild(homeBtn);
    shell.append(screen, home, el('div', 'phone-hint', '↑↓ navigate · Enter open · Backspace back · Esc close'));
    this.el.appendChild(shell);
    this.el.addEventListener('click', (e) => {
      if (e.target === this.el) this.close();
    });
    this.nav.onMove = () => this.game.audio.play('ui.blip');
    this.feed.onChange = () => {
      if (this.opened && this.app === 'feed') this.render();
    };
  }

  get isOpen(): boolean {
    return this.opened;
  }

  open(): void {
    if (this.opened) return;
    this.opened = true;
    (document.getElementById('ui') ?? document.body).appendChild(this.el);
    this.game.input.pushBlocker('phone');
    this.game.input.exitPointerLock();
    this.app = 'home';
    this.render();
    this.game.audio.play('ui.phone.open');
    this.game.events.emit('phone:opened', {});
  }

  close(): void {
    if (!this.opened) return;
    this.opened = false;
    this.el.remove();
    this.game.input.popBlocker('phone');
    this.game.audio.play('ui.phone.close');
    this.game.events.emit('phone:closed', {});
    this.game.input.requestPointerLock();
  }

  toggle(): void {
    if (this.opened) this.close();
    else this.open();
  }

  openApp(id: string): void {
    if (!this.opened) this.open();
    this.app = isApp(id) ? id : 'home';
    this.selectedContact = null;
    this.render();
  }

  /** Captures a screenshot (also used by the debug action / autotest). */
  async snap(): Promise<string> {
    const url = this.snapmatic.capture();
    if (this.opened && this.app === 'snap') this.render();
    return url;
  }

  update(): void {
    const input = this.game.input;
    if (!this.opened) {
      if (input.wasPressed('ArrowUp') && !input.isBlocked && !(this.game.ui?.isModal ?? false)) {
        input.consume('ArrowUp');
        this.open();
      }
      return;
    }
    if (input.wasPressed('Escape')) {
      // Shops run before the UI system: without consuming, the same Escape edge also opens the pause menu.
      input.consume('Escape');
      this.close();
      return;
    }
    if (input.wasPressed('Backspace')) {
      input.consume('Backspace');
      if (this.app === 'home') this.close();
      else if (this.selectedContact) {
        this.selectedContact = null;
        this.render();
      } else this.openApp('home');
      return;
    }
    this.nav.handleInput(input);
    this.refreshStatus();
  }

  // --- Rendering ---------------------------------------------------------------------

  private refreshStatus(): void {
    const t = this.game.time;
    const h = Math.floor(t.hour);
    const m = Math.floor((t.hour - h) * 60);
    setText(this.clock, `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}  ·  day ${t.day}`);
    setText(this.cash, formatCash(this.game.state.cash));
  }

  private render(): void {
    this.refreshStatus();
    this.body.replaceChildren();
    switch (this.app) {
      case 'home':
        return this.renderHome();
      case 'contacts':
        return this.renderContacts();
      case 'map':
        return this.renderMap();
      case 'snap':
        return this.renderSnap();
      case 'feed':
        return this.renderFeed();
    }
  }

  private renderHome(): void {
    setText(this.screenTitle, 'iFruit');
    const grid = el('div', 'phone-grid');
    const buttons = APPS.map((app) => {
      const b = el('button', 'phone-app');
      b.type = 'button';
      b.dataset.app = app.id;
      const icon = el('span', 'phone-app-icon', app.glyph);
      icon.style.background = app.color;
      b.append(icon, el('span', '', app.label));
      b.addEventListener('click', () => {
        this.game.audio.play('ui.select');
        this.openApp(app.id);
      });
      grid.appendChild(b);
      return b;
    });
    this.body.appendChild(grid);
    this.nav.setItems(buttons);
  }

  private renderContacts(): void {
    setText(this.screenTitle, 'Contacts');
    const missions = this.game.missions;
    const list = missions ? missions.available() : [];
    if (this.selectedContact) {
      const c = this.selectedContact;
      this.body.append(el('div', 'phone-post', `${c.contact}: "${c.name}". You in?`));
      const call = this.row('Call', c.contact, () => {
        const ok = missions?.start(c.id) ?? false;
        this.game.audio.play(ok ? 'ui.phone.hangup' : 'ui.error');
        if (ok) this.close();
      });
      this.body.appendChild(call);
      this.nav.setItems([call]);
      return;
    }
    if (list.length === 0) {
      this.body.appendChild(el('div', 'phone-empty', missions ? 'No contacts yet. Meet some people.' : 'No contacts yet.'));
      this.nav.setItems([]);
      return;
    }
    const rows = list.map((m) =>
      this.row(m.contact, m.name, () => {
        this.selectedContact = m;
        this.render();
      }),
    );
    this.body.append(...rows);
    this.nav.setItems(rows);
  }

  private renderMap(): void {
    setText(this.screenTitle, 'Map');
    const open = this.row('Open full map', 'M also works in-game', () => {
      this.close();
      try {
        this.game.debug.run('ui.map');
      } catch {
        this.game.events.emit('notification', { text: 'Map is not available.', kind: 'info' });
      }
    });
    this.body.appendChild(open);
    this.nav.setItems([open]);
  }

  private renderSnap(): void {
    setText(this.screenTitle, 'Snapmatic');
    const last = this.snapmatic.lastDataUrl;
    if (last) {
      const img = el('img', 'phone-snap');
      img.src = last;
      img.alt = 'Last snapshot';
      this.body.appendChild(img);
    } else {
      this.body.appendChild(el('div', 'phone-empty', 'No snaps yet.'));
    }
    const take = this.row('Take photo', `${this.snapmatic.count} taken`, () => void this.snap());
    const save = this.row('Save last photo', 'downloads a PNG', () => {
      this.snapmatic.download();
      this.render();
    });
    save.disabled = last === null;
    this.body.append(take, save);
    this.nav.setItems([take, save]);
  }

  private renderFeed(): void {
    setText(this.screenTitle, 'Feed');
    if (this.feed.posts.length === 0) {
      this.body.appendChild(el('div', 'phone-empty', 'Nothing happening. Suspicious.'));
    }
    for (const p of this.feed.posts) {
      const post = el('div', 'phone-post');
      const head = el('div', 'phone-post-handle');
      head.append(el('span', '', p.handle), el('span', 'phone-post-likes', `♥ ${p.likes}  ${p.time}`));
      post.append(head, el('div', '', p.text));
      this.body.appendChild(post);
    }
    this.nav.setItems([]);
  }

  private row(title: string, sub: string, onClick: () => void): HTMLButtonElement {
    const b = el('button', 'phone-row');
    b.type = 'button';
    b.append(el('span', '', title), el('span', 'phone-row-sub', sub));
    b.addEventListener('click', () => {
      this.game.audio.play('ui.select');
      onClick();
    });
    return b;
  }

  dispose(): void {
    this.close();
    this.feed.dispose();
  }
}

function isApp(id: string): id is AppId {
  return id === 'home' || APPS.some((a) => a.id === id);
}
