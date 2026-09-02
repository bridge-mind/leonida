import type { Game } from '../../core/Game';
import { el, setText } from '../dom';
import '../menu.css';
import { MenuNav } from '../MenuNav';
import './missions.css';

const BLOCKER_ID = 'missionRetry';

/**
 * "Mission failed — Retry / Quit" panel shown after the failure card. Keyboard only
 * (arrows/W-S + Enter, R = retry, Q = quit) so the pointer lock stays untouched:
 * releasing it would read as a pause request to the UI system.
 */
export class RetryPrompt {
  readonly el: HTMLElement;
  private readonly title: HTMLElement;
  private readonly reason: HTMLElement;
  private readonly nav = new MenuNav();
  private resolve: ((choice: 'retry' | 'quit') => void) | null = null;

  constructor(
    private readonly game: Game,
    parent: HTMLElement,
  ) {
    this.el = el('div', 'mission-retry');
    this.el.hidden = true;
    const panel = el('div', 'mission-retry-panel');
    this.title = el('div', 'mission-retry-title', 'Mission Failed');
    this.reason = el('div', 'mission-retry-reason');
    const retry = el('button', 'menu-button', 'Retry');
    const quit = el('button', 'menu-button', 'Quit');
    retry.addEventListener('click', () => this.choose('retry'));
    quit.addEventListener('click', () => this.choose('quit'));
    const buttons = el('div', 'mission-retry-buttons');
    buttons.append(retry, quit);
    panel.append(this.title, this.reason, buttons, el('div', 'mission-retry-hint', 'Enter · R retry · Q quit'));
    this.el.append(panel);
    parent.append(this.el);
    this.nav.setItems([retry, quit]);
    this.nav.onMove = () => game.audio.play('ui.blip');
  }

  get visible(): boolean {
    return !this.el.hidden;
  }

  show(missionName: string, reason: string): Promise<'retry' | 'quit'> {
    if (this.resolve) this.choose('quit');
    setText(this.title, `${missionName} — Failed`);
    setText(this.reason, reason);
    this.el.hidden = false;
    this.nav.select(0, false);
    this.game.input.pushBlocker(BLOCKER_ID);
    return new Promise((resolve) => {
      this.resolve = resolve;
    });
  }

  /** Programmatic choice (debug action / autotest). */
  choose(choice: 'retry' | 'quit'): void {
    const r = this.resolve;
    if (!r) return;
    this.resolve = null;
    this.el.hidden = true;
    this.game.input.popBlocker(BLOCKER_ID);
    this.game.audio.play('ui.select');
    r(choice);
  }

  update(): void {
    if (this.el.hidden) return;
    const input = this.game.input;
    if (this.game.ui?.isModal) return;
    if (input.wasPressed('KeyR')) this.choose('retry');
    else if (input.wasPressed('KeyQ')) this.choose('quit');
    else this.nav.handleInput(input);
  }

  dispose(): void {
    if (this.resolve) this.choose('quit');
    this.el.remove();
  }
}
