import type { Game } from '../core/Game';
import { UiConfig } from '../systems/ui/config';
import { el, setVisible, type UiComponent } from './dom';
import './menu.css';
import { MenuNav } from './MenuNav';
import './title-screen.css';

export interface TitleActions {
  newGame(): void;
  continueGame(): void;
}

/**
 * Boot screen. While shown the camera drifts over Vice Beach in cinematic mode and the
 * simulation keeps running underneath; input is blocked by the UI system.
 */
export class TitleScreen implements UiComponent {
  readonly el: HTMLElement;
  visible = false;
  private readonly nav = new MenuNav();
  private readonly continueButton: HTMLButtonElement;

  constructor(
    private readonly game: Game,
    actions: TitleActions,
  ) {
    this.el = el('div', 'title');
    this.el.hidden = true;
    this.el.append(el('div', 'title-logo', 'Leonida'), el('div', 'title-sub', 'Vice City, Leonida'));
    const buttons = el('div', 'title-buttons');
    const newGame = el('button', 'menu-button', 'New Game');
    newGame.addEventListener('click', () => actions.newGame());
    this.continueButton = el('button', 'menu-button', 'Continue');
    this.continueButton.addEventListener('click', () => actions.continueGame());
    buttons.append(newGame, this.continueButton);
    this.nav.setItems([newGame, this.continueButton]);
    this.nav.onMove = () => game.audio.play('ui.blip', { volume: 0.5 });
    const footer = el('div', 'title-footer');
    const left = el('span');
    left.append(key('W A S D'), 'Move  ', key('Mouse'), 'Look  ', key('F'), 'Enter vehicle  ', key('M'), 'Map  ', key('Esc'), 'Pause');
    footer.append(left, el('span', undefined, 'Non-commercial fan project'));
    this.el.append(buttons, footer);
  }

  show(): void {
    if (this.visible) return;
    this.visible = true;
    this.continueButton.disabled = !this.game.save.hasSave();
    setVisible(this.el, true);
    this.nav.select(0, false);
    if (this.game.systems.has('camera')) {
      const t = UiConfig.title;
      this.game.camera.setMode('cinematic');
      this.game.camera.cinematic.orbit(t.orbitCenter, t.orbitRadius, t.orbitHeight, t.orbitSpeed);
    }
  }

  hide(): void {
    if (!this.visible) return;
    this.visible = false;
    setVisible(this.el, false);
  }

  handleInput(): void {
    this.nav.handleInput(this.game.input);
  }

  update(): void {
    /* static while shown; the camera system animates the backdrop */
  }

  dispose(): void {
    this.el.remove();
  }
}

function key(label: string): HTMLElement {
  return el('span', 'menu-key', label);
}
