import type { GameState } from '../core/GameState';
import './bottom-stack.css';
import { el, setVisible, type UiComponent } from './dom';
import { renderMarkup } from './format';

/** Mission objective line with `~y~noun~y~` highlights. Re-renders only when the text changes. */
export class Objective implements UiComponent {
  readonly el: HTMLElement;
  private last: string | null = null;

  constructor() {
    this.el = el('div', 'objective');
    this.el.hidden = true;
  }

  update(state: GameState): void {
    const text = state.objective;
    if (text === this.last) return;
    this.last = text;
    if (!text) {
      setVisible(this.el, false);
      return;
    }
    renderMarkup(this.el, text);
    setVisible(this.el, true);
  }
}
