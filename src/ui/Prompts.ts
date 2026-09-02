import type { GameState } from '../core/GameState';
import './bottom-stack.css';
import { el, setVisible, type UiComponent } from './dom';

const KEY_LABELS: Record<string, string> = {
  Space: 'SPACE',
  ShiftLeft: 'SHIFT',
  Escape: 'ESC',
  ArrowUp: '\u2191',
  ArrowDown: '\u2193',
  Mouse0: 'LMB',
  Mouse2: 'RMB',
  Tab: 'TAB',
  CapsLock: 'CAPS',
};

/** `[F] Enter vehicle` list from `state.prompts`; the DOM is rebuilt only when the joined text changes. */
export class Prompts implements UiComponent {
  readonly el: HTMLElement;
  private lastKey = '';

  constructor() {
    this.el = el('div', 'prompts');
    this.el.hidden = true;
  }

  update(state: GameState): void {
    const prompts = state.prompts;
    let key = '';
    for (let i = 0; i < prompts.length; i++) key += `${prompts[i].key}\u0001${prompts[i].text}\u0002`;
    if (key === this.lastKey) return;
    this.lastKey = key;
    this.el.replaceChildren();
    if (prompts.length === 0) {
      setVisible(this.el, false);
      return;
    }
    for (const p of prompts) {
      const row = el('span', 'prompt');
      row.append(el('span', 'prompt-key', keyLabel(p.key)), el('span', 'prompt-text', p.text));
      this.el.appendChild(row);
    }
    setVisible(this.el, true);
  }
}

/** 'KeyF' → 'F', 'Digit1' → '1', otherwise a friendly alias or the raw key. */
function keyLabel(code: string): string {
  if (code.startsWith('Key') && code.length === 4) return code.charAt(3);
  if (code.startsWith('Digit') && code.length === 6) return code.charAt(5);
  return KEY_LABELS[code] ?? code.toUpperCase();
}
