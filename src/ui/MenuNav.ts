import type { Input } from '../core/Input';

/**
 * Arrow/Enter keyboard navigation over a vertical list of buttons, merged with mouse
 * hover so the highlighted item is always the one Enter will activate.
 */
export class MenuNav {
  private items: HTMLButtonElement[] = [];
  private index = 0;
  onMove: (() => void) | null = null;

  setItems(items: HTMLButtonElement[]): void {
    this.items = items;
    for (let i = 0; i < items.length; i++) {
      items[i].addEventListener('mouseenter', () => this.select(i, false));
    }
    this.select(Math.min(this.index, items.length - 1), false);
  }

  get current(): HTMLButtonElement | null {
    return this.items[this.index] ?? null;
  }

  select(index: number, playSound = true): void {
    if (this.items.length === 0) return;
    const n = this.items.length;
    let i = ((index % n) + n) % n;
    // Skip disabled entries in the direction of travel.
    for (let guard = 0; guard < n && this.items[i].disabled; guard++) i = (i + (index >= this.index ? 1 : n - 1)) % n;
    this.index = i;
    for (let k = 0; k < n; k++) this.items[k].classList.toggle('selected', k === i);
    if (playSound) this.onMove?.();
  }

  /** Reads the scripted/real key edges for this frame. Returns true when Enter activated an item. */
  handleInput(input: Input): boolean {
    if (input.wasPressed('ArrowDown') || input.wasPressed('KeyS')) this.select(this.index + 1);
    if (input.wasPressed('ArrowUp') || input.wasPressed('KeyW')) this.select(this.index - 1);
    if (input.wasPressed('Enter') || input.wasPressed('NumpadEnter')) {
      const item = this.current;
      if (item && !item.disabled) {
        item.click();
        return true;
      }
    }
    return false;
  }
}
