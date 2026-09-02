import type { GameState } from '../core/GameState';

/** One HUD/overlay piece. `update` runs every render frame with the unscaled delta. */
export interface UiComponent {
  readonly el: HTMLElement;
  update(state: GameState, realDt: number): void;
  dispose?(): void;
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Writes `text` only when it differs from what is already rendered. */
export function setText(node: HTMLElement, text: string): boolean {
  if (node.textContent === text) return false;
  node.textContent = text;
  return true;
}

export function setVisible(node: HTMLElement, visible: boolean): void {
  if (node.hidden === !visible) return;
  node.hidden = !visible;
}

export function toggleClass(node: HTMLElement, name: string, on: boolean): void {
  if (node.classList.contains(name) === on) return;
  node.classList.toggle(name, on);
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
