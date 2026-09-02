import type { GameState } from '../../core/GameState';

/**
 * One `[E] text` slot in `state.prompts`. Each system owns a slot object and
 * toggles it so prompts from interiors, shops and services never overwrite each
 * other (the UI's keyed `prompt()` API would collide on the shared 'E' key).
 */
export class PromptSlot {
  private readonly entry: { key: string; text: string };
  private shown = false;

  constructor(
    private readonly state: GameState,
    key = 'KeyE',
  ) {
    this.entry = { key, text: '' };
  }

  set(text: string | null): void {
    const prompts = this.state.prompts;
    if (text === null) {
      if (!this.shown) return;
      const idx = prompts.indexOf(this.entry);
      if (idx >= 0) prompts.splice(idx, 1);
      this.shown = false;
      return;
    }
    this.entry.text = text;
    if (!this.shown) {
      prompts.push(this.entry);
      this.shown = true;
    }
  }

  get visible(): boolean {
    return this.shown;
  }
}
