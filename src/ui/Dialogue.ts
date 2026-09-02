import type { Game } from '../core/Game';
import type { DialogueLine } from '../core/peers';
import { UiConfig } from '../systems/ui/config';
import './dialogue.css';
import { el, setText, setVisible, type UiComponent } from './dom';
import { nameColor } from './format';

const D = UiConfig.dialogue;

interface Run {
  lines: DialogueLine[];
  index: number;
  timer: number;
  resolve: () => void;
}

/**
 * Subtitle box for cutscenes and mission chatter. Space advances a line; the UI system
 * decides when Space may be consumed (not while paused) and calls `advance()`.
 */
export class Dialogue implements UiComponent {
  readonly el: HTMLElement;
  private readonly speaker: HTMLElement;
  private readonly text: HTMLElement;
  private run: Run | null = null;

  constructor(private readonly game: Game) {
    this.el = el('div', 'dialogue');
    this.el.hidden = true;
    this.speaker = el('span', 'dialogue-speaker');
    this.text = el('span', 'dialogue-text');
    this.el.append(this.speaker, this.text, el('span', 'dialogue-hint', 'Space'));
  }

  get active(): boolean {
    return this.run !== null;
  }

  show(lines: DialogueLine[]): Promise<void> {
    this.hide();
    if (lines.length === 0) return Promise.resolve();
    return new Promise((resolve) => {
      this.run = { lines, index: -1, timer: 0, resolve };
      this.advance();
    });
  }

  hide(): void {
    const run = this.run;
    if (!run) return;
    this.run = null;
    this.el.classList.remove('show');
    setVisible(this.el, false);
    run.resolve();
  }

  /** Moves to the next line (or ends the run). */
  advance(): void {
    const run = this.run;
    if (!run) return;
    run.index++;
    if (run.index >= run.lines.length) {
      this.hide();
      return;
    }
    const line = run.lines[run.index];
    run.timer = line.seconds ?? Math.min(D.maxSeconds, Math.max(D.minSeconds, line.text.length * D.secondsPerChar));
    setText(this.speaker, line.speaker);
    this.speaker.style.color = nameColor(line.speaker);
    setText(this.text, line.text);
    setVisible(this.el, true);
    this.el.classList.add('show');
    this.game.audio.play('dialogue.tick');
  }

  update(_state: unknown, realDt: number): void {
    const run = this.run;
    if (!run) return;
    run.timer -= realDt;
    if (run.timer <= 0) this.advance();
  }

  dispose(): void {
    this.hide();
    this.el.remove();
  }
}
