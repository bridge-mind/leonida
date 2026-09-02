import type { Game } from '../core/Game';
import type { CardKind } from '../core/peers';
import type { SoundId } from '../systems/audio/sounds';
import { UiConfig } from '../systems/ui/config';
import './cards.css';
import { el, setText, toggleClass, type UiComponent } from './dom';

const C = UiConfig.cards;
const LABELS: Record<CardKind, string> = {
  wasted: 'Wasted',
  busted: 'Busted',
  missionPassed: 'Mission Passed',
  missionFailed: 'Mission Failed',
  custom: '',
};
const SOUNDS: Record<CardKind, SoundId | null> = {
  wasted: 'wanted.wasted',
  busted: 'wanted.busted',
  missionPassed: 'mission.passed',
  missionFailed: 'mission.failed',
  custom: null,
};

interface Active {
  kind: CardKind;
  timer: number;
  fading: boolean;
  resolve: () => void;
}

/**
 * WASTED / BUSTED / MISSION cards: letterbox bars, huge display text, world desaturated,
 * simulation at quarter speed. `letterbox(on)` drives the bars alone for cutscenes.
 */
export class Cards implements UiComponent {
  readonly el: HTMLElement;
  private readonly text: HTMLElement;
  private readonly sub: HTMLElement;
  private active: Active | null = null;
  private letterboxOn = false;
  private readonly gameCanvas: HTMLElement | null;

  constructor(private readonly game: Game) {
    this.el = el('div', 'cards');
    this.el.append(el('div', 'letterbox top'), el('div', 'letterbox bottom'));
    this.text = el('div', 'card-text');
    this.sub = el('div', 'card-sub');
    this.el.append(this.text, this.sub);
    this.gameCanvas = document.getElementById('game');
  }

  get isActive(): boolean {
    return this.active !== null;
  }

  get activeKind(): CardKind | null {
    return this.active?.kind ?? null;
  }

  show(kind: CardKind, text: string, seconds: number = C.defaultSeconds): Promise<void> {
    if (this.active) this.finish();
    const label = LABELS[kind] || text;
    const subtitle = kind === 'custom' || text.trim().toLowerCase() === label.toLowerCase() ? '' : text;
    setText(this.text, label);
    setText(this.sub, subtitle);
    this.text.className = `card-text ${kind}`;
    this.sub.className = 'card-sub';
    // Reflow so the show transition starts from the hidden state even on back-to-back cards.
    void this.text.offsetWidth;
    this.text.classList.add('show');
    toggleClass(this.sub, 'show', subtitle !== '');
    this.applyLetterbox();
    this.gameCanvas?.classList.add('ui-desaturate');
    const time = this.game.time;
    if (time.scale > C.slowMoScale) time.scale = C.slowMoScale;
    const sound = SOUNDS[kind];
    if (sound) this.game.audio.play(sound);
    return new Promise((resolve) => {
      this.active = { kind, timer: Math.max(0.1, seconds), fading: false, resolve };
    });
  }

  letterbox(on: boolean): void {
    this.letterboxOn = on;
    this.applyLetterbox();
  }

  private applyLetterbox(): void {
    toggleClass(this.el, 'letterbox-on', this.letterboxOn || this.active !== null || this.text.classList.contains('show'));
  }

  update(_state: unknown, realDt: number): void {
    const a = this.active;
    if (!a) return;
    a.timer -= realDt;
    if (a.timer > 0) return;
    if (!a.fading) {
      a.fading = true;
      a.timer = C.fadeSeconds;
      this.text.classList.add('fade');
      this.sub.classList.remove('show');
      return;
    }
    this.finish();
  }

  private finish(): void {
    const a = this.active;
    if (!a) return;
    this.active = null;
    this.text.classList.remove('show', 'fade');
    this.sub.classList.remove('show');
    this.gameCanvas?.classList.remove('ui-desaturate');
    // Only undo our own slow-mo; a weapon wheel or ability may have taken over meanwhile.
    if (this.game.time.scale === C.slowMoScale) this.game.time.scale = 1;
    this.applyLetterbox();
    a.resolve();
  }

  dispose(): void {
    this.finish();
    this.gameCanvas?.classList.remove('ui-desaturate');
    this.el.remove();
  }
}
