import './bottom-stack.css';
import { clamp01, el, setText, setVisible, type UiComponent } from './dom';

/** Thin labelled bar for robberies / timed actions. Driven by `ui.progressBar.show(label, fraction)`. */
export class ProgressBar implements UiComponent {
  readonly el: HTMLElement;
  private readonly label: HTMLElement;
  private readonly fill: HTMLElement;
  private lastFraction = -1;
  private visible = false;

  constructor() {
    this.el = el('div', 'progress-bar');
    this.el.hidden = true;
    this.label = el('div', 'progress-bar-label');
    const track = el('div', 'progress-bar-track');
    this.fill = el('div', 'progress-bar-fill');
    track.appendChild(this.fill);
    this.el.append(this.label, track);
  }

  show(label: string, fraction: number): void {
    setText(this.label, label);
    const f = Math.round(clamp01(fraction) * 1000) / 1000;
    if (f !== this.lastFraction) {
      this.lastFraction = f;
      this.fill.style.transform = `scaleX(${f})`;
    }
    if (!this.visible) {
      this.visible = true;
      setVisible(this.el, true);
    }
  }

  hide(): void {
    if (!this.visible) return;
    this.visible = false;
    setVisible(this.el, false);
  }

  update(): void {
    /* state-less: driven by show/hide */
  }
}
