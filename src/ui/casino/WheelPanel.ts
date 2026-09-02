import { WHEEL_SEGMENT_COUNT, WHEEL_SEGMENTS, type WheelSpin } from '../../casino/LuckyWheel';
import type { CasinoSystem } from '../../systems/casino';
import { CasinoConfig } from '../../systems/casino/config';
import { CasinoPanel } from './Panel';

/**
 * Daily Lucky Wheel. The 3D disc in the interior does the real spin; this panel mirrors
 * it with a segment readout that eases from a blur of prizes to the winning one.
 */
export class WheelPanel extends CasinoPanel {
  private spinning = false;
  private display!: HTMLElement;
  private label!: HTMLElement;
  private sub!: HTMLElement;
  private spinBtn!: HTMLButtonElement;
  private readonly segEls: HTMLElement[] = [];
  private raf = 0;
  private loop: ReturnType<CasinoSystem['game']['audio']['loop']> | null = null;
  lastSpin: WheelSpin | null = null;

  constructor(casino: CasinoSystem) {
    super(casino, 'wheel', 'Lucky Wheel');
  }

  protected build(): void {
    this.display = this.div('lw-display');
    this.label = this.div('lw-label', 'SPIN');
    this.label.dataset.field = 'prize';
    this.sub = this.div('lw-sub', 'One free spin per day');
    this.display.append(this.label, this.sub);
    const grid = this.div('lw-segments');
    WHEEL_SEGMENTS.forEach((s, i) => {
      const el = this.div('lw-seg', s.label);
      el.style.background = s.color;
      el.dataset.segment = String(i);
      grid.appendChild(el);
      this.segEls.push(el);
    });
    this.spinBtn = this.button('Spin', () => this.spin(), '', 'spin');
    this.body.append(this.display, grid, this.row(this.spinBtn), this.msgEl);
    this.casino.game.audio.play('ui.wheel.open');
    this.refreshAvailability();
  }

  protected override canClose(): boolean {
    return !this.spinning;
  }

  protected override onClosed(): void {
    cancelAnimationFrame(this.raf);
    if (this.loop) this.casino.game.audio.stop(this.loop);
  }

  private refreshAvailability(): void {
    const canSpin = this.casino.wheel.canSpin(this.casino.game.time.day);
    this.spinBtn.disabled = this.spinning || !canSpin;
    if (!canSpin && !this.spinning) this.message('Come back tomorrow for another free spin.', 'info');
  }

  /** Returns false when today's spin is already used. */
  spin(): boolean {
    if (this.spinning) return false;
    const day = this.casino.game.time.day;
    if (!this.casino.wheel.canSpin(day)) {
      this.fail('Come back tomorrow.');
      this.refreshAvailability();
      return false;
    }
    const rigged = this.casino.rig.takeWheel();
    const result = this.casino.wheel.spin(day, rigged ?? undefined);
    if (!result) {
      this.fail('Come back tomorrow.');
      return false;
    }
    this.lastSpin = result;
    this.spinning = true;
    this.spinBtn.disabled = true;
    this.message('');
    const audio = this.casino.game.audio;
    this.loop = audio.loop('casino.wheel.spin.loop');
    const seconds = CasinoConfig.wheel.spinSeconds;
    this.casino.stage.spinWheel(result.index, seconds, () => audio.play('casino.wheel.tick'));
    // Mirror the disc: total travel = whole turns + the winning segment, eased out.
    const totalSegments = CasinoConfig.wheel.turns * WHEEL_SEGMENT_COUNT + result.index;
    const start = performance.now();
    let shown = -1;
    const tick = (): void => {
      const t = Math.min(1, (performance.now() - start) / (seconds * 1000));
      const eased = 1 - Math.pow(1 - t, 3);
      const index = Math.floor(eased * totalSegments) % WHEEL_SEGMENT_COUNT;
      if (index !== shown) {
        shown = index;
        this.showSegment(index);
      }
      if (t >= 1) this.finish(result);
      else this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
    return true;
  }

  private showSegment(index: number): void {
    const s = WHEEL_SEGMENTS[index];
    this.label.textContent = s.label;
    this.display.style.background = s.color;
    for (let i = 0; i < this.segEls.length; i++) this.segEls[i].classList.toggle('current', i === index);
  }

  private finish(result: WheelSpin): void {
    const audio = this.casino.game.audio;
    if (this.loop) audio.stop(this.loop);
    this.loop = null;
    this.showSegment(result.index);
    audio.play('casino.wheel.stop');
    audio.play('casino.wheel.prize');
    const text = this.casino.applyWheelPrize(result.segment);
    this.sub.textContent = text;
    this.message(text, 'win');
    this.spinning = false;
    this.refreshAvailability();
  }
}
