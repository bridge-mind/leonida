import { Config } from './Config';
import type { Events } from './Events';

/**
 * Game clock. `scale` implements slow-mo (affects the simulation, not UI).
 * `hour` is a float in [0, 24); one full day takes `Config.time.dayLengthSeconds`.
 */
export class Time {
  /** Simulation speed multiplier. 1 = realtime, 0.2 = weapon-wheel slow-mo, 0 = paused. */
  scale = 1;
  /** Unscaled seconds since the last frame (UI animations use this). */
  realDt = 0;
  /** Scaled seconds since the last frame. */
  dt = 0;
  /** Scaled seconds since start. */
  elapsed = 0;
  /** Unscaled seconds since start. */
  realElapsed = 0;
  /** Render frame counter. */
  frame = 0;

  hour: number = Config.time.startHour;
  day: number = Config.time.startDay;
  /** Game clock speed multiplier (debug fast-forward); 1 = 1 game minute per real second. */
  clockScale = 1;
  /** When false the day/night clock stops but the simulation keeps running. */
  clockRunning = true;

  constructor(private readonly events: Events) {}

  /** Called once per render frame by Game with the clamped unscaled delta. */
  advance(realDt: number): void {
    this.realDt = realDt;
    this.dt = realDt * this.scale;
    this.realElapsed += realDt;
    this.elapsed += this.dt;
    this.frame++;
    if (this.clockRunning) {
      const hoursPerSecond = 24 / Config.time.dayLengthSeconds;
      this.addHours(this.dt * hoursPerSecond * this.clockScale);
    }
  }

  setHour(hour: number): void {
    const prevHour = Math.floor(this.hour);
    this.hour = ((hour % 24) + 24) % 24;
    if (Math.floor(this.hour) !== prevHour) this.events.emit('time:hourChanged', { hour: Math.floor(this.hour) });
  }

  addHours(hours: number): void {
    if (hours === 0) return;
    const prevHour = Math.floor(this.hour);
    this.hour += hours;
    while (this.hour >= 24) {
      this.hour -= 24;
      this.day++;
      this.events.emit('time:dayChanged', { day: this.day });
    }
    while (this.hour < 0) {
      this.hour += 24;
      this.day = Math.max(1, this.day - 1);
    }
    if (Math.floor(this.hour) !== prevHour) this.events.emit('time:hourChanged', { hour: Math.floor(this.hour) });
  }

  /** "18:30" style label for the HUD. */
  get clockLabel(): string {
    const h = Math.floor(this.hour);
    const m = Math.floor((this.hour - h) * 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
  }

  get isNight(): boolean {
    return this.hour < Config.dayNight.sunriseHour || this.hour > Config.dayNight.sunsetHour;
  }

  serialize(): { hour: number; day: number } {
    return { hour: this.hour, day: this.day };
  }

  deserialize(data: { hour: number; day: number }): void {
    this.hour = data.hour;
    this.day = data.day;
  }
}
