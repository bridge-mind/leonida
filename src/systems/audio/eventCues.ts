import type { Game } from '../../core/Game';
import type { AudioApi } from './api';
import { AudioConfig } from './config';

/**
 * Plays cues for core events that no other system owns a sound for: player
 * hurt/death, water splashes and interior transitions. Everything else is played
 * by the system that owns the event.
 */
export class EventCues {
  private readonly offs: (() => void)[] = [];
  private lastHurt = -Infinity;

  constructor(
    private readonly game: Game,
    private readonly audio: AudioApi,
  ) {}

  attach(): void {
    const { events, state } = this.game;
    this.offs.push(
      events.on('player:damaged', ({ amount }) => {
        const now = this.game.time.realElapsed;
        if (amount <= 0 || now - this.lastHurt < AudioConfig.cues.hurtInterval) return;
        this.lastHurt = now;
        this.audio.play('player.hurt', { position: state.playerPos, volume: Math.min(1, 0.5 + amount / 60) });
      }),
      events.on('player:died', ({ position }) => this.audio.play('player.death', { position })),
      events.on('water:entered', ({ entity }) => {
        const big = entity.kind === 'vehicle';
        if (entity.kind === 'player' || big) this.audio.play(big ? 'water.splash.big' : 'water.splash', { position: entity.position });
      }),
      events.on('interior:entered', () => this.audio.play('interior.enter')),
      events.on('interior:exited', () => this.audio.play('interior.exit')),
    );
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.offs.length = 0;
  }
}
