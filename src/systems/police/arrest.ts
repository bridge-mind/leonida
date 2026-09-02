import type { IPed } from '../../core/entities';
import type { Game } from '../../core/Game';
import { PoliceConfig } from './config';
import type { CopBrain } from './CopBrain';

const K = PoliceConfig.cop;
const PROMPT_KEY = 'police.arrest';

/**
 * BUSTED logic: while the level is ≤ 3 and the player holds still (on foot, or in a
 * car that has stopped) with a cop inside 2.5 m who has line of sight, a contact timer
 * runs; at 1 s the player is arrested. The player system handles the respawn at the
 * police station and the bail fee (`kill('busted')`), the UI shows the card on
 * `player:busted`, and the wanted system clears on `player:respawned`.
 */
export class Arrest {
  timer = 0;
  busting = false;
  private prompted = false;

  constructor(private readonly game: Game) {}

  reset(): void {
    this.timer = 0;
    this.busting = false;
    this.clearPrompt();
  }

  update(dt: number, level: number, cops: readonly IPed[], brains: ReadonlyMap<IPed, CopBrain>, playerSpeed: number, inVehicle: boolean): void {
    if (this.busting) return;
    const game = this.game;
    if (level === 0 || level > 3 || !game.systems.has('player') || !game.player.entity.isAlive) {
      this.timer = 0;
      this.clearPrompt();
      return;
    }
    const still = inVehicle ? playerSpeed < K.vehicleStillSpeed : playerSpeed < K.fleeSpeed * 0.6;
    let contact = false;
    for (let i = 0; i < cops.length; i++) {
      if (brains.get(cops[i])?.arrestContact) {
        contact = true;
        break;
      }
    }
    if (contact && still) {
      this.timer += dt;
      if (!this.prompted) {
        this.prompted = true;
        game.ui?.prompt(PROMPT_KEY, 'Freeze! Hands where I can see them');
      }
      if (this.timer >= K.arrestSeconds) this.bust();
    } else {
      // Contact lost: decay quickly rather than snap to zero, so a stumble does not reset it.
      this.timer = Math.max(0, this.timer - dt * 2);
      if (this.timer === 0) this.clearPrompt();
    }
  }

  bust(): void {
    if (this.busting) return;
    this.busting = true;
    this.clearPrompt();
    const game = this.game;
    const p = game.player.position;
    const position = { x: p.x, y: p.y, z: p.z };
    // Leave the car first so the respawn does not teleport a "mounted" player.
    if (game.vehicles?.playerVehicle) game.vehicles.exitCurrent();
    // `kill` emits player:died (UI: Wasted card); player:busted right after replaces it with the
    // Busted card, which plays the `wanted.busted` sting itself.
    game.player.kill('busted');
    game.events.emit('player:busted', { position });
    game.wanted?.clear('busted');
  }

  private clearPrompt(): void {
    if (!this.prompted) return;
    this.prompted = false;
    this.game.ui?.clearPrompt(PROMPT_KEY);
  }
}
