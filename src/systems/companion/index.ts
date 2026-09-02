import { Vector3 } from 'three';
import type { IPed } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { CharacterId } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { CompanionBrain } from './CompanionBrain';
import { CompanionConfig, teleportPed } from './config';

const tmp = new Vector3();

/**
 * Summons the other protagonist as a follower NPC (missions 1, 2 and 8). Owns the
 * hostile memory the brain reads and re-spawns the ped if something kills it outright.
 */
export class CompanionSystem implements GameSystem {
  ped: IPed | null = null;
  character: CharacterId | null = null;
  private readonly hostiles = new Map<IPed, number>();
  private brain!: CompanionBrain;
  private readonly offs: (() => void)[] = [];
  private respawnTimer = -1;
  private liveCheck = 1;

  constructor(private readonly game: Game) {}

  init(): void {
    this.brain = new CompanionBrain(this.game, this.hostiles);
    const game = this.game;
    this.offs.push(
      game.events.on('ped:attackedPlayer', (e) => this.hostiles.set(e.ped, game.time.realElapsed)),
      game.events.on('ped:died', (e) => {
        this.hostiles.delete(e.ped);
        if (e.ped === this.ped) {
          this.ped = null;
          this.respawnTimer = CompanionConfig.respawnDelay;
        }
      }),
      game.events.on('interior:exited', () => {
        if (this.ped && game.systems.has('player')) teleportPed(this.ped, this.besidePlayer(), game.player.heading);
      }),
    );
  }

  get active(): boolean {
    return this.character !== null;
  }

  get isDowned(): boolean {
    return this.brain.isDowned;
  }

  /** Spawns the companion next to the player; replaces any current one. */
  summon(character: CharacterId): IPed | null {
    const peds = this.game.peds;
    if (!peds || !this.game.systems.has('player')) {
      console.warn('[companion] peds/player system missing: companion unavailable');
      return null;
    }
    if (this.ped) this.despawnPed();
    this.character = character;
    this.brain.reset();
    const ped = peds.spawn(character, this.besidePlayer(), { brain: this.brain, heading: this.game.player.heading, weaponId: CompanionConfig.combat.weapon });
    ped.state = 'scripted';
    ped.health = CompanionConfig.pinnedHealth;
    this.ped = ped;
    this.respawnTimer = -1;
    return ped;
  }

  dismiss(): void {
    this.character = null;
    this.respawnTimer = -1;
    this.despawnPed();
  }

  private despawnPed(): void {
    const ped = this.ped;
    this.ped = null;
    if (ped) this.game.peds?.despawn(ped);
  }

  private besidePlayer(): Vector3 {
    const player = this.game.player;
    const h = player.heading;
    tmp.set(-Math.cos(h), 0, Math.sin(h)).multiplyScalar(CompanionConfig.spawnOffset).add(player.position);
    return tmp;
  }

  fixedUpdate(dt: number): void {
    // The ped pool recycles instances silently (capacity pressure); notice and re-summon.
    this.liveCheck -= dt;
    if (this.liveCheck <= 0) {
      this.liveCheck = 1;
      if (this.ped && this.game.peds && !this.game.peds.all.includes(this.ped)) {
        this.ped = null;
        this.respawnTimer = CompanionConfig.respawnDelay;
      }
    }
    if (this.respawnTimer >= 0) {
      this.respawnTimer -= dt;
      if (this.respawnTimer < 0 && this.character && !this.ped) this.summon(this.character);
    }
    // Forget attackers the brain no longer cares about so the map cannot grow unbounded.
    if (this.hostiles.size > 64) {
      const now = this.game.time.realElapsed;
      for (const [p, t] of this.hostiles) if (now - t > CompanionConfig.combat.hostileMemory) this.hostiles.delete(p);
    }
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.dismiss();
  }

  debugActions = {
    summon: (who: unknown): boolean => this.summon(who === 'lucia' ? 'lucia' : 'jason') !== null,
    dismiss: (): void => this.dismiss(),
    state: (): { character: CharacterId | null; alive: boolean; downed: boolean; inVehicle: boolean; hp: number } => ({
      character: this.character,
      alive: this.ped?.isAlive ?? false,
      downed: this.brain.isDowned,
      inVehicle: !!this.ped?.vehicle,
      hp: this.brain.hp,
    }),
    hurt: (amount: unknown): void => {
      this.ped?.applyDamage({ amount: Number(amount ?? 50), type: 'bullet' });
    },
  };
}

const module: SystemModule = {
  id: 'companion',
  priority: PRIORITY.companion,
  create: (game) => new CompanionSystem(game),
};

export default module;
