import { Vector3 } from 'three';
import type { IPed } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { Random } from '../../core/Random';
import type { StoreDef } from '../../data/shops';
import type { StoreInterior } from '../../world/interiors/stores';
import { ClerkBrain } from './brains/ClerkBrain';
import { GuardBrain } from './brains/GuardBrain';

/**
 * Per-store live state: the interior it owns, the peds staffing it while the
 * player is inside, and its closed-until day. Peds are spawned on entry and
 * despawned on exit so ambient ped budgets are never consumed by empty stores.
 */
export class StoreRuntime {
  clerk: IPed | null = null;
  clerkBrain: ClerkBrain | null = null;
  readonly guards: Array<{ ped: IPed; brain: GuardBrain }> = [];
  /** In-game day on which the store reopens; 0 = open. */
  closedUntilDay = 0;
  /** Rolled once per visit so the 25% is "per store per visit", not per frame. */
  shotgunRolled = false;
  /** Debug/autotest override for the shotgun roll (null = roll normally). */
  shotgunOverride: boolean | null = null;
  private readonly tmp = new Vector3();

  constructor(
    readonly def: StoreDef,
    readonly interior: StoreInterior,
  ) {}

  get layout() {
    return this.interior.layout;
  }

  isClosed(day: number): boolean {
    return this.closedUntilDay > day;
  }

  /** Spawns the clerk and guards for a visit. Safe without the ped system (the store is simply unstaffed). */
  staff(
    game: Game,
    origin: Vector3,
    rng: Random,
    hooks: { onClerkShotgun(): void; onGuardWarning(): void; onGuardHostile(): void },
  ): void {
    this.unstaff(game);
    const peds = game.systems.has('peds') ? game.peds : undefined;
    if (!peds) return;
    const layout = this.layout;

    if (this.def.clerkType && layout.clerk) {
      this.shotgunRolled = this.shotgunOverride ?? (this.def.shotgunChance > 0 && rng.chance(this.def.shotgunChance));
      const brain = new ClerkBrain(layout.clerk, this.shotgunRolled, { onDrewShotgun: hooks.onClerkShotgun });
      this.tmp.set(origin.x + layout.clerk.pos.x, origin.y + layout.clerk.pos.y, origin.z + layout.clerk.pos.z);
      const ped = peds.spawn(this.def.clerkType, this.tmp, { brain, heading: layout.clerk.heading });
      ped.state = 'scripted';
      this.clerk = ped;
      this.clerkBrain = brain;
    }

    for (let i = 0; i < Math.min(this.def.guards, layout.guards.length); i++) {
      const post = layout.guards[i];
      const brain = new GuardBrain(post, { onWarning: hooks.onGuardWarning, onHostile: hooks.onGuardHostile });
      this.tmp.set(origin.x + post.pos.x, origin.y + post.pos.y, origin.z + post.pos.z);
      const ped = peds.spawn('guard', this.tmp, { brain, heading: post.heading, weaponId: 'pistol' });
      ped.state = 'scripted';
      this.guards.push({ ped, brain });
    }
  }

  unstaff(game: Game): void {
    const peds = game.systems.has('peds') ? game.peds : undefined;
    if (this.clerk && peds) peds.despawn(this.clerk);
    this.clerk = null;
    this.clerkBrain = null;
    for (const g of this.guards) peds?.despawn(g.ped);
    this.guards.length = 0;
  }

  /** Per-frame: mirror the "player has a weapon out" flag into the brains. */
  update(_dt: number, playerArmed: boolean): void {
    if (this.clerkBrain) this.clerkBrain.playerArmed = playerArmed;
    for (const g of this.guards) g.brain.playerArmed = playerArmed;
  }

  get anyGuardHostile(): boolean {
    for (const g of this.guards) if (g.brain.isHostile) return true;
    return false;
  }
}
