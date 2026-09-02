import { Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { CameraTarget, DamageInfo, Entity, IEnterable, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { Player, type PlayerMoveInput } from '../../entities/Player';

const P = Config.player;

/**
 * Drives the Player entity from input, handles death/respawn, and publishes
 * player state. Public API used by other agents: `entity`, `mount`, `dismount`,
 * `teleport`, `applyDamage`, `position`, `heading`, `cameraTarget`.
 */
export class PlayerSystem implements GameSystem {
  entity!: Player;
  private readonly moveInput: PlayerMoveInput = { x: 0, y: 0, cameraYaw: 0, sprint: false, walk: false, jump: false, aiming: false };
  private readonly axis = { x: 0, y: 0 };
  private respawnPending = false;

  constructor(private readonly game: Game) {}

  init(): void {
    const spawn = this.resolveSpawn();
    this.entity = new Player(this.game, spawn, spawn.heading);
    this.publishState();
  }

  /** Scene queries only see colliders after the first physics step, so snap on the first tick. */
  private needsGroundSnap = true;

  private resolveSpawn(): { x: number; y: number; z: number; heading: number } {
    const raw = this.game.params.get('spawn');
    if (raw) {
      const parts = raw.split(',').map(Number);
      if (parts.length >= 2 && parts.every((n) => Number.isFinite(n))) {
        return { x: parts[0], y: 0, z: parts[1], heading: parts[2] ?? P.spawn.heading };
      }
    }
    return { ...P.spawn };
  }

  // --- Public API --------------------------------------------------------------

  get position(): Vector3 {
    return this.entity.position;
  }

  get heading(): number {
    return this.entity.heading;
  }

  get velocity(): Vector3 {
    return this.entity.velocity;
  }

  get isAlive(): boolean {
    return this.entity.isAlive;
  }

  get health(): number {
    return this.entity.health;
  }

  get armor(): number {
    return this.entity.armor;
  }

  get cameraTarget(): CameraTarget {
    return this.entity.cameraTarget;
  }

  get isMounted(): boolean {
    return this.entity.mounted !== null;
  }

  get mountedVehicle(): (IEnterable & Entity) | null {
    return this.entity.mounted?.vehicle ?? null;
  }

  applyDamage(info: DamageInfo): void {
    this.entity.applyDamage(info);
  }

  heal(amount: number): void {
    this.entity.health = Math.min(P.maxHealth, this.entity.health + amount);
  }

  addArmor(amount: number): void {
    this.entity.armor = Math.min(P.maxArmor, this.entity.armor + amount);
  }

  /** Teleports to `position` (y is snapped to the ground when `snapToGround`). */
  teleport(position: V3, heading?: number, snapToGround = true): void {
    let y = position.y;
    if (snapToGround) {
      const ground = this.game.physics.groundHeightAt(position.x, position.z);
      if (ground !== null) y = ground + 0.05;
    }
    this.entity.teleport({ x: position.x, y, z: position.z }, heading);
    if (this.game.systems.has('camera')) this.game.camera.snap();
  }

  /** Hides the character and parents it to the vehicle; the camera follows the vehicle target. */
  mount(vehicle: IEnterable & Entity, seat = 0): void {
    this.entity.mount(vehicle, seat);
    const cam = this.game.systems.has('camera') ? this.game.camera : undefined;
    const target = (vehicle as unknown as { cameraTarget?: CameraTarget }).cameraTarget;
    if (cam && target) cam.setTarget(target);
  }

  dismount(position: V3): void {
    this.entity.dismount(position);
    const cam = this.game.systems.has('camera') ? this.game.camera : undefined;
    if (cam) cam.setTarget(this.entity.cameraTarget);
  }

  /** Kills the player immediately (used by BUSTED/debug). `reason` selects the respawn point. */
  kill(reason: 'wasted' | 'busted' = 'wasted'): void {
    this.entity.applyDamage({ amount: 1e6, type: 'fall' });
    this.pendingReason = reason;
  }

  private pendingReason: 'wasted' | 'busted' = 'wasted';

  // --- Loop ------------------------------------------------------------------

  fixedUpdate(dt: number): void {
    if (this.needsGroundSnap) {
      this.needsGroundSnap = false;
      const p = this.entity.position;
      const ground = this.game.physics.groundHeightAt(p.x, p.z);
      if (ground !== null && Math.abs(ground - p.y) > 0.02) this.entity.teleport({ x: p.x, y: ground + 0.05, z: p.z });
    }
    const input = this.game.input;
    const blocked = input.isBlocked;
    input.moveAxis(this.axis);
    const cam = this.game.systems.has('camera') ? this.game.camera : undefined;
    const mi = this.moveInput;
    mi.x = blocked ? 0 : this.axis.x;
    mi.y = blocked ? 0 : this.axis.y;
    mi.cameraYaw = cam ? cam.yaw : this.entity.heading + Math.PI;
    mi.sprint = !blocked && (input.isDown('ShiftLeft') || input.isDown('ShiftRight'));
    mi.walk = !blocked && input.isDown('AltLeft');
    mi.jump = !blocked && input.isDown('Space');
    mi.aiming = !blocked && input.aim && this.entity.isAlive;
    this.entity.fixedUpdate(dt, mi);

    if (cam && !this.entity.mounted && (cam.mode === 'thirdPerson' || cam.mode === 'aim')) {
      cam.setMode(mi.aiming ? 'aim' : 'thirdPerson');
    }

    if (!this.entity.isAlive && this.entity.deathTimer <= 0 && !this.respawnPending) {
      this.respawnPending = true;
      this.respawn(this.pendingReason);
      this.pendingReason = 'wasted';
      this.respawnPending = false;
    }
  }

  private respawn(reason: 'wasted' | 'busted'): void {
    // 'wasted' hospital fee is owned by the economy system (on `player:respawned`); only bail is billed here.
    const economy = this.game.economy;
    if (economy && reason === 'busted') {
      const loss = Math.min(P.deathCashMax, Math.floor(economy.cash * P.deathCashFraction));
      if (loss > 0) economy.trySpend(loss, 'bail');
    }
    const point = this.respawnPoint(reason);
    this.entity.respawn(point, point.heading, reason);
    if (this.game.systems.has('camera')) this.game.camera.snap();
    // Autosave after the wallet fee and the teleport so Continue resumes outside the hospital.
    this.game.save.save();
  }

  private respawnPoint(reason: 'wasted' | 'busted'): { x: number; y: number; z: number; heading: number } {
    const world = this.game.systems.has('world') ? this.game.world : undefined;
    const kind = reason === 'wasted' ? 'hospital' : 'police';
    const landmark = world?.landmarks.nearestOfKind(this.entity.position, kind);
    if (landmark) {
      const ground = this.game.physics.groundHeightAt(landmark.door.x, landmark.door.z) ?? 0;
      return { x: landmark.door.x, y: ground + 0.05, z: landmark.door.z, heading: landmark.doorHeading };
    }
    return { ...P.spawn };
  }

  update(dt: number, alpha: number): void {
    this.entity.update(dt, alpha);
    this.publishState();
  }

  private publishState(): void {
    const s = this.game.state;
    s.health = this.entity.health;
    s.armor = this.entity.armor;
    s.playerPos.x = this.entity.position.x;
    s.playerPos.y = this.entity.position.y;
    s.playerPos.z = this.entity.position.z;
    s.playerHeading = this.entity.heading;
  }

  dispose(): void {
    this.entity.dispose();
  }

  serialize(): unknown {
    const p = this.entity.position;
    return { x: p.x, y: p.y, z: p.z, heading: this.entity.heading, health: this.entity.health, armor: this.entity.armor };
  }

  deserialize(data: unknown): void {
    const d = data as { x: number; y: number; z: number; heading: number; health: number; armor: number };
    if (!d || typeof d.x !== 'number') return;
    this.teleport({ x: d.x, y: d.y, z: d.z }, d.heading);
    // A save written mid-death would otherwise resume with 0 hp but `isAlive` true.
    if (typeof d.health === 'number' && Number.isFinite(d.health)) this.entity.health = Math.max(1, d.health);
    if (typeof d.armor === 'number' && Number.isFinite(d.armor)) this.entity.armor = Math.max(0, d.armor);
  }

  debugActions = {
    teleport: (x: unknown, z: unknown, heading?: unknown): V3 => {
      this.teleport({ x: Number(x), y: 0, z: Number(z) }, heading === undefined ? undefined : Number(heading));
      return { x: this.position.x, y: this.position.y, z: this.position.z };
    },
    pos: (): V3 => ({ x: this.position.x, y: this.position.y, z: this.position.z }),
    damage: (amount: unknown): number => {
      this.applyDamage({ amount: Number(amount), type: 'bullet' });
      return this.entity.health;
    },
    kill: (reason?: unknown): void => this.kill(reason === 'busted' ? 'busted' : 'wasted'),
    heal: (): void => {
      this.entity.health = P.maxHealth;
    },
    armor: (amount: unknown): void => {
      this.entity.armor = Math.min(P.maxArmor, Number(amount));
    },
    face: (heading: unknown): void => {
      this.entity.heading = Number(heading);
    },
  };
}

const module: SystemModule = {
  id: 'player',
  priority: PRIORITY.player,
  create: (game) => new PlayerSystem(game),
};

export default module;
