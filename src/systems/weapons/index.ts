import { MathUtils, Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { Entity, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IWeaponsPeer, WeaponId } from '../../core/peers';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { VEHICLE_WEAPONS, WEAPON_ORDER, WEAPONS, shotInterval } from '../../data/weapons';
import { Crosshair, type CrosshairView } from '../../ui/Crosshair';
import { WeaponWheel } from '../../ui/WeaponWheel';
import { AimTracker } from './Aim';
import { WeaponsConfig as C } from './config';
import { createWeaponDebugActions } from './debug';
import { TargetDummy } from './Dummy';
import { Hitscan, type FireParams } from './Hitscan';
import { Inventory } from './Inventory';
import { Melee } from './Melee';
import { npcSpreadHalfAngle, spreadHalfAngle } from './spread';
import './events';

/**
 * Player weapons: inventory, equip, hitscan firing with spread and recoil, reload, melee,
 * aim target tracking + soft aim-assist, sniper scope, firing from a car window, the
 * weapon wheel and crosshair UI, and `npcFire` for cops/gangs/companion.
 */
export class WeaponSystem implements GameSystem, IWeaponsPeer {
  readonly ammo = new Inventory();
  readonly dummies: TargetDummy[] = [];
  private readonly hitscan: Hitscan;
  private readonly aim: AimTracker;
  private readonly melee: Melee;
  private wheel: WeaponWheel | null = null;
  private crosshair: Crosshair | null = null;

  private aiming = false;
  private aimingPrev = false;
  private fireHeldPrev = false;
  /** Fire edge captured in `update` so a press is never lost when a frame has no fixed step. */
  private fireLatch = false;
  private emptyClicked = false;
  private fireCooldown = 0;
  private reloadTimer = 0;
  private reloadTotal = 0;

  private readonly origin = new Vector3();
  private readonly direction = new Vector3();
  private readonly muzzle = new Vector3();
  private readonly tmp = new Vector3();
  private readonly fireParams: FireParams;
  private readonly crosshairView: CrosshairView = { visible: false, reload01: null, scope: false };
  private readonly offs: (() => void)[] = [];

  constructor(private readonly game: Game) {
    this.hitscan = new Hitscan(game);
    this.aim = new AimTracker(game);
    this.melee = new Melee(game, this.hitscan);
    this.fireParams = {
      def: WEAPONS.fists,
      muzzle: this.muzzle,
      origin: this.origin,
      direction: this.direction,
      halfAngle: 0,
      exclude: null,
      shooter: 'player',
      exactFirst: false,
    };
  }

  init(): void {
    const ui = document.getElementById('ui') ?? document.body;
    this.wheel = new WeaponWheel(this.game, ui, this.ammo, (id) => this.equip(id));
    this.crosshair = new Crosshair(this.game, ui);
    if (C.startWithPistol) this.give('pistol', WEAPONS.pistol.magazine * 3);
    this.offs.push(
      this.game.events.on('player:died', () => this.cancelReload()),
      this.game.events.on('player:respawned', () => this.cancelReload()),
    );
    this.publishState();
  }

  // --- IWeaponsPeer -------------------------------------------------------------------

  give(weaponId: WeaponId, ammo: number): void {
    const isNew = this.ammo.give(weaponId, ammo);
    if (isNew) this.game.audio.play('pickup.weapon');
    else this.game.audio.play('pickup.ammo');
    if (isNew && C.autoEquipOnGive && this.ammo.current === 'fists') this.equip(weaponId);
  }

  equip(weaponId: WeaponId): void {
    if (!this.ammo.has(weaponId) || weaponId === this.ammo.current) return;
    this.cancelReload();
    this.ammo.current = weaponId;
    this.fireCooldown = Math.max(this.fireCooldown, 0.25);
    this.emptyClicked = false;
    this.game.audio.play('weapon.switch');
    this.game.events.emit('weapon:equipped', { weaponId });
  }

  addAmmo(weaponId: WeaponId, amount: number): void {
    this.ammo.addAmmo(weaponId, amount);
  }

  has(weaponId: WeaponId): boolean {
    return this.ammo.has(weaponId);
  }

  get current(): WeaponId {
    return this.ammo.current;
  }

  get isAiming(): boolean {
    return this.aiming;
  }

  get aimTarget(): Entity | null {
    return this.aim.target;
  }

  /** Seconds the current aim target has been under the crosshair. */
  get aimSeconds(): number {
    return this.aim.seconds;
  }

  get isReloading(): boolean {
    return this.reloadTimer > 0;
  }

  inventory(): Partial<Record<WeaponId, number>> {
    return this.ammo.summary();
  }

  npcFire(shooter: Entity, weaponId: WeaponId, origin: V3, direction: V3, accuracy: number): void {
    const def = WEAPONS[weaponId];
    if (!def || def.rpm === 0) return;
    const len = Math.hypot(direction.x, direction.y, direction.z) || 1;
    this.direction.set(direction.x / len, direction.y / len, direction.z / len);
    this.origin.set(origin.x, origin.y, origin.z);
    this.muzzle.copy(this.origin);
    const N = C.npc;
    const p = this.fireParams;
    p.def = def;
    p.halfAngle = npcSpreadHalfAngle(def.spreadDeg, accuracy, N.spreadPenalty, N.missChance, N.missOffsetRad);
    p.exclude = shooter;
    p.shooter = shooter;
    p.exactFirst = false;
    this.hitscan.fire(p);
  }

  // --- Loop ---------------------------------------------------------------------------

  update(dt: number): void {
    if (!this.hasPlayerAndCamera()) return;
    const game = this.game;
    const input = game.input;
    const player = game.player;
    const usable = this.canUse();
    const mounted = player.isMounted;

    this.wheel?.update(usable);
    const wheelOpen = this.wheel?.isOpen === true;

    // Discrete input lives here: `wasPressed` edges are per render frame.
    if (usable && !wheelOpen) {
      if (input.wasPressed('KeyR')) this.startReload();
      for (let i = 0; i < WEAPON_ORDER.length; i++) {
        if (input.wasPressed(`Digit${i + 1}`)) this.equip(WEAPON_ORDER[i]);
      }
      if (input.wheelDelta !== 0) this.equip(this.ammo.cycle(input.wheelDelta > 0 ? 1 : -1));
      const fire = input.fire;
      if (fire && !this.fireHeldPrev) this.fireLatch = true;
      if (!fire) this.emptyClicked = false;
      this.fireHeldPrev = fire;
    } else {
      this.fireHeldPrev = input.fire;
      this.fireLatch = false;
    }

    this.aiming = usable && !wheelOpen && !mounted && input.aim;
    const def = WEAPONS[this.ammo.current];
    if (this.aiming) {
      this.computeAimRay();
      const exclude = this.excludeEntity();
      if (!this.aimingPrev && C.aimAssist) {
        game.camera.three.getWorldDirection(this.tmp);
        if (this.aim.assist(player.position, this.tmp, this.dummies)) this.computeAimRay();
      }
      this.aim.update(dt, this.origin, this.direction, exclude);
      if (def.id === 'sniper') this.scopeKick();
    } else if (this.aim.target) {
      this.aim.clear();
    }
    this.aimingPrev = this.aiming;

    const view = this.crosshairView;
    view.visible = usable && (this.aiming || (mounted && VEHICLE_WEAPONS.has(def.id)));
    view.reload01 = this.reloadTimer > 0 ? 1 - this.reloadTimer / this.reloadTotal : null;
    view.scope = this.aiming && def.id === 'sniper';
    this.crosshair?.update(view);
    this.publishState();
  }

  fixedUpdate(dt: number): void {
    if (!this.hasPlayerAndCamera()) return;
    this.melee.update(dt);
    if (this.fireCooldown > 0) this.fireCooldown -= dt;
    if (this.reloadTimer > 0) {
      this.reloadTimer -= dt;
      if (this.reloadTimer <= 0) this.finishReload();
    }
    if (!this.canUse() || this.wheel?.isOpen) {
      this.fireLatch = false;
      return;
    }
    const def = WEAPONS[this.ammo.current];
    const held = this.game.input.fire;
    const wantsShot = this.fireLatch || (def.automatic && held);
    if (!wantsShot) return;
    if (this.game.player.isMounted && !VEHICLE_WEAPONS.has(def.id)) {
      this.fireLatch = false;
      return;
    }
    if (def.id === 'fists') {
      if (!this.game.player.isMounted && this.melee.punch()) this.fireLatch = false;
      return;
    }
    // A press during the cooldown/reload stays queued so fast clicks fire as soon as allowed.
    if (this.fireCooldown > 0 || this.reloadTimer > 0) return;
    this.fireLatch = false;
    this.fireOnce();
  }

  // --- Firing -------------------------------------------------------------------------

  /** Fires the current weapon once along the current aim ray (ignores cooldown; debug + input path). */
  fireOnce(): boolean {
    const def = WEAPONS[this.ammo.current];
    if (def.rpm === 0) return false;
    const slot = this.ammo.currentSlot;
    if (slot.mag <= 0) {
      if (!this.emptyClicked) {
        this.emptyClicked = true;
        if (def.sounds.empty) this.game.audio.play(def.sounds.empty);
        this.startReload();
      }
      return false;
    }
    const game = this.game;
    const player = game.player;
    this.computeAimRay();
    this.computeMuzzle();
    const p = this.fireParams;
    p.def = def;
    p.halfAngle = spreadHalfAngle(def.spreadDeg) * (this.aiming || player.isMounted ? 1 : C.hipFireSpreadMul);
    p.exclude = this.excludeEntity();
    p.shooter = 'player';
    p.damageMultiplier = player.entity.stats.damageMultiplier;
    p.exactFirst = C.exactScriptedAim && game.input.mouseWorld !== null;
    this.hitscan.fire(p);

    slot.mag--;
    this.fireCooldown = shotInterval(def);
    const stats = game.state.stats;
    stats.shotsFired = (stats.shotsFired ?? 0) + 1;
    if (game.systems.has('camera')) {
      const cam = game.camera;
      const recoil = def.recoil * player.entity.stats.recoilMultiplier;
      cam.pitch = MathUtils.clamp(cam.pitch - recoil, Config.camera.pitchMin, Config.camera.pitchMax);
      cam.fovKick(C.recoil.fovKick);
      cam.shake(C.recoil.shake, C.recoil.shakeSeconds);
    }
    // Fresh position: the wanted system keeps crime positions for hot scenes.
    game.events.emit('crime:committed', {
      type: 'shot_fired',
      position: { x: player.position.x, y: player.position.y, z: player.position.z },
      actor: 'player',
      loud: true,
    });
    if (slot.mag === 0 && C.autoReload) this.startReload();
    return true;
  }

  startReload(): boolean {
    const id = this.ammo.current;
    if (this.reloadTimer > 0 || !this.ammo.canReload(id)) return false;
    const def = WEAPONS[id];
    this.reloadTotal = def.reloadSeconds;
    this.reloadTimer = def.reloadSeconds;
    if (def.sounds.reload) this.game.audio.play(def.sounds.reload, { position: this.game.player.position });
    return true;
  }

  private finishReload(): void {
    this.reloadTimer = 0;
    const id = this.ammo.current;
    this.ammo.completeReload(id);
    this.emptyClicked = false;
    this.game.audio.play('weapon.cock', { position: this.game.player.position });
    this.game.events.emit('weapon:reloaded', { weaponId: id });
  }

  private cancelReload(): void {
    this.reloadTimer = 0;
  }

  private canUse(): boolean {
    const game = this.game;
    return game.player.isAlive && !game.input.isBlocked && game.ui?.isModal !== true;
  }

  /** Player input and aim rays need both; `npcFire` works without them. */
  private hasPlayerAndCamera(): boolean {
    return this.game.systems.has('player') && this.game.systems.has('camera');
  }

  /** The player's own collider (or car) so the ray does not start inside it. */
  private excludeEntity(): Entity | null {
    return this.game.vehicles?.playerVehicle ?? this.game.player.mountedVehicle ?? this.game.player.entity;
  }

  /** Origin = camera; direction = toward the scripted aim point when set, else the camera's look direction. */
  private computeAimRay(): void {
    const cam = this.game.camera.three;
    this.origin.copy(cam.position);
    const mw = this.game.input.mouseWorld;
    if (mw) this.direction.set(mw.x - cam.position.x, mw.y - cam.position.y, mw.z - cam.position.z).normalize();
    else cam.getWorldDirection(this.direction);
  }

  private computeMuzzle(): void {
    const player = this.game.player;
    const cam = this.game.camera;
    const M = C.muzzle;
    if (player.isMounted) {
      const v = player.mountedVehicle;
      this.muzzle.set(v ? v.position.x : player.position.x, (v ? v.position.y : player.position.y) + M.vehicleUp, v ? v.position.z : player.position.z);
      cam.right(this.tmp);
      this.muzzle.addScaledVector(this.tmp, M.vehicleRight);
      return;
    }
    this.muzzle.set(player.position.x, player.position.y + M.up, player.position.z);
    cam.right(this.tmp);
    this.muzzle.addScaledVector(this.tmp, M.right);
    cam.forward(this.tmp);
    this.muzzle.addScaledVector(this.tmp, M.forward);
  }

  /** Per-frame kick whose steady state is `sniperScopeFovDelta` given the camera's exponential decay. */
  private scopeKick(): void {
    const e = Math.exp(-6 * this.game.time.realDt);
    this.game.camera.fovKick((C.sniperScopeFovDelta * (1 - e)) / Math.max(1e-3, e));
  }

  private publishState(): void {
    const s = this.game.state;
    const slot = this.ammo.currentSlot;
    if (!s.weapon) s.weapon = { id: this.ammo.current, ammoInMag: slot.mag, ammoReserve: slot.reserve, isReloading: false };
    s.weapon.id = this.ammo.current;
    s.weapon.ammoInMag = slot.mag;
    s.weapon.ammoReserve = slot.reserve;
    s.weapon.isReloading = this.reloadTimer > 0;
  }

  // --- Dummies (isolation testing) -------------------------------------------------------

  spawnDummy(x: number, z: number): TargetDummy {
    const y = this.game.physics.groundHeightAt(x, z) ?? 0;
    const dummy = new TargetDummy(this.game, { x, y, z });
    this.dummies.push(dummy);
    return dummy;
  }

  clearDummies(): void {
    for (const d of this.dummies) d.dispose();
    this.dummies.length = 0;
  }

  // --- Save / debug -----------------------------------------------------------------------

  serialize(): unknown {
    return this.ammo.serialize();
  }

  deserialize(data: unknown): void {
    this.ammo.deserialize(data);
    this.cancelReload();
    this.publishState();
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.clearDummies();
    this.wheel?.dispose();
    this.crosshair?.dispose();
    this.game.state.weapon = null;
  }

  debugActions = createWeaponDebugActions(this);
}

const module: SystemModule = {
  id: 'weapons',
  priority: PRIORITY.weapons,
  create: (game) => new WeaponSystem(game),
};

export default module;
