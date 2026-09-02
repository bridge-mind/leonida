import { Vector3 } from 'three';
import type { Game } from '../../core/Game';
import { Vehicle } from '../../entities/Vehicle';
import { VehicleConfig } from './config';

function isVehicle(e: unknown): e is Vehicle {
  return e instanceof Vehicle;
}

const I = VehicleConfig.interaction;

interface PendingEnter {
  vehicle: Vehicle;
  timer: number;
  carjack: boolean;
}

/**
 * The player's side of vehicles: the F prompt (enter / carjack / exit), driving
 * input → `vehicle.controls`, horn / lights, and `game.state.vehicle`.
 */
export class PlayerControl {
  /** Vehicle the player is currently driving. */
  vehicle: Vehicle | null = null;
  private pending: PendingEnter | null = null;
  private readonly prompt = { key: 'F', text: '' };
  private promptShown = false;
  private readonly stateInfo = { id: 'sedan', speedKmh: 0, health: 0 };
  private readonly door = new Vector3();

  constructor(
    private readonly game: Game,
    private readonly nearestEnterable: (pos: Vector3, radius: number) => Vehicle | null,
  ) {}

  private get canInteract(): boolean {
    const g = this.game;
    if (!g.systems.has('player')) return false;
    return g.player.isAlive && g.state.interior === null && !g.input.isBlocked;
  }

  fixedUpdate(dt: number): void {
    const g = this.game;
    if (!g.systems.has('player')) return;
    const player = g.player;
    const input = g.input;

    if (this.vehicle) {
      const v = this.vehicle;
      if (!v.isAlive || !v.playerDriven) {
        // Exploded or despawned under us: the FX code already dismounted the player.
        this.vehicle = null;
        return;
      }
      if (!player.isAlive) {
        // Died in the seat (shot, BUSTED): Player.fixedUpdate skips the death timer while
        // mounted, so leave the car now; the camera follows the on-foot target automatically.
        this.exitCurrent();
        return;
      }
      const blocked = input.isBlocked;
      const c = v.controls;
      c.throttle = !blocked && input.isDown('KeyW') ? 1 : 0;
      c.brake = !blocked && input.isDown('KeyS') ? 1 : 0;
      c.steer = blocked ? 0 : (input.isDown('KeyD') ? 1 : 0) - (input.isDown('KeyA') ? 1 : 0);
      c.handbrake = !blocked && input.isDown('Space');
      if (!blocked && input.wasPressed('KeyH')) this.honk(v);
      if (!blocked && input.wasPressed('KeyL')) v.lightsOverride = !v.state.lightsOn;
      // KeyC (camera distance cycle) is intentionally a no-op: the camera exposes no distance API yet.
      if (!blocked && input.wasPressed('KeyF')) this.exitCurrent();
      return;
    }

    if (this.pending) {
      this.tickPending(dt);
      return;
    }

    // Mounted by someone else (missions, debug): adopt the car so driving input works.
    const mounted = player.mountedVehicle;
    if (mounted && isVehicle(mounted) && mounted.playerDriven) {
      this.vehicle = mounted;
      this.setPrompt(null);
      return;
    }

    if (!this.canInteract || player.isMounted) {
      this.setPrompt(null);
      return;
    }
    const target = this.nearestEnterable(player.position, I.enterRadius);
    if (!target) {
      this.setPrompt(null);
      return;
    }
    const carjack = target.driver !== null || target.driverEntity !== null;
    this.setPrompt(`${carjack ? 'Steal' : 'Enter'} ${target.archetype.name}`);
    if (input.wasPressed('KeyF')) this.beginEnter(target, carjack);
  }

  private tickPending(dt: number): void {
    const p = this.pending as PendingEnter;
    const player = this.game.player;
    const v = p.vehicle;
    const far = player.position.distanceTo(v.position) > I.cancelDistance + v.length / 2;
    if (!v.isAlive || !player.isAlive || far || !v.canEnter(player.entity)) {
      this.pending = null;
      return;
    }
    p.timer -= dt;
    if (p.timer > 0) return;
    this.pending = null;
    this.completeEnter(v, p.carjack);
  }

  private beginEnter(v: Vehicle, carjack: boolean): void {
    this.setPrompt(null);
    this.pending = { vehicle: v, timer: carjack ? I.carjackSeconds : I.enterSeconds, carjack };
    this.game.audio.play('vehicle.door.open', { position: v.seatDoorPosition(0, this.door) });
  }

  private completeEnter(v: Vehicle, carjack: boolean): void {
    const g = this.game;
    const door = v.seatDoorPosition(0, this.door);
    if (carjack) {
      const driverTypeId = v.driverTypeId ?? (v.isPolice ? 'cop' : 'business');
      v.setDriver(null);
      v.driverTypeId = null;
      const pos = { x: door.x, y: door.y, z: door.z };
      g.events.emit('vehicle:carjacked', { vehicle: v, position: pos, heading: v.heading, driverTypeId });
      g.events.emit('crime:committed', { type: v.isPolice ? 'steal_police_vehicle' : 'carjack', position: pos, actor: 'player', loud: false });
    }
    const seat = v.enter(g.player.entity, 0);
    if (seat < 0) return;
    g.player.mount(v, seat);
    this.vehicle = v;
    g.audio.play('vehicle.door.close', { position: door });
    g.audio.play('vehicle.engine.start', { position: v.position });
  }

  /** Same as pressing F next to a car; returns the vehicle being entered (mount completes after the timer). */
  enterNearest(): Vehicle | null {
    if (this.vehicle) return this.vehicle;
    if (!this.canInteract || this.game.player.isMounted) return null;
    const target = this.nearestEnterable(this.game.player.position, I.enterRadius);
    if (!target) return null;
    this.beginEnter(target, target.driver !== null || target.driverEntity !== null);
    return target;
  }

  exitCurrent(): void {
    const v = this.vehicle;
    if (!v || !this.game.systems.has('player')) return;
    const g = this.game;
    const speed = v.state.speedMs;
    const door = v.seatDoorPosition(0, this.door);
    v.exit(g.player.entity);
    g.player.dismount(door);
    g.audio.play('vehicle.door.open', { position: door });
    this.vehicle = null;
    if (speed > I.exitDamageSpeed) g.player.applyDamage({ amount: speed * I.exitDamagePerMs, type: 'fall', source: v });
  }

  /** Called by the system when a vehicle is despawned while the player is inside it. */
  forceExit(v: Vehicle): void {
    if (this.vehicle === v) this.exitCurrent();
    if (this.pending?.vehicle === v) this.pending = null;
  }

  private honk(v: Vehicle): void {
    this.game.audio.play(v.archetype.hornSound, { position: v.position });
    this.game.events.emit('vehicle:horn', { vehicle: v });
  }

  /** Per render frame: HUD read model. */
  update(): void {
    const v = this.vehicle;
    const s = this.game.state;
    if (v) {
      this.stateInfo.id = v.archetypeId;
      this.stateInfo.speedKmh = Math.round(v.state.speedMs * 3.6);
      this.stateInfo.health = Math.max(0, Math.round(v.state.health));
      s.vehicle = this.stateInfo;
    } else if (s.vehicle === this.stateInfo) {
      s.vehicle = null;
    }
  }

  private setPrompt(text: string | null): void {
    const prompts = this.game.state.prompts;
    if (text === null) {
      if (this.promptShown) {
        const idx = prompts.indexOf(this.prompt);
        if (idx >= 0) prompts.splice(idx, 1);
        this.promptShown = false;
      }
      return;
    }
    this.prompt.text = text;
    if (!this.promptShown) {
      prompts.push(this.prompt);
      this.promptShown = true;
    }
  }

  dispose(): void {
    this.setPrompt(null);
    if (this.game.state.vehicle === this.stateInfo) this.game.state.vehicle = null;
  }
}
