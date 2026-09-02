import { Vector3 } from 'three';
import type { IVehicle } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { Random } from '../../core/Random';
import { resprayCost } from '../../data/prices';
import { storeDef } from '../../data/shops';
import { VEHICLES, isVehicleArchetypeId } from '../../data/vehicles';
import { ResprayConfig } from './config';
import { PromptSlot } from './prompt';

/**
 * Vice Sprays drive-in. A volume in front of the garage door; stop a vehicle in
 * it and press E to pay for a fresh coat. Clears the wanted level only when no
 * cop currently has eyes on the player — otherwise the shop won't touch it.
 */
export class Respray {
  private readonly centre = new Vector3();
  private readonly prompt: PromptSlot;
  private available = false;
  private busy = false;
  private vehicle: IVehicle | null = null;
  private cost = 0;

  constructor(
    private readonly game: Game,
    private readonly rng: Random,
  ) {
    this.prompt = new PromptSlot(game.state);
    const def = storeDef('respray');
    const landmark = def && game.systems.has('world') ? game.world.landmarks.byId(def.landmarkId) : undefined;
    if (landmark) {
      const h = landmark.doorHeading;
      this.centre.set(
        landmark.door.x + Math.sin(h) * ResprayConfig.forward,
        landmark.door.y,
        landmark.door.z + Math.cos(h) * ResprayConfig.forward,
      );
      this.available = true;
    }
  }

  get position(): Vector3 {
    return this.centre;
  }

  /** The vehicle currently parked in the volume, if any. */
  get parked(): IVehicle | null {
    return this.vehicle;
  }

  update(): void {
    if (!this.available || this.busy) return;
    const vehicles = this.game.vehicles;
    const v = vehicles?.playerVehicle ?? null;
    const inside =
      v !== null &&
      !this.game.input.isBlocked &&
      v.position.distanceToSquared(this.centre) < ResprayConfig.radius * ResprayConfig.radius &&
      Math.abs(v.state.speedMs) < ResprayConfig.maxSpeed;
    if (!inside) {
      this.vehicle = null;
      this.prompt.set(null);
      return;
    }
    this.vehicle = v;
    this.cost = resprayCost(this.game.wanted?.level ?? 0);
    this.prompt.set(`Respray $${this.cost.toLocaleString('en-US')}`);
    if (this.game.input.wasPressed('KeyE')) void this.spray();
  }

  /** Runs the job on the parked vehicle. Returns false when refused (seen by cops, broke, no car). */
  async spray(): Promise<boolean> {
    const v = this.vehicle;
    if (!v || this.busy) return false;
    const wanted = this.game.wanted;
    if (wanted && wanted.level > 0 && wanted.playerSeen) {
      this.game.events.emit('notification', { text: 'They can see you — lose them first.', kind: 'warning' });
      this.game.audio.play('shop.denied');
      return false;
    }
    const cost = resprayCost(wanted?.level ?? 0);
    const economy = this.game.economy;
    if (!economy || !economy.trySpend(cost, 'respray')) {
      this.game.events.emit('notification', { text: `Respray costs $${cost.toLocaleString('en-US')}.`, kind: 'warning' });
      return false;
    }
    this.busy = true;
    this.prompt.set(null);
    this.game.audio.play('respray.spray.loop');
    const cleared = !!wanted && wanted.level > 0;
    const finish = (): void => {
      v.color = this.pickColor(v);
      if (cleared) wanted?.clear('respray');
    };
    const interiors = this.game.interiors;
    if (interiors) await interiors.fadeThrough(finish, ResprayConfig.holdSeconds);
    else finish();
    this.game.audio.play('respray.done');
    this.game.events.emit('notification', { text: cleared ? 'Fresh paint. The heat is off.' : 'Fresh paint.', kind: 'info' });
    this.game.events.emit('respray:done', { cost, cleared });
    this.busy = false;
    return true;
  }

  private pickColor(v: IVehicle): number {
    const palette = isVehicleArchetypeId(v.archetypeId) ? VEHICLES[v.archetypeId].palette : [0xffffff];
    if (palette.length < 2) return palette[0] ?? 0xffffff;
    let color = this.rng.pick(palette);
    // A respray should visibly change the car.
    for (let i = 0; i < 4 && color === v.color; i++) color = this.rng.pick(palette);
    return color;
  }

  dispose(): void {
    this.prompt.set(null);
  }
}
