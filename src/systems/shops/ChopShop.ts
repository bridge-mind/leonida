import { Vector3 } from 'three';
import type { IVehicle } from '../../core/entities';
import type { Game } from '../../core/Game';
import type { IChopShopPeer } from '../../core/peers';
import { Random } from '../../core/Random';
import { PRIORITY, type GameSystem, type SystemModule } from '../../core/System';
import { chopValue } from '../../data/prices';
import { storeDef } from '../../data/shops';
import type { VehicleArchetypeId } from '../../core/peers';
import { VEHICLES, VEHICLE_IDS, isVehicleArchetypeId } from '../../data/vehicles';
import { ChopShopConfig as C } from './chopConfig';
import { PromptSlot } from './prompt';

/**
 * Chop shop at the Docks. Drive any non-police vehicle into the volume in front
 * of the garage and stop: the prompt shows the payout (archetype chop value ×
 * health multiplier, doubled for today's wanted archetype). Selling dismounts the
 * player, fades, despawns the car and pays out.
 */
export class ChopShopSystem implements GameSystem, IChopShopPeer {
  private readonly centre = new Vector3();
  private readonly prompt: PromptSlot;
  private available = false;
  private busy = false;
  private parked: IVehicle | null = null;
  private wanted: VehicleArchetypeId = 'sports';
  private readonly unsubs: Array<() => void> = [];

  constructor(private readonly game: Game) {
    this.prompt = new PromptSlot(game.state);
  }

  init(): void {
    const def = storeDef('chop_shop');
    const landmark = def && this.game.systems.has('world') ? this.game.world.landmarks.byId(def.landmarkId) : undefined;
    if (landmark) {
      const h = landmark.doorHeading;
      this.centre.set(landmark.door.x + Math.sin(h) * C.forward, landmark.door.y, landmark.door.z + Math.cos(h) * C.forward);
      this.available = true;
    }
    this.rollWanted(this.game.time.day);
    this.unsubs.push(
      this.game.events.on('time:dayChanged', ({ day }) => {
        this.rollWanted(day);
        this.game.events.emit('notification', { text: `Chop shop wants a ${VEHICLES[this.wanted].name} today (2×).`, kind: 'info' });
        this.refreshBlip();
      }),
      this.game.events.on('game:ready', () => this.refreshBlip()),
    );
  }

  dispose(): void {
    for (const u of this.unsubs) u();
    this.unsubs.length = 0;
    this.prompt.set(null);
  }

  // --- IChopShopPeer -----------------------------------------------------------------

  get wantedArchetype(): VehicleArchetypeId {
    return this.wanted;
  }

  valueOf(vehicle: IVehicle): number {
    if (vehicle.isPolice || !isVehicleArchetypeId(vehicle.archetypeId)) return 0;
    const archetype = VEHICLES[vehicle.archetypeId];
    const health01 = archetype.health > 0 ? vehicle.state.health / archetype.health : 0;
    return chopValue(archetype.chopValue, health01, vehicle.archetypeId === this.wanted);
  }

  get position(): Vector3 {
    return this.centre;
  }

  // --- Loop --------------------------------------------------------------------------

  update(): void {
    if (!this.available || this.busy) return;
    const v = this.game.vehicles?.playerVehicle ?? null;
    const inside =
      v !== null &&
      !this.game.input.isBlocked &&
      v.position.distanceToSquared(this.centre) < C.radius * C.radius &&
      Math.abs(v.state.speedMs) < C.maxSpeed;
    if (!inside) {
      this.parked = null;
      this.prompt.set(null);
      return;
    }
    this.parked = v;
    if (v.isPolice) {
      this.prompt.set('Chop shop: no police vehicles');
      return;
    }
    const value = this.valueOf(v);
    if (value <= 0) {
      this.prompt.set('Chop shop: too wrecked to sell');
      return;
    }
    this.prompt.set(`Sell for $${value.toLocaleString('en-US')}${v.archetypeId === this.wanted ? ' (wanted 2×)' : ''}`);
    if (this.game.input.wasPressed('KeyE')) void this.sell();
  }

  /** Sells the vehicle parked in the volume. Returns the payout (0 when refused). */
  async sell(): Promise<number> {
    const v = this.parked;
    const vehicles = this.game.vehicles;
    const economy = this.game.economy;
    if (!v || !vehicles || !economy || this.busy) return 0;
    const value = this.valueOf(v);
    if (value <= 0) {
      this.game.audio.play('shop.denied');
      return 0;
    }
    this.busy = true;
    this.prompt.set(null);
    const archetypeId = v.archetypeId as VehicleArchetypeId;
    const finish = (): void => {
      // despawn() force-exits the player first, so he ends up standing at the door.
      vehicles.despawn(v);
      economy.add(value, 'chopshop');
    };
    this.game.audio.play('chopshop.crush');
    const interiors = this.game.interiors;
    if (interiors) await interiors.fadeThrough(finish, C.holdSeconds);
    else finish();
    this.game.audio.play('chopshop.paid');
    this.game.events.emit('chopshop:sold', { archetypeId, amount: value });
    this.game.events.emit('notification', { text: `Sold the ${VEHICLES[archetypeId].name} for $${value.toLocaleString('en-US')}.`, kind: 'money' });
    this.parked = null;
    this.busy = false;
    return value;
  }

  private rollWanted(day: number): void {
    const pool = VEHICLE_IDS.filter((id) => !C.excluded.includes(id));
    this.wanted = Random.fromString(`chopshop:${day}`).pick(pool);
  }

  private refreshBlip(): void {
    const nav = this.game.navigation;
    const def = storeDef('chop_shop');
    const landmark = def && this.game.systems.has('world') ? this.game.world.landmarks.byId(def.landmarkId) : undefined;
    if (!nav || !def || !landmark) return;
    nav.addBlip({ id: landmark.id, kind: landmark.kind, position: landmark.door, label: `${def.name} — wants ${VEHICLES[this.wanted].name}` });
  }

  debugActions = {
    value: (): number => (this.parked ? this.valueOf(this.parked) : (this.game.vehicles?.playerVehicle ? this.valueOf(this.game.vehicles.playerVehicle) : 0)),
    sell: (): Promise<number> => {
      // Debug path: accept the player's current car wherever it is.
      if (!this.parked) this.parked = this.game.vehicles?.playerVehicle ?? null;
      return this.sell();
    },
    wanted: (): VehicleArchetypeId => this.wanted,
    setWanted: (id: unknown): VehicleArchetypeId => {
      if (isVehicleArchetypeId(String(id))) this.wanted = String(id) as VehicleArchetypeId;
      this.refreshBlip();
      return this.wanted;
    },
    position: (): { x: number; y: number; z: number } => ({ x: this.centre.x, y: this.centre.y, z: this.centre.z }),
  };
}

/**
 * Module descriptor, discovered by `Game.start` through `src/systems/chopshop/index.ts`
 * (a re-export). Other systems reach it via `game.systems.get<IChopShopPeer>('chopshop')`.
 */
export const CHOPSHOP_MODULE: SystemModule = {
  id: 'chopshop',
  priority: PRIORITY.robbery + 5,
  create: (game) => new ChopShopSystem(game),
};
