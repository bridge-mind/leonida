/**
 * Typed event bus. Systems add events without editing this file:
 *
 *   declare module '../../core/Events' {
 *     interface GameEvents {
 *       'vehicle:entered': { vehicle: Vehicle; ped: Ped | Player };
 *     }
 *   }
 *
 * Naming convention: `domain:pastTenseVerb`.
 */
import type { DamageInfo, Entity, IPed, IVehicle, V3 } from './entities';
import type { WantedState } from './GameState';
import type { CharacterId, PickupKind, VehicleArchetypeId, WeaponId } from './peers';

export type CrimeType =
  | 'shot_fired'
  | 'kill_ped'
  | 'kill_cop'
  | 'carjack'
  | 'steal_police_vehicle'
  | 'rob_ped'
  | 'rob_store'
  | 'hit_cop_vehicle'
  | 'hit_cop'
  | 'explosion'
  | 'attack_helicopter'
  | 'kill_swat';

export interface GameEvents {
  'game:ready': { systems: string[] };
  'game:systemDisabled': { id: string; error: unknown };

  'crime:committed': { type: CrimeType; position: V3; actor: 'player' | 'companion'; loud: boolean };

  'player:died': { position: V3 };
  'player:respawned': { position: V3; reason: 'wasted' | 'busted' };
  'player:damaged': { amount: number; source: Entity | 'player' | null };
  'player:mounted': { vehicle: Entity; seat: number };
  'player:dismounted': { position: V3 };

  'time:hourChanged': { hour: number };
  'time:dayChanged': { day: number };

  'interior:entered': { id: string };
  'interior:exited': { id: string };

  'economy:cashChanged': { cash: number; delta: number; reason: string };

  notification: { text: string; kind?: 'info' | 'money' | 'warning' };

  'physics:contactStarted': { a: Entity | null; b: Entity | null; colliderA: number; colliderB: number };
  'physics:contactEnded': { a: Entity | null; b: Entity | null; colliderA: number; colliderB: number };

  'water:entered': { entity: Entity };
  'water:exited': { entity: Entity };

  'save:saved': { slot: string };
  'save:loaded': { slot: string };

  // --- Cross-system events (payload shapes are binding; see docs/TEAM-PLAN.md §3.3) ---
  'vehicle:entered': { vehicle: IVehicle; who: Entity; seat: number };
  'vehicle:exited': { vehicle: IVehicle; who: Entity };
  'vehicle:damaged': { vehicle: IVehicle; amount: number; source: Entity | 'player' | null };
  'vehicle:exploded': { vehicle: IVehicle; position: V3 };
  'vehicle:collided': { vehicle: IVehicle; other: Entity | null; impulse: number };
  /** The player pulled a driver out; the peds system (if loaded) spawns the fleeing driver. */
  'vehicle:carjacked': { vehicle: IVehicle; position: V3; heading: number; driverTypeId: string };
  'vehicle:horn': { vehicle: IVehicle };

  'ped:spawned': { ped: IPed };
  'ped:died': { ped: IPed; source: Entity | 'player' | null };
  'ped:knockedDown': { ped: IPed; source: Entity | null };
  'ped:cowered': { ped: IPed };
  'ped:fled': { ped: IPed };
  'ped:attacked': { ped: IPed; source: Entity | 'player' | null };
  /** A civilian witness finished calling it in (4 s after the crime). */
  'ped:reportedCrime': { ped: IPed; position: V3 };
  /** A ped punched or shot at the player (UI hit direction, audio). */
  'ped:attackedPlayer': { ped: IPed; melee: boolean };
  /** A gang ped or robbery victim dropped cash. */
  'ped:droppedCash': { ped: IPed; amount: number };

  'weapon:fired': { weaponId: WeaponId; origin: V3; direction: V3; shooter: Entity | 'player' };
  'weapon:reloaded': { weaponId: WeaponId };
  'weapon:equipped': { weaponId: WeaponId };
  'weapon:hit': { target: Entity; killed: boolean };
  /** Emitted every frame the player aims at an entity, with the accumulated time on that target. */
  'weapon:aimedAt': { target: Entity; seconds: number };

  'damage:applied': { target: Entity; info: DamageInfo };
  'entity:killed': { target: Entity; source: Entity | 'player' | null };

  'pickup:collected': { kind: PickupKind; value: number; weaponId?: WeaponId };

  'wanted:changed': { level: number; state: WantedState; previousLevel: number };
  'wanted:cleared': { reason: string };
  'player:busted': { position: V3 };

  'casino:entered': { id: 'casino' };
  'casino:exited': { id: 'casino' };
  'casino:won': { game: string; amount: number };
  'casino:lost': { game: string; amount: number };

  'shop:purchased': { itemId: string; price: number };
  'robbery:started': { storeId: string };
  'robbery:completed': { storeId: string; amount: number };
  'chopshop:sold': { archetypeId: VehicleArchetypeId; amount: number };

  'mission:started': { id: string };
  'mission:passed': { id: string; reward: number };
  'mission:failed': { id: string; reason: string };
  'mission:objective': { id: string; text: string };

  'character:switched': { to: CharacterId };
  'ability:activated': { character: CharacterId };
  'ability:ended': { character: CharacterId };

  'ui:newGame': Record<string, never>;
  'ui:continue': Record<string, never>;
  'ui:paused': Record<string, never>;
  'ui:resumed': Record<string, never>;
}

export type EventName = keyof GameEvents;
export type Listener<K extends EventName> = (payload: GameEvents[K]) => void;

type AnyListener = (payload: unknown) => void;

export class Events {
  private readonly listeners = new Map<string, AnyListener[]>();
  private emitting = 0;

  on<K extends EventName>(name: K, fn: Listener<K>): () => void {
    let list = this.listeners.get(name);
    if (!list) {
      list = [];
      this.listeners.set(name, list);
    }
    // Copy-on-write while emitting keeps in-flight iteration stable.
    if (this.emitting > 0) {
      list = list.slice();
      this.listeners.set(name, list);
    }
    list.push(fn as AnyListener);
    return () => this.off(name, fn);
  }

  once<K extends EventName>(name: K, fn: Listener<K>): () => void {
    const off = this.on(name, (payload) => {
      off();
      fn(payload);
    });
    return off;
  }

  off<K extends EventName>(name: K, fn: Listener<K>): void {
    const list = this.listeners.get(name);
    if (!list) return;
    const idx = list.indexOf(fn as AnyListener);
    if (idx < 0) return;
    if (this.emitting > 0) {
      const copy = list.slice();
      copy.splice(idx, 1);
      this.listeners.set(name, copy);
    } else {
      list.splice(idx, 1);
    }
  }

  emit<K extends EventName>(name: K, payload: GameEvents[K]): void {
    const list = this.listeners.get(name);
    if (!list || list.length === 0) return;
    this.emitting++;
    try {
      for (let i = 0; i < list.length; i++) {
        try {
          list[i](payload);
        } catch (err) {
          console.error(`[events:${name}]`, err);
        }
      }
    } finally {
      this.emitting--;
    }
  }

  listenerCount(name: EventName): number {
    return this.listeners.get(name)?.length ?? 0;
  }

  clear(): void {
    this.listeners.clear();
  }
}
