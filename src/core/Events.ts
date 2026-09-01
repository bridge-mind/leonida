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
import type { Entity, V3 } from './entities';

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
