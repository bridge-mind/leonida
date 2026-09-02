import { Vector3, type Object3D } from 'three';
import { allocEntityId, type Entity, type V3 } from '../core/entities';
import type { PickupKind, WeaponId } from '../core/peers';

/**
 * Pooled world pickup (cash, weapon, ammo, armor, health). No physics: the pickup
 * system distance-tests against the player each frame. Rendered by instancing.
 */
export class Pickup implements Entity {
  readonly id = allocEntityId();
  readonly kind = 'pickup' as const;
  object3d: Object3D | null = null;
  readonly position = new Vector3();
  isAlive = false;

  pickupKind: PickupKind = 'cash';
  value = 0;
  weaponId: WeaponId | undefined = undefined;
  /** Seconds since spawn (expiry, bob phase). */
  age = 0;
  /** Rest height; the bob animates around it. */
  baseY = 0;
  spin = 0;
  /** Slot in the renderer for this kind; -1 while pooled. */
  instanceIndex = -1;
  active = false;

  spawn(kind: PickupKind, position: V3, value: number, weaponId: WeaponId | undefined, spin: number): void {
    this.pickupKind = kind;
    this.value = value;
    this.weaponId = weaponId;
    this.position.set(position.x, position.y, position.z);
    this.baseY = position.y;
    this.age = 0;
    this.spin = spin;
    this.isAlive = true;
    this.active = true;
  }

  despawn(): void {
    this.isAlive = false;
    this.active = false;
    this.instanceIndex = -1;
  }
}
