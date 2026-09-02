import type { Collider } from '@dimforge/rapier3d-compat';
import { BoxGeometry, Mesh, MeshStandardMaterial, Vector3 } from 'three';
import { allocEntityId, type DamageInfo, type Entity, type IDamageable, type V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { Groups } from '../../physics/Physics';
import { WeaponsConfig } from './config';

const D = WeaponsConfig.dummy;
let sharedGeometry: BoxGeometry | null = null;

/**
 * Shooting-range target for isolation runs: a static box on the PED collision group so
 * rays treat it like a ped. Turns red as it takes damage and falls over when destroyed.
 */
export class TargetDummy implements Entity, IDamageable {
  readonly id = allocEntityId();
  readonly kind = 'prop' as const;
  readonly position = new Vector3();
  readonly headHeight = D.headHeight;
  isAlive = true;
  health = D.health;
  hits = 0;
  readonly object3d: Mesh;
  private readonly collider: Collider;
  private readonly material: MeshStandardMaterial;

  constructor(
    private readonly game: Game,
    feet: V3,
  ) {
    this.position.set(feet.x, feet.y, feet.z);
    sharedGeometry ??= new BoxGeometry(D.width, D.height, D.depth);
    this.material = new MeshStandardMaterial({ color: 0xd8c9a0, roughness: 0.8 });
    this.object3d = new Mesh(sharedGeometry, this.material);
    this.object3d.position.set(feet.x, feet.y + D.height / 2, feet.z);
    this.object3d.castShadow = true;
    game.worldRoot.add(this.object3d);
    this.collider = game.physics.addStaticCuboid(
      { x: feet.x, y: feet.y + D.height / 2, z: feet.z },
      { x: D.width / 2, y: D.height / 2, z: D.depth / 2 },
      { entity: this, groups: Groups.PED },
    );
  }

  /** Point autotests aim at for a body shot. */
  chest(out: V3): V3 {
    out.x = this.position.x;
    out.y = this.position.y + 1.0;
    out.z = this.position.z;
    return out;
  }

  /** Point autotests aim at for a head shot (above `headHeight`). */
  head(out: V3): V3 {
    out.x = this.position.x;
    out.y = this.position.y + D.height - 0.12;
    out.z = this.position.z;
    return out;
  }

  applyDamage(info: DamageInfo): void {
    if (!this.isAlive) return;
    this.hits++;
    this.health = Math.max(0, this.health - info.amount);
    const k = this.health / D.health;
    this.material.color.setRGB(0.85, 0.2 + 0.6 * k, 0.15 + 0.5 * k);
    if (this.health <= 0) {
      this.isAlive = false;
      // Tip over along the hit direction so a kill is visible without any animation system.
      const dx = info.direction?.x ?? 0;
      const dz = info.direction?.z ?? 1;
      this.object3d.rotation.set(Math.PI / 2 * Math.abs(dz), 0, -Math.PI / 2 * Math.abs(dx) * Math.sign(dx));
      this.object3d.position.y = this.position.y + D.depth / 2;
      this.collider.setEnabled(false);
    }
  }

  dispose(): void {
    this.game.worldRoot.remove(this.object3d);
    this.game.physics.removeCollider(this.collider);
    this.material.dispose();
  }
}

export function isDummy(e: Entity | null | undefined): e is TargetDummy {
  return e instanceof TargetDummy;
}
