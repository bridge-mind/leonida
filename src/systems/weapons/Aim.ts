import { Vector3 } from 'three';
import { Config } from '../../core/Config';
import type { Entity, IPed, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { Groups, type RaycastHit } from '../../physics/Physics';
import { WeaponsConfig as C } from './config';
import { isDummy, type TargetDummy } from './Dummy';

/**
 * What the crosshair is over (`weapon:aimedAt` with accumulated seconds) and the one-shot
 * soft aim-assist snap when aiming starts.
 */
export class AimTracker {
  target: Entity | null = null;
  seconds = 0;

  private readonly hit: RaycastHit = {
    entity: null,
    point: new Vector3(),
    normal: new Vector3(),
    toi: 0,
    collider: null as unknown as RaycastHit['collider'],
  };
  private readonly peds: IPed[] = [];
  private readonly toTarget = new Vector3();

  constructor(private readonly game: Game) {}

  /** Raycast (no spread) along the aim ray; accumulates time while the same target stays under the crosshair. */
  update(dt: number, origin: V3, dir: V3, exclude: Entity | null): void {
    const h = this.game.physics.raycastInto(this.hit, origin, dir, C.aimTargetRange, {
      groups: Groups.STATIC | Groups.VEHICLE | Groups.PED,
      exclude: exclude ?? undefined,
    });
    const e = h && h.entity && isAimTarget(h.entity) ? h.entity : null;
    if (e !== this.target) {
      this.target = e;
      this.seconds = 0;
    }
    if (!e) return;
    this.seconds += dt;
    this.game.events.emit('weapon:aimedAt', { target: e, seconds: this.seconds });
  }

  clear(): void {
    this.target = null;
    this.seconds = 0;
  }

  /**
   * Snaps the camera orbit to the best ped/dummy inside the assist cone. Returns true when
   * a snap happened.
   */
  assist(playerPos: V3, forward: V3, dummies: readonly TargetDummy[]): boolean {
    const peds = this.peds;
    peds.length = 0;
    this.game.peds?.query(playerPos, C.aimAssistRange, peds, (p) => p.isAlive && p.state !== 'dead');
    let best: Entity | null = null;
    let bestAngle = C.aimAssistConeRad;
    const consider = (e: Entity): void => {
      this.toTarget.set(e.position.x - playerPos.x, e.position.y + C.aimAssistChestHeight - (playerPos.y + Config.camera.pivotHeight), e.position.z - playerPos.z);
      const dist = this.toTarget.length();
      if (dist < 0.5 || dist > C.aimAssistRange) return;
      this.toTarget.divideScalar(dist);
      const dot = this.toTarget.x * forward.x + this.toTarget.y * forward.y + this.toTarget.z * forward.z;
      const angle = Math.acos(Math.min(1, Math.max(-1, dot)));
      if (angle < bestAngle) {
        bestAngle = angle;
        best = e;
      }
    };
    for (let i = 0; i < peds.length; i++) consider(peds[i]);
    for (let i = 0; i < dummies.length; i++) if (dummies[i].isAlive) consider(dummies[i]);
    if (!best) return false;
    const b: Entity = best;
    const dx = b.position.x - playerPos.x;
    const dz = b.position.z - playerPos.z;
    const dy = b.position.y + C.aimAssistChestHeight - (playerPos.y + Config.camera.pivotHeight);
    // Camera yaw 0 looks toward -Z; positive pitch looks down.
    const yaw = Math.atan2(-dx, -dz);
    const pitch = -Math.atan2(dy, Math.hypot(dx, dz));
    this.game.camera.setOrbit(yaw, pitch);
    return true;
  }
}

/** Entities the crosshair reports: peds, vehicles and target dummies. */
export function isAimTarget(e: Entity): boolean {
  return e.kind === 'ped' || e.kind === 'vehicle' || isDummy(e);
}
