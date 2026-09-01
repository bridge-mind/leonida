import { Vector3 } from 'three';
import type { V3 } from '../../core/entities';
import type { CameraSystem } from './index';

interface MoveTo {
  kind: 'moveTo';
  fromPos: Vector3;
  fromLook: Vector3;
  toPos: Vector3;
  toLook: Vector3;
  duration: number;
  elapsed: number;
  resolve: () => void;
}

interface Orbit {
  kind: 'orbit';
  center: Vector3;
  radius: number;
  height: number;
  speed: number;
  angle: number;
}

/**
 * Cinematic camera moves used by cutscenes and the title screen. Calling any
 * method switches the camera to `cinematic` mode; `stop()` restores the previous mode.
 */
export class CinematicController {
  private active: MoveTo | Orbit | null = null;
  fov = 55;

  constructor(private readonly camera: CameraSystem) {}

  /** Smoothly moves from the current view to `pos` looking at `lookAt` over `seconds`. */
  moveTo(pos: V3, lookAt: V3, seconds: number, fov?: number): Promise<void> {
    this.camera.setMode('cinematic');
    if (fov !== undefined) this.fov = fov;
    return new Promise((resolve) => {
      const prev = this.active;
      if (prev && prev.kind === 'moveTo') prev.resolve();
      this.active = {
        kind: 'moveTo',
        fromPos: this.camera.position.clone(),
        fromLook: this.camera.lookAt.clone(),
        toPos: new Vector3(pos.x, pos.y, pos.z),
        toLook: new Vector3(lookAt.x, lookAt.y, lookAt.z),
        duration: Math.max(0, seconds),
        elapsed: 0,
        resolve,
      };
    });
  }

  /** Orbits `center` at `radius`/`height`, `speed` in radians per second, looking at the center. */
  orbit(center: V3, radius: number, height: number, speed: number, fov?: number): void {
    this.camera.setMode('cinematic');
    if (fov !== undefined) this.fov = fov;
    const startAngle = this.active && this.active.kind === 'orbit' ? this.active.angle : this.camera.yaw;
    this.active = { kind: 'orbit', center: new Vector3(center.x, center.y, center.z), radius, height, speed, angle: startAngle };
  }

  /** Ends the cinematic and returns to the previous camera mode. */
  stop(): void {
    if (this.active && this.active.kind === 'moveTo') this.active.resolve();
    this.active = null;
    this.camera.restoreMode();
    this.camera.snap();
  }

  get isActive(): boolean {
    return this.active !== null;
  }

  update(dt: number, outPos: Vector3, outLook: Vector3): void {
    const a = this.active;
    if (!a) {
      outPos.copy(this.camera.position);
      outLook.copy(this.camera.lookAt);
      return;
    }
    if (a.kind === 'moveTo') {
      a.elapsed += dt;
      const t = a.duration <= 0 ? 1 : Math.min(1, a.elapsed / a.duration);
      const s = t * t * (3 - 2 * t);
      outPos.lerpVectors(a.fromPos, a.toPos, s);
      outLook.lerpVectors(a.fromLook, a.toLook, s);
      if (t >= 1) {
        a.resolve();
        a.resolve = () => {};
      }
    } else {
      a.angle += a.speed * dt;
      outPos.set(a.center.x + Math.sin(a.angle) * a.radius, a.center.y + a.height, a.center.z + Math.cos(a.angle) * a.radius);
      outLook.copy(a.center);
    }
  }
}
