import { Vector3 } from 'three';
import type { IPed, IVehicle, PedBrain, PedContext, PedEvent, V3 } from '../../core/entities';
import { asBrainEvent } from '../../systems/peds/events';
import { RouteDriver } from '../drivers/RouteDriver';

const tmp = new Vector3();

/**
 * Bad Debt target: runs from the player on foot for `fleeSeconds`, then jumps into the
 * getaway sedan and drives the pre-computed route. Cowers (hands up) while the player
 * holds a gun on him, which is the intimidation window.
 */
export class DebtorBrain implements PedBrain {
  private elapsed = 0;
  private started = false;
  private driving = false;
  private sinceAimed = Infinity;
  private readonly runSpeed = 5.5;

  constructor(
    private readonly car: () => IVehicle | null,
    private readonly route: readonly V3[],
    private readonly fleeSeconds = 8,
  ) {}

  get isDriving(): boolean {
    return this.driving;
  }

  update(ped: IPed, ctx: PedContext, dt: number): void {
    // Stay 'scripted' even behind the wheel: PedSpawner culls any other state past its
    // despawn radius, which would turn a long chase into a vanished target.
    ped.state = 'scripted';
    if (ped.vehicle) return;
    const player = ctx.playerPosition;
    // Nothing happens until the player is close enough to be noticed.
    if (!this.started) {
      if (ctx.playerDistance > 25) {
        ped.stop();
        ped.face(player);
        return;
      }
      this.started = true;
    }
    this.sinceAimed += dt;
    if (this.sinceAimed < 0.4) {
      ped.handsUp = true;
      ped.stop();
      ped.face(player);
      return;
    }
    ped.handsUp = false;
    this.elapsed += dt;
    const car = this.car();
    if (this.elapsed >= this.fleeSeconds && car && car.isAlive && !car.state.wrecked) {
      const door = car.seatDoorPosition(0, tmp);
      const d = Math.hypot(door.x - ped.position.x, door.z - ped.position.z);
      if (d < 2.2) {
        if (ped.enterVehicle(car, 0)) {
          this.driving = true;
          car.setDriver(new RouteDriver(this.route));
        }
        return;
      }
      ped.moveToward(door, this.runSpeed);
      return;
    }
    tmp.set(ped.position.x - player.x, 0, ped.position.z - player.z);
    if (tmp.lengthSq() < 1e-4) tmp.set(1, 0, 0);
    tmp.normalize().multiplyScalar(8).add(ped.position);
    ped.moveToward(tmp, this.runSpeed);
  }

  onEvent(ped: IPed, pedEvent: PedEvent): void {
    const event = asBrainEvent(pedEvent);
    if (event.type === 'aimedAt' && !ped.vehicle) this.sinceAimed = 0;
    else if (event.type === 'damaged') this.sinceAimed = Infinity;
  }
}
