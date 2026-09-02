import type { WeaponId } from '../../core/peers';
import { dispatchTier } from '../../data/police';
import type { PoliceSystem } from './index';

export interface UnitsReport {
  cruisers: number;
  swat: number;
  helicopters: number;
  roadblocks: number;
  cops: number;
  footCops: number;
  arrestTimer: number;
}

/** `game.debug.run('police.<action>', ...)` handlers. */
export function policeDebugActions(sys: PoliceSystem): Record<string, (...args: unknown[]) => unknown> {
  const game = sys.game;
  const heading = (): number => (game.systems.has('player') ? game.player.heading : 0);
  return {
    /** Spawns a cruiser (or 'swat' van) 40 m behind the player, already chasing. */
    spawnCruiser: (kind?: unknown): number | null => {
      if (!sys.ready) return null;
      const k = kind === 'swat' ? 'swat' : 'police';
      const p = sys.playerPosition();
      const h = heading();
      const at = { position: { x: p.x - Math.sin(h) * 40, y: p.y, z: p.z - Math.cos(h) * 40 }, heading: h };
      const ground = game.physics.groundHeightAt(at.position.x, at.position.z);
      if (ground !== null) at.position.y = ground + 0.05;
      const unit = sys.spawnUnit(k, k === 'swat' ? 'rifle' : sys.tier.copWeapon, at);
      return unit ? unit.vehicle.id : null;
    },
    /** Foot cop at (x, z) — default 10 m ahead — facing the player, armed with `weapon` (default pistol). */
    spawnCop: (x?: unknown, z?: unknown, weapon?: unknown): number | null => {
      if (!sys.ready) return null;
      const p = sys.playerPosition();
      const h = heading();
      const px = x === undefined ? p.x + Math.sin(h) * 10 : Number(x);
      const pz = z === undefined ? p.z + Math.cos(h) * 10 : Number(z);
      const y = game.physics.groundHeightAt(px, pz) ?? p.y;
      const w = typeof weapon === 'string' && isWeaponId(weapon) ? weapon : 'pistol';
      const ped = sys.dispatcher.spawnFootCop('cop', { x: px, y, z: pz }, Math.atan2(p.x - px, p.z - pz), w);
      return ped ? ped.id : null;
    },
    units: (): UnitsReport => ({
      cruisers: sys.cruisers.filter((u) => u.kind === 'police').length,
      swat: sys.cruisers.filter((u) => u.kind === 'swat').length,
      helicopters: sys.helicopters.length,
      roadblocks: sys.roadblocks.length,
      cops: sys.cops.length,
      footCops: sys.cops.filter((c) => !c.vehicle).length,
      arrestTimer: sys.arrest.timer,
    }),
    /** Places a roadblock 120–200 m ahead; `spikes` forces a spike strip. */
    roadblock: (spikes?: unknown): boolean => {
      if (!sys.ready) return false;
      const tier = { ...dispatchTier(Math.max(3, sys.level)), spikes: spikes === true || spikes === 'true' || sys.tier.spikes };
      return sys.spawnRoadblock(tier) !== null;
    },
    heli: (): number | null => (sys.ready ? sys.spawnHelicopter(sys.playerPosition()).id : null),
    /** Removes every police unit immediately. */
    clear: (): void => sys.despawnAll(),
  };
}

function isWeaponId(v: string): v is WeaponId {
  return v === 'fists' || v === 'pistol' || v === 'smg' || v === 'shotgun' || v === 'rifle' || v === 'sniper';
}
