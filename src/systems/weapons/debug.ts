import type { WeaponId } from '../../core/peers';
import { WEAPONS, isWeaponId } from '../../data/weapons';
import { WeaponsConfig as C } from './config';
import type { WeaponSystem } from './index';

/** `game.debug.run('weapons.<action>', ...)` handlers; see docs/progress/weapons.md. */
export function createWeaponDebugActions(ws: WeaponSystem): Record<string, (...args: unknown[]) => unknown> {
  return {
    give: (id: unknown, ammo: unknown): boolean => {
      const wid = String(id);
      if (!isWeaponId(wid)) return false;
      ws.give(wid, Number(ammo ?? WEAPONS[wid].magazine * 3));
      return true;
    },
    equip: (id: unknown): WeaponId => {
      const wid = String(id);
      if (isWeaponId(wid)) ws.equip(wid);
      return ws.current;
    },
    fire: (): boolean => ws.fireOnce(),
    reload: (): boolean => ws.startReload(),
    state: (): Record<string, unknown> => ({
      current: ws.current,
      ...ws.ammo.currentSlot,
      aiming: ws.isAiming,
      reloading: ws.isReloading,
      aimTarget: ws.aimTarget ? { id: ws.aimTarget.id, kind: ws.aimTarget.kind, seconds: ws.aimSeconds } : null,
      inventory: ws.ammo.serialize().slots,
    }),
    spawnDummy: (x: unknown, z: unknown): number => ws.spawnDummy(Number(x), Number(z)).id,
    dummies: (): { id: number; health: number; alive: boolean }[] => ws.dummies.map((d) => ({ id: d.id, health: d.health, alive: d.isAlive })),
    clearDummies: (): void => ws.clearDummies(),
    aimAssist: (on: unknown): boolean => {
      C.aimAssist = on === undefined ? !C.aimAssist : on === true || on === 'true' || on === 1;
      return C.aimAssist;
    },
  };
}
