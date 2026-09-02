import type { IPed, V3 } from '../core/entities';
import type { CutsceneDef, CutsceneSystem } from '../systems/cutscene';
import { MissionsConfig } from '../systems/missions/config';
import { action, custom, horizontal, playerPos, TargetTracker } from './objectives';
import type { MissionContext, Objective } from './types';

/**
 * Objectives that lean on optional peers (wanted, robbery, casino, cutscene). Each one
 * degrades to an immediate 'done' with a single console.warn when its peer is missing,
 * so a mission never stalls in an isolation run.
 */

const warned = new Set<string>();
function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(`[missions] ${message}`);
}

/**
 * Done once the wanted level is back to 0. Heat may arrive a beat after the previous
 * objective (the `rob_store` crime is only reported when the player leaves the store), so
 * the objective arms on the first star and otherwise waits a short grace period outdoors
 * before accepting "no cops at all".
 */
export function escapeWanted(text = 'Lose the ~y~cops~y~'): Objective {
  let armed = false;
  let calm = 0;
  return custom({
    text,
    start(ctx) {
      armed = (ctx.game.wanted?.level ?? 0) > 0;
      calm = 0;
    },
    update(ctx, dt) {
      const wanted = ctx.game.wanted;
      if (!wanted) {
        warnOnce('wanted', 'wanted system missing: escapeWanted completes immediately');
        return 'done';
      }
      if (wanted.level > 0) {
        armed = true;
        calm = 0;
        return 'running';
      }
      if (armed) return 'done';
      if (ctx.game.state.interior !== null) return 'running';
      calm += dt;
      return calm >= MissionsConfig.escape.graceSeconds ? 'done' : 'running';
    },
    skip(ctx) {
      ctx.game.wanted?.clear('mission');
      armed = true;
    },
  });
}

/** Waits for the player to reach `level` stars; completes at once without a wanted system. */
export function reachWanted(level: number, text: string): Objective {
  const obj = custom({
    text,
    update(ctx) {
      const wanted = ctx.game.wanted;
      if (!wanted) {
        warnOnce('wanted', 'wanted system missing: reachWanted completes immediately');
        return 'done';
      }
      obj.text = `${text} ~y~${Math.min(level, wanted.level)}/${level}~y~`;
      return wanted.level >= level ? 'done' : 'running';
    },
    skip(ctx) {
      ctx.game.wanted?.setLevel(level);
    },
  });
  return obj;
}

/**
 * Rob a store (`storeId` or any). The register cash is added to the mission reward via
 * `ctx.reward`. Listens to `robbery:completed`; the `rob_store` crime is only accepted as a
 * fallback when the robbery system is not loaded (with it loaded the crime also fires for
 * an aborted hold-up on the way out, which must not pass the objective).
 */
export function robStore(storeId: string | null, target: V3 | null, text: string): Objective {
  let done = false;
  let off: (() => void)[] = [];
  return custom({
    text,
    blipKind: 'objective',
    target: () => target,
    start(ctx) {
      done = false;
      const hasRobbery = ctx.game.systems.has('robbery');
      off = [
        ctx.game.events.on('robbery:completed', (e) => {
          if (done || (storeId && e.storeId !== storeId)) return;
          done = true;
          ctx.reward += Math.max(0, Math.floor(e.amount));
        }),
        ctx.game.events.on('crime:committed', (e) => {
          if (!hasRobbery && e.type === 'rob_store' && e.actor === 'player') done = true;
        }),
      ];
    },
    update: () => (done ? 'done' : 'running'),
    skip() {
      done = true;
    },
    stop() {
      for (const fn of off) fn();
      off = [];
    },
  });
}

/** Net casino winnings (won − lost) must reach `amount`; going broke fails the mission. */
export function winAtCasino(amount: number, target: V3 | null, text: string): Objective {
  let net = 0;
  let off: (() => void)[] = [];
  const obj = custom({
    text,
    blipKind: 'casino',
    target: () => target,
    start(ctx) {
      net = 0;
      off = [
        ctx.game.events.on('casino:won', (e) => {
          net += e.amount;
        }),
        ctx.game.events.on('casino:lost', (e) => {
          net -= e.amount;
        }),
      ];
    },
    update(ctx) {
      obj.text = `${text} ~y~$${Math.max(0, net).toLocaleString()} / $${amount.toLocaleString()}~y~`;
      if (net >= amount) return 'done';
      const casino = ctx.game.casino;
      const cash = ctx.game.economy?.cash ?? Infinity;
      if (casino && casino.chips <= 0 && cash < MissionsConfig.casino.brokeCash && net < 0) {
        ctx.game.ui?.notify(MissionsConfig.casino.brokeTaunt, 'warning');
        ctx.fail('you went broke');
        return 'failed';
      }
      return 'running';
    },
    skip() {
      net = amount;
    },
    stop() {
      for (const fn of off) fn();
      off = [];
    },
  });
  return obj;
}

export type IntimidateResult = 'intimidated' | 'killed' | null;

/**
 * Bad Debt beat: hold the crosshair on `target` for `seconds` within `range` (intimidate)
 * or kill him. The outcome is exposed through `result()` for the reward.
 */
export function intimidateOrKill(target: () => IPed | null, seconds: number, range: number, text: string): Objective & { result(): IntimidateResult } {
  let aimed = 0;
  let result: IntimidateResult = null;
  const tracker = new TargetTracker(() => {
    const t = target();
    return t ? [t] : [];
  });
  const obj = {
    text,
    blipKind: 'objective' as const,
    target: () => target()?.position ?? null,
    result: () => result,
    start(ctx: MissionContext) {
      aimed = 0;
      result = null;
      tracker.start(ctx);
    },
    update(ctx: MissionContext, dt: number) {
      if (result) return 'done';
      const ped = target();
      if (!ped) {
        ctx.fail('target lost');
        return 'failed';
      }
      if (!tracker.isLive(ped)) {
        // Killed counts as done; a despawned/recycled target was lost, not dealt with.
        if (tracker.wasKilled(ped)) {
          result = 'killed';
          return 'done';
        }
        ctx.fail('target lost');
        return 'failed';
      }
      const p = playerPos(ctx);
      if (!p) return 'running';
      const dist = horizontal(p, ped.position);
      if (dist > MissionsConfig.targetLostDistance) {
        ctx.fail('target lost');
        return 'failed';
      }
      const weapons = ctx.game.weapons;
      const aiming = !!weapons && weapons.isAiming && weapons.aimTarget === ped && dist <= range;
      aimed = aiming ? aimed + dt : 0;
      obj.text = aiming ? `${text} ~y~${Math.max(0, seconds - aimed).toFixed(1)}s~y~` : text;
      if (aimed >= seconds) {
        result = 'intimidated';
        return 'done';
      }
      return 'running';
    },
    skip() {
      aimed = seconds;
      result = 'intimidated';
    },
    stop() {
      tracker.stop();
    },
  };
  return obj;
}

/**
 * Timed loot grab: the progress bar fills while the player stands within `radius` of
 * `counter()`; leaving pauses it. Done when full.
 */
export function lootGrab(counter: () => V3 | null, seconds: number, radius: number, text: string): Objective {
  let progress = 0;
  return custom({
    text,
    target: counter,
    start() {
      progress = 0;
    },
    update(ctx, dt) {
      const c = counter();
      const p = playerPos(ctx);
      const near = !!c && !!p && horizontal(p, c) <= radius;
      if (near) progress += dt / seconds;
      ctx.game.ui?.progressBar.show('Filling bag', Math.min(1, progress));
      return progress >= 1 ? 'done' : 'running';
    },
    skip() {
      progress = 1;
    },
    stop(ctx) {
      ctx.game.ui?.progressBar.hide();
    },
  });
}

/** Plays a cutscene (built lazily when given a function); fast-forward skips it through the cutscene system. */
export function cutscene(scene: CutsceneDef | (() => CutsceneDef)): Objective {
  let system: CutsceneSystem | undefined;
  return action(
    (ctx) => {
      const def = typeof scene === 'function' ? scene() : scene;
      system = ctx.game.systems.get<CutsceneSystem>('cutscene');
      if (!system) {
        warnOnce('cutscene', 'cutscene system missing: dialogue-only fallback');
        for (const step of def.steps) if ('run' in step) step.run(ctx.game);
        return ctx.game.ui?.dialogue.show(def.steps.flatMap((s) => ('dialogue' in s ? s.dialogue : [])));
      }
      return system.play(def, { fast: ctx.fast });
    },
    '',
    () => system?.skip(),
  );
}
