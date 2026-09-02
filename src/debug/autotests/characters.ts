import type { Game } from '../../core/Game';
import { PED_PALETTES } from '../../gfx/PedBuilder';
import { CharactersConfig } from '../../systems/characters/config';
import { assert, assertClose, Suite, waitFor, waitFrames } from './util';

interface CharState {
  active: 'lucia' | 'jason';
  charge: number;
  abilityActive: boolean;
  switching: boolean;
  wheelOpen: boolean;
}

/**
 * Dual-protagonist acceptance (docs/TEAM-PLAN.md Agent 10, SYSTEM_PROMPT §15).
 * Run: npm run smoke -- --systems=world,player,camera,ui,weapons,characters --name=characters --autotest=characters
 */
export default async function characters(game: Game): Promise<void> {
  const suite = new Suite('characters', game);
  assert(game.systems.has('characters'), 'characters system missing');
  assert(game.systems.has('player'), 'player system missing');
  const peer = game.characters;
  assert(peer, 'game.characters peer missing');
  const state = (): CharState => game.debug.run('characters.state') as CharState;
  const errorsAtStart = game.debug.errors.length;
  const modeBefore = game.camera.mode;

  if (peer.active !== 'lucia') {
    await peer.switchTo('lucia');
    await waitFrames(game, 2);
  }

  await suite.check('switch-to-jason', async () => {
    const got = { switched: null as string | null };
    const off = game.events.on('character:switched', (e) => (got.switched = e.to));
    try {
      const before = { x: game.player.position.x, z: game.player.position.z };
      await peer.switchTo('jason');
      await waitFrames(game, 2);
      assert(got.switched === 'jason', 'character:switched not emitted');
      assert(peer.active === 'jason', `active() ${peer.active}`);
      assert(game.state.activeCharacter === 'jason', `state.activeCharacter ${game.state.activeCharacter}`);
      assert(game.camera.mode === modeBefore, `camera mode ${game.camera.mode}, expected ${modeBefore}`);
      assert(!state().switching, 'still switching');
      assert(!game.input.isBlocked, 'input blocker left behind');
      const rig = game.player.entity.rig;
      assert(rig.palette.shirt === PED_PALETTES.jason.shirt, 'rig palette is not jason');
      const stats = game.player.entity.stats;
      assertClose(stats.recoilMultiplier, CharactersConfig.stats.jason.recoilMultiplier, 1e-6, 'recoil');
      assertClose(stats.sprintMultiplier, CharactersConfig.stats.jason.sprintMultiplier, 1e-6, 'sprint');
      const p = game.player.position;
      const moved = Math.hypot(p.x - before.x, p.z - before.z);
      assert(moved > 1, `player did not move to Jason's position (${moved.toFixed(1)} m)`);
    } finally {
      off();
    }
  });

  await suite.check('switch-back-restores-lucia', async () => {
    await peer.switchTo('lucia');
    await waitFrames(game, 2);
    assert(peer.active === 'lucia', 'not back on lucia');
    const rig = game.player.entity.rig;
    assert(rig.palette.shirt === PED_PALETTES.lucia.shirt, 'rig palette is not lucia');
    assertClose(game.player.entity.stats.sprintMultiplier, CharactersConfig.stats.lucia.sprintMultiplier, 1e-6, 'sprint');
    assert(game.camera.mode === modeBefore, `camera mode ${game.camera.mode}`);
  });

  await suite.check('adrenaline-slow-mo', async () => {
    game.debug.run('characters.charge', 1);
    const got = { activated: false, ended: false };
    const offs = [
      game.events.on('ability:activated', () => (got.activated = true)),
      game.events.on('ability:ended', () => (got.ended = true)),
    ];
    try {
      assert(peer.abilityCharge >= 1, 'charge not full');
      assert(peer.activateAbility(), 'activateAbility returned false');
      await waitFrames(game, 2);
      assert(got.activated, 'ability:activated not emitted');
      assertClose(game.time.scale, CharactersConfig.ability.timeScale, 0.01, 'time.scale during ability');
      assert(state().abilityActive, 'state says ability inactive');
      assert(!peer.activateAbility(), 'ability re-activated while running');
      await waitFor(() => got.ended, (CharactersConfig.ability.seconds + 1.5) * 1000, 'ability:ended');
      await waitFrames(game, 2);
      assertClose(game.time.scale, 1, 1e-6, 'time.scale after ability');
      assert(peer.abilityCharge < 1, `charge ${peer.abilityCharge} did not drop`);
      assertClose(game.state.abilityCharge, peer.abilityCharge, 0.05, 'state.abilityCharge');
    } finally {
      for (const off of offs) off();
    }
  });

  await suite.check('recharge-progresses', async () => {
    const a = peer.abilityCharge;
    await waitFrames(game, 30);
    assert(peer.abilityCharge > a, 'charge does not recharge');
    assert(!peer.activateAbility(), 'ability fired with partial charge');
  });

  await suite.check('no-errors', () => {
    const n = game.debug.errors.length - errorsAtStart;
    assert(n === 0, `${n} console errors: ${game.debug.errors.slice(errorsAtStart).join(' | ')}`);
  });

  suite.finish();
}
