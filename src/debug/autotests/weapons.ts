import type { Entity, V3 } from '../../core/entities';
import type { Game } from '../../core/Game';
import { WEAPONS } from '../../data/weapons';
import type { WeaponSystem } from '../../systems/weapons';
import type { TargetDummy } from '../../systems/weapons/Dummy';
import { assert, assertClose, Suite, wait, waitFor, waitFrames } from './util';

/**
 * Weapons / damage / particles acceptance (TEAM-PLAN Agent 4).
 * Run: npm run smoke -- --systems=world,player,camera,weapons,damage,particles --name=weapons --autotest=weapons
 * The game is driven only through debug actions and scripted input.
 */
export default async function weapons(game: Game): Promise<void> {
  const suite = new Suite('weapons', game);
  const ws = game.systems.get<WeaponSystem>('weapons');
  assert(ws, 'weapons system missing');
  const player = game.player;
  const input = game.input;
  const events = game.events;

  // Vice Beach sand strip: open ground, nothing between the camera and the targets.
  const base = { x: 918, z: 120 };
  player.teleport({ x: base.x, y: 0, z: base.z }, Math.PI);
  game.camera.setOrbit(0, 0.05); // yaw 0 looks toward -Z
  await waitFrames(game, 5);

  let playerShots = 0;
  let npcShots = 0;
  let crimes = 0;
  let killed = 0;
  const damage: { target: Entity; amount: number; part: string | undefined; point: V3 | null }[] = [];
  const offs = [
    events.on('weapon:fired', (e) => (e.shooter === 'player' ? playerShots++ : npcShots++)),
    events.on('crime:committed', (e) => {
      if (e.type === 'shot_fired') crimes++;
    }),
    events.on('entity:killed', () => killed++),
    events.on('damage:applied', ({ target, info }) => {
      damage.push({ target, amount: info.amount, part: info.part, point: info.point ? { ...info.point } : null });
    }),
  ];
  const point = { x: 0, y: 0, z: 0 };
  const hitsOn = (d: TargetDummy): typeof damage => damage.filter((h) => h.target === d);

  /** One scripted press of the fire button; resolves once the game had a chance to process it. */
  const pressFire = async (): Promise<void> => {
    input.setFire(true);
    await waitFrames(game, 2);
    input.setFire(false);
    await waitFrames(game, 2);
  };

  await suite.check('give-and-equip-pistol', () => {
    assert(game.debug.run('weapons.give', 'pistol', 48) === true, 'give refused');
    game.debug.run('weapons.equip', 'pistol');
    assert(ws.current === 'pistol', `current is ${ws.current}`);
    assert(ws.has('pistol') && !ws.has('smg'), 'inventory wrong');
    const w = game.state.weapon;
    assert(w && w.id === 'pistol' && w.ammoInMag === 12 && w.ammoReserve === 36, `state.weapon ${JSON.stringify(w)}`);
  });

  const dummyA = ws.spawnDummy(base.x, base.z - 20);
  await waitFrames(game, 2);

  await suite.check('pistol-body-shot-25', async () => {
    input.setMouseWorld(dummyA.chest(point));
    input.setAim(true);
    await waitFrames(game, 3);
    assert(ws.isAiming, 'not aiming');
    await pressFire();
    await waitFor(() => dummyA.health < 100, 1500, 'dummy to take damage');
    await waitFrames(game, 2);
    assert(dummyA.health === 75, `dummy health ${dummyA.health}, expected 75`);
    const hits = hitsOn(dummyA);
    assert(hits.length === 1 && hits[0].part === 'body' && hits[0].amount === 25, `hits ${JSON.stringify(hits.map((h) => [h.part, h.amount]))}`);
    assert(game.state.weapon?.ammoInMag === 11, `mag ${game.state.weapon?.ammoInMag}`);
  });

  await suite.check('pistol-headshot-x3-kills', async () => {
    await wait(250); // semi-auto cooldown (300 rpm)
    input.setMouseWorld(dummyA.head(point));
    await pressFire();
    await waitFor(() => !dummyA.isAlive, 1500, 'headshot kill');
    const hits = hitsOn(dummyA);
    assert(hits.length === 2 && hits[1].part === 'head' && hits[1].amount === 75, `second hit ${JSON.stringify(hits[1])}`);
    assert(dummyA.health === 0, `health ${dummyA.health}`);
    assert(killed === 1, `entity:killed count ${killed}`);
  });

  await suite.check('reload-timing', async () => {
    let reloaded = false;
    const off = events.on('weapon:reloaded', () => (reloaded = true));
    const t0 = game.time.elapsed;
    input.press('KeyR', 100);
    await waitFor(() => game.state.weapon?.isReloading === true, 1000, 'reload start');
    await waitFor(() => reloaded, 4000, 'weapon:reloaded');
    off();
    const took = game.time.elapsed - t0;
    const expected = WEAPONS.pistol.reloadSeconds;
    // ±10 % plus two fixed steps for the input edge and the timer granularity.
    assertClose(took, expected, expected * 0.1 + 2 / 60, 'reload seconds');
    assert(game.state.weapon?.ammoInMag === 12 && game.state.weapon.ammoReserve === 34, `ammo after reload ${JSON.stringify(game.state.weapon)}`);
  });

  await suite.check('smg-automatic-rate', async () => {
    game.debug.run('weapons.give', 'smg', 90);
    game.debug.run('weapons.equip', 'smg');
    await wait(350); // weapon switch delay
    const before = playerShots;
    const t0 = game.time.elapsed;
    input.setFire(true);
    await waitFor(() => game.time.elapsed - t0 >= 2, 6000, '2 s of fire');
    input.setFire(false);
    await waitFrames(game, 2);
    const shots = playerShots - before;
    assert(shots >= 24 && shots <= 30, `SMG fired ${shots} rounds in 2 s (expected 24–30)`);
  });

  const dummyB = ws.spawnDummy(base.x, base.z - 3);
  await waitFrames(game, 2);

  await suite.check('shotgun-pellets', async () => {
    game.debug.run('weapons.give', 'shotgun', 16);
    game.debug.run('weapons.equip', 'shotgun');
    await wait(350);
    input.setMouseWorld(dummyB.chest(point));
    await waitFrames(game, 2);
    const before = playerShots;
    await pressFire();
    await waitFor(() => playerShots > before, 1500, 'shotgun shot');
    await waitFrames(game, 2);
    assert(playerShots - before === 1, `weapon:fired ${playerShots - before} times for one trigger pull`);
    const hits = hitsOn(dummyB);
    assert(hits.length >= 1 && hits.length <= 8, `${hits.length} pellet hits`);
    const distinct = new Set(hits.map((h) => (h.point ? `${h.point.x.toFixed(3)},${h.point.y.toFixed(3)},${h.point.z.toFixed(3)}` : 'none')));
    assert(distinct.size >= 2, `pellets did not spread (${distinct.size} distinct impact points from ${hits.length} hits)`);
    assert(hits.every((h) => h.amount === 12 || h.amount === 36), 'pellet damage wrong');
  });

  await suite.check('explode-radius-falloff', () => {
    // Along the sand strip, 60 m from the player: C sits 3 m from the blast, D 12 m.
    const c = ws.spawnDummy(base.x, base.z - 63);
    const d = ws.spawnDummy(base.x, base.z - 72);
    let explosions = 0;
    const off = events.on('crime:committed', (e) => {
      if (e.type === 'explosion') explosions++;
    });
    game.debug.run('damage.explode', base.x, base.z - 60, 6, 100);
    off();
    assert(c.health < 100 && c.health > 0, `dummy at 3 m has health ${c.health}`);
    assert(d.health === 100, `dummy at 12 m has health ${d.health}`);
    assert(explosions === 1, 'crime:committed explosion not emitted');
  });

  await suite.check('weapon-wheel-slowmo-and-select', async () => {
    input.setAim(false);
    input.setMouseWorld(null);
    const el = document.querySelector<HTMLElement>('#ui .weapon-wheel');
    assert(el, 'wheel element missing');
    assert(el.hidden, 'wheel visible before Tab');
    input.hold('Tab');
    await waitFrames(game, 3);
    assert(!el.hidden, 'wheel not shown while Tab is held');
    assert(game.time.scale === 0.2, `time.scale ${game.time.scale}`);
    // Pistol sits at slot 2 = 30° below the +X axis on the ring.
    input.look(60, -35);
    await waitFrames(game, 2);
    assert(el.querySelector('.weapon-wheel-slot.selected .ww-name')?.textContent === 'Pistol', 'pistol slot not selected');
    input.release('Tab');
    await waitFrames(game, 3);
    assert(el.hidden, 'wheel still visible after release');
    const restored: number = game.time.scale;
    assert(restored === 1, `time.scale after release ${restored}`);
    assert(ws.current === 'pistol', `wheel equipped ${ws.current}`);
  });

  await suite.check('quick-select-digits', async () => {
    input.press('Digit4', 100);
    await waitFor(() => ws.current === 'shotgun', 1000, 'digit 4 to equip shotgun');
    input.press('Digit6', 100);
    await waitFrames(game, 4);
    assert(ws.current === 'shotgun', 'unowned sniper equipped');
  });

  await suite.check('aimed-at-event', async () => {
    // Offset sideways so the ray clears the shotgun dummy standing 3 m ahead.
    const f = ws.spawnDummy(base.x + 6, base.z - 25);
    await waitFrames(game, 2);
    let last: { target: Entity; seconds: number } | null = null;
    const off = events.on('weapon:aimedAt', (e) => (last = e));
    input.setMouseWorld(f.chest(point));
    input.setAim(true);
    await waitFrames(game, 20);
    off();
    input.setAim(false);
    const seen = last as { target: Entity; seconds: number } | null;
    assert(seen && seen.target === f, 'weapon:aimedAt did not report the dummy');
    assert(seen.seconds > 0.1, `aimed seconds only ${seen.seconds.toFixed(3)}`);
    assert(ws.aimTarget === null, 'aimTarget not cleared after aiming stopped');
  });

  await suite.check('melee-combo', async () => {
    input.setMouseWorld(null);
    game.debug.run('weapons.equip', 'fists');
    game.debug.run('player.face', Math.PI); // rig heading π faces -Z
    const g = ws.spawnDummy(base.x, base.z - 1.3);
    await waitFrames(game, 2);
    let combos = 0;
    const off = events.on('weapon:meleeSwung', (e) => {
      if (e.hit === g) combos = Math.max(combos, e.combo);
    });
    for (let i = 0; i < 3; i++) {
      await pressFire();
      await wait(380);
    }
    off();
    assert(g.health === 55, `dummy health after 3 punches ${g.health} (expected 55)`);
    assert(combos === 3, `combo reached ${combos}`);
  });

  await suite.check('npc-fire-hits-player', async () => {
    const healthBefore = player.health + player.armor;
    const shooter = ws.dummies[0];
    const p = player.position;
    // Just in front of the player's capsule (the melee dummy stands farther out at -1.3 m).
    const origin = { x: p.x, y: p.y + 1.3, z: p.z - 0.9 };
    const dir = { x: 0, y: -0.05, z: 1 };
    const before = npcShots;
    ws.npcFire(shooter, 'pistol', origin, dir, 1);
    await waitFrames(game, 2);
    assert(npcShots === before + 1, 'weapon:fired not emitted for the NPC');
    assert(player.health + player.armor < healthBefore, 'player took no damage from npcFire');
  });

  await suite.check('particles-alive-and-bounded', () => {
    if (!game.systems.has('particles')) return; // not part of this isolation set
    const counts = game.debug.run('particles.count') as { particles: number; tracers: number };
    game.debug.run('particles.emit', 'explosion', base.x, base.z - 10);
    const after = game.debug.run('particles.count') as { particles: number; tracers: number };
    assert(after.particles > counts.particles, 'explosion emitted no particles');
    assert(after.particles <= 4096, 'particle pool exceeded capacity');
  });

  await suite.check('crimes-match-shots', () => {
    assert(crimes === playerShots, `shot_fired crimes ${crimes} vs player shots ${playerShots}`);
    assert(game.state.stats.shotsFired === playerShots, `stats.shotsFired ${game.state.stats.shotsFired}`);
  });

  await suite.check('no-errors-after-run', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  for (const off of offs) off();
  input.setAim(false);
  input.setFire(false);
  input.setMouseWorld(null);
  suite.finish();
}
