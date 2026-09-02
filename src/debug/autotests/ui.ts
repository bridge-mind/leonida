import type { Game } from '../../core/Game';
import type { WantedState } from '../../core/GameState';
import { assert, horizontalDistance, Suite, wait, waitFor, waitFrames } from './util';

/**
 * UI + navigation acceptance (docs/TEAM-PLAN.md Agent 5).
 * Run: npm run smoke -- --systems=world,player,camera,economy,navigation,ui --name=ui --autotest=ui
 */
export default async function ui(game: Game): Promise<void> {
  const suite = new Suite('ui', game);
  const player = game.player;
  const uiState = (): Record<string, unknown> => game.debug.run('ui.state') as Record<string, unknown>;
  // Read through functions so TypeScript does not narrow `scale`/`mode` after the assignments below.
  const timeScale = (): number => game.time.scale;
  const cameraMode = (): string => game.camera.mode;

  await suite.check('systems-present', () => {
    assert(game.systems.has('ui'), 'ui system missing');
    assert(game.systems.has('navigation'), 'navigation system missing');
    assert(document.querySelector('#ui .hud'), 'HUD not mounted');
    assert(document.querySelector('#ui .minimap canvas'), 'minimap canvas not mounted');
    assert(uiState().title === false, 'title screen should be hidden during autotests');
  });

  // Downtown: dense road grid so a 600 m waypoint has a real road route.
  player.teleport({ x: 0, y: 0, z: 0 }, Math.PI);
  await waitFrames(game, 3);

  await suite.check('waypoint-route', async () => {
    const start = player.position;
    const target = { x: start.x + 600, y: 0, z: start.z + 30 };
    game.debug.run('navigation.waypoint', target.x, target.z);
    await waitFrames(game, 2);
    const route = game.state.route;
    assert(route.length > 5, `route has ${route.length} points`);
    const last = route[route.length - 1];
    const d = horizontalDistance(last, target);
    assert(d <= 15, `route ends ${d.toFixed(1)} m from the target`);
    assert(game.state.waypoint !== null, 'state.waypoint not set');
    assert(game.state.blips.some((b) => b.kind === 'waypoint'), 'waypoint blip missing');
    const info = game.debug.run('navigation.route') as { length: number; distance: number };
    assert(info.length >= 600, `route length ${info.length} m is shorter than the crow flies`);
    assert(info.distance > 0, 'distance along route is zero');
  });

  await suite.check('landmark-blips', () => {
    const kinds = new Set(game.state.blips.map((b) => b.kind));
    for (const k of ['safehouse', 'hospital', 'police', 'casino', 'gunstore', 'store'] as const) assert(kinds.has(k), `no ${k} blip`);
  });

  await suite.check('minimap-renders', async () => {
    await waitFrames(game, 3);
    const canvas = document.querySelector<HTMLCanvasElement>('#ui .minimap canvas');
    assert(canvas, 'minimap canvas missing');
    const ctx = canvas.getContext('2d');
    assert(ctx, 'no 2d context');
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let opaque = 0;
    for (let i = 3; i < data.length; i += 4 * 37) if (data[i] > 0) opaque++;
    assert(opaque > 50, `minimap looks empty (${opaque} sampled opaque pixels)`);
    const distance = document.querySelector('#ui .minimap-distance')?.textContent ?? '';
    assert(/\d+ m|\d+\.\d km/.test(distance), `distance text '${distance}'`);
  });

  await suite.check('waypoint-clears-on-arrival', async () => {
    const wp = game.state.waypoint;
    assert(wp, 'no waypoint');
    player.teleport({ x: wp.x + 3, y: 0, z: wp.z }, 0);
    await waitFor(() => game.state.waypoint === null, 3000, 'waypoint auto-clear');
    assert(game.state.route.length === 0, 'route not cleared');
    player.teleport({ x: 0, y: 0, z: 0 }, Math.PI);
  });

  await suite.check('wanted-star-classes', async () => {
    const stars = Array.from(document.querySelectorAll<HTMLElement>('#ui .hud-star'));
    assert(stars.length === 6, `expected 6 star slots, got ${stars.length}`);
    const wanted = game.state.wanted;
    const prevLevel = wanted.level;
    const prevState = wanted.state;
    wanted.level = 3;
    for (const state of ['responding', 'searching', 'active'] as WantedState[]) {
      wanted.state = state;
      await waitFrames(game, 2);
      for (let i = 0; i < 3; i++) assert(stars[i].classList.contains(state), `star ${i} lacks class '${state}' (has '${stars[i].className}')`);
      for (let i = 3; i < 6; i++) assert(stars[i].className === 'hud-star', `star ${i} should be dim (has '${stars[i].className}')`);
    }
    game.events.emit('wanted:cleared', { reason: 'test' });
    wanted.level = 0;
    wanted.state = 'none';
    await waitFrames(game, 2);
    assert(stars[0].classList.contains('flash'), 'stars do not flash on wanted:cleared');
    await waitFor(() => !stars[0].classList.contains('flash'), 4000, 'flash to end');
    assert(stars[0].className === 'hud-star', `star 0 not reset after flash ('${stars[0].className}')`);
    wanted.level = prevLevel;
    wanted.state = prevState;
  });

  await suite.check('notification-shows-and-expires', async () => {
    game.debug.run('ui.notify', 'Autotest notification', 'warning');
    await waitFrames(game, 2);
    const node = document.querySelector('#ui .notification.warning');
    assert(node, 'notification element not shown');
    assert(node.textContent?.includes('Autotest notification'), 'notification text wrong');
    await wait(5000);
    assert(!document.querySelector('#ui .notification.warning'), 'notification still visible after 5 s');
  });

  await suite.check('money-notification-for-cash-change', async () => {
    const economy = game.economy;
    if (!economy) return;
    economy.add(500, 'pickup');
    await waitFrames(game, 2);
    const node = document.querySelector('#ui .notification.money');
    assert(node, 'no money notification for a $500 pickup');
    assert(node.textContent?.includes('Cash picked up'), `money notification text '${node.textContent}'`);
    economy.add(-500, 'test');
  });

  await suite.check('card-slow-mo', async () => {
    game.time.scale = 1;
    const promise = game.debug.run('ui.card', 'custom', 'TEST', 1) as Promise<void>;
    await waitFrames(game, 2);
    assert(timeScale() === 0.25, `time scale during card is ${timeScale()}`);
    const text = document.querySelector<HTMLElement>('#ui .card-text');
    assert(text && text.textContent === 'TEST' && text.classList.contains('show'), 'card text not shown');
    assert(document.getElementById('game')?.classList.contains('ui-desaturate'), 'world not desaturated');
    await promise;
    assert(timeScale() === 1, `time scale after card is ${timeScale()}`);
    assert(!document.getElementById('game')?.classList.contains('ui-desaturate'), 'desaturation not removed');
  });

  await suite.check('objective-markup', async () => {
    game.debug.run('ui.objective', 'Go to the ~y~safehouse~y~ and ~r~lose the cops~r~');
    await waitFrames(game, 2);
    const objective = document.querySelector<HTMLElement>('#ui .objective');
    assert(objective && !objective.hidden, 'objective not visible');
    const yellow = objective.querySelector('.markup-yellow');
    assert(yellow && yellow.textContent === 'safehouse', 'yellow span missing');
    assert(objective.querySelector('.markup-red')?.textContent === 'lose the cops', 'red span missing');
    assert(objective.textContent === 'Go to the safehouse and lose the cops', `markup leaked: '${objective.textContent}'`);
    game.debug.run('ui.objective', '');
    await waitFrames(game, 2);
    assert(objective.hidden, 'objective still visible after clearing');
  });

  await suite.check('prompts-render', async () => {
    const uiSys = game.ui;
    assert(uiSys, 'ui peer missing');
    uiSys.prompt('KeyF', 'Enter vehicle');
    await waitFrames(game, 2);
    const prompt = document.querySelector<HTMLElement>('#ui .prompt');
    assert(prompt, 'prompt not rendered');
    assert(prompt.querySelector('.prompt-key')?.textContent === 'F', 'prompt key label wrong');
    assert(prompt.textContent?.includes('Enter vehicle'), 'prompt text wrong');
    uiSys.clearPrompt('KeyF');
    await waitFrames(game, 2);
    assert(!document.querySelector('#ui .prompt'), 'prompt not removed');
    assert(game.state.prompts.length === 0, 'state.prompts not cleaned up');
  });

  await suite.check('progress-bar', async () => {
    game.debug.run('ui.progress', 0.5);
    await waitFrames(game, 2);
    const bar = document.querySelector<HTMLElement>('#ui .progress-bar');
    assert(bar && !bar.hidden, 'progress bar hidden');
    assert(bar.querySelector<HTMLElement>('.progress-bar-fill')?.style.transform === 'scaleX(0.5)', 'fill not at 50%');
    game.debug.run('ui.progress', -1);
    await waitFrames(game, 2);
    assert(bar.hidden, 'progress bar not hidden');
  });

  await suite.check('dialogue-advances-with-space', async () => {
    const uiSys = game.ui;
    assert(uiSys, 'ui peer missing');
    const done = uiSys.dialogue.show([
      { speaker: 'Lucia', text: 'Line one', seconds: 30 },
      { speaker: 'Jason', text: 'Line two', seconds: 30 },
    ]);
    await waitFrames(game, 2);
    assert(uiSys.dialogue.active, 'dialogue not active');
    const box = document.querySelector<HTMLElement>('#ui .dialogue');
    assert(box && box.textContent?.includes('Line one'), 'first line not shown');
    game.input.press('Space', 60);
    await waitFrames(game, 6);
    assert(box.textContent?.includes('Line two'), 'Space did not advance the dialogue');
    game.input.press('Space', 60);
    await done;
    assert(!uiSys.dialogue.active, 'dialogue still active after last line');
  });

  await suite.check('pause-resume', async () => {
    const uiSys = game.ui;
    assert(uiSys, 'ui peer missing');
    game.time.scale = 1;
    game.debug.run('ui.pause');
    await waitFrames(game, 2);
    assert(uiSys.isPaused && uiSys.isModal, 'not paused / modal');
    assert(timeScale() === 0, `time scale while paused is ${timeScale()}`);
    assert(game.input.isBlocked, 'input not blocked while paused');
    assert(!document.querySelector<HTMLElement>('#ui .pause')?.hidden, 'pause menu hidden');
    game.debug.run('ui.resume');
    await waitFrames(game, 2);
    assert(!uiSys.isPaused && !uiSys.isModal, 'still paused after resume');
    assert(timeScale() === 1, `time scale after resume is ${timeScale()}`);
    assert(!game.input.isBlocked, 'input still blocked after resume');
  });

  await suite.check('full-map-toggle', async () => {
    const uiSys = game.ui;
    assert(uiSys, 'ui peer missing');
    assert(game.debug.run('ui.map') === true, 'map did not open');
    await waitFrames(game, 3);
    assert(uiSys.isModal && game.input.isBlocked, 'map open but not modal/blocking');
    const canvas = document.querySelector<HTMLCanvasElement>('#ui .fullmap canvas');
    assert(canvas && canvas.width > 10, 'full map canvas not sized');
    game.input.press('KeyM', 60);
    await waitFor(() => !uiSys.isModal, 2000, 'M to close the map');
    assert(!game.input.isBlocked, 'blocker left behind after closing the map');
  });

  await suite.check('title-screen-flow', async () => {
    const uiSys = game.ui;
    assert(uiSys, 'ui peer missing');
    game.debug.run('ui.title');
    await waitFrames(game, 3);
    assert(uiSys.isModal && game.input.isBlocked, 'title not modal');
    assert(cameraMode() === 'cinematic', `camera mode on title is ${cameraMode()}`);
    let started = 0;
    const off = game.events.on('ui:newGame', () => started++);
    const button = document.querySelector<HTMLButtonElement>('#ui .title .menu-button');
    assert(button, 'New Game button missing');
    button.click();
    off();
    await waitFrames(game, 3);
    assert(started === 1, 'ui:newGame not emitted');
    assert(!uiSys.isModal && !game.input.isBlocked, 'title still modal after New Game');
    assert(cameraMode() === 'thirdPerson', `camera mode after New Game is ${cameraMode()}`);
  });

  await suite.check('no-errors-after-run', () => {
    assert(game.debug.errorCount === 0, `${game.debug.errorCount} console errors: ${game.debug.errors.join(' | ')}`);
  });

  // Leave a waypoint and route on screen for the screenshot review.
  player.teleport({ x: 575, y: 0, z: 40 }, -Math.PI / 2);
  game.debug.run('navigation.waypoint', -400, 100);
  await waitFrames(game, 3);
  suite.finish();
}
