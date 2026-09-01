import RAPIER from '@dimforge/rapier3d-compat';
import { Game } from './core/Game';

async function boot(): Promise<void> {
  // The compat build embeds the WASM; it must be initialised before any world exists.
  await RAPIER.init();
  const canvas = document.getElementById('game') as HTMLCanvasElement | null;
  if (!canvas) throw new Error('missing #game canvas');
  const game = new Game(canvas);
  (window as unknown as { __game: Game }).__game = game;
  await game.start();
}

boot().catch((err) => {
  console.error('[boot]', err);
  const el = document.createElement('div');
  el.className = 'hint';
  el.textContent = `LEONIDA failed to start: ${err instanceof Error ? err.message : String(err)}`;
  document.body.appendChild(el);
});
