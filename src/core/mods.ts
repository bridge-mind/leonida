import type { Game } from './Game';

/**
 * A compiled-in mod: `src/mods/<id>/index.ts` default-exports one of these.
 * The folder name is the id. Nothing is loaded from the network or from a
 * folder the player drops next to the page.
 */
export interface GameMod {
  id: string;
  setup(game: Game): void | Promise<void>;
}

export interface ModError {
  id: string;
  reason: string;
}

type ModLoader = () => Promise<{ default?: GameMod }>;

/** Loads the modules Vite discovered. A bad module is reported and skipped. */
export async function loadMods(loaders: Record<string, ModLoader>): Promise<{ mods: GameMod[]; errors: ModError[] }> {
  const mods: GameMod[] = [];
  const errors: ModError[] = [];
  const paths = Object.keys(loaders).sort();
  for (const path of paths) {
    const id = path.split('/').slice(-2, -1)[0] ?? path;
    try {
      const loaded = await loaders[path]();
      if (!loaded.default || loaded.default.id !== id) {
        errors.push({ id, reason: `index.ts must default-export a GameMod with id '${id}'` });
        continue;
      }
      mods.push(loaded.default);
    } catch (err) {
      errors.push({ id, reason: err instanceof Error ? err.message : String(err) });
    }
  }
  return { mods, errors };
}

/** Runs `setup` after the built-in systems exist. A throw skips that mod and leaves the game running. */
export async function startMods(game: Game, mods: GameMod[]): Promise<{ ids: string[]; errors: ModError[] }> {
  const ids: string[] = [];
  const errors: ModError[] = [];
  for (const mod of mods) {
    try {
      await mod.setup(game);
      ids.push(mod.id);
    } catch (err) {
      errors.push({ id: mod.id, reason: err instanceof Error ? err.message : String(err) });
    }
  }
  return { ids, errors };
}
