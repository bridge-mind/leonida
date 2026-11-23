import { describe, expect, it } from 'vitest';
import type { Game } from '../src/core/Game';
import { loadMods, startMods, type GameMod } from '../src/core/mods';

function loader(mod?: GameMod, boom?: string): () => Promise<{ default?: GameMod }> {
  return async () => {
    if (boom) throw new Error(boom);
    return { default: mod };
  };
}

describe('compiled-in mods', () => {
  it('loads a mod whose id matches the folder and skips the rest', async () => {
    const hello: GameMod = { id: 'hello', setup() {} };
    const report = await loadMods({
      '../mods/hello/index.ts': loader(hello),
      '../mods/broken/index.ts': loader(undefined, 'cannot parse'),
      '../mods/mismatch/index.ts': loader({ id: 'other', setup() {} }),
      '../mods/empty/index.ts': async () => ({}),
    });
    expect(report.mods.map((m) => m.id)).toEqual(['hello']);
    expect(report.errors.map((e) => e.id).sort()).toEqual(['broken', 'empty', 'mismatch']);
    expect(report.errors.find((e) => e.id === 'mismatch')?.reason).toContain("id 'mismatch'");
  });

  it('starts mods in path order and keeps going when setup throws', async () => {
    const seen: string[] = [];
    const mods: GameMod[] = [
      { id: 'a', setup() { seen.push('a'); } },
      { id: 'b', setup() { throw new Error('nope'); } },
      { id: 'c', async setup() { seen.push('c'); } },
    ];
    const started = await startMods({} as Game, mods);
    expect(started.ids).toEqual(['a', 'c']);
    expect(seen).toEqual(['a', 'c']);
    expect(started.errors).toEqual([{ id: 'b', reason: 'nope' }]);
  });

  it('loads nothing when the mods folder is empty', async () => {
    const report = await loadMods({});
    expect(report.mods).toEqual([]);
    expect(report.errors).toEqual([]);
  });
});
