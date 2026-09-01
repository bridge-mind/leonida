import { describe, expect, it } from 'vitest';
import { Pool } from '../src/core/Pool';
import { Random } from '../src/core/Random';
import { SpatialHash } from '../src/core/SpatialHash';

interface Item {
  id: number;
  x: number;
  z: number;
}

describe('SpatialHash', () => {
  it('returns exactly the items within the radius', () => {
    const hash = new SpatialHash<Item>(20);
    const rng = new Random(99);
    const items: Item[] = [];
    for (let i = 0; i < 2000; i++) {
      const item = { id: i, x: rng.range(-500, 500), z: rng.range(-500, 500) };
      items.push(item);
      hash.insert(item, item.x, item.z);
    }
    const out: Item[] = [];
    for (let q = 0; q < 20; q++) {
      const qx = rng.range(-500, 500);
      const qz = rng.range(-500, 500);
      const r = rng.range(5, 120);
      hash.query(qx, qz, r, out);
      const expected = items.filter((it) => Math.hypot(it.x - qx, it.z - qz) <= r).map((it) => it.id).sort();
      expect(out.map((it) => it.id).sort()).toEqual(expected);
    }
  });

  it('tracks updates and removals', () => {
    const hash = new SpatialHash<Item>(10);
    const a = { id: 1, x: 0, z: 0 };
    hash.insert(a, 0, 0);
    const out: Item[] = [];
    expect(hash.query(0, 0, 1, out)).toHaveLength(1);
    hash.update(a, 100, 100);
    expect(hash.query(0, 0, 1, out)).toHaveLength(0);
    expect(hash.query(100, 100, 1, out)).toHaveLength(1);
    hash.remove(a);
    expect(hash.query(100, 100, 1, out)).toHaveLength(0);
    expect(hash.count).toBe(0);
  });

  it('does not allocate the output array', () => {
    const hash = new SpatialHash<Item>(10);
    hash.insert({ id: 1, x: 1, z: 1 }, 1, 1);
    const out: Item[] = [];
    const returned = hash.query(0, 0, 5, out);
    expect(returned).toBe(out);
  });
});

describe('Pool', () => {
  it('reuses released objects', () => {
    let created = 0;
    const pool = new Pool(
      () => ({ n: created++ }),
      (o) => {
        o.n = -1;
      },
      2,
    );
    expect(pool.size).toBe(2);
    const a = pool.acquire();
    const b = pool.acquire();
    const c = pool.acquire();
    expect(created).toBe(3);
    pool.release(b);
    expect(pool.freeCount).toBe(1);
    const d = pool.acquire();
    expect(d).toBe(b);
    expect(d.n).toBe(-1);
    expect(pool.activeCount).toBe(3);
    expect([a, c].includes(d)).toBe(false);
  });
});

describe('Random', () => {
  it('is deterministic per seed', () => {
    const a = new Random(5);
    const b = new Random(5);
    for (let i = 0; i < 100; i++) expect(a.next()).toBe(b.next());
    const c = Random.fromString('leonida');
    const d = Random.fromString('leonida');
    expect(c.int(0, 1000)).toBe(d.int(0, 1000));
  });
});
