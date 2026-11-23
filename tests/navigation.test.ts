import { Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import type { Game } from '../src/core/Game';
import { NavigationSystem } from '../src/systems/navigation';

/** Three intersections 100 m apart. A player part-way along the first edge is still far from the route once that node is skipped. */
function harness(playerX: number) {
  const nodes = [
    { id: 0, position: new Vector3(0, 0, 0) },
    { id: 1, position: new Vector3(100, 0, 0) },
    { id: 2, position: new Vector3(200, 0, 0) },
  ];
  const state = { route: [] as Vector3[], waypoint: null as Vector3 | null, blips: [] as { id: string }[] };
  const game = {
    state,
    time: { realDt: 0.016 },
    systems: { has: () => true },
    world: {
      roadGraph: {
        nodes,
        nearestNode(p: { x: number; z: number }) {
          let best = nodes[0];
          let bestD = Infinity;
          for (const n of nodes) {
            const d = (n.position.x - p.x) ** 2 + (n.position.z - p.z) ** 2;
            if (d < bestD) {
              bestD = d;
              best = n;
            }
          }
          return best;
        },
        astar(from: number, to: number) {
          const path: number[] = [];
          const step = from <= to ? 1 : -1;
          for (let i = from; step > 0 ? i <= to : i >= to; i += step) path.push(i);
          return path;
        },
      },
    },
    player: { position: new Vector3(playerX, 0, 0) },
  };
  return { nav: new NavigationSystem(game as unknown as Game), state };
}

describe('navigation replan', () => {
  it('does not recurse when the fresh route is still farther than the stray radius', () => {
    const { nav, state } = harness(40);
    expect(() => nav.setWaypoint({ x: 200, y: 0, z: 0 })).not.toThrow();
    expect(state.route.map((p) => p.x)).toEqual([100, 200]);
    expect(nav.distanceToWaypoint).toBeCloseTo(160);
  });

  it('keeps a finite distance when the player is already on the route', () => {
    const { nav } = harness(100);
    nav.setWaypoint({ x: 200, y: 0, z: 0 });
    expect(nav.distanceToWaypoint).toBeCloseTo(100, 0);
  });
});
