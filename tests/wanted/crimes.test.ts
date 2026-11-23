import { describe, expect, it } from 'vitest';
import { isFreshPoliceTheft } from '../../src/systems/wanted/crimes';

const cruiser = { isPolice: true };
const sedan = { isPolice: false };
const player = { kind: 'player' };
const ped = { kind: 'ped' };

describe('police theft on enter', () => {
  it('counts the first time the player sits in a cruiser', () => {
    expect(isFreshPoliceTheft(cruiser, player, false)).toBe(true);
  });

  it('does not count getting back into a cruiser the player already took', () => {
    expect(isFreshPoliceTheft(cruiser, player, true)).toBe(false);
  });

  it('ignores civilian cars and a ped climbing into a cruiser', () => {
    expect(isFreshPoliceTheft(sedan, player, false)).toBe(false);
    expect(isFreshPoliceTheft(cruiser, ped, false)).toBe(false);
  });
});
