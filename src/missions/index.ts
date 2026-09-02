import { jobChopShop, jobRobbery } from './jobs';
import { m1 } from './m1_release';
import { m2 } from './m2_gasMoney';
import { m3 } from './m3_repo';
import { m4 } from './m4_badDebt';
import { m5 } from './m5_mudClub';
import { m6 } from './m6_highRoller';
import { m7 } from './m7_heat';
import { m8 } from './m8_theScore';
import type { MissionDef } from './types';

/** Story order, then the repeatable jobs unlocked by the finale. */
export const MISSIONS: readonly MissionDef[] = [m1, m2, m3, m4, m5, m6, m7, m8, jobChopShop, jobRobbery];

export const MISSION_BY_ID: ReadonlyMap<string, MissionDef> = new Map(MISSIONS.map((m) => [m.id, m]));
