import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { contactMarker, sceneTalk, sidewalkNear } from './helpers';
import { action, goto, survive } from './objectives';
import { cutscene, escapeWanted, reachWanted } from './objectivesWorld';
import type { MissionDef } from './types';

const story = STORY_BY_ID.get('m7')!;
const DOWNTOWN_AREA = { x: 150, y: 0, z: 0 };
const TARGET_LEVEL = 4;
const SURVIVE_SECONDS = 180;
const REWARD_PER_STAR = 1000;

/** 7 — Heat: reach four stars Downtown, live through three minutes of it, then vanish. */
export const m7: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m7'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build(ctx) {
    let peak = 0;
    let off: (() => void) | null = null;
    const start = sidewalkNear(ctx.game, DOWNTOWN_AREA);
    return [
      cutscene(sceneTalk('m7_start', dialogue('m7_start'))),
      action((c) => {
        peak = c.game.wanted?.level ?? 0;
        off = c.game.events.on('wanted:changed', (e) => {
          if (e.level > peak) peak = e.level;
        });
        c.onCleanup(() => off?.());
      }),
      goto(start, 6, 'Get to ~y~Downtown~y~'),
      reachWanted(TARGET_LEVEL, 'Raise hell until you hit ~y~4 stars~y~ — shoot, steal, crash'),
      survive(SURVIVE_SECONDS, 'Stay alive and free for', { showTimer: true }),
      escapeWanted('Now ~y~lose them~y~ — respray or break line of sight'),
      action((c) => {
        const stars = Math.max(TARGET_LEVEL, peak, c.game.wanted ? 0 : TARGET_LEVEL);
        c.reward = REWARD_PER_STAR * stars;
      }),
      cutscene(sceneTalk('m7_done', dialogue('m7_done'))),
    ];
  },
};
