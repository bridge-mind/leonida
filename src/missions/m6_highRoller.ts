import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { contactMarker, doorFront, sceneOrbit, sceneTalk } from './helpers';
import { goto } from './objectives';
import { cutscene, winAtCasino } from './objectivesWorld';
import type { MissionDef } from './types';

const story = STORY_BY_ID.get('m6')!;
const TARGET_WINNINGS = 2000;

/** 6 — High Roller: meet Dominic at Leonida Palms and walk out $2,000 up. */
export const m6: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m6'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build() {
    const entrance = doorFront('casino', 5);
    return [
      goto(entrance, 6, 'Go to ~y~Leonida Palms~y~ Downtown', { blip: 'casino' }),
      cutscene(sceneOrbit('m6_start', entrance, dialogue('m6_start'))),
      winAtCasino(TARGET_WINNINGS, entrance, 'Win at the ~y~tables~y~ inside — blackjack counts'),
      cutscene(sceneTalk('m6_done', dialogue('m6_done'))),
    ];
  },
};
