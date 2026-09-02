import { dialogue } from '../data/dialogue';
import { STORY_BY_ID } from '../data/story';
import { contactMarker, doorFront, sceneTalk, withCompanion } from './helpers';
import { goto } from './objectives';
import { cutscene, escapeWanted, robStore } from './objectivesWorld';
import type { MissionDef } from './types';

const story = STORY_BY_ID.get('m2')!;

/** 2 — Gas Money: hold up Gellhorn Gas with Jason covering the door, then lose the cops. */
export const m2: MissionDef = {
  id: story.id,
  name: story.name,
  contact: story.contact,
  position: contactMarker('m2'),
  unlockedAfter: story.unlockedAfter,
  reward: story.reward,
  repeatable: false,
  build() {
    const station = doorFront('gas_station', 6);
    return [
      withCompanion(),
      cutscene(sceneTalk('m2_start', dialogue('m2_start'))),
      goto(station, 8, 'Drive to ~y~Gellhorn Gas~y~ in Port Gellhorn', { blip: 'gas' }),
      robStore('gas_station', station, 'Rob the ~y~register~y~ — walk in and aim at the clerk'),
      escapeWanted(),
      cutscene(sceneTalk('m2_done', dialogue('m2_done'))),
    ];
  },
};
