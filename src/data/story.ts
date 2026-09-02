/**
 * Story data (SYSTEM_PROMPT §14): the eight missions plus the post-finale jobs.
 * Beats live in `src/missions/m*.ts`; this table holds what the phone, the map and
 * the mission list need without loading any mission code.
 */
export type ContactId = 'jason' | 'manny' | 'ray' | 'dominic';

export interface ContactDef {
  id: ContactId;
  name: string;
  /** Landmark whose door the contact hangs out at. */
  landmarkId: string;
  /** Meters from the door along the door heading where the marker sits (keeps the E prompt off the door). */
  doorOffset: number;
}

export const CONTACTS: Record<ContactId, ContactDef> = {
  jason: { id: 'jason', name: 'Jason', landmarkId: 'safehouse', doorOffset: 6 },
  manny: { id: 'manny', name: 'Manny Ochoa', landmarkId: 'chop_shop', doorOffset: 7 },
  ray: { id: 'ray', name: 'Ray Tillman', landmarkId: 'waffle_hut', doorOffset: 6 },
  dominic: { id: 'dominic', name: 'Dominic Reyes', landmarkId: 'casino', doorOffset: 7 },
};

export interface StoryMissionDef {
  id: string;
  name: string;
  contact: ContactId;
  /** Overrides the contact's landmark for the marker (mission 1 starts at the prison gate). */
  markerLandmarkId?: string;
  markerOffset?: number;
  unlockedAfter?: string;
  /** Base cash reward; some missions compute the final amount from what happened. */
  reward: number;
  repeatable: boolean;
  /** One-line pitch for the phone contact list. */
  blurb: string;
}

export const STORY_MISSIONS: readonly StoryMissionDef[] = [
  { id: 'm1', name: 'Release', contact: 'jason', markerLandmarkId: 'prison', markerOffset: 8, reward: 500, repeatable: false, blurb: 'Lucia walks free. Jason is waiting.' },
  { id: 'm2', name: 'Gas Money', contact: 'jason', unlockedAfter: 'm1', reward: 300, repeatable: false, blurb: "We're broke. The Gellhorn gas station isn't." },
  { id: 'm3', name: 'Repo', contact: 'manny', unlockedAfter: 'm2', reward: 1500, repeatable: false, blurb: 'A sports coupe at a Vice Beach hotel has a new owner.' },
  { id: 'm4', name: 'Bad Debt', contact: 'manny', unlockedAfter: 'm3', reward: 800, repeatable: false, blurb: 'A debtor in Little Habana owes Manny. Collect.' },
  { id: 'm5', name: 'Mud Club', contact: 'ray', unlockedAfter: 'm4', reward: 2000, repeatable: false, blurb: 'Run a package out to the Gator Pit in Grassrivers.' },
  { id: 'm6', name: 'High Roller', contact: 'dominic', unlockedAfter: 'm5', reward: 2500, repeatable: false, blurb: 'Meet the inside man at Leonida Palms. Win big.' },
  { id: 'm7', name: 'Heat', contact: 'dominic', unlockedAfter: 'm6', reward: 1000, repeatable: false, blurb: 'Find out how hard VCPD really hits back.' },
  { id: 'm8', name: 'The Score', contact: 'jason', unlockedAfter: 'm7', reward: 25000, repeatable: false, blurb: 'Vice Rocks Jewelers. Jason. One shot.' },
  { id: 'job_chopshop', name: 'Wanted Wheels', contact: 'manny', unlockedAfter: 'm8', reward: 3000, repeatable: true, blurb: "Bring Manny today's wanted ride." },
  { id: 'job_robbery', name: 'Register Run', contact: 'ray', unlockedAfter: 'm8', reward: 500, repeatable: true, blurb: 'Knock over any store and get away clean.' },
];

export const STORY_BY_ID: ReadonlyMap<string, StoryMissionDef> = new Map(STORY_MISSIONS.map((m) => [m.id, m]));

/** Story order used for unlock checks and the "next mission" hint. */
export const STORY_ORDER: readonly string[] = STORY_MISSIONS.filter((m) => !m.repeatable).map((m) => m.id);
