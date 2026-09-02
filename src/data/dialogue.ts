import type { DialogueLine } from '../core/peers';

/**
 * Every cutscene / mission line, keyed by scene. Speaker names match `nameColor()`
 * in the UI so Lucia is pink and Jason cyan. Lines without `seconds` use reading time.
 */
export const DIALOGUE = {
  m1_gate: [
    { speaker: 'Jason', text: 'Two years. You look good for two years.' },
    { speaker: 'Lucia', text: "Don't. Just drive." },
    { speaker: 'Jason', text: 'Muscle car, full tank, no plates. Welcome back to Leonida.' },
  ],
  m1_safehouse: [
    { speaker: 'Jason', text: "It's not much. Habana rent, Habana walls." },
    { speaker: 'Lucia', text: "It's got a door that locks from the inside. That's plenty." },
    { speaker: 'Jason', text: 'Sleep. Tomorrow we figure out money.' },
  ],
  m2_start: [
    { speaker: 'Jason', text: "We're broke. Like, counting-coins broke." },
    { speaker: 'Lucia', text: 'The gas station out in Gellhorn. One register, one clerk.' },
    { speaker: 'Jason', text: 'I cover the door. You do the talking. Aim, take the bag, we walk.' },
  ],
  m2_done: [
    { speaker: 'Jason', text: 'Told you. Easy money.' },
    { speaker: 'Lucia', text: 'Two stars is not easy money, Jason.' },
  ],
  m3_start: [
    { speaker: 'Manny Ochoa', text: 'Manny Ochoa. I turn cars into cash and questions into silence.' },
    { speaker: 'Manny Ochoa', text: 'A red coupe at a hotel lot on the beach. Bring it here in one piece, without a tail.' },
    { speaker: 'Lucia', text: 'And the owner?' },
    { speaker: 'Manny Ochoa', text: "Has insurance. Everybody in Vice Beach has insurance." },
  ],
  m3_done: [{ speaker: 'Manny Ochoa', text: 'Clean. You want more work, you know where the docks are.' }],
  m4_start: [
    { speaker: 'Manny Ochoa', text: "Guy in Little Habana owes me eight hundred and thinks I forgot." },
    { speaker: 'Manny Ochoa', text: "Remind him. Scare him, break him, I don't care. Just make him remember." },
    { speaker: 'Lucia', text: 'Scaring is cheaper.' },
  ],
  m4_intimidated: [{ speaker: 'Manny Ochoa', text: 'He paid. Crying, but he paid. Nice work.' }],
  m4_killed: [{ speaker: 'Manny Ochoa', text: 'Dead men pay slow, but the message got around. Here.' }],
  m5_start: [
    { speaker: 'Ray Tillman', text: 'Ray. Waffle Hut is a front, the swamp is the business.' },
    { speaker: 'Ray Tillman', text: "Package downtown, deliver it to the Gator Pit mud club out in Grassrivers. Don't open it." },
    { speaker: 'Lucia', text: "What's in it?" },
    { speaker: 'Ray Tillman', text: 'Reasons not to open it.' },
  ],
  m5_ambush: [
    { speaker: 'Lucia', text: "That's not the club's welcome party." },
    { speaker: 'Ray Tillman', text: "Rifle's in the truck bed. Hold them off, then take the pickup and go!" },
  ],
  m5_done: [{ speaker: 'Ray Tillman', text: "You're still breathing. Around here that's a reference." }],
  m6_start: [
    { speaker: 'Dominic Reyes', text: 'Dominic Reyes. Floor manager, Leonida Palms. I know where every dollar in this building sleeps.' },
    { speaker: 'Dominic Reyes', text: 'Prove you can keep your head at a table. Walk out two grand up and we talk about something bigger.' },
    { speaker: 'Lucia', text: 'And if I lose?' },
    { speaker: 'Dominic Reyes', text: "Then you're just another tourist." },
  ],
  m6_done: [{ speaker: 'Dominic Reyes', text: 'Two thousand. Cool hands. Come see me again; I have a jeweler in mind.' }],
  m7_start: [
    { speaker: 'Dominic Reyes', text: 'Before the big one I need to know what VCPD brings when you really annoy them.' },
    { speaker: 'Dominic Reyes', text: 'Four stars, three minutes on your feet, then vanish. I pay per star you survive.' },
    { speaker: 'Lucia', text: "You're insane." },
    { speaker: 'Dominic Reyes', text: "I'm thorough." },
  ],
  m7_done: [{ speaker: 'Dominic Reyes', text: 'SWAT, a helicopter and you still walked away. Good. Then we are ready.' }],
  m8_start: [
    { speaker: 'Jason', text: 'Vice Rocks. Downtown. Dominic kills the cameras for ninety seconds.' },
    { speaker: 'Lucia', text: 'Guards?' },
    { speaker: 'Jason', text: 'Two. We grab what we can, then it is five stars all the way to the marina. Boat is waiting.' },
    { speaker: 'Lucia', text: 'After this we are done, Jason.' },
    { speaker: 'Jason', text: 'After this we are rich.' },
  ],
  m8_end: [
    { speaker: 'Jason', text: 'Twenty-five thousand. And nobody followed us to the water.' },
    { speaker: 'Lucia', text: 'Leonida looks better from out here.' },
    { speaker: 'Jason', text: 'Everything does.' },
  ],
  job_chopshop: [{ speaker: 'Manny Ochoa', text: "Board says what I'm paying double for today. Bring me one." }],
  job_robbery: [{ speaker: 'Ray Tillman', text: 'Any register in the city. Empty it, lose the cops, come back.' }],
} satisfies Record<string, DialogueLine[]>;

export type DialogueKey = keyof typeof DIALOGUE;

export function dialogue(key: DialogueKey): DialogueLine[] {
  return DIALOGUE[key];
}
