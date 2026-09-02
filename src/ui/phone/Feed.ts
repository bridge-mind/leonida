import type { Game } from '../../core/Game';
import { Random } from '../../core/Random';
import { formatCash } from '../format';

export interface FeedPost {
  handle: string;
  text: string;
  /** In-game clock when posted, e.g. "14:32". */
  time: string;
  likes: number;
}

const HANDLES = [
  '@viceb1tch',
  '@leonida_eyes',
  '@gatorwatch',
  '@portgellhorn_pd_scanner',
  '@habana_hot_takes',
  '@swampthing305',
  '@keysdrifter',
  '@downtownDoomer',
  '@chopchopdocks',
  '@weazel_news_intern',
];

const CRIME_LINES: Record<string, string[]> = {
  rob_store: ['someone just held up a store on my block?? clerk is FINE, register is not', 'store robbery. again. this city man'],
  rob_ped: ['got mugged at gunpoint, wallet gone, dignity gone', 'someone is robbing people downtown watch ur pockets'],
  shot_fired: ['gunshots near me. not the 4th of july.', 'anyone else hear that?? sounded like a shotgun'],
  carjack: ['bro just took someone\u2019s car with them still holding the keys', 'carjacking outside the waffle hut. the WAFFLE HUT.'],
  hit_ped: ['saw a car plough through a crosswalk. driver did NOT stop', 'pedestrians: 0. sedan: 1.'],
  kill_ped: ['there\u2019s a body on the sidewalk and traffic is just going around it', 'RIP to whoever that was on Ocean Drive'],
  kill_cop: ['a cop got shot. every siren in Leonida is on', 'do NOT go downtown rn. cops are furious'],
  explosion: ['a car just EXPLODED next to the marina lol', 'fireball on the highway, my uber is cancelled'],
};

const DEFAULT_LINES = ['something is going down in Leonida', 'sirens. so many sirens.'];

/**
 * Parody social feed built from game events. Kept in memory only (not saved),
 * capped to the most recent posts.
 */
export class Feed {
  readonly posts: FeedPost[] = [];
  private readonly rng = Random.unseeded();
  private readonly offs: Array<() => void> = [];
  private lastCrimeAt = -Infinity;
  onChange: (() => void) | null = null;

  constructor(
    private readonly game: Game,
    private readonly max = 20,
  ) {
    const ev = game.events;
    this.offs.push(
      ev.on('crime:committed', ({ type, actor }) => {
        if (actor !== 'player') return;
        // Gunfights spam shot_fired; one post per few seconds keeps the feed readable.
        const now = game.time.elapsed;
        if (type === 'shot_fired' && now - this.lastCrimeAt < 8) return;
        this.lastCrimeAt = now;
        this.post(this.rng.pick(CRIME_LINES[type] ?? DEFAULT_LINES));
      }),
      ev.on('casino:won', ({ game: table, amount }) => {
        if (amount >= 500) this.post(`someone just took the Diamond for ${formatCash(amount)} at ${table}. house is NOT happy`);
      }),
      ev.on('economy:cashChanged', ({ delta, reason }) => {
        if (Math.abs(delta) < 2000) return;
        if (delta > 0) this.post(reason === 'chopshop' ? `docks were busy tonight. ${formatCash(delta)} kind of busy` : `who just came into ${formatCash(delta)}?? asking for a friend`);
        else this.post(`watched someone drop ${formatCash(-delta)} without blinking. rich people are weird`);
      }),
      ev.on('robbery:completed', ({ amount }) => this.post(`store on the corner just got emptied for like ${formatCash(amount)}. clerk hands still up`)),
      ev.on('chopshop:sold', ({ archetypeId }) => this.post(`if you owned a ${archetypeId} in Leonida... you don't anymore`)),
      ev.on('mission:passed', () => this.post('heard some people just pulled off something big. no details. respect.')),
    );
    this.post('welcome to Leonida. don\u2019t drink the canal water.');
  }

  post(text: string, handle = this.rng.pick(HANDLES)): void {
    const h = Math.floor(this.game.time.hour);
    const m = Math.floor((this.game.time.hour - h) * 60);
    this.posts.unshift({
      handle,
      text,
      time: `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`,
      likes: this.rng.int(3, 4200),
    });
    if (this.posts.length > this.max) this.posts.length = this.max;
    this.onChange?.();
  }

  dispose(): void {
    for (const off of this.offs) off();
    this.offs.length = 0;
  }
}
