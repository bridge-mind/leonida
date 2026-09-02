# Audio (Agent 7) — progress

## Works

- `src/systems/audio/index.ts` replaces the stub: `AudioSystem implements GameSystem, AudioApi`, module `{ id: 'audio', priority: PRIORITY.audio }`. `api.ts` untouched; `sounds.ts` untouched (no ids added or removed).
- Graph (`graph.ts`): `AudioContext` created lazily on the first `play`/`loop` (the ambient manager triggers this ~0.5 s after ready), `resume()` on `keydown`/`mousedown`/`pointerdown` and on `ui:newGame`/`ui:continue`. Buses `sfx/engine/ambient/ui/music → master → DynamicsCompressor → destination`; shared convolver reverb (generated 1.6 s impulse) fed by a per-voice send. Bus volumes persisted as JSON in `localStorage['leonida.audio']`.
- Headless / no Web Audio: `AudioGraph.tryCreate` returns null → every API call is a no-op with `NullAudio` semantics (unknown ids warn once, handles returned, `stop` sets `alive = false`). No API method throws; recipe failures are caught and `console.warn`ed once per id.
- Positional audio (`voice.ts`): `PlayOptions.position` → `PannerNode` (equalpower, inverse, refDistance 6, maxDistance 220, rolloff 1.2). Listener from `setListener` via `positionX/forwardX/upX.value` with a `setPosition/setOrientation` fallback. One-shots farther than 220 m are not scheduled. Positional loops get a cheap doppler (radial speed from successive `setLoopParams` positions, ±12 % max).
- Voice pool (`voicePool.ts`, pure logic): 40 one-shots + 16 loops; steals the quietest (then oldest) voice when full; `stolen` counter; `activeVoices`/`voiceStats()` exposed. One-shots expire by AudioContext time, or by wall clock when the context is suspended (so nothing piles up before the first gesture).
- Every one of the 279 `SoundId`s maps to a recipe (`recipes/*.ts`); the registry type `Record<SoundId, Recipe>` makes a missing id a compile error, and `tests/audio/recipes.test.ts` checks it at runtime too.
- Engine model (`engine.ts`): two detuned saws + triangle sub + bandpassed intake noise → lowpass → tanh saturation; `engine(handle, rpm01, throttle01, speedMs)` ramps frequency/cutoff/gain with 45 ms `setTargetAtTime`; gears 1..5 quantised from speed (0/9/17/27/38 m/s), in-gear fraction blended 40 % into the rpm, 180 ms × 0.7 dip on each shift. Variants: car, motorcycle (square+saw, 95 Hz), boat (38 Hz, heavy sub), airboat (prop-wash AM), helicopter (12 Hz pulsed brown noise + turbine whine; LFO rate follows rpm).
- Siren loop with `siren(handle, on)` fade; skid, scrape, alarm, tire-flat loops with `setLoopParams` pitch following.
- Footstep driver (`footstepDriver.ts`): stride accumulator (0.75 m walk → 1.4 m sprint), surface classified every 0.5 s by a downward raycast against `StaticEntity.surface` (+ `world.isInWater`), ±10 % pitch, volume by speed; jump/land/land.hard from grounded transitions.
- Event cues (`eventCues.ts`): `player:damaged` → `player.hurt` (throttled 0.25 s), `player:died` → `player.death`, `water:entered` → `water.splash(.big for vehicles)`, `interior:entered/exited` → `interior.enter/exit`.
- Ambient manager (`ambientManager.ts`): polls every 0.5 s, bed from `districtAt(playerPos).ambient` (fallback `ambient.city.loop`), `ambient.interior.loop` / `casino.ambient.loop` while `state.interior` is set, 2 s crossfades, `ambient.night.crickets.loop` scaled by `world.night01` outside Downtown/Vice Beach. All on the `ambient` bus, non-positional.
- Radio (`recipes/radio.ts`): 120 BPM 16-step sequencer (Am–F–C–G): saw bass, detuned saw pad, kick (sine pitch drop), snare, hats, delayed triangle arpeggio on odd bars. `radio(on)` plays `radio.switch`/`radio.off` and starts/stops the loop on the `music` bus.
- Debug actions: `audio.playAll`, `audio.voices`, `audio.radio:<on>`, `audio.bus:<bus>:<volume>`, `audio.state`.
- Tests: `tests/audio/recipes.test.ts`, `engine.test.ts`, `voicePool.test.ts` (16 tests, no AudioContext). Autotest `src/debug/autotests/audio.ts`.

## Cut or deferred

- No listening pass in a real browser this session (browser automation was forbidden). Parameter choices below are from the design brief and Web Audio experience; expect a tuning pass on levels once heard together.
- Doppler is an approximation on the loop pitch (radial speed derived from positions), not a physically modelled listener/source velocity pair; `setListener`'s velocity argument is accepted but unused.
- Footstep surface uses `StaticEntity.surface` only; dynamic surfaces (vehicle roofs) fall back to the last classified surface.
- No per-ped chatter scheduling or crowd density coupling; `ped.chatter.loop`/`ambient.crowd.loop` are self-contained beds for whoever starts them.
- No dedicated weather manager: `ambient.rain.loop`, `ambient.wind.loop`, `ambient.thunder` exist as recipes for a later weather system.
- `serialize/deserialize` not implemented; settings live in localStorage, nothing else is worth saving.

## Known bugs

- Until the first user gesture the context is `suspended`; ambient/radio loops started before that begin on resume (intended), one-shots scheduled before it are dropped by the wall-clock expiry (intended) — but a loop's first ~0.2 s of periodic events can bunch up right at resume.
- `play()` of a `.loop` id plays it for 1.5 s then fades (useful for `playAll`); callers should use `loop()` for real loops.
- `AudioParam.value` writes for the listener every frame can zipper on very fast camera cuts; not audible in normal play.
- Stealing picks by estimated loudness (volume × distance falloff), not measured RMS, so a quiet-but-important one-shot can be stolen during a 40-voice burst.

## How to test

- Unit tests: `npx vitest run tests/audio`
- Typecheck: `npm run typecheck` (audio files: 0 errors)
- Isolation URL: `?systems=world,player,camera,audio`
- Autotest (when browser runs are allowed again): `npm run smoke -- --systems=world,player,camera,audio --name=audio --autotest=audio`
- Manual, in the browser console (`__game`):
  - `__game.debug.run('audio.playAll')` → 279, every id staggered 30 ms
  - `__game.debug.run('audio.voices')` → `{ oneShots, loops, maxOneShots, maxLoops, stolen }`
  - `__game.debug.run('audio.state')` → context state, bus volumes, current ambient bed, radio flag
  - `__game.debug.run('audio.radio', true)` / `false`
  - `__game.debug.run('audio.bus', 'music', 0.3)`
  - URL form: `&actions=audio.radio:true,audio.bus:sfx:0.5`
  - Walk (WASD/Shift) to hear footsteps change with speed and surface; teleport with `world.goto:<landmark>` to hear district beds crossfade; `time.set:22` for crickets.
  - Engine: `h = __game.audio.loop('vehicle.engine.loop', { position: __game.state.playerPos }); __game.audio.engine(h, 0.7, 1, 20)`

## Integration requests

1. **Vehicles (Agent 2), `src/systems/vehicles/**`** — drive the engine per vehicle:
   ```ts
   // on spawn / engine start
   this.engineLoop = game.audio.loop(archetype.engineSound ?? 'vehicle.engine.loop', { position: this.position, volume: 0.9 });
   // every update while the loop is alive
   game.audio.engine(this.engineLoop, this.state.rpm01, this.controls.throttle, this.state.speedMs);
   game.audio.setLoopParams(this.engineLoop, { position: this.position });
   // despawn / explode
   game.audio.stop(this.engineLoop);
   ```
   Siren loops start audible: call `game.audio.siren(handle, false)` right after `loop('vehicle.siren.loop', …)` if the light bar is off, then `siren(handle, true)` when it turns on. Skid: `loop('vehicle.skid.loop')` and set `volume` by slip, `stop` when grip returns.
2. **UI (Agent 5), settings menu** — bind sliders to `game.audio.getBusVolume(bus)` / `setBusVolume(bus, v)` for `master, sfx, engine, ambient, ui, music`; persistence is handled by audio. Do not play `interior.enter/exit`, `player.hurt/death`, `water.splash*`, `player.jump/land*` or footsteps — audio already plays those from core events.
3. **Foundation (Agent 1), `src/core/Game.ts` (optional)** — none required. If you want the listener to follow the smoothed camera rather than the raw `viewCamera` quaternion, nothing changes on my side; the camera system already calls `setListener` each frame.
4. **Casino (Agent 6)** — `casino.ambient.loop` is started automatically while `state.interior` contains `'casino'`; do not start it yourself. Start `casino.reel.spin.loop` / `casino.wheel.spin.loop` / `casino.roulette.ball.loop` with `loop()` and slow them with `setLoopParams(h, { pitch })` (tick rate follows pitch), then `stop()`.

## Public API

Exactly `AudioApi` from `src/systems/audio/api.ts`:

```ts
play(id: SoundId, opts?: { position?: V3; volume?: number; pitch?: number; bus?: AudioBus }): void
loop(id: SoundId, opts?: PlayOptions): LoopHandle          // { id, sound, alive }
stop(handle: LoopHandle): void                               // 80 ms fade; handle.alive = false
setLoopParams(handle, { pitch?, volume?, position? }): void  // 40 ms ramps
engine(handle, rpm01, throttle01, speedMs): void
siren(handle, on: boolean): void
setListener(position: V3, quaternion: Quaternion, velocity: V3): void
setBusVolume(bus: AudioBus, volume: number): void            // persisted
getBusVolume(bus: AudioBus): number
radio(on: boolean): void
resume(): void
```

Extras on the concrete `AudioSystem` (`game.systems.get<AudioSystem>('audio')`): `fadeLoop(handle, volume, seconds)`, `activeVoices`, `voiceStats()`, `radioOn`. Pure helpers for other agents' tests: `engineHz`, `gearFor`, `gearFraction`, `EngineModel` from `src/systems/audio/engine.ts`.

Default bus by id: `ui.*` → ui; `ambient.*`, `casino.ambient.loop` → ambient; `vehicle.engine.*.loop` → engine; `radio.*` → music; everything else → sfx (`opts.bus` overrides).

### Listening notes per family (parameter choices)

- **Weapons** — white-noise burst 12–40 ms through a lowpass sweeping 4–7 kHz → 300–600 Hz, plus a sine thump gliding to 45 %: pistol 140 Hz/120 ms, SMG 170 Hz/60 ms (tight, 12 ms noise), shotgun 70 Hz/350 ms + 180 ms brown tail, rifle 110 Hz + 3.2 kHz crack, sniper 55 Hz/500 ms + crack + 300 ms tail, reverb send 0.45. Reloads are 2–4 bandpass clicks (Q 4, ±8 % pitch) at 80–200 ms spacing with lowpass slide noises for pumps/bolts; empties are a two-click dry snap. Melee swing = bandpass whoosh 500→2600 Hz/180 ms; hits = 120/80 Hz thumps + lowpassed noise. Shell drop = two 4.2/5.1 kHz Q 40 pings.
- **Impacts** — concrete/asphalt bandpass 2.6/1.9 kHz 80–90 ms with 180/150 Hz thump; metal = 2.2 + 1.5 kHz Q 20 rings; glass = four 3.2–7.5 kHz pings; wood 700 Hz Q 1.5; dirt/sand/flesh pink noise lowpass 700–1200 Hz; water lowpass sweeping 500→3500 Hz; ricochet = 3 kHz→700 Hz chirp; whizz = bandpass 3.8 kHz→600 Hz at 1.4× rate; shatter = 12 random pings over 450 ms.
- **Footsteps** — concrete/asphalt bandpass 2.2/1.7 kHz 50–60 ms + 130–140 Hz thump; grass/sand/dirt pink noise lowpass 700–1200 Hz 90–100 ms; wood 600 Hz Q 2.5; metal adds a 1.9 kHz ring; water is a lowpass sweep. Driver adds ±10 % pitch and 0.3–0.7 volume by speed.
- **Vehicles** — engine idle 55 Hz (×3.5 at full rpm), second saw at ×1.012, sub at ×0.5, intake brown noise bandpass 320 Hz, lowpass 320 + 2200·rpm + 600·throttle Hz, drive 1.8, idle lope LFO 7 Hz. Skid = white noise bandpass 1.8 kHz Q 2.2 with a 3.1 Hz ±220 Hz wobble. Horn = 440 + 554 Hz squares through 2.4 kHz lowpass (0.22/0.6/1.6 s). Siren = saw + square at 900 Hz ±300 Hz triangle LFO 0.8 Hz through bandpass 1.4 kHz. Crashes = lowpass sweep 3.2 kHz→250 Hz over 150/300/550 ms with 90/65/45 Hz thumps and metal rings on medium/heavy.
- **Explosions** — brown noise lowpass 3.2 kHz→140 Hz over 1.8 s, 46 Hz boom gliding to 25 Hz over 1.3 s, 1.8 kHz crack, rumble tail (lowpass 220→60 Hz) from +0.3 s for 2.2 s, reverb 0.5; small is 0.7 s; distant is lowpass 320→60 Hz over 2.5 s with a 36 Hz boom. Fire = brown lowpass 420 Hz + white bandpass 2.6 kHz hiss + 42 Hz saw rumble + random 1.8–5 kHz crackle ticks every 40–220 ms.
- **Peds** — sawtooth through a formant bandpass (Q 2.5) with a three-point pitch envelope: yelp 300→520→380 Hz/250 ms, scream 380→700→500 Hz/900 ms with 6.5 Hz vibrato, female 550→950→700 Hz, grunt 140→125→100 Hz. Cop radio = bandpass 1.8 kHz static 0.5 s + 1.4 kHz squelch; megaphone = three syllables through drive 3 + bandpass 1.2 kHz Q 2.5; handcuffs = six 25 ms ratchet clicks + 2.8 kHz ping.
- **Pickups / economy** — cash.gain = 784/988/1175 Hz sines 70 ms apart, cash.loss = 660/523/415 Hz, cash.tick = 3 kHz click + 2 kHz 30 ms blip; register = 200 Hz clunk + 2.8 kHz bell (Q 40 + 2.76× partial); bagfill = pink bandpass 2 kHz bed with 50–120 ms rustles every 100–280 ms; store.bell = 2.4 + 3.6 kHz bells with reverb 0.3.
- **UI** — sine/triangle blips 8 ms attack, 80–200 ms decay at 0.2–0.25 gain; select/back 880↔1320 Hz; notification 988→1319 Hz with reverb 0.2; error = 110 + 115 Hz squares (beating) through lowpass 600 Hz; wheel.tick 2.5 kHz click + 1.8 kHz 30 ms; hitmarker 2.8 kHz click + 1.4 kHz 40 ms, kill 2 kHz + 900 Hz 90 ms; title stinger = A3 C#4 E4 A4 detuned saw pad (±7 cents) through 1.2 kHz lowpass, 400 ms per note, 55 Hz sub, reverb 0.5.
- **Wanted / mission** — star.gain 660→990 Hz glide; lost = 659/784/988 Hz triangle 130 ms apart; searching = 110 + 55 Hz pulses; busted/wasted = detuned saw pairs (220/262 Hz and 196/233 Hz) gliding down an octave over 1.2/1.6 s through lowpass 800 Hz plus 50→30 Hz boom; passed = C5 E5 G5 triangle sting + C4/E4/G4 saw pad + C3 sub, reverb 0.5; failed = A4 G#4 F4 saws over 900 ms with a 55→35 Hz boom.
- **Casino** — chips = 2.3–2.9 kHz Q 5 clicks + 2.9–3.5 kHz 50 ms sine; card swishes = bandpass sweeps 1.8→4 kHz 50–150 ms; reel/wheel/ball loops are periodic ticks (60/120/60–100 ms, rate follows `pitch`); jackpot = 12-note C-major run over 1.4 s + 14 random 2.4–4.8 kHz bells + detuned pad, reverb 0.5; ambient = pink lowpass 800 Hz + murmur voice blips every 120–300 ms + a chip clink every 1–4 s.
- **Ambient beds** — beach: pink lowpass 900 Hz with 0.1 Hz swell + white bandpass 3 kHz at 0.13 Hz, seagull cries every 6–15 s; city: brown lowpass 260 Hz + 60 Hz hum + distant two-tone honks every 4–12 s; habana: warmer brown lowpass 360 Hz + soft pentatonic triangle pulses every 0.5 s; suburb: pink lowpass 600 Hz + bird chirps (2.2–3.6 kHz glides) every 0.4–2.2 s; docks: brown lowpass 200 Hz + 50 Hz hum + clank every 3–9 s; swamp: white bandpass 5.2 kHz Q 8 with 25 Hz tremolo + brown bed + 180→120 Hz frog pulses every 0.8–2.5 s; crickets: bandpass 4.5 kHz Q 10 noise with 30 Hz square tremolo + 4.2 kHz chirp bursts every 0.3–0.6 s; club: 55 Hz saw bass + 65→40 Hz kick every 0.5 s all behind a 200 Hz lowpass; wind: pink bandpass 500 Hz with 0.15 Hz gain and 0.11 Hz ±300 Hz filter LFOs.
- **Radio** — 120 BPM, 16 sixteenths per bar, Am–F–C–G; bass saw (lowpass 650 Hz Q 1.5) on steps 0 2 3 6 8 10 11 14 with octave hops; pad = two saws ±8 cents through lowpass 1.3 kHz per bar; kick 150→45 Hz on 0 4 8 12; snare bandpass 1.8 kHz on 4 12; hats highpass 7 kHz on 2 6 10 14; triangle arpeggio through a 3-step delay (0.375 s, feedback 0.35) on odd bars. Music bus default 0.6.
