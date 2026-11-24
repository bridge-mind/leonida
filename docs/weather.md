# Weather XML in `common.rpf`

This note measures the one `CContentsOfWeatherXmlFile` document that drives weather (106,816 bytes inflated at
stream offset 0x207000 in `common.rpf`). `gamedata.md` has the element tree and the link checks; this page adds
what the `Mod*` axes correlate with, the exact vocabulary mismatches, and the measured behavior of the numeric
fields. Counts, schema names and small constants only, no values dump. Everything is reproducible from any
copy of the Xbox 1.0.2.1 package with the tooling in `gamedata.md`: carve with `tools/rpf9_carve.py`, then
`python -I tools/gamedata_reader.py <work>/carve/common --out <scratch>` and read
`CContentsOfWeatherXmlFile_207000.json` (game data, stays local).

The root has 18 direct children. The load-bearing ones:

| Child | Count | What it is |
|---|---:|---|
| `WeatherTypes` | 168 | named weather states the game can select |
| `WeatherTypeDatas` | 25 | numeric field sets a type borrows via `DataName` (72 fields each) |
| `WeatherAppTypeDatas` | 26 | weather-app presentation entries (19 of them referenced) |
| `WeatherCycles` | 2 | named cycles with a `TimeMult` |
| `WeatherTimeCycleFileNames` | 25 | `COMMON:/DATA/TIMECYCLE/W_<family>.XML` references |
| 8 `Mod*List`s | 3..17 | vocabularies for the `Mod*` tags |
| `WeatherTypeIndirectionList` | 15 | `From`/`To` pairs (13 complete, per `gamedata.md`) |
| `SkipDebugCycling` | 1 | top-level flag, `True` |
| `SinglePlayerOnlyWeathers` | 1 | an empty string |

## The 168 types and their `Mod*` tags

Every type carries `Name`, `ShortName`, `DataName`, `SkipDebugCycling`, and either all eight or none of the
`Mod*` tags: `ModAtmosphere`, `ModPrecipitation`, `ModTemperature`, `ModHumidity`, `ModWind`, `ModFetch`,
`ModMisc`, `ModNiceness`. The split is 151 fully tagged, 17 with nothing, and it is exactly a family split.
The 151 tagged types live in the nine `CLOUDSCAPE_*` families (2 to 38 types each by `DataName`). The 17
untagged ones are the 16 single-type families plus one stray variant of a `CLOUDSCAPE_*` data set. Eleven of
the 17 have `Name == DataName` exactly; the other six differ only in case (like `MISTY` borrowing `misty`).

The `Mod*` axes are not restatements of the numeric data. All eight axes change *within* a family; seven of
the eight do it in at least 6 of the nine `CLOUDSCAPE_*` families (`ModTemperature` and `ModFetch` in 8 of 9,
`ModMisc` only in 3). Every one of the 151 tagged
types references one of 19 distinct `WeatherAppDataName` entries; none of the 17 untagged types has one.
`SkipDebugCycling` per type is `True` for 137 and `False` for 31; all 31 `False` ones are tagged (120 of the
151), and every untagged type is `True`.

### The vocabularies, exact case

Comparing type tokens to their list case-sensitively:

| Axis | List | Used | Off-list (exact case) |
|---|---:|---:|---|
| `ModAtmosphere` | 17 | 14 | none |
| `ModPrecipitation` | 16 | 11 | none |
| `ModTemperature` | 9 | 9 | `W_MOD_temperature_cool` x1 |
| `ModHumidity` | 5 | 5 | none |
| `ModWind` | 11 | 14 | `_Light` x3, `_Mid` x1 (case), `_camlLight` x1 (typo) |
| `ModFetch` | 5 | 8 | `_Low` x10, `_High` x26, `_mid` x1 (case) |
| `ModMisc` | 7 | 6 | `W_MOD_misc_None` x135, `W_MOD_misc_High` x2 (real misses) |
| `ModNiceness` | 3 | 3 | none |

Case-insensitively only three tokens remain off-list: `W_MOD_wind_camlLight` (one type; two characters from
the list's `calmLight`, a genuine typo), and the two `ModMisc` values above, which is what `gamedata.md`
already flags (137 of 151 types use a `ModMisc` no list defines). A loader should rank-match case-insensitively
and never reject a type over `ModMisc`.

The vocabularies themselves read as ordered scales, with two catches. `ModPrecipitation` is 15 tokens
`W_MOD_precip_{20..100 step 20}` x `{drizzle, rain, HeavyRain}` plus a bare `W_MOD_precip_0`, so the intensity
is the middle number, not the list position. `ModHumidity` and `ModNiceness` are written top-down
(`VeryHigh` first, `good` first), the other lists bottom-up.

## What each axis tracks in the numbers

Ranking each axis by its own scale (precipitation by the embedded intensity; humidity and niceness
bottom-up; the rest by list order; temperature excluding its one `TEST_Mid` entry) and correlating with the
fields of the borrowed data set (Spearman, average ranks, over the 151 tagged types):

| Axis | Best correlation | Value |
|---|---|---:|
| `ModNiceness` | `Niceness` field | +0.621 |
| `ModHumidity` | `WindMax` | +0.598 |
| `ModWind` | `WindMin` (n=150) | +0.496 |
| `ModPrecipitation` | `WindMax` | +0.451 |
| `ModFetch` | `WindGustsSpeedMax` | +0.309 |
| `ModTemperature` | `WindMax` | +0.292 |
| `ModAtmosphere` | `Rain` | +0.149 |

Three readings fall out of this. The `ModNiceness` label is a coarse view of the numeric `Niceness` field
(bucket means 5.47 bad / 6.69 neutral / 9.85 good, values 1 to 10). Humidity, precipitation and wind all
correlate with wind strength rather than with their nominal subject, which says the 25 data sets are ordered
mostly by storminess and the axes ride along. And `ModAtmosphere` correlates with nothing, consistent with it
being a categorical label (pressure, clarity, haze, mist groups) rather than a scale.

The `Rain` field does not follow the top precipitation token. It is nonzero in exactly 7 of the 25 data sets
(`DRIZZLE` 0.4, `Rain` 0.6, `THUNDERSTORM` 1.0, `HURRICANE` 0.8, `Shower` 0.2, `HAIL` 0.3, `CLOUDSCAPE_HURRICANE`
0.8), and read per type through the borrowed set, 8 of the 168 types are nonzero: the 6 untagged storm names
(`DRIZZLE`, `RAIN`, `THUNDERSTORM`, `HURRICANE`, `SHOWER`, `HAIL`) plus `AM_Hurricane` and `AM_Hurricane_Au`.
12 of the 14 types tagged `W_MOD_precip_100_HeavyRain` borrow a data set whose `Rain` is 0, so `Rain > 0` is a
property of the borrowed set, not of the precipitation token.

## The 25 data sets

Each `WeatherTypeDatas` entry is 72 numeric fields borrowed by name. Measured ranges: `WindMin` 0.5 to 26,
`WindMax` 0.9 to 32, `WindGustsChance` 10 to 100, `WindGustsSpeedMax` 0.5 to 20, `CloudType` 0.01 to 0.67,
`Niceness` 1 to 10 (8 distinct), `Fog` binary, `MaxWeatherAccumulation` two values (0 and 0.12). Dead fields
in this build: `Honor` is 0.0 in all 25, `IsBlizzard` 0 in all 25, and `MudWashSpeed` and `OceanWaveWindScale`
are constant (0 and 0.7). The `IsNice` flag is not a threshold on `Niceness`: the values 5 and 6 each appear
with the flag set both ways.

## `WeatherCycles` and `TimeMult`

Two cycles: `SUNNY` with `TimeMult` 2 and `HIGHPRESSURE` with `TimeMult` 1. `TimeMult` is the only numeric
field the block has. Each cycle name is the `@name` of one of the 25 referenced `timecycle_keyframe_data`
documents, and those documents are structurally ordinary (2 regions, 1,142 variables per region, same as the
other 23). Nothing in the archive carries a duration, stage count or keyframe spacing that `TimeMult` could
multiply, so it is a pure runtime scalar; what it scales is a runtime question, but it demonstrably changes
nothing about the cycle's keyframe structure.

## The `Honor*` values

`WeatherMiscData` holds four values: `HonorThresholdForGraph` 0.1, `HonorThresholdForWeatherChange` 0.1,
`HonorDecreaseWaitTime` 30.0, `HonorWeatherChangeTime` 30.0. Read plainly: an honor change has to move at
least 0.1 to show on the graph or to trigger a weather change, and both paths wait 30 units. With the per-type
`Honor` field at 0.0 in all 25 data sets, there is nothing for these thresholds to gate, so the
honor-driven weather path is inert as shipped.

## App entries and keyframe documents

`WeatherAppTypeDatas` has 26 uniquely named entries; the 151 types reference 19 of them. The 7 unreferenced
ones (like `PatchyMist` and `Seafog`) carry empty icons and descriptions, placeholders for app states this
build never selects.

All 25 `TimeCycleName` values match a `timecycle_keyframe_data` document `@name` case-insensitively (21 match
exactly; the four mixed-case families `misty`, `Fog`, `Rain`, `Shower` have uppercase timecycle names). The 27
keyframe documents are those 25 plus two small ones no weather type references: `underwater_deep` (2 regions,
515 variables each) and `sleet` (2 regions, 520 variables each).

## How to reproduce

Carve and dump as above, then count what this note claims: the 151/17 tag split and its family alignment, the
exact-case off-list table, the 7 nonzero-`Rain` data sets and the 8 types that read them, the seven Spearman
figures (average ranks, borrowed family fields), the constant `Honor`/`IsBlizzard` fields, and the two
unreferenced keyframe documents. None of it needs a key and none of it leaves the local copy.
