# Mission state tables in the script archives

`story000_script.rpf` and `script.rpf` are mostly not scripts in the sense of an instruction stream. The
containers that carry mission structure are **state machines**: arrays of 32-byte records keyed by a joaat
hash of a state name, each record pointing at a pool element that carries world transforms and an
archetype hash. This page records the layout, the vocabulary that resolved, and the shape of the flows.
Counts and offsets only — no payload, and only the small name excerpt needed to explain the key column.

## Where they are

| | |
|---|---|
| containers | `PSIN`-tagged, 2,309 of them across the two archives |
| record chunks | 8,986 |
| records | 40,804 |
| distinct record hashes | 6,065 |
| files with a record array | 2,035 |

Note that `PSIN` here is the same chunk family the reader meets elsewhere; the mission arrays live in the
`LSNS`-style payload side of those containers, not in the `PSIN` descriptor side.

## Record layout

32 bytes:

```
u32 key │ u32 0 │ u32 colB │ u32 0 │ u32 (u16 kind, u16 offset BE) │ u32 band │ u32 (u16 n, u16 n) │ u32 0
```

Two details cost a false start and are worth stating:

- **`band` is not zero.** It is an ascending band counter (`0,0,1,2,2,…`). A validator that requires
  `field[5] == 0` misclassifies the real tables as opaque blobs.
- **`offset` is a byte offset, big-endian in its half**, and it is **file-absolute for most records**:
  2,087 of 2,207 entity-bearing records resolve that way, 429 resolve through the "next even-tagged pool
  chunk" rule, which only fits one table family. The proof of byte-ness: consecutive deltas equal
  `count * 4`, and values pack back to back from 0.

## The pool element

64 bytes, 16 dwords:

```
u32 archetype hash │ 0 │ 0 │ 0 │ f32 pos.xyz │ f32 pos.xyz (dup) │ rot.a,b,c (dup) │ u32 hash │ u32 linkHash │ 0 │ 0
```

The 16-dword stride is a histogram peak (319 windows against 25/25 for neighbours), and the duplicated
archetype hash plus the duplicated position components are what identify the shape. Positions are true
world coordinates in RAGE space, Z up.

## The key column

Keys are **joaat of the state name, lower-cased first**: joaat here folds case, which is not true of every
implementation of it. Measured on 340 ground-truth pairs: a folding implementation reproduces **340/340**,
a case-sensitive one only **91/340** (329 reference names contain an uppercase byte). Any name lookup must
fold before hashing, or it will invent phantom names.

Resolved keys fall in four provenance tiers, kept separate because they are not equally strong:

| tier | count | how it was resolved |
|---|---:|---|
| document-proven | 22 | the name appears literally in the package's own text metadata |
| register-measured | 178 | found by grammar search, in-register, discounting ≈3.6 expected chance collisions |
| earlier register | 114 | same, from a weaker pass |
| positional | 4 | still unnamed, but distinguishable by position in the machine |

Coverage: 167 → 485 of 6,065 distinct hashes (2.75 % → 8.00 %); record keys 5,846 → 8,314 of 40,804
(14.33 % → 20.38 %); real (non-structural) states named **53.5 %** (2,675 of 4,996). 26.1 % of real states
are still hex, and 65 % of records sit in structural states rather than gameplay states.

## Shape of the flows

The corpus is highly templated: **113 identical-state-set clusters cover 1,676 of 2,035 files (82.4 %)**.
Naming one cluster therefore names dozens of files. Repeated clusters seen so far include
`g1798–g1804`, `g2209–g2212`, `g0986–g0995`.

Flows read as ordered states with an entity count per state. Representative shapes (names where resolved,
hex otherwise; `<…>` marks a positional label):

```
g0082/83   guardaggro → startRobbery → fleeRobbery → robberyCombat → disrupt → INTRO
g1599      stealvehicle → delivervehicle
g0554      fleestore → selfEscape
g1774–79   stealboat → dropoff → boatchase → deliver
g1096–1126 Approach → Leave → <pre-HOPOUT> → HOPOUT → INTRO
```

**135 initiation→terminal shaped jobs** were listed; only 3 are fully bounded end to end.

Phase tagging over the named states (primary match): setup 958, exit 400, action 344, cutscene 130,
aftermath 97, law reaction 29. Law reaction being the rarest phase is a property of the corpus, not of the
tagging.

## What a flow does and does not say

A flow gives **ordering, states and entities**. It carries no conditions, objectives, or durations: there
are no opcodes here, and no jump table. Two consequences for a runtime:

1. State order in the file is not necessarily play order.
2. The remaining unnamed hashes still have positional meaning — e.g. one is the single looping beat
   between two cutscenes, another is the "get the car into position" beat before a vehicle exit — so a
   trace is interpretable even where the name is missing.

## Mission identity

The package's own text metadata carries a 145-entry mission table. Two of those ids now have a
reproducible binding to one of these tables, so most outlines are still unattributed. Joining ids to
tables is an open problem; a plain joaat of the ids does not appear in the program data (0 of 290
variants in 170.9 MB), so the join is not by name hash.

## Binding a mission id to its tables

This binding does not join on the id. It uses a **character** as the bridge: the mission's own text names
a person, a name dictionary turns that name into a 32-bit hash, and the same hash appears as the
`linkHash` of pool elements in a state table.

1. **Start from a name the mission text uses.** `COM_KBOS_NEIGHB` is the replay-table id for the mission
   publicly catalogued as **Pipework** (Mandy's rent). The text table at `data.rpf` stream
   `0x19957A0` contains the line "You brought the cops to Mandy's house" and the line "Bert died".
   The stream carries no mission id; the id-to-text step rests on the public name and the cast, not an
   archive key.
2. **Resolve the character and its schedule.** `common.rpf` stream `0x4C270` lists
   `KeyLento/Perschars/Peds/Unique/Mandy` and `.../Bert` under the pack `PSP_KeyLento_perschars_peds`.
   SixNG's name dictionary, which is lower-case joaat, resolves the pack to `0x865F0DA9`, Mandy to
   `0x03A885E6` and Bert to `0xE34ACBC8`.
3. **Scan the PSINs for the link hash.** In the 64-byte element above, `linkHash` is the dword at byte
   `+0x34`, not the archetype dword. Inflating the five `story000_script.rpf` streams at `0x3FE4000`,
   `0x3FE8000`, `0x3FEB000`, `0x3FEE000` and `0x3FF0000` and searching for those two hashes finds 28
   elements: 15 Mandy and 13 Bert. A scan of all 2,309 PSIN streams finds those 28 hits in exactly these
   five files.
4. **Read the element.** Its start is `linkHash - 52`. The first dword is the archetype/state hash
   described above. It does not resolve in the name dictionary, and the same character takes three
   values, so it is an archetype/state hash rather than a character model name.

| stream | Mandy elements | Mandy archetype | Bert elements | Bert archetype |
|---|---:|---|---:|---|
| `0x3FE4000` | 5 | `0x18BB2550` | 5 | `0xA5A1E67D` |
| `0x3FE8000` | 3 | `0x73E83886` | 3 | `0x780A05A5` |
| `0x3FEB000` | 3 | `0x73E83886` | 4 | `0x780A05A5` |
| `0x3FEE000` | 3 | `0x73E83886` | 0 | - |
| `0x3FF0000` | 1 | `0x57DDFFC2` | 1 | `0xCBA48304` |

Mandy has three distinct archetype values in this set and Bert three; none resolves in the name
dictionary.

The five tables are bound to `COM_KBOS_NEIGHB` through two people the mission's own text names, both
present as link hashes in the same five streams. It is a binding by cast, not by id: the public
`Pipework` name is the outside link between the id and the text, and a mission whose cast is not in the
dictionary still has no bridge.

It does **not** turn `0x18BB2550`, `0x73E83886`, `0x57DDFFC2`, `0xA5A1E67D`, `0x780A05A5` or
`0xCBA48304` into model names. None resolves in the same dictionary, and the first dword changes per
state; it is an archetype/state hash, not a Mandy or Bert drawable. The table that would name those
archetypes is not in the readable archives.

This is **one more binding, not a general solution**. The other 143 ids in the mission table remain
unattributed unless a character, schedule or other stable name gives them a bridge; joining ids to
tables is still an open problem.

## Reading these tables

`tools/mission_tables.py` reads the 32-byte record tables and the 64-byte pool elements they point
at, from inflated PSIN streams or a directory of carved `.bin` streams:

```
python -I tools/mission_tables.py <file-or-dir> [--json] [--limit N]
```

A lone path is the PSIN file: a file without the `PSIN` magic is one `ERROR: <path>: not a PSIN container`
line and exit 2, while a folder is the place for the bare carved streams and is scanned as before. A PSIN
file that holds no records is still a clean run.

It prints per-file record, chunk, state and entity counts, plus each resolved entity's archetype
hash, link hash, position and rotation. Pointers are scored in three encodings: **file-absolute**,
**linked pool** (`pool_block_base + (offset >> 4)`), and **block-indexed** (`kind` is a 1-based
block number and the byte offset is `(band << 12) | (poolOffset >> 4)`). A slot that does not match
the 64-byte position/rotation layout is reported as unresolved or skipped, not guessed. The tests
are synthetic (`python -I tests/test_mission_tables.py`) and need no game files.
