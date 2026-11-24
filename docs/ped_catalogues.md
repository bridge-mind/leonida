# Ped Appearance Catalogues (`0x8dcb4cf5`)

Compiled ped and wardrobe appearance catalogues share one schema rooted at struct `0x8dcb4cf5`, stored in two container framings:

1. **Resource-packed `0DRP` v122 containers** inside `packs.rpf` (per-archetype ped catalogues).
2. **Standalone `PSIN` containers** (e.g. the player/wardrobe catalogue, whose multi-megabyte body mixes deflate and raw 64 KiB chunks inside the archive and is read once extracted as a `.pso` file).

Both framings expose a 1-based block table and use the same internal 64-bit pointer encoding:

```text
block_1based = ptr & 0xfffff
byte_offset  = ptr >> 20
```

Parser and CLI: `tools/ped_catalogues.py` (stdlib only, `python -I`). Synthetic test suite: `tests/test_ped_catalogues.py`.

---

## Container framings

### Resource-packed `0DRP` v122 (`0x7a`)

Uses the standard `0x60`-byte standalone `0DRP`/`0x7a` header validated by `tools/odrp_7a.py` (see [page_directory.md](page_directory.md)):

| Offset | Type | Field | Notes |
|---:|---|---|---|
| `+0x00` | `u64` | zero prefix | Always `0` |
| `+0x08` | `u64` | page-map pointer | `0x50000000`-based virtual pointer |
| `+0x10` | `char[4]` | magic | `b"0DRP"` (`0x50524430`) |
| `+0x14` | `u32` | version | `122` (`0x7a`) |
| `+0x18` | `u64` | reserved | Always `0` |
| `+0x20` | `u32` | `root_block` | 1-based index into block directory (`+0x24` is `0`) |
| `+0x28` | `u64` | struct table ptr | `0x50000000`-based virtual pointer (`ns` x 32 B) |
| `+0x30` | `u64` | enum table ptr | `0` or `0x50000000`-based virtual pointer (`ne` x 24 B) |
| `+0x38` | `u64` | block directory ptr | `0x50000000`-based virtual pointer (`nb` x 16 B) |
| `+0x40` | `u64` | optional table ptr | `0` or `0x50000000`-based virtual pointer |
| `+0x48` | `4 x u16` | `(ns, ne, nb, u4e)` | Struct, enum, and block counts |
| `+0x50` | `u64` | optional ptr | `0` or `0x50000000`-based virtual pointer |
| `+0x58` | `u64` | reserved | Always `0` |

Each 16-byte block directory row at `+0x38` is `(u32 type_hash, u32 byte_size, u64 virtual_ptr)`. Virtual pointers resolve to `virtual_ptr - 0x50000000` within the inflated resource buffer. When a resource spans two contiguous deflate streams, the block directory or a block payload extends past the first stream's end; continuation streams must only be joined while the resource is truncated and must stop before a new `0DRP`/`PSIN` header (a complete resource whose size happens to be a multiple of 64 KiB can sit immediately before another `0DRP` catalogue).

### Standalone `PSIN`

Uses the standard `PSIN` header, `0xaa`-filled 16-byte block table, `0x70` table terminator, `LSCH`/`LENM` schema sections, and `LRSC` `0DRP` v122 summary record (`u32@+0x10` = 1-based root block index), decoded via `tools/psin_data.py` (see [psin.md](psin.md)).

---

## Schema and record layouts

All offsets below were read directly from `LSCH` and verified across every `0x8dcb4cf5` catalogue in `packs.rpf`.

### Root catalogue — `0x8dcb4cf5` (192 bytes)

| Offset | Size | Field hash | Type | Meaning |
|---:|---:|---|---|---|
| `+0x00` | 96 B | `0x05269a03` | `0x50` (`0x60bcb6f3[2]`) | Two 48-byte preset buckets; each holds a `0x5c` array (`0x593bc9c3` at `+0x08`) of `0xf795e8ba` presets |
| `+0x60` | 96 B | `0x9bcf5f90` | `0x50` (`0x4c70a132[2]`) | Two 48-byte secondary buckets; each holds a `0x5c` array (`0x593bc9c3` at `+0x08`) of `0xa9d57751` records |

### Preset record — `0xf795e8ba` (88 bytes)

| Offset | Size | Field hash | Type | Meaning |
|---:|---:|---|---|---|
| `+0x00` | 4 B | `0x65121749` | `0x4a` (`u32`) | `preset_id` (unique within a catalogue) |
| `+0x04` | 4 B | `0x080eec39` | `0x4a` (`u32`) | `parent_id` (`0` = root preset, else parent `preset_id` in the same catalogue) |
| `+0x08` | 16 B | `0xebacf18c` | `0x5c` (`{u32 count, u32 cap, u64 ptr}`) | Array of 184-byte `0x37e52277` drawable/texture state records |
| `+0x18` | 4 B | `0xb53667be` | `0x4a` (`u32`) | Category hash (e.g. `0x378ad10c` head; `0x3817131d`, `0x44925bfd`, `0x9d283797` body skin) |
| `+0x1c` | 4 B | `0x5ecf9331` | `0x4a` (`u32`) | Secondary hash |
| `+0x20` | 4 B | `0xc4d75d79`..`0xa41d353b` | `3 x u8` + `0x60` enum | Preset flags and `0x25529dba` enum byte |
| `+0x24` | 4 B | `0xe13ebaf8` | `0x4a` (`u32`) | Auxiliary hash |
| `+0x28` | 24 B | `0x12f85293` | `0x71` sorted map | Preset tag map: `u64 flag` (`+0x28`), `u64 ptr` (`+0x30`), `u16 count` (`+0x38`), `u16 cap` (`+0x3a`) |
| `+0x40` | 24 B | `0x9a289423` | `0x71` sorted map | `u32` hash -> `array<s32>` map |

Each slot in the `0x12f85293` tag map is **24 bytes** inside a synthetic anonymous map-slot block (`0x100..0x1ff`):

```text
+0x00  u32 key        (tag key hash, e.g. 0x4bfd27dd or 0x1c114e61 for drawable joaat32)
+0x04  u32 pad
+0x08  u32 count      (number of u32 values)
+0x0c  u32 capacity
+0x10  u64 value_ptr  (internal pointer into a type-0x4a block of u32 hashes)
```

Synthetic block IDs (`0x100..0x1ff`) depend on per-file schema emission order. Across `packs.rpf`, `0x12f85293` uses `0x100` in 1,685 `0DRP` catalogues and `0x101` in 1,030 catalogues; hardcoding `0x100` drops 37.9% of catalogues.

### Drawable / texture state record — `0x37e52277` (184 bytes)

| Offset | Size | Field hash | Type | Meaning |
|---:|---:|---|---|---|
| `+0x00` | 4 B | `0x3eaddcd7` | `0x4a` (`u32`) | State ID / slot hash |
| `+0x04` | 4 B | `0x46d26d93` | `0x15` (`u32`) | State flags |
| `+0x08` | 56 B | `0xe7cb9e1b` | `0x05` (`0x147fdab5`) | Embedded drawable/texture + extension-map record (see below) |
| `+0x40` | 4 B | `0x1d52bc91` | `0x4a` (`u32`) | Auxiliary hash |
| `+0x48` | 16 B | `0x83698d93` | `0x5c` (`array<0x4a>`) | Auxiliary hash list |
| `+0x58` | 72 B | `0x35ff19d7`, `0x3b7601de`, `0x7e43b9a9` | `3 x 0x71` maps | Bitset/mask maps |
| `+0xa0` | 24 B | `0x12f85293` | `0x71` sorted map | State-level tag map (`u64 ptr` at `+0xa8`, `u16 count` at `+0xb0`) |

### Embedded state body (`0x147fdab5`, 56 bytes at `0x37e52277 + 0x08`)

| Local offset | `0x37e52277` offset | Field hash | Type | Meaning |
|---:|---:|---|---|---|
| `+0x00` | `+0x08` | `0x41acf8eb` | `0x4f` (`u64`) | Primary drawable `murmur64` (`h0`, MurmurHash3 x64_128 low word, seed 0) |
| `+0x08` | `+0x10` | `0xead733fc` | `0x4f` (`u64`) | Albedo texture `murmur64` |
| `+0x10` | `+0x18` | `0xd689b17f` | `0x4f` (`u64`) | Normal texture `murmur64` |
| `+0x18` | `+0x20` | `0x9d1000d8` | `0x4f` (`u64`) | Fourth 64-bit asset hash (`extra_hash`) |
| `+0x20` | `+0x28` | `0xb70a22c1` | `0x71` sorted map | Extension map: `u64 flag` (`+0x28`), `u64 ptr` (`+0x30`), `u16 count` (`+0x38`), `u16 cap` (`+0x3a`) |

### `cull_drawable` chain (`0x147fdab5` -> `0xb70a22c1` -> `0x46e1a048` -> `0x3d780b74`)

Each slot in the `0xb70a22c1` extension map is **16 bytes** inside a synthetic anonymous block (`0x100..0x1ff` — measured across `0x100`, `0x101`, `0x102`, `0x103` in `packs.rpf`):

```text
+0x00  u32 key         (extension key hash; 0x46e1a048 == joaat("cull_drawable"))
+0x04  u32 pad
+0x08  u64 target_ptr  (0x07 polymorphic struct pointer: block = p & 0xfffff, off = p >> 20)
```

When `key == 0x46e1a048`, `target_ptr` resolves into a block of type `0x3d780b74` (**16-byte** records):

| Offset | Size | Field hash | Type | Meaning |
|---:|---:|---|---|---|
| `+0x00` | 8 B | — | `u64` | Zero header (always `0` in all 43,626 measured records) |
| `+0x08` | 8 B | `0x7deba6f8` | `0x4f` (`u64`) | Alternate culled drawable `murmur64` |

---

## Preset inheritance and resolution algorithm

Within a single catalogue, `0xf795e8ba` presets form a forest keyed by `preset_id` via `parent_id`. Child presets commonly carry texture overrides (e.g. face albedo and normal) while placing the drawable tag (`0x4bfd27dd` or `0x1c114e61`, `joaat32` of the lowercased drawable name) on a parent or grandparent preset with zero texture hashes.

`tools/ped_catalogues.py` resolves each state in deterministic precedence order:

1. **Direct preset drawable tag** (`0x4bfd27dd` / `0x1c114e61` on the preset's `+0x28` tag map, `joaat32`).
2. **Direct state drawable tag** (`0x4bfd27dd` / `0x1c114e61` on the state's `+0xa0` tag map, `joaat32`).
3. **Inherited ancestor drawable tag**: walk `parent_id` within `presets_by_id`, maintaining a `visited` set (cycle guard) and depth cap (`MAX_PARENT_DEPTH = 16`; corpus maximum is 3), stopping at the nearest ancestor defining a drawable tag.
4. **State drawable hash**: direct `murmur64` `drawable_hash` (`h0` at `0x37e52277 + 0x08`).
5. **`cull_drawable` target**: resolved through `0x147fdab5` -> `0xb70a22c1` (`0x46e1a048`) -> `0x3d780b74 + 0x08` (`murmur64`), attached to every resolved state binding and searchable directly.

---

## Verified corpus counts (Xbox package `1.0.2.1`, `packs.rpf`)

Measured read-only with `tools/ped_catalogues.py` (0 parse errors, 0 warnings):

| Metric | Resource-packed `0DRP` (`packs.rpf`) | Standalone `PSIN` (`00011ccb2000.pso`) | Combined |
|---|---:|---:|---:|
| `0x8dcb4cf5` catalogues | 2,715 | 1 | 2,716 |
| `0xf795e8ba` presets | 612,950 | 19,828 | 632,778 |
| Root-bucket (`0x60bcb6f3`) presets | 612,950 (100%) | 19,828 (100%) | 632,778 (100%) |
| Duplicate `preset_id`s within a catalogue | 0 | 0 | 0 |
| Presets with `parent_id != 0` | 334,146 | 11,634 | 345,780 |
| Maximum `parent_id` chain depth | 3 | 3 | 3 |
| Parent cycles / unresolved parents | 0 / 0 | 0 / 0 | 0 / 0 |
| `0x37e52277` states (reached / in blocks) | 878,374 / 878,374 | 17,741 / 17,741 | 896,115 / 896,115 |
| States with non-zero drawable `h0` | 559,139 | 7,227 | 566,366 |
| States with non-zero albedo | 623,350 | 7,642 | 630,992 |
| States with non-zero normal | 598,104 | 7,248 | 605,352 |
| Catalogues with `cull_drawable` | 2,206 (81.25%) | 1 | 2,207 |
| Resolved `0x46e1a048` -> `0x3d780b74` links | 42,882 | 744 | 43,626 |
| `0x3d780b74` block records (all non-zero `murmur64`) | 42,882 (100%) | 744 (100%) | 43,626 (100%) |
| Observed `0x12f85293` tag-map block IDs | `0x100` (1,685), `0x101` (1,030) | `0x100` (1) | `0x100`, `0x101` |
| Observed `0xb70a22c1` ext-map block IDs | `0x100` (1,030), `0x101` (1,411), `0x102` (3), `0x103` (1) | `0x101` (1) | `0x100`..`0x103` |

---

## CLI usage

```powershell
python -I tools/ped_catalogues.py census  <file|dir|archive.rpf> --json
python -I tools/ped_catalogues.py inspect <catalogue.pso|catalogue.0drp> --limit 10
python -I tools/ped_catalogues.py resolve <file|dir|archive.rpf> --name <drawable_name> --cull-only --json
```
