# Save files

The game's save containers, measured from one save set (two story saves, a profile record and settings) taken from a
retail copy, with every file already plain on disk. `tools/saves.py` reads, rebuilds and imports them using only
the standard library. Nothing here decrypts anything. The one high-entropy layer (`SLAYER1`, below) is described
and left alone.

Confidence: **confirmed** means it holds on every sample or is proven by a byte-identical rebuild. **likely** means
it's consistent with the data. **hypothesis** means it's a guess.

Never commit or attach real saves, extracted blocks or `save.pretty.json`. They're the player's data, and the block
contents are game data.

## Overview

| Container | Holds | Where it's seen |
|---|---|---|
| `JSAV` v3 | The story save: an index plus one JSON document | One file per story slot (manual and autosave) |
| `ASAV` v1 | Generic header + payload | Profile record (kind 3), settings (kind 2) |
| `SLAYER1` | A named wrapper around a packed `ASAV` (kind 1) | Next to each story save: the same save packed, with its slot name and title |

## JSAV: story save (confirmed)

```
0x00  char[4]  "JSAV"
0x04  u32      version            = 3
0x08  u32      n_code_blocks
0x0C  u32      n_script_blocks
0x10  entry[n_code + n_script], 20 bytes each:
        u32 hash      block id, also the block's inner key
        u32 flags     1 on every sample
        u32 reserved  0
        u32 offset    absolute file offset of the block's JSON text
        u32 size      byte length of the block's JSON text
      zero padding to a 16-byte boundary
body  UTF-8 JSON, CRLF between blocks:
        {"CodeSaveBlocks":[\r\n  b0,\r\n b1 ... bN],\r\n"ScriptSaveBlocks":[\r\n  ... bM]\r\n}
      zero padding to a 16-byte boundary
```

Both samples have 38 code and 44 script blocks. `tools/saves.py verify` rebuilds them byte for byte from the parsed
blocks, which confirms the framing, padding and offsets. The JSAV carries no checksum of its own.

Block conventions:
- Each block is `{"$0x<group>":{"$0x<hash>":{...}}}`, with one group hash for code blocks and one for script blocks. The inner hash equals the index entry's hash. **confirmed**
- Keys are 32-bit hashes written as `"$0x%08x"`: the **case-sensitive** joaat (seed 0, no lower-casing) of the engine or script name, the same rule as the member hashes in [world_names.md](world_names.md). **confirmed**, see [Key names](#key-names).
- Reserved, unhashed keys **(confirmed)**:
  - `$x $y $z $w` vectors and quaternions
  - `$content` + `$data` typed arrays (`int_array`, `int64_array`, `char_array`, `float_array`, `short_array`)
  - `$data:"0x…"` a hash value stored as a string
  - `$key`/`$value` map entries
  - `$type`, `$bits`
  - `$v` version (settings)
- Floats are printed `%f` (six decimals) and ints in decimal. A JSON library re-serialises them differently, so `pack` copies block text verbatim. **confirmed**
- Diffing an autosave against a manual save taken about a minute later shows 62 of 82 blocks identical. The other 20 are positions, counters and stats, so blocks are independent units. **confirmed**

## ASAV: header + payload

```
0x000 char[4]  "ASAV"
0x004 u32      version = 1
0x008 u32      kind: 1 story (packed), 2 settings, 3 profile               likely
0x00C u32      1 plain, 3 packed                                           hypothesis: storage mode
0x010 u32      0 plain, 0x180 packed                                       hypothesis
0x014 u32      payload offset = 0x170                                      confirmed
0x018 u32      0 plain, stored size packed                                 likely
0x01C u32      file size (plain), unpacked size (packed)                   confirmed plain / likely packed
0x020 u64      unix time, seconds, UTC                                     confirmed
0x028 u16[128] display title, UTF-16LE                                     confirmed
0x128 char[56] slot name, ASCII                                            confirmed
0x160 u32      checksum: algorithm unknown
0x164 u32      0
0x168 u64      file size again (plain)
0x170          payload
```

The checksum at 0x160 matched none of these:
- CRC-32, CRC-32C, Adler-32, FNV-1/1a, joaat (seeds 0, `0x3fac7125`, `0xffffffff`), byte sum, MD5, SHA-1 or SHA-256 (truncated to 32 bits)
- tried over every 4-aligned sub-range, with the field zeroed and raw, in both byte orders

It's left as **open**. Copy the header verbatim whenever the payload is unchanged.

Payloads: settings (kind 2) are hash-keyed JSON text with tabs and CRLF **(confirmed)**. The profile record (kind 3)
is 1,644 bytes (first 112 bytes non-zero, starting `u32 10, u32 3`, followed by 1,532 bytes of zero padding) **(confirmed)**.

## Profile record (kind 3, confirmed)

```
0x000 u32         version = 10                                         confirmed
0x004 u32         kind = 3                                             confirmed
0x008 u32         subversion / format tag                              likely
0x00C u32         flags                                                likely
0x010 u64         account / profile identifier                         likely
0x018 u32[22]     profile settings & attributes (first 112 bytes non-zero)
0x070 byte[1532]  zero padding (reserved area)                         confirmed
```

The profile record (kind 3) has a fixed 1,644-byte payload. Only the first 112 bytes (0x70) contain active profile
fields, starting with format version 10 and kind 3; the remaining 1,532 bytes (0x5FC) are zero padding reserved for
future profile expansion and cloud sync.

`tools/saves.py info` reports the profile version, active bytes and verifies the zero-padding boundary;
`tools/saves.py extract` writes `profile.json`.

## SLAYER1: packed copy (not decoded)

```
0x00  char[8]  "SLAYER1\0"
0x08  u32      version = 1
0x0C  u32      name length n
0x10  char[n]  name, no terminator (so the rest may be unaligned)
...   u64      payload size
...   ASAV, kind 1: header as above (title, slot name, time), then 16 high-entropy bytes at +0x170
      and a high-entropy body at +0x180 (~30% of the JSAV's size)
```

The body is compressed and/or encrypted. Following AGENTS.md, it's recorded here and not decoded. The plain JSAV next
to it holds the same save. The wrapper's header is the only place the autosave's slot name and title appear, so
`saves.py` reads it for those.

## Importing into the runtime

The runtime's offline `XGameSaveImpl` (`runtime/shims/xgameruntime_gamesave.cpp`,
[runtime_imports.md](runtime_imports.md)) keeps `XGameSave` data in
`LEONIDA_SAVE_DIR\local-user\<container>\<blob>`, defaulting to `%LOCALAPPDATA%\Leonida\saves`, with names
%-encoded. `python -I tools/saves.py import <save folder>... [--out DIR] [--map OLD=CONTAINER/BLOB] [--with-packed]`
writes payloads in that layout with the same encoding, plus `import-manifest.json` (source, size, SHA-256).
`--with-packed` also imports `chives.dat`, the packed `SLAYER1` copy of the same save (below), as a blob of its own;
without it only the plain payload is imported.

Checked earlier with the shim of the open #279, which used the same layout: four imported blobs (2 KB to 1.2 MB)
read back byte-identical through its `XGameSaveReadBlobData`.

**VERIFY:** the title's real container and blob names. The default is container = slot name and blob = payload file
name, and `--map` overrides it. Confirm or fix them from the executable's `XGameSaveCreateContainer` and
`XGameSaveSubmitBlobWrite` calls.

## Key names

The hash was found from structure, before trying a dictionary: the code and script group keys are the joaat of names
derived from the clear-text JSON wrappers around them, with case kept. Measured on the sample set (two story saves
plus settings, **1,059 distinct keys**), with a dictionary of 1.14 million identifier-like strings harvested by
`tools/save_keys.py` from the raw bytes and zlib streams of four of the game's own data archives (`common`, `script`,
`data`, `appdata0`):

| Method | Keys resolved | Expected by chance |
|---|---|---|
| joaat, case-sensitive, strings as found | 128 | 0.3 |
| joaat, lower-cased | 45 (1 not found case-sensitively) | 0.3 |
| FNV-1, FNV-1a, MurmurHash3-32 | 0 | 0.3 |
| CRC-32, MurmurHash3-64 low or high half | 1 each | 0.3 |
| + one- and two-word Pascal/camel/snake names from the 3,000 most frequent words (`--compose 3000`) | +134 | about 7 |
| + block ids only (82 targets): one or two PascalCase words; script blocks share a fixed name prefix | 33 of 82 (14 code, 19 script) | 0.6 |
| **Total** | **278 of 1,059 (26%)**, 1 ambiguous | at most about 8 |

Every family except case-sensitive joaat stays at chance level. Code block ids are bare PascalCase nouns, script block
ids add a common prefix, code fields are PascalCase and many settings fields are camelCase **(likely)**. The other ~780
names aren't in the data archives: they're code-only or script-local names, so the executable's strings should
resolve most of them. The resolved names are game data and stay local; only counts belong here.

`save_keys.py resolve` prints these counts for any dictionary. `--out` writes the hash -> name table for local use.

## Tool

`verify` reports on every supplied file. Its exit status is 2 if any file could not be read or parsed,
otherwise 1 if any rebuild differed, otherwise 0.

```powershell
python -I tools/saves.py info    <file or folder>...
python -I tools/saves.py verify  <JSAV>...              # parse every block, rebuild, compare byte for byte
python -I tools/saves.py extract <JSAV|ASAV> <outdir>   # blocks/*.json, index.json, save.pretty.json
python -I tools/saves.py pack    <outdir> <JSAV>        # after editing blocks/*.json
python -I tools/saves.py diff    <JSAV a> <JSAV b>
python -I tools/saves.py import  <save folder>... [--out DIR] [--map OLD=CONTAINER/BLOB] [--with-packed] [--dry-run] [--force]
python -I tools/save_keys.py collect <save file or extract folder>...           # distinct $0x… keys
python -I tools/save_keys.py resolve <saves>... --dict <files>... [--compose N] [--out FILE]
```

`tests/test_saves.py` and `tests/test_save_keys.py` cover all of it on synthetic containers and names built in the
tests.
