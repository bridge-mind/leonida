# The pipeline library in `shaders.rpf`

**Status: measured** on the 1.0.2.1 package. `shaders.rpf` holds one more file behind its 26,503 shader entries:
a library of precompiled pipelines, with 25,471 graphics and 2,693 compute pipeline records and the 21,522 native
RDNA2 programs they run. `tools/shader_carve.py` reads only the raw-deflate shader entries, so until now nothing
read this file. No program, blob or record is reproduced here; the tables hold counts, sizes and offsets.

## 1. Where it is

The archive's data area has two parts. The first 274 MB are the shader entries `shader_carve.py` reads (name,
kind, sizes, raw deflate). The rest, from file offset `0x10579000`, is 4,205 zlib streams: 4,204 inflate to
65,536 bytes and the last to 27,436. Concatenated, they form one 275,540,780-byte file. That is the 33.4%
stream coverage [world_census.md](world_census.md) measured for `shaders.rpf`. The gaps between the streams are 0
to 12 bytes.

`tools/shader_library.py` finds the first stream that inflates to the header below and appends streams until it
has the size the header gives. It fails if the sizes don't match exactly.

**No shaders elsewhere.** The other 74 RPF9 and TTPF archives of the package (214 GB: the root archives except
`movies.rpf`, and everything under `pack/` and `data/`) were scanned raw and through every zlib stream (about 420 GB
inflated) for `DXBC`, `XBCH`, `FSSM`, `FSPX` and the shader entries' `_Wrapped` names. There are no shader entries,
and none of the 148 `XBCH`/`FSSM` hits is a record: no `XBCH` inflates to an `FSSM` blob, and no `FSSM` has a code
offset with an `s_endpgm` after it. The same checks accept the 52 records of `XboxBvhBuilder.dll` and 200 of 200
carved FSSM blobs. No `DXBC` hit has a valid header. So the game's shader programs are the shader entries, this
library and the two DLLs' containers ([embedded_shaders.md](embedded_shaders.md)), as far as raw and zlib data go.

## 2. Layout

All values little-endian.

| offset | size | field |
|---|---:|---|
| 0x00 | 4 | `'P' 'S' 00 09` |
| 0x04 | 32 | name, NUL-padded: `0x65F41EDD` |
| 0x24 | 36 | version string, NUL-padded: `930.048.020` |
| 0x48 | 8 | hash |
| 0x50 | 4 × 4 | graphics pipelines G = 25,471, compute pipelines C = 2,693, blobs B = 54,576, programs P = 21,522 |
| 0x60 | 4 | blob section size, 31,120,020 |
| 0x64 | 4 | code section size, 240,894,464 |
| 0x80 | G × 0x70 | graphics pipeline records |
| | C × 24 | compute pipeline records |
| | B × 8 | blob table: (offset, size) into the blob section |
| | P × 8 | program table: (offset, size) into the code section; every offset is 256-byte aligned |
| | | blob section, then code section |

The sections add up to the file size exactly: 0x80 + 25,471 × 0x70 + 2,693 × 24 + 54,576 × 8 + 21,522 × 8 +
31,120,020 + 240,894,464 = 275,540,780. Every blob and program entry lies inside its section. The blob entries
are contiguous; the program entries are in increasing offset order.

**Graphics record** (0x70 bytes): u64 hash, u32 state blob, u32 pipeline blob, 40 bytes not decoded, then seven
(u32 program, u32 stage blob) pairs. 0xFFFFFFFF means no program. The slot gives the stage. Slot 0 is the pixel
shader, slot 1 the vertex shader and slot 2 the mesh shader. The source containers named in §3 have those stages
in every use §3 links: 24,090 of the 24,111 slot-0 uses, 13,388 of 13,409 slot-1 and all 12,062 slot-2. No linked
use names a source of another stage. Slot 6 is used by 2 records with one program and no stage blob; it is not
identified. Slots 3 to 5 are never used.

| stages | pipelines |
|---|---:|
| VS + PS | 13,383 |
| MS + PS | 10,726 |
| MS only | 1,336 |
| VS only | 24 |
| VS + PS + slot 6 | 2 |

The state blob is 388 to 472 bytes (5,680 distinct). The pipeline blob holds the input layout, with element
names such as `POSITION`, `COLOR`, `TEXCOORD`, `NORMAL`, `TANGENT`, `BLENDWEIGHT` and `BLENDINDICES` for vertex
pipelines (80 bytes, with no elements, for mesh pipelines).

**Compute record** (24 bytes): u64 hash, u32 state blob, u32 blob, u32 program, u32 layout blob. The four parts
have the sizes and shapes of the four [FSPX](fspx.md) kinds:

| record field | size | FSPX kind |
|---|---|---|
| state blob | 28 bytes in all 2,693 | kind 0, dispatch state: thread-group size in words 3-4, wave32 flag (word 4 bit 20), RSRC1, RSRC2 |
| blob | 20 bytes in all 2,693 | kind 1 |
| program | | kind 6 |
| layout blob | 56 to 10,552 bytes | kind 7 |

1,529 of the 2,684 compute programs set the wave32 flag. 90 set RSRC2 `TG_SIZE_EN` (bit 10), which none of the
2,689 carved compute shaders do.

**Programs.** 52 to 239,464 bytes (median 3,968; 238,829,764 in total), padded to 256 bytes with `s_code_end`.
21,463 of them start with the 36-byte prologue FSPX programs have, three 12-byte VOP3 opcode-0x180 words with
literals (`fspx_unpack.code_range` finds it); 59 have none. The code runs from there to the last 4-byte-aligned
`s_endpgm`, as for FSSM blobs. 579 of the first 3,000 pixel programs have more than one. Ending at the first one,
as for FSPX, cut off code their branches reach: 2,281 "branch target outside code" and 1,662 "branch into data" over
all stages.
Each program belongs to exactly one stage: 9,996 PS, 7,498 MS, 2,684 CS, 1,343 VS and the one slot-6 program.

## 3. Where the programs come from

A graphics stage blob and a compute layout blob share one layout. Word 0 is the kind (2 for a graphics stage, 3
for compute). Then comes a head of 16 bytes (kind 2) or 24 bytes (kind 3), whose last word is tagged `0x40`. Then
come 32-byte records tagged `0x41` or `0x42`, then a word `0x43`, a 16-byte digest and 12 zero bytes. All 20,318
distinct stage and layout blobs follow this. In FSPX, the 16 bytes after kind 7's `0x43` word are the container's
`HASH` digest ([fspx.md](fspx.md)). Here they are too: **21,509 of the 21,522 programs** have the `HASH` digest of
one of the carved containers, and every linked program has its container's stage (2,680 CS, 9,991 PS, 7,498 MS,
1,340 VS; the carve index's stage comes from the container's `DXIL` program header). They come from 6,522
distinct carved shaders, and a source has 1 to 257 programs.

None of the programs is byte-identical to its source's FSSM code. Only 47 distinct code bodies, from
`s_inst_prefetch` to the first `s_endpgm`, occur in both. So the library holds **pipeline builds**: the same
shaders compiled again with their pipeline's state. The 13 programs whose digest names no carved container (4 CS,
5 PS, 3 VS, the slot-6 program) are the only ones with no source in the shader entries.

## 4. Tool

```sh
python -I tools/shader_library.py <package>/shaders.rpf --out <work>/library --disasm --carve <work>/shaders/dxbc
python -I tools/rdna2spirv.py <work>/library/disasm --out <work>/library/spirv
```

* It writes `fspx/<name>.code` and `fspx/<name>.json`, the layout `fspx_unpack.py` writes and `rdna2spirv.py`
  reads (`native_blob`, `cs_dispatch_state`). It also writes `index.json` with each program's stage, and
  `library.json` with the header and both pipeline tables, stage to program name. Programs are named
  `lib_<stage>_<index>`.
* `--disasm` writes `disasm/<name>.s` for the code after the prologue, as `fspx_unpack.py --disasm` does.
* `--carve` names each program's source shaders. It copies the first source container to `<name>.dxbc`, because
  the translator reads the stage's signatures and `PSV0` from there. Pipeline compilation keeps the D3D12 stage
  interface.
* `--save` writes the reassembled library. A saved file is also accepted as input.
* `--stages` limits the output to some stages.

The output is game data and must stay in your work folder. `tests/test_shader_library.py` builds libraries and
archives by hand and covers the header, both tables, the size checks, the search in an archive and the digest
walk. It also checks the `--carve` copy and the translator's reads of the compute sidecar.

## 5. Translation

With `--carve` and `rdna2spirv.py` at `main` `aa2988cc` (llvm-mc 23, `gfx1013`), **17,955 of the 21,522 programs
translate and pass `spirv-val`**:

| stage | programs | pass `spirv-val` |
|---|---:|---:|
| VS | 1,343 | 1,333 (99.3%) |
| MS | 7,498 | 7,059 (94.1%) |
| PS | 9,996 | 7,719 (77.2%) |
| CS | 2,684 | 1,844 (68.7%) |
| slot 6 | 1 | 0 (stage not identified) |

What stops the rest, by the translator's message:

* **2,533 programs: "reachable instruction with normalized encoding".** llvm-mc rejects scalar register tuples that
  run into `vcc` (SGPRs 106 and 107): `s_load_dwordx8`/`x4`/`x16` writing `s[100:107]`, `s[104:107]` or
  `s[92:107]`, and image instructions whose resource is `s[100:107]`. In the compute programs alone that is 37,952
  SMEM, 22,248 MIMG and 2,298 MUBUF words. The same words with the register field one tuple lower decode; with
  `gfx1030` they don't either. The FSSM shaders don't use these tuples; the pipeline builds use them often. A decoder
  fallback for these encodings would cover most of the remaining programs. The translator already reads
  `[s100,...,vcc_lo,vcc_hi]` lists (design.md *Scalar/buffer memory*).
* **443: an instruction llvm-mc prints with another encoding than the blob's** (`v_fma_mix*`, `v_interp_p2_f16`):
  the existing `check_printed_encodings` refusal.
* **90 CS: "CS thread-group-size SGPR"**: RSRC2 `TG_SIZE_EN`, which no FSSM compute shader sets.
* **68 translate but fail `spirv-val`**, mostly "requires a previous definition".
* Fewer than 30 each: `ds_append`, `s_bitreplicate_b64_b32` and `s_flbit_i32_b64` on a lane mask, `v_cmp` and mask
  destinations, 25 branches into data, 10 undecodable words that can't be re-anchored, and a few `vcc`/`ttmp`
  operands.
* Programs with no source container: a PS or MS one has no signatures and is `Unsupported` (the reads now raise
  that instead of `FileNotFoundError`, which stopped the whole run). A VS that fetches no inputs translates without
  one.

None of the programs has been run on a GPU.

## 6. What this changes

* **Pipeline pairing is in the package.** [d3d12_vulkan.md](d3d12_vulkan.md) question 14 asks whether a PSO pairs
  two code-base stages, and says the pairing is in the executable. The graphics records give it for 25,471
  pipelines: each names its VS or MS and its PS. Each source can be found through §3.
* **A second compiled form of 6,522 shaders.** The pipeline builds differ from the FSSM code. Which form the game
  runs for a pipeline is not measured here (that needs the executable). Either way, a translated pipeline build can
  be compared with the translated FSSM shader of the same pipeline.
* **Not decoded.** The graphics state blob, the 40 bytes in each graphics record, the compute kind-1 blob, the
  records inside the stage blobs, and slot 6.
