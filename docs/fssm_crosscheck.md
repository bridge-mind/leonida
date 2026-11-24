# FSSM / RDNA2 cross-check against SCARAB

Status: facts checked 2026-11-21 against `p-portela/SCARAB` (`d304fa2`) and the Leonida
tree at `793b06f`. This note records **facts and measurements, not code**. SCARAB is
GPL-3.0-only; Leonida is GPL-3.0-or-later. No SCARAB code was copied. Any later code
adaptation needs the GPL-3.0 notice and attribution to Pedro Portela / SCARAB.

The corpus used for the measurements is the local Leonida extraction:
**26,503 `.dxbc` containers and 26,503 decompressed `.fssm` blobs**. Only counts and
container fields are reported here; no game data is included. SCARAB's own counts
(25,321 extracted streams, 25,782 parseable blobs, 25,889 payloads) come from a different
title build and are not directly comparable.

The short version:

- The FSSM unwrap is **not the same** as `tools/fssm_unpack.py`. Leonida parses the XDAT
  directory and selects tag `0x0D`; SCARAB scans for `XBCH`. Both are correct on the
  common shape, but they disagree on duplicate entries (below).
- SCARAB's **code start is 24 bytes early**: on the Leonida corpus the code starts at
  `u32[0xC0] + 0x28`, not `+0x10`. The bytes at `+0x10` are a fixed two-record descriptor.
- SCARAB's **"RDNA2 / gfx10.3"** label is not supported by the encoding. The stream decodes
  as **gfx10.1.3 (`gfx1013`)**, the PS5/custom RDNA1.5 target; `gfx1030` is markedly worse.
- SCARAB's **"XBSC ISA"** is the name of the vendor DLL (`xbsc_xs.dll`), not a distinct ISA.
  The extracted files are a 1,542-name AMD mnemonic list plus 856 prose sentences; they
  contain no numeric encodings apart from the standard VOP3 base-plus-offset convention.
- SCARAB's family classifier is sound; its VOP1/VOP2/SOP1 **mnemonic tables and length
  model** are not. Leonida's `llvm-mc` path is the better decoder.
- **Leonida bug found, since fixed:** `xdat_entries` kept only the last of a container's two
  `0x0D` entries. 195/26,503 containers have two distinct `XBCH` payloads. #2479 added
  `xdat_entries_all` and a `payload=` choice; `tests/test_fssm_crosscheck.py` pins both.

## 1. Container and unwrap

| Claim | SCARAB source | Leonida behaviour / measurement | Verdict |
|---|---|---|---|
| Outer file is a DXBC-shaped container; chunk table at `0x1C` (count) / `0x20` (offsets); chunks are `SFI0;ISG1;OSG1;PSV0;HASH;DXIL;PRIV`. | `components/shader-bridge/fssm_lib.py` docstring; `fssm_to_rdna2.py` docstring | 26,503/26,503 have exactly that chunk set; `fssm_unpack.parts()` uses the same table. | **Verified** |
| `DXIL` chunk payload starts with `u32 0x00010066`, `u32 0x61`, `DXIL`, `u32 0x106`, `containerSize`, `partCount`. | `fssm_lib.py` docstring | The first word is the DxilProgramHeader **ProgramVersion**, not a marker: `0x66` PS (12,101), `0x10066` VS (2,236), `0x20066` GS (195), `0x50066` CS (2,689), `0xD0066` MS (9,282). The next word is that shader's program-header **size** (0x61 in SCARAB's sample, not constant). `0x106` is constant, `0x10 == DXIL chunk size - 12`, `0x14 == part count` (3 or 4). | **Corrected** |
| `XDAT` holds the part table; `XBCH` is one of its sections. | `fssm_lib.py` docstring (describes `XDAT`, but the code scans for `XBCH` and never parses the directory) | `tools/fssm_unpack.py` parses `XDAT`: count/header at `+4/+8`, 12-byte entries at `+12`; tags `0x07` (DxilProgramHeader/NOBC), `0x0D` (XBCH), `0x12` (hash). 26,503/26,503 have that tag set. The header word is `0x24` (36) in all 26,503, but 195 containers have `count = 4` and a fourth entry beyond that declared header size; the count-based loop is what reads it. | **Corrected** |
| `XBCH` = fourcc, `u32 version=3`, `u32 compressedSize`, `u32 0`, `u32 uncompressedSize`, `u32 0`, then zlib. | `fssm_lib.py`, `fssm_to_rdna2.py` | 26,503/26,503: version 3; both zero words 0; `compressedSize == entry size - 24`; decompressed length `== uncompressedSize`; zlib magic `78 9c`; result starts `FSSM` v0x101. | **Verified** |
| The unwrap is a full chain `.dxbc -> DXIL chunk -> XBCH -> zlib -> FSSM`. | `fssm_lib.fssm_from_dxbc()` | `fssm_unpack.unpack()` does the same, but selects the `XBCH` by XDAT tag and checks the decompressed length. SCARAB scans for `XBCH` (the extractor additionally requires version 3) and does not check the length. | **Equivalent in result, different in mechanism** |
| A bare `FSSM` blob is accepted as input. | `fssm_to_rdna2.unwrap()` | `fssm_unpack.unpack()` accepts only `.dxbc`; the rest of Leonida (`shader_disasm.py`) accepts a bare `.fssm`. | **Minor input-form gap** |
| `FSSM` magic is asserted after decompression. | `fssm_lib.fssm_from_dxbc()` | `fssm_unpack.unpack()` also rejects a payload that doesn't start with `FSSM` (`ValueError`, counted as a problem, nothing written). All 26,503 are `FSSM`, so there is no corpus failure. | **Verified** |
| **Duplicate XDAT tag `0x0D`.** | `fssm_lib` / `fssm_to_rdna2` pick the **first** version-3 `XBCH` | `xdat_entries` was a dict, so the **last** tag-`0x0D` entry won. 195/26,503 containers (all `$VS_*` VoxelizeGeneric) have **two different** tag-`0x0D` payloads; the two FSSM payloads differ in code size and content. Since #2479 `xdat_entries_all` returns every entry in directory order and `unpack(payload=…)` picks one; `xdat_entries` keeps the last-wins view for old callers. | **Corrected — real Leonida bug, fixed in #2479** |

### The duplicate-XBCH exception (the important one)

`tools/fssm_unpack.py` does:

```python
ents = {}
for k in range(count):
    tag, off, size = ...
    ents[tag] = dxil[off:off + size]
```

For the 195 `$VS_*` containers the directory holds two entries with tag `0x0D` (`count = 4`,
but the header word stays `0x24`, so the fourth entry is beyond the declared header size).
The first payload and the second payload both decompress to `FSSM` v0x101, but their code
differs.
In the measured pair the first is 3,679 bytes and starts with a GS/ES-like prologue
(`s_bfm_b64`, `s_barrier`, no `exp`); the second is 3,551 bytes and has two `exp`
instructions. The 24-byte descriptor at `code_off + 0x10` is identical in the measured
pairs, so they belong to the same container record.

Leonida writes the second payload as `<name>.fssm` by default; `fssm_unpack.py --payload first`
writes the first instead, which is the one SCARAB takes when it scans from the start. The
default is unchanged, so the two projects still extract **different payloads** for these
containers unless that flag is passed. The authoritative mapping (which payload the game's
pipeline uses) is not established here.

Covered by `tests/test_fssm_crosscheck.py::DuplicateXbchTest`:
`test_xdat_entries_all_preserves_both_payloads_in_order` pins the directory order,
`test_xdat_entries_keeps_the_last_payload_per_tag` the compatibility view, and
`test_unpack_selects_first_or_last_payload` the `payload=` choice.

## 2. FSSM header and code range

| Claim | SCARAB source | Leonida behaviour / measurement | Verdict |
|---|---|---|---|
| `0x00 FSSM`, `0x04 u32 0x101`, `0x08 u64 len-16`, `0x10 u64 6`. | `fssm_parse.py`, `docs/fssm-analysis.md` | 26,503/26,503 match all four. | **Verified** |
| `0xC0 u64 code_off`, `0xC8 u64 code_size`, `0xE8 u64 code_end`, `0xF0 u64 meta_ptr`. | `fssm_parse.py` | Same fields. `code_off` is 8-aligned: 11,971 blobs are `0 mod 16`, 14,532 are `8 mod 16`. `code_size` is `0 mod 16` in all 26,503. `meta_ptr` is `0xDEADBEEFDEADBEEF` in 635. | **Verified** |
| `code_end == code_off + code_size` in 99.9%. | `fssm_parse.is_code_consistent()`, `docs/fssm-analysis.md` | 26,486/26,503. The 17 exceptions are not malformed: `code_end` is the `0xDEADBEEFDEADBEEF` "unset" sentinel. `code_off + code_size` is still valid in all 17. SCARAB's `fssm_to_rdna2.code_words()` uses `code_end` as the slice end, so for these 17 it would read from `code_off + 0x10` to EOF, including the metadata and name record. | **Verified claim; exception missing** |
| `code_off + 0x00..0x10` is a 16-byte sub-header, `code_off + 0x10` is raw instructions. | `fssm_parse.py`, `fssm_to_rdna2.py`, `include/shader_bridge.h`, `docs/fssm-analysis.md`, `docs/shader-abi-notes.md` | **Wrong on this corpus.** The code starts at `code_off + 0x28`. `code_off + 0x10..0x28` is a fixed 24-byte block: dwords 0 and 3 have `0xD580` in their high 16 bits in 26,503/26,503; dwords 1 and 4 are `0x03FE0000` or `0x03FF0000`; dwords 2 and 5 vary. LLVM rejects the single word at `+0x10` in 100/100 sampled blobs; the single word at `+0x28` is a valid one-word instruction in 100/100. The family census over all 26,503 gives SOPP 25,715 (all opcode 32 = `s_inst_prefetch`) and SOP1 788 (all opcode 4 = `s_swappc_b64`). | **Corrected** |
| The 16-byte sub-header is `(0,0,0,0)` or `(8,1,0,0)`. | `fssm_to_rdna2.py` docstring, `fssm_parse.py` | Those two values cover 14,965/26,503 (56.5%). Other common values: `(0x2110080,8,0,0)` 8,119, `(1,1,0,0)` 1,794, `(0x2100080,8,0,0)` 767. The tuple is not a two-value enum. | **Incomplete** |
| `0x18`/`0x20` are sentinel-filled slots. | `docs/fssm-analysis.md` | Both are `0xDEADBEEFDEADBEEF` in 26,503/26,503. | **Verified** |
| `0x78` is a `(2,N)` pair; `0x80` is `0x44`/`0x45`. | `docs/fssm-analysis.md` | `0x78` low word is `2` in all; high word is `0` (14,532), `2` (9,282), or `3` (2,689). `0x80` is `0x44` (17,221) or `0x45` (9,282). | **Verified** |
| Code is padded with `0xBF9F0000` (`s_code_end`) up to `code_end`; `code_end` is the code end. | `fssm_parse.py`, `fssm_to_rdna2.py` | The last `s_endpgm` is immediately followed by a run of `s_code_end` words up to `code_end` in 26,486/26,486 valid blobs. The last `s_endpgm` is the real program end; the filler is not code. Leonida stops at the last `s_endpgm`, SCARAB includes the filler. | **Behavioural difference; Leonida's end is cleaner** |
| Multiple programs per blob. | Not stated in SCARAB; `branch_report` walks the whole section | In the code range of the 26,486 blobs with a valid `code_end`: `s_endpgm` count is 1 in 18,558; 2 in 5,799; 3 in 1,535; 4 in 232; 5+ in 362. Leonida's last-`s_endpgm` range covers all of them. | **Missing fact in SCARAB** |
| The post-`code_end` tail is 16 bytes of `0xBF9F0000` filler, a u32 header and a zero word, a 20-byte-record table ended by `0xFFFFFFFF`, then `u32 0, u16 0, u16 len, name`. | `fssm_header.binding_table` | 26,503/26,503 blobs parse. 5,640,704 records (mean 212.8, max 385, 17 empty). Header is 4 in 24,572, 3 in 1,914 and 0 in the 17 `code_end = DEAD` blobs. The tail name is a substring of the container name in all 26,503 cases, exact in 691. At least 40 bytes follow the name and are not parsed. | **Verified; layout corrected** |

### Why `+0x28` matters

SCARAB's extractor emits `payload[code_off + 0x10 : code_end]`. On the Leonida corpus
that is 6 dwords of the fixed descriptor followed by the real program, then 28-42+
`s_code_end` filler dwords. The descriptor is not harmless padding: its first word fails
LLVM in 100/100 sampled blobs, so it changes the first decoded instruction and the
instruction count. SCARAB's observation that its walk is "never lower" than the vendor is
consistent with this: both decoders see the same extra 6 words, and the filler adds more.
The constant 0x18 shift does not invalidate relative branch targets, which is why
`branch_report` still looks healthy.

Leonida's rule is `u32[0xC0] + 0x28` to the last `s_endpgm` (also `docs/design.md`,
"Disassembling shaders"). The 17 `code_end = DEAD` blobs are handled because the last
`s_endpgm` is used instead of `code_end`.

### Initial float mode (SHADER-190)

The 1.0.2.1 scan tested 88 aligned little-endian words in each of the 26,503 indexed FSSM blobs as possible
packed `PGM_RSRC1` registers carrying `FLOAT_MODE`, `DX10_CLAMP` and `IEEE`. It covered 78 word starts at
FSSM offsets `0x00` through `0x134` and 10 at `code_off + 0x00` through `code_off + 0x24`, inclusive.
Local Python scripts walked the exported DXBC `index.json`, opened the matching `fssm/<stem>.fssm` for each row,
and read each candidate with `struct.unpack_from("<I", ...)`. For the standalone check, the scripts intersected
the sets of header offsets containing `0xC0` across all rows.

The VGPR allocation field provides an independent check on that interpretation. Of 2,689 compute shaders, 1,675
had zero reported invalid words and at least one explicit VGPR in the instruction listing. For each of these,
the minimum allocation must cover the highest numbered VGPR, including range endpoints. The tested capacity was
the low six bits of the candidate word plus one, multiplied by 8 for wave32 or 4 for wave64 (FSSM flag at
`0xD4`, bit 5). Graphics stages can share register allocations across merged stages and were excluded from this check.

Of the 88 positions, 63 gave too little capacity for at least one qualifying compute shader. Each of the other 25
exceeded the minimum whole-granule allocation in all 1,675. Over-allocation is possible, and the mode fields could
be stored separately or supplied by pipeline state. The initial float mode remains unresolved.

#### Standalone FLOAT_MODE candidate at +0x104

@465d6a0eac proposed that FSSM stores the eight mode bits in their own word. Across all 26,503 indexed
1.0.2.1 headers, `+0x104` is the only aligned word below `+0x138` that always equals `0x000000C0`.
Reading it as packed RSRC1 gives too few VGPRs in 1,578 of the 1,675 qualifying compute shaders.
A standalone mode field makes no VGPR allocation prediction. The contributor also reports the same value in
52 BVH-builder DLL shaders, outside this replay.

The adjacent words provide two cross-checks. The low byte at `+0x100` matches the stage in every row
(GS 3, VS 4, PS 5, CS 6, MS 7). The VS word is `0x00020004`; the GS word is `0x00020203`.
Only the 2,236 VS have zero in bits 15:8 and match the full `0x000200NN` pattern.
The corresponding byte is 1 for PS, 3 for CS and 6 for MS. The word at `+0x108` is always zero. For every one of the 2,689 CS,
rounding the byte count at `+0x10C` up to 512-byte blocks predicts the independently decoded
`COMPUTE_PGM_RSRC2.LDS_SIZE` at `+0x138[23:15]`, including all 485 nonzero allocations. Five MS also
have a nonzero count. This varying LDS measurement locates stage state near the candidate.

Under the AMD FLOAT_MODE layout, `0xC0` selects round-to-nearest for both precision groups, flushes f32
denormals on input and output, and preserves f16/f64 denormals on both. All 29 [FSPX programs](fspx.md)
have the same mode byte in an inferred packed RSRC1 whose VGPR allocation varies. That supports a plausible
compiler default across formats. The 102,365 reported f32 output modifiers agree with f32 flushing.
The absence of f16 output modifiers leaves f16 preservation open, and explicit rounding updates give no
initial value. IEEE and DX10_CLAMP lie outside the proposed byte.

A fixed header-layout value also fits the census: `0xC0` is the offset of the established `code_off` field.
Whether `+0x104` refers to that field is itself unverified. A loader read mapping `+0x104` to the mode bits,
or an FSSM with the same layout and an independently known different mode correctly predicted there, would
distinguish the readings. SHADER-190 remains open.

## 3. ISA, RDNA2 classification and the "XBSC" notes

| Claim | SCARAB source | Leonida behaviour / measurement | Verdict |
|---|---|---|---|
| The FSSM body is AMD **RDNA2 / gfx10.3** machine code. | `README.md`, `docs/fssm-analysis.md` | The stream decodes as **gfx10.1.3 (`gfx1013`)**, not gfx10.3. On 100 random shaders (208,917 decoded words), LLVM 21 `gfx1013` rejects 729 words (0.35%); `gfx1030` rejects 3,567 (1.7%). A 40-shader sample gave the same 10 rejects for `gfx1013`, `gfx1010` and `gfx1012`. The residue is the known code-embedded constant tables. SCARAB's own ABI note calls its upstream recompiler "PS5 (gfx1013, RDNA2)", which is the right encoding. | **Corrected label** |
| "`xbsc_xs.dll` embeds an ISA reference and the vendor's encoding rules"; 1,542 mnemonics, 856 spec sentences. | `docs/isa/shader-isa-notes.md`, `docs/isa/xbsc-isa-mnemonics.txt`, `docs/isa/xbsc-isa-docs.txt` | The names are standard AMD GCN/RDNA mnemonics (`s_*`, `v_*`, `buffer_*`, `image_*`, `ds_*`), not a proprietary Xbox ISA. The list is a union across generations: it contains gfx9-only `s_cbranch_g_fork`, `s_cbranch_i_fork`, `s_cbranch_join`, `s_set_gpr_idx_*`; gfx10-era `v_dot2c_f32_f16`, `v_dot4c_i32_i8`, `image_msaa_load`, `s_waitcnt_depctr`; and no RDNA3 FP8/BF8 or `s_delay_alu`. It contains **no numeric encodings**. | **Verified as AMD names; corrected as "ISA"** |
| The VOP3 encoding is base-plus-offset: VOP1 `+0x180`, VOP2 `+0x100`; VOPC/VINTRP offsets exist but their values are unproven. | `docs/isa/shader-isa-notes.md` §3 | Standard published AMD convention; LLVM implements it. Leonida's translator consumes the mnemonic text from `llvm-mc`, so it never needs the offset arithmetic. | **Verified; not actionable for Leonida** |
| `family()` classifies VOP1/VOP2/VOPC, SOP1/SOP2/SOPC/SOPP/SOPK, VOP3/VOP3P, VINTRP, DS, FLAT, MUBUF/MTBUF, MIMG, SMEM and EXP by the gfx10 top-bit fields. | `rdna2_classify.py` | Checked against 14 assembled gfx1013 instructions (`v_mov`, `v_add`, `v_cmp`, `v_add_e64`, `v_pk_add_f16`, `v_interp`, `s_mov`, `s_add`, `s_cmp`, `s_movk`, `s_endpgm`, `s_load`, `ds_read`, `exp`): all correct. It is a top-bit family split, not a validity check (it labels the invalid `+0x10` header word `VOP3`); the mnemonic tables are separately wrong. | **Verified** |
| `name_of()` gives the instruction mnemonic for SOPP/VOP1/VOP2/SOP1. | `rdna2_classify.py` | The SOPP table is correct (e.g. `s_endpgm`, `s_branch`, `s_cbranch_scc0`, `s_waitcnt`). The VOP1/VOP2/SOP1 tables are not gfx10: `v_mov_b32` (word `0x7e000301`) is named `V_NOP` (VOP1 reads the vdst field at `[24:17]`; the opcode is `[16:9]`); `v_add_f32` (`0x06000501`) is named `V_SUBREV_F32`; `s_mov_b32` (`0xbe800301`) is named `S_CMOV_B64`; `s_swappc_b64` (`0xbe842104`) is named `S_SETPC_B64`. | **Corrected** |
| The walker is length-aware; base widths plus a literal dword for 1-dword families. | `rdna2_classify.instr_length()` | Base widths are right. Confirmed gaps: `s_setreg_imm32_b32` is 2 dwords but `instr_length()` returns 1; 68 of 100 sampled shaders contain it (150 occurrences), so the walk drifts at the first one. `v_add_f32_e64 v0, v1, 0x12345678` is 3 dwords but VOP3 is always 2. `mimg_nsa()` is a stub and MIMG is always 2 words. `fssm_parse.instructions()` does not use `instr_length()` at all; it uses base widths only. | **Incomplete** |
| Branch targets resolve in-range 12,786/12,789 (SCARAB's corpus). | `fssm_parse.branch_report()`, `docs/fssm-analysis.md` | The SOPP branch opcodes (2, 4-9) and `i + 1 + simm16` are correct for gfx10. The check is self-consistent but cannot detect a constant `+0x10` start shift; Leonida's CFG builder additionally rejects a branch target outside the code range. | **Verified logic; not a correctness proof** |
| The vendor disassembler (`XBSC_DisassembleRawShader`) is the oracle; all 25,321 streams disassemble with no fault. | `docs/isa/shader-isa-notes.md`, `tools/xbsc_probe_notes.md` | Cannot be re-run here (no vendor DLL in this workspace). The claim is plausible; the output header's `asic(CI)` label is not reconciled with gfx1013 and should not be read as a distinct ISA. | **Unknown (not re-run)** |
| The 1,542-name list documents opcodes the Leonida translator lacks. | `docs/isa/xbsc-isa-mnemonics.txt` | It documents names, not encodings. Names present in the list and **not handled** by `rdna2spirv.py`: `v_sat_pk_u8_i16`, `v_screen_partition_4se_b32`, `v_mad_mix_f32`, `v_dot2c_f32_f16`, `v_dot4c_i32_i8`, `v_mul_hi_i32_i24`, `v_mul_hi_u32_u24`, `s_absdiff_i32`, `v_add_i32`, `v_sub_i32`, `image_msaa_load`, `s_ttracedata_imm`, `s_cbranch_g_fork`, `s_cbranch_i_fork`, `s_cbranch_join`, `s_set_gpr_idx_on`, `s_set_gpr_idx_off`, `s_set_gpr_idx_mode`. None of these is in the translated 26,503-shader corpus (otherwise translation would have rejected the shader), so they are future-proofing, not current blockers. | **Names only; not actionable now** |
| SCARAB's `rdna2_sweep` locates the first decoder rejection and decodes VOP3 modifier fields. | `src/rdna2_sweep.cpp` | It is a front-end coverage/boundary diagnostic: opcode coverage, word length, VOP3 `abs`/`op_sel`/`clamp`/`omod`/`neg` legality. It does **not** check register liveness, dataflow, control-flow semantics or output values. `spirv-val` checks the output module, not the input stream; neither one proves the translation. The differential-validation plan on Leonida's in-progress `shader-val` branch (`docs/shader_validation.md`, `tools/cpu_ref.py`) is the stronger class of check. | **Useful triage; not a substitute for `spirv-val` or differential tests** |
| SCARAB's `rdna2_recompile` produces SPIR-V for the corpus. | `src/rdna2_recompile.cpp` | In practice it always builds a **compute** request (`numThreads 64x1x1`, no header, no memory) even when `--stage` is parsed; the `stage` variable is not used by `BuildComputeRequest`. It cannot validate the VS/PS/GS/mesh ABI. | **Incomplete for this corpus** |
| The prior-art note concludes FSSM is inside the DXIL container and is not a direct `rgxe`/`SGD2` descendant; `FSSM`/`0x101`/`0xDEADBEEF` have no public prior art. | `docs/shader-container-prior-art.md` | The nesting is confirmed: `DXBC` precedes `FSSM`, and `FSSM` is the decompressed `XBCH` payload inside the `DXIL` chunk. The lineage and negative prior-art claims were not re-run. | **Nesting verified; lineage unknown** |
| The shader container's binding model is the PSO root signature; the FSSM code is `words[code_off..code_end]`, keeping the 16-byte sub-header. | `docs/shader-abi-notes.md` | The root-signature model is out of scope here. The code slice repeats the `+0x10` error; the ABI note's "PS5 (gfx1013, RDNA2)" is a useful correction to the README's gfx10.3 label. | **Code-range claim corrected; ABI not re-checked** |

## 4. What to adopt, and what not to

Adopt:

1. **Keep `+0x28`.** Leonida already does. The regression test in
   `tests/test_fssm_crosscheck.py` pins it, including the `code_end = DEAD` case.
2. **Preserve duplicate XBCH payloads.** This is the one real Leonida bug found, and #2479
   did it: `xdat_entries_all` keeps every entry for a tag in directory order,
   `unpack(payload="first"|"last")` and the CLI's `--payload` choose between them, and
   `xdat_entries` keeps the last-wins view for old callers.
3. **Treat `0xDEADBEEFDEADBEEF` as an unset `code_end`** (and `meta_ptr`). If a future tool
   reads `0xE8`, it must fall back to `code_off + code_size` or the last `s_endpgm`.
   Leonida's last-`s_endpgm` rule already does this implicitly; document it.
4. **Record the header fields in the FSSM index.** `code_off`, `code_size`, `code_end`,
   `meta_ptr`, `0x78`, `0x80`, and the `code_end` sentinel would make corpus triage and
   future format changes visible. This is an index-only change; it does not touch
   `fssm_unpack.py`'s extraction.
5. **Keep the XDAT tag lookup.** Leonida's `0x0D`/`0x07`/`0x12` selection is more precise than
   SCARAB's scan. Optionally check `XBCH` version 3 and `compressedSize == entry - 24` for
   robustness; both hold in 26,503/26,503.
6. **Qualify "RDNA2".** The GPU is RDNA2, but the shader encoding is gfx10.1.3 (`gfx1013`).
   Do not switch to `gfx1030`; the existing `--mcpu gfx1013` is the measured best target.
7. **Use `llvm-mc` as the disassembler authority.** Do not adopt SCARAB's VOP1/VOP2/SOP1
   name tables or its length model. If a lightweight classifier is kept for statistics,
   fix the field extraction (`VOP1` opcode `[16:9]`, `SOP1` `[15:8]`) and the VOP2/SOP1
   tables, and add `s_setreg_imm32_b32` (2 words) and VOP3/VOP3P literals (3 words).
8. **Adopt the sweep's diagnostic shape, not its conclusion.** When the translator rejects
   a word, reporting the first rejected PC, the raw word, the VOP3 modifier fields and the
   unmasked decode is cheap and useful. The sweep does not replace `spirv-val`, and
   `spirv-val` does not replace a differential test.
9. **Leave the XBSC mnemonic list as a name reference.** It has no encodings; AMD's ISA XML
   or LLVM's AMDGPU tables are the encoding source.

Do not adopt:

- SCARAB's `+0x10` code start, or its `words[code_off..code_end]` slice.
- SCARAB's "RDNA2 / gfx10.3" wording for the encoding.
- SCARAB's VOP1/VOP2/SOP1 mnemonic tables (verified wrong on assembled gfx1013 words).
- SCARAB's claim that the remaining boundary mismatch is only "stream alignment"; the
  length model has named gaps that cause it (`s_setreg_imm32_b32`, present in 68/100
  sampled shaders, VOP3 literals, MIMG NSA).
- SCARAB's scan-for-`XBCH` unwrap in place of Leonida's XDAT tag lookup.

## 5. Licence note

SCARAB is GPL-3.0-only (SPDX headers in every file; repository `LICENSE` is GPL-3.0).
Leonida is GPL-3.0-or-later. The licences are compatible for a combined GPL-3.0 work, but
this note deliberately contains **no SCARAB code**: only measured facts, field layouts and
counts. The synthetic test is original. If any SCARAB implementation is adapted later,
carry the GPL-3.0 notice and attribute Pedro Portela / SCARAB in the file header and the
PR description.

## 6. Unknowns

- We did not run SCARAB's own blobs. The `+0x28` correction and the gfx1013 numbers are
  measured on the Leonida corpus; the shared FSSM v0x101/DXBC shape makes the same layout
  likely, but it is not directly verified against SCARAB's title.
- Which of the two `$VS_*` XBCH payloads the game's pipeline actually uses is not
  established against pipeline bytes. The 24-byte descriptor is the same in both, so
  it does not disambiguate, but the PAL stage descriptors do name two different
  hardware stages for them (`tools/pal_stages.py`): the first payload is stage 2
  (`es`) and the last stage 4 (`vs`), and `scripts/vs_layout_census.py` confirms the
  code matches those stages — the ES payload has no `exp` (it writes the vertex
  outputs to LDS for the geometry stage), takes its lane id from `v_mbcnt` and reads
  `v5`, `v9`-`v11` and `v13`-`v14` on entry, while the VS payload is the legacy non-NGG VS of
  `docs/design.md`. `fssm_unpack.py` defaults to the last payload, so the translated
  one is the VS variant; which payload a GS-bound pipeline selects is still open.
- The meaning of the 24-byte descriptor (two 12-byte records with a `0xD580` tag) is not
  established. It is a fixed header, not code.
- The vendor disassembler's `asic(CI)` label is unexplained; it does not override the
  gfx1013 decode result.
- The post-`code_end` binding table is parsed, but the meaning of its five words and the bytes
  after the name are not established.
- The negative prior-art claim (`FSSM`/`0x101`/`0xDEADBEEF` have no public prior art) was
  not re-run; it is a web-search result, not a format fact.

## Repro

```
python -I tests/test_fssm_crosscheck.py
# 6 tests, all pass (the duplicate-XBCH payloads are pinned; no expected failure)
```

The test uses synthetic `FSSM` and `DXBC` bytes only. No game data is read or written.
