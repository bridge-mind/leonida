# FSPX: the FidelityFX DLL's native compute programs

The package's FSR upscaler, `amd_fidelityfx_upscaler_xbox.dll`, carries 29 compute shaders as `DXBC` containers with
`SFI0`, `ISG1`, `OSG1`, `PSV0`, `RTS0`, `HASH` and `ACCC` parts and no `DXIL` ([d3d12_command_objects.md](d3d12_command_objects.md)
§3.5a). The `ACCC` part is an `FSPX` payload: a small directory around a precompiled RDNA2 program. It is not the
`XBCH`/`FSSM` format of the game's own shaders ([design.md](design.md)), but it holds the same kind of program, and
`tools/fspx_unpack.py` lays it out so `tools/rdna2spirv.py` translates it like a game compute shader.

Everything below was measured on the 1.0.2.1 package's DLL (SHA-256 `faff02f2…a75fa`). "Read from the DLL" means
Ghidra 12.2 output (through REA) for the named function; "measured" means checked on all 29 programs unless a count is
given; anything else is marked as inferred.

## Layout

```
u32  'FSPX'
u32  version          1 in all 29; the DLL does not read it
u32  count            4 in all 29
u32  total size       == the ACCC part size in all 29
count x { u32 kind, u32 offset, u32 size, u32 0 }    offset from the start of the payload
```

| Kind | Size | What it is |
|---|---|---|
| 0 | 28 bytes in all 29 | Dispatch state, see below. Required by the DLL. |
| 1 | 20 bytes in all 29 | Unknown. Optional for the DLL. |
| 6 | 768 to 41,536 bytes | The RDNA2 program, at offset 0x100 in all 29. Required by the DLL. |
| 7 | 152 to 696 bytes | Resource and hash words, see below. Optional for the DLL. |

### How the DLL reads it

The DLL's reader (RVA 0x1a20, read from the DLL) clamps the total to the buffer, then returns:

- 1 for a null or shorter-than-16-byte buffer, 2 when the magic is not `FSPX`,
- 4 when `16 + 16 * count` does not fit the total,
- 5 when kind 0 or kind 6 is missing or empty, else 0.

It skips an entry whose range starts inside the directory or ends past the total, and keeps the last entry of each
kind 0, 1, 6 and 7; other kinds are ignored. `tools/fspx_unpack.py`'s `parse_fspx` follows the same rules and codes.

The pipeline function (RVA 0x4aa0, read from the DLL) accepts either a bare `FSPX` payload or a `DXBC` container whose
`ACCC` part is one. It copies kind 6 into an `XMemAlloc` block (size rounded up to 256 bytes, attributes
`0xa8004000`) and calls device slot 67 with:

| Offset | Field |
|---|---|
| +0x00 | root signature |
| +0x08, +0x10 | kind 0 pointer, size |
| +0x18, +0x20 | kind 1 pointer, size |
| +0x28, +0x30 | the copied kind 6, its size |
| +0x38, +0x40 | kind 7 pointer, size |

The size at `+0x30` is kind 6's own size, not the rounded allocation: the DLL rounds it into a separate register for
`XMemAlloc` and stores the original, which is also the copy's length (read from the DLL's code with `objdump`, `0x5649` to
`0x56d5`). So
the bytes at `+0x28` with that size are exactly kind 6, and their SHA-256 is the key below.

**How the table was pinned** (read from the DLL's code with `objdump`; checked by running the DLL's reader on all 29
payloads):
- **The reader (`0x1a20`).** It zeroes four (pointer, size) pairs, then stores kind 0 in pair 0, kind 1 in pair 1,
  kind 6 in pair 2 and kind 7 in pair 3 (`0x1bcb` to `0x1bfb`). Each pointer points into the payload itself.
- **The container branch (`0x5510` to `0x55ea`).** For a `DXBC` container the pipeline function runs the same loop
  inline over the `ACCC` part and fills the same four places. Unlike the reader, it does not zero the kind-1 and
  kind-7 places first; all 29 payloads have both kinds.
- **The copy into the description (`0x567d` to `0x56d5`).** Pair 0 goes to `+0x08`, pair 1 to `+0x18`, the kind-6
  copy to `+0x28` with pair 2's size at `+0x30`, and pair 3 to `+0x38`.
- **Code and state come from one payload.** Kind 0 and kind 6 of one call therefore always belong to the same
  program, so the run-time lookup by code key and state key (below) has the pairing the build gives it.
- **Run.** A harness mapped the DLL without running it and called its reader on each of the 29 payloads. It built the
  description that way and called slot 67 on the Vulkan device with the 26 programs translated on `main` at
  `aa2988cc`:
  - all 26 resolve to their own entry with their own state, and the other 3 are refused with `E_INVALIDARG`;
  - the same on llvmpipe, the Ryzen 9 7950X3D's integrated GPU and the RX 7900 XTX (Mesa 26.3.1, under Wine).

`d3d12_fspx_compute_test` builds its slot-67 descriptions the same way, from a container's own directory.

**What happens when a program is not translated.** The DLL creates its pipelines one after another as a dispatch
needs them. When one of them has no translation, slot 67 fails, and so does the slot-62 fallback with the same
container. The DLL then records the null pipeline and faults at `0x65ff`, as on a PC device. The proxy logs every
failed slot-67 call with its number (`slot 67 call N failed`), because only the first call of a slot is announced.

The DLL's own messages name that call `DeserializeComputePipelineStateX` and the path "ACO". When the call fails,
the message says it falls back to "NewBE", and the DLL sets the environment variable `FFX_FSR4_XBOX_ACO_ACTIVE` to
`0`. Its next pipeline call is slot 62 (measured in #864), so "NewBE" is inferred to be that path, the one a PC
device refuses (§3.5a).

## Kind 0: dispatch state

Seven u32 words.

| Word | Meaning | How far it is checked |
|---|---|---|
| 0, 1 | Unknown | The same two values in all 29. |
| 2 | 0 | All 29. |
| 3 | `NumThreadsX` (bits 15..0), `NumThreadsY` (31..16) | Measured: equal to the container's `PSV0` thread-group size in all 29. |
| 4 | `NumThreadsZ` (15..0); bit 20 set = wave32 | Z measured against `PSV0` in all 29. Wave size: bit 20 is set in 24 programs and clear in 5, and nothing measured contradicts it (below the table). The translator takes it as the wave size. |
| 5 | `COMPUTE_PGM_RSRC1` | Inferred. `FLOAT_MODE` 0xC0, `DX10_CLAMP` and `MEM_ORDERED` in all 29. With a VGPR granule of 8 for wave32 and 4 for wave64, the highest VGPR the code uses is inside the allocation in all 21 programs that decode with no invalid word, at its last register in several. |
| 6 | `COMPUTE_PGM_RSRC2` | Inferred. `LDS_SIZE` covers the largest `ds_*` offset in all 29. `USER_SGPR` plus the enabled `TGID` SGPRs cover the SGPRs the code reads before writing them in 20 of those 21 (8 or 14 user SGPRs here). In the 21st, a wave64 program, the only SGPRs past them are the carry-out destination of a `v_add_co_ci_u32_e64`, which the check counted as a read. |

`rdna2spirv.py` reads words 3, 4 and 6 from `fspx/<name>.json` where it reads the FSSM header for a game compute
shader: the thread-group size, `RSRC2`, and the wave size from word 4 bit 20 (the `wave32` field).

**Word 4 bit 20 is the wave size.** Over the 29 programs of 1.0.2.1 (`tests/semantic/test_fspx_wave_size.py` has
the rule):

- **Strict markers (measured).** A strict wave64 marker (`rdna2spirv`'s `WAVE64_RE`: `exec_hi`, `v_mbcnt_hi`,
  `*_saveexec_b64` and the 64-bit lane-mask SOP ops) is in 4 of the 5 programs with the bit clear and in none of the
  24 with it set. A wave32 marker (`*_saveexec_b32`, `*_wrexec_b32`) is in 4 programs with the bit set and in none
  with it clear.
- **VGPR granule (measured).** In the 21 programs that decode with no invalid word, the highest VGPR is inside the
  `RSRC1` allocation with the granule the bit gives (8 for wave32, 4 for wave64) in all 21. All 18 of those with the
  bit set use a VGPR past what the wave64 granule would allocate. The 3 with the bit clear would fit either granule.
- **`PSV0` (measured).** Its expected wave lane counts are stated in 5 programs and agree with the bit in all 5 (64 in
  3 programs with it clear, 32 in 2 with it set); the other 24 leave them open (0 and the maximum).
- **Thread groups (measured; the reading is a hypothesis).** The programs with the bit clear have 64 or 256 threads
  per group; 17 with it set have 32, which as one wave64 would leave half its lanes idle.
- **`_b64` ops on `vcc` are not markers (measured counts; the reading is a hypothesis).** The three wave32 programs
  the old heuristic called wave64 write `vcc` from an SGPR pair with `s_mov_b64` (2 to 5 times each) and otherwise use
  `vcc_lo` and `vcc_hi`, including `vcc_lo` as the lane mask of `v_cmp`/`v_cndmask`: `vcc` held as an ordinary pair.
- **The one program with no evidence beyond the bit.** It has the bit clear and no marker either way, and its highest
  VGPR is the last register of the wave64 allocation, so the bit alone decides.

A strict marker against the bit makes the translation refuse the program, as a contradicting FSSM header does. A
program whose kind 0 is too short for word 4 keeps the code heuristic.

## Kind 6: the program

- **Not compressed.** Each program holds exactly one `s_endpgm`, followed by padding (mostly `s_nop 0`).
- **A 36-byte prologue** comes first. It is three 12-byte VOP3 words with opcode 0x180 (bits 31..16 = `0xD580`), plus
  literals.
  - They carry the container's shader hash. Measured: the low halves of words 0 and 3 and the literals in words 2 and
    5 equal the 16 bytes after kind 7's `0x43` word, which are the digest in the container's `HASH` part, in all 29.
  - `llvm-mc` rejects them. Measured on RDNA2 hardware (`gfx1031`, AMD Radeon RX 6700 XT, driver 26.11.2, through
    `tools/fspx_prologue_probe.py` / `tools/hw_oracle`): in wave32 and wave64 across 5 synthetic 36-byte prologues
    (50 positive checks and 8 negative controls), executing the raw words leaves VGPRs (`v0`, `v1`, `v8`..`v15`),
    SGPRs (`s[0:1]`, `s4`..`s15`), `M0`, `VCC`, `EXEC` (including inactive lanes across 64 threads) and `SCC` (both 0
    and 1) unchanged, and execution resumes cleanly at `+36` bytes. So they are hardware no-ops on RDNA2.
  - `tools/fspx_unpack.py` disassembles from the end of the prologue.
- **Target.** The programs use `v_dot4_i32_i8` (1,458 times) and `v_dot4c_i32_i8` (25,813 times), which fits the
  provider's name (`FSR4_Int8`). `llvm-mc` (LLVM 23.1.5) decodes them for `gfx1011`, `gfx1012`, `gfx1030`, `gfx1031`,
  `gfx1033`, `gfx1034` and `gfx1036`, but not for `gfx1010`, `gfx1013` (the target `shader_disasm.py` uses for the
  game) or `gfx1100`. Measured on one 10,432-byte program, prologue included: 13 invalid words with the first group,
  1,276 with `gfx1010` and `gfx1013`, 1,290 with `gfx1100`.
  - `fspx_unpack.py --disasm` defaults to `gfx1030`.
- **What llvm-mc still rejects.** With `gfx1030`, 21 of the 29 decode with no invalid word. The other 89 words, in 8
  programs, are all VOP3P opcode 0x16 with `op_sel` 2 and `op_sel_hi` 5. That is the opcode of the 1,458
  `v_dot4_i32_i8` that `llvm-mc` does decode, all of which have `op_sel` 0.
  - Measured on RDNA2 hardware (gfx1036, through `tools/hw_oracle`, the instruction given as two raw words): with
    `op_sel` 2 and `op_sel_hi` 5 the result is the dot product with the two 16-bit halves of the second source
    exchanged, in all 5 operand sets (4 of them give another word under the plain rule). So `op_sel` selects halves
    here as it does for the packed 16-bit ops. The words are in `tests/semantic/test_alu_dot4_i8.py`.

## Kind 7: records

- **Format.** It starts with the word `0x403` in all 29, and the words `0x41`, `0x42` and `0x43` appear in all 29.
  That they tag records is inferred from the layout around them.
- **Records `0x41` and `0x42`.** The word 3 after each `0x41` or `0x42` is a register number: the 121 such numbers
  in the 29 programs are all among the register numbers the container's `RTS0` declares (measured; the check ignores
  the register class, so it is weaker than it looks). That the records list the resources the program uses is
  inferred: in one program, checked by hand, a declared SRV does not appear.
- **Record `0x43`.** The 16 bytes after it are the container's shader hash (its `HASH` part), the value embedded in
  the program's prologue.
- **Not read by the translator,** which does not read the game's FSSM binding table either. The programs load their
  descriptors with `s_load` and `s_buffer_load`, as the game's shaders do; that this is all they need is inferred.

## Build and run-time lookup

**Build.** `make_game.py`'s `shaders` stage, after the game's own shaders, runs `tools/fspx_unpack.py --disasm` on
the package's `amd_fidelityfx_upscaler_xbox.dll` into `<work>/shaders/fspx/` and `tools/rdna2spirv.py` on that
disassembly into `<work>/shaders/spirv/fspx/`, with the same `--wave64-emulate` and fallback flags. Each index entry
of a program without an FSSM blob gets two keys (`tools/shader_meta.py`):

- `fspx_code_sha256`, the SHA-256 of `fspx/<name>.code`: the whole kind-6 entry, prologue and padding included;
- `fspx_state_sha256`, the SHA-256 of `fspx/<name>.state`: the kind-0 entry (dispatch state) as it is.

The same kind-6 bytes can come with different kind-0 entries, for example two thread-group sizes, and then translate
to different modules. Such a pair is allowed: the run time tells them apart by the state key. Two entries with the
same code key and the same state key must have identical modules; when they do not, the step fails and names both
programs, and leaves no `index.json` for the runtime to load. In the 1.0.2.1 DLL the 29 kind-6 programs are 29
different byte strings, so every code key has one entry (measured).

The step is redone as a whole when the DLL, the unpacking or translating tools, `llvm-mc`, the wave64 mode or the
flags change (`fspx_build.json`, written last); it never makes the game's shaders translate again. A package without
the DLL skips it.

**Run time.** `ShaderLibrary::load(dir)` (`runtime/src/shader_library.h`) reads `<dir>/index.json` and then, when it
is there, `<dir>/fspx/index.json`, whose entries are named `fspx/<name>`. A missing or unreadable `fspx/index.json`
leaves the main index loaded. Entries are found by `fssm_sha256` or by `fspx_code_sha256`, in two separate tables.
`compute_program()` decides what a compute PSO's bytecode is looked up by:

| Bytecode | Key | Who passes it |
|---|---|---|
| `FSSM` blob, bare or in a `DXIL` part | `fssm_sha256` of the blob | the game (unchanged) |
| `DXBC` container whose `ACCC` part is an FSPX payload | `fspx_code_sha256` of its kind-6 bytes, then only the entries whose `fspx_state_sha256` is that of its kind-0 entry | the DLL, slot 62 |
| bare FSPX payload | the same two keys | the DLL's pipeline function accepts one (none measured at a slot) |
| bare kind-6 program: 4-byte multiple, at least 36 bytes, the prologue's words 0, 3 and 6 with bits 31..16 = `0xD580` | `fspx_code_sha256` of the bytes as they are; kind 0 is not passed | the slot-67 description at `+0x28`, size at `+0x30` |
| anything else | none: `E_INVALIDARG`, as before | |

With kind 0 (the first two rows), an input whose state key matches no entry for its code key has no program and gets
`E_INVALIDARG`, like any untranslated shader. Without it (a bare kind-6 program), the entries for the code key are
used when they all have the same module; when their modules differ, the program is ambiguous without its kind-0
state, so the call is refused with `E_INVALIDARG` and a log line saying so, instead of taking one of them.

The payload is parsed with the DLL reader's rules above (sizes and offsets checked in 64-bit arithmetic against the
clamped total), so a malformed directory, or one without kind 0 or kind 6, gives no program and `E_INVALIDARG`. The
pipeline's code base, which a translated shader reads its embedded constants through, is the kind-6 bytes: the
translator's code offsets are into `fspx/<name>.code`.

**Slot 67.** The proxy maps slot 67's description onto `CreateComputePipelineState` (#1714). When kind 0 is in the
description, which the DLL always passes (its reader refuses a payload without one), the proxy builds one FSPX
payload from the four pointer and size pairs (`leonida::fspx_payload_from_parts`) and passes that as `CS`, the
bare-payload row above. Two programs with the same code and different dispatch state therefore stay apart at slot 67
too. Each pair is checked before it is read: a pointer without a size or a size without a pointer, a size over
16 MiB, or a missing kind 6 refuses the call with `E_INVALIDARG`. Only a description without kind 0 still passes the
kind-6 bytes alone (the bare-program row).

**Slot 11.** After either path the DLL calls the pipeline's console slot 11. The Vulkan device's pipeline answers
with a blob holding the dword 0, so the DLL creates no compute scratch buffer; why 0 is justified is in
[d3d12_command_objects.md](d3d12_command_objects.md) §3.5a.

Not done here: using kind 0 at run time beyond what the translated module already carries (its thread-group size
and `RSRC2`).

## Translation status

With `tools/fspx_unpack.py --disasm` and `tools/rdna2spirv.py` on the 29 programs (`main` at `aa2988cc` with the
`v_fma_mix*` literal repair in [design.md](design.md), "Printed encodings"):
**Wave size.** On `main` at `aa2988cc`, `make_game.py`'s FSPX step translates 26 of the 29, and each now gets the
wave size of its dispatch state. Before, the code heuristic gave 4 of the 26 the other size: one wave64 program
was translated as wave32, and three wave32 programs as wave64, because they write `vcc` with `s_mov_b64`. The
translated and `spirv-val`-valid count is the same (26) either way. The breakdown below was measured when 17 translated.

With `tools/fspx_unpack.py --disasm` and `tools/rdna2spirv.py` on the 29 programs:

- **27 translate and pass `spirv-val`**, 11 of the 13 programs that use dot4 among them (design.md, "Signed-byte
  dot products"). Without the repair, 26: the `v_fma_mix_f32` it is for is refused.
- **The other 2** (both use dot4) stop at a reachable `v_dot4_i32_i8` VOP3P word with `op_sel` 2 that `llvm-mc` does
  not decode (the VOP3P words above).

These counts predate SHADER-813. A program whose reachable code falls through into those undecodable VOP3P words used
to translate with the path cut at the gap, so its later stores were missing from a module that passed `spirv-val`;
it is now refused (`fall-through into undecodable words`): programs 04, 05, 06, 09, 12 and 15 of package 1.0.2.1
(found by @aslan2142). Recount by @CuteBooty with LLVM 23.1.5 (gfx1030 listings, `spirv-val --target-env vulkan1.2`):
26 of 29 translated and passed before the change, **20 of 29** after it; the 20 kept give byte-identical SPIR-V, and
8 of the 9 refused stop at a fall-through into undecodable words.

## Unknown

- Kind 0 words 0 and 1.
- Kind 1.
- The exact layout of the kind-7 records.
- What the prologue does on the GPU.
- Whether the game itself ships FSPX programs. Its shaders measured so far are `FSSM`.
