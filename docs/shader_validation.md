# Differential shader validation

This note describes a value-level check for the RDNA2 → SPIR-V translator; it is not a claim that
the corpus is correct. The CPU differential oracle in `tools/shader_diff.py` is implemented and
tested (`tests/test_shader_diff.py`): it compares the ISA model in `tools/cpu_ref.py` against
translated SPIR-V run in `tests/semantic/spirv_mini_interp.py` over seeded straight-line programs.
Run it with `python -I tools/shader_diff.py --programs 200 --seed 7` (`--self-test` checks the
harness itself). The GPU-side runners are described below: `tools/valu_fuzz.py` (16-bit and memory ops), `tools/lds_fuzz.py`, `tools/crosslane_fuzz.py`, `tools/cf_diff.py` and `tools/ab_control_flow.py` (corpus pixel shaders in both control-flow forms).

## Why `spirv-val` is not enough

`spirv-val` checks that a module is structurally valid SPIR-V for a target environment. It does not
check that the module computes what the RDNA2 program computes. Those are different questions, and
this repository already has cases where only the second question had a useful answer:

- All 26,503 shaders translate and pass `spirv-val` (Vulkan 1.2) at `43d4703`, after `dcd8ecf` fixed
  the ID-bound regression (#881); the intervening `dedd75c` run was 26,501, and `793b06f` before
  #223's LLVM 23 source-field fix was 25,032. Yet the SDWA source/destination selects were silently
  ignored until 2026-11-20: validation passed for 3,826 of the 13,926 shaders translated at the time
  and produced wrong values (`docs/design.md`, "SDWA").
- The compressed-export bug was found by a GPU test, not by validation: LLVM prints `v4, v4, v5, v5`
  for a compressed pair, and the translated shader exported `(r, g, r, g)`.
- `v_cvt_pkrtz_f16_f32` with round-to-nearest overflowed a finite value to infinity; the GPU smoke
  test found it, `spirv-val` could not.
- The other project's execution harness found a read-after-store coherence bug: a shader that stored
  a texel and loaded it back got stale data because the read and write used different bindings. That
  is a well-formed, validator-clean pair of instructions with the wrong observable result.

A validator answers "is this a legal module?" A differential test answers "does this module produce
the same bits as the original program for the same inputs?" Only the second one can find a wrong
instruction, a wrong operand order, a missing modifier, or a wrong rounding mode.

## What "differential" means here

Two independent computations start from the same architectural state:

1. **Reference.** A CPU interpreter executes the original RDNA2 instructions and records the
   registers (and, later, memory) the program writes.
2. **System under test.** The translator turns the same listing into SPIR-V; the existing Vulkan test
   tools dispatch it on a real GPU and read back the same registers, buffers or images.

The comparison is bit-for-bit on every value the test declares observable. A mismatch is reported
with the first differing lane/register, both bit patterns, the random seed, and (in the future
runner) the RDNA2 instruction that last wrote the value. The translator is the path under test; the
CPU model and the SPIR-V interpreter are the two independent oracles in the shipped CPU runner. The
translator → `spirv-as`/`spirv-val` → Vulkan tools path is what the future GPU oracle will run, not
a reference to be trusted. A hand-written SPIR-V twin of a test program is a useful third control: it
checks the harness and driver without involving the translator, so a harness failure is not blamed on
translation.

Assembling needs no external `spirv-as` binary: `tools/spirv_asm.py` encodes SPIR-V assembly text into binary
words in process, as a module (`assemble(text, target_env="vulkan1.2")`) or from the command line
(`python -I tools/spirv_asm.py in.spvasm -o out.spv`, with `--target-env` choosing the environment and
`vulkan1.2` the default), using the Khronos SPIRV-Headers grammar kept in `tools/spirv_asm_tables.py`.
`tools/rdna2spirv.py --assembler python` takes that path instead of the `spirv-as` subprocess and its
temporary `.spvasm` file; `spirv-val` stays the external validator. `tests/test_spirv_asm.py` and
`tests/test_spirv_asm_tables.py` cover the assembler and the tables.

The CPU model is deliberately not a second translator. It interprets the listing's mnemonics and
operands; it never consumes SPIR-V. That independence is what makes a disagreement meaningful.

## The two sides

### CPU interpreter (`tools/cpu_ref.py`, prototype)

- Input is a `.s` listing in the format produced by `tools/shader_disasm.py` and by
  `runtime/test/shaders/build.py`: a header line, then tab-indented instructions with an optional
  `; encoding: [...]` comment per line. `tools/shader_ir.py` parses the listing and treats line 1 as
  the header, so a raw `llvm-mc` dump has to be given the same header `build.py` writes.
- State is an SGPR file (32-bit values), a VGPR file (one 32-bit value per lane), VCC, EXEC, SCC and
  a program counter. Float registers keep raw bits; only float instructions reinterpret them.
  Uninitialized registers read as zero, matching the translator's deterministic register arrays.
- One lane or 64 lanes. Scalar instructions execute once; vector instructions execute for lanes whose
  EXEC bit is set, and inactive lanes keep their old value. EXEC and VCC are 64-bit masks even in
  one-lane mode, so scalar reads of `exec_lo`/`exec_hi` stay meaningful.
- The implemented subset is intentionally small: `s_mov_b32`, `v_mov_b32`, `v_add_f32`, `v_sub_f32`,
  `v_mul_f32`, `v_fma_f32`, `v_cndmask_b32`, `v_and_b32`, `v_or_b32`, `v_xor_b32`,
  `v_lshlrev_b32`, `v_lshrrev_b32`, `v_ashrrev_i32`, `s_add_u32`, `s_lshl_b32`, `s_lshr_b32`,
  `s_endpgm`. Anything else raises `Unsupported` with the mnemonic and PC. It never guesses an
  instruction's meaning.
- f32 add/sub/mul produce correctly rounded binary32 results. FMA is evaluated exactly with rational
  arithmetic and rounded once to binary32, so it cannot inherit a double-rounding error from a host
  `float`.
- NaNs are canonicalized to `0x7fc00000`; the comparison layer should treat all NaNs as equal by
  default because Vulkan does not promise a NaN payload.
- Denormals are preserved. A GPU that flushes f32 denormals will disagree on the denormal cases, and
  the harness must report that as an environment mismatch, not a translator bug.

### GPU side (existing tools, future driver)

The repository already has the pieces: `runtime/test/shaders/build.py` assembles `.asm` with
`llvm-mc`, `tools/rdna2spirv.py` translates the resulting `.s`, and the `runtime/test/vk_*_test.cpp`
programs dispatch SPIR-V on Vulkan and read buffers/images back. A GPU-side differential runner would:

1. assemble the `.asm` to a `.s` listing with `llvm-mc -show-encoding`;
2. run the CPU model on that listing with the test's inputs;
3. translate the same listing, `spirv-as` + `spirv-val`, then dispatch;
4. read back the declared outputs and compare raw bits;
5. print the first mismatch and the seed; exit nonzero on any mismatch.

Keeping one listing for both sides matters: if the CPU model and the translator parse different text,
the test measures the parser, not the translation.

## Building test programs

Hand-written RDNA2 assembly is the source of truth. Each test is an `.asm` file whose header comment
states the inputs, the observable outputs and the expected bits, matching the existing convention in
`runtime/test/shaders/*.asm`. The CPU expected values are computed by the interpreter and pinned in
the test table; the GPU must produce the same bits. Pinning the values catches a regression in the
CPU model as well as in the translator.

`llvm-mc` is already a prerequisite for the shader pipeline. The test build is the same one
`build.py` uses:

```
llvm-mc -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding test.asm
```

A minimal case, with the expected value in the header comment like the existing GPU samples:

```
; f32_add_nan  hand-written test; expected v2 = 0x7fc00000
v_mov_b32 v0, 0x7f800000      ; +inf
v_mov_b32 v1, 0xff800000      ; -inf
v_add_f32 v2, v0, v1
s_endpgm
```

The prototype can check the CPU side of that case directly:

```
python -I tools/cpu_ref.py <listing.s> --parse-only
python -I tools/cpu_ref.py <listing.s> --json
python -I tests/test_cpu_ref.py
```

The scalar ALU has its own differential fuzzer (FUZZ-4), `tools/salu_diff.py`: a separate reference for 87
SOP1/SOP2/SOPC/SOPK opcodes with SCC kept explicit, seeded programs, the translated SPIR-V run in
`tests/semantic/spirv_mini_interp.py`, and failing programs shrunk to the instructions that matter.
64-bit mask ops are left out by default (the translator models them as wave masks), and `s_bfe_*` control
words stay inside offset + width <= 32 unless `--bfe-any`. Its docstring lists the known divergences.

`--branches` puts SCC into control flow, one lane: after an SCC producer comes either a chain of 1..3 consumers
(`s_cselect_b32`, `s_addc_u32`, `s_subb_u32`) or `s_cbranch_scc0`/`scc1` forward over the next 1..3 scalar
instructions (no loops or nesting), and half the instructions are SCC producers. The listing carries
`; encoding:` placeholders of each instruction's size, which places the branch targets. It is a flag so that the
default seeds keep their programs; translator oracle only.

```
python -I tools/salu_diff.py --programs 2000 --seed 1
python -I tools/salu_diff.py --branches --programs 2000 --seed 1
python -I tests/test_salu_diff.py
```

The `_sdwa` VALU forms have one too, `tools/sdwa_diff.py`: a separate reference for 92 VOP1/VOP2/VOPC opcodes in an
integer and an f16 family (source selects, `sext()`, `dst_sel` with `dst_unused` PAD/SEXT/PRESERVE, SGPR and
inline-constant sources, `clamp`, f16 neg/abs), seeded one-lane programs that chain through v1..v7 and VCC, the same
interpreter, shrinking and exit codes as `salu_diff.py`. Forms where the translator is known to differ
(`KNOWN_DIVERGENCES` in its docstring, which also lists what is modelled and skipped) are left out unless `--known`.
`--oracle hw` runs the programs on an RDNA1/RDNA2 GPU only (GFX11 has no SDWA); every rule of the reference was
checked that way on an RDNA2 iGPU (gfx1036), including two that differ from the non-SDWA forms: a 16-bit result
with `dst_sel:DWORD` zeroes the high half, and a float inline constant in an f16 op is narrowed by `src_sel` as the
f16 value.

```
python -I tools/sdwa_diff.py --programs 2000 --seed 1
python -I tools/sdwa_diff.py --oracle hw --programs 500 --seed 1
python -I tests/test_sdwa_diff.py
```

### Hardware oracle (optional)

`--oracle hw` also runs the same programs natively on an AMD GPU and compares the hardware's three SGPRs and SCC
with the reference (`--oracle translator,hw` does both; `hw` alone leaves the translator out and then also
runs the 64-bit mask ops, which the hardware reads as plain integers). It tests the reference model, not the
translator. No tool is needed unless you pass `--oracle hw`; when one is missing the run prints
`hardware oracle unavailable: <what>` and exits 77, so CI is unaffected.

```
python -I tools/salu_diff.py --oracle hw --programs 5000 --seed 1
python -I tools/hw_oracle/runner.py        # checks the setup: prints the device, or what is missing (exit 77)
```

Setup on Windows, about 800 MB and no admin rights:

- **Driver.** The AMD Windows driver ships the HIP runtime as `amdhip64_7.dll` in System32; nothing else is
  loaded from the SDK. `LEONIDA_HIP_DLL` overrides the path. Drivers from 2024 ship `amdhip64_6.dll` instead and
  refuse code object v6 (LLVM 23's default) with error 209 (`hipErrorNoBinaryForGpu`) or 303 (`hipErrorSharedObjectInitFailed`);
  the oracle then assembles as v5 (seen with driver 32.0.11020.1002 on gfx1103 and 31.0.24002.92 on gfx1036).
- **true16.** LLVM 23 wants `v8.l`, not `v8`, for 16-bit operands on GFX11. The oracle asks the assembler once and
  uses the spelling it takes, for the ops its tools run: the two f16/f32 conversions and `f16_trans_sweep.py`'s
  seven. Another f16 op needs adding to `asm.true16_spelling`.
- **Compiler.** AMD clang with the amdgcn backend and `ld.lld`. The stock LLVM from winget has no amdgcn
  target. The ROCm wheel has one:
  `py -m venv .venv` then `.venv\Scripts\python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm-sdk-core==10.0.0"`;
  `clang.exe` ends up in `.venv\Lib\site-packages\_rocm_sdk_core\lib\llvm\bin`. A `.venv` in the working
  directory or repository is found by itself, and so is the wheels' `amdclang` on PATH: that launcher has no
  `ld.lld` beside it, so the oracle uses the `clang.exe` in the folder it reports as `InstalledDir`. Otherwise set
  `LEONIDA_AMD_CLANG` to that `clang.exe`.
- **Linux** has a launcher of its own, without ROCm: see the next list. `hw_oracle.c` is the Windows one (it loads
  the driver's DLL).
- **Host C compiler.** `tools/hw_oracle/hw_oracle.c` is built once into the temp folder by MinGW gcc (on PATH or in
  `C:\msys64`), else by MSVC: a `cl` on PATH (a Developer shell), else the Visual Studio that `vswhere` finds, through
  its `vcvars64.bat`. The Visual Studio the runtime needs is therefore enough. `LEONIDA_HOST_CC` overrides the choice
  (gcc or `cl.exe`). By hand: `gcc -O1 tools/hw_oracle/hw_oracle.c -o hw_oracle.exe` or `cl /O1 tools\hw_oracle\hw_oracle.c`.
- `LEONIDA_HW_ARCH` overrides the detected architecture (`amdgpu-arch`, else `gfx1150`). GFX10 spellings
  (`s_load_dwordx2`, `global_store_dwordx4`) are used for `gfx10xx`, so the same kernels assemble for an RDNA2 card
  (checked by assembling for gfx1030 with clang and gfx1013 with `llvm-mc`, and run on gfx1036 on Linux).
  `gfx12xx` (RDNA4) uses the GFX11 spellings. An architecture outside gfx10, gfx11 and gfx12 is `unavailable`
  (exit 77), like a missing tool.

Setup on Linux, without ROCm. `tools/hw_oracle/hw_oracle_drm.c` submits each kernel as a raw compute dispatch
through the kernel driver (`libdrm_amdgpu`), so nothing sits between the instructions and the GPU:

- **GPU.** GFX10 only, which is RDNA1 and RDNA2 (compute engine version 10); the launcher answers anything else
  with exit 77. Name the render node, for example `LEONIDA_HW_DRM_NODE=/dev/dri/renderD129`
  (`ls -l /dev/dri/by-path` shows which node is which PCI device). There is no default device, so a raw command
  stream never goes to a GPU by accident.
- **Not the GPU you are looking at.** A raw stream bypasses the Vulkan driver, and a kernel that hangs makes the
  kernel driver reset that GPU. A node with a display connected is refused unless
  `LEONIDA_HW_DRM_ALLOW_DISPLAY=1`. An integrated GPU next to a discrete card is the convenient case.
- **Tools.** `clang` with the amdgcn target (the distribution's LLVM has it; `LEONIDA_AMD_CLANG` overrides), `gcc`
  and the libdrm headers. The launcher is built once into the temp folder. By hand:
  `gcc -O1 tools/hw_oracle/hw_oracle_drm.c -o hw_oracle_drm $(pkg-config --cflags --libs libdrm_amdgpu)`.
- **Access.** Your user must be able to open the render node (the `render` or `video` group on most
  distributions).
- **The command stream** is libdrm's own compute dispatch test for GFX10 (`amdgpu_test_dispatch_memset`), with what
  belongs to the kernel put in: its entry point, its group size, and what its kernel descriptor says.
  - `COMPUTE_PGM_RSRC1`, `COMPUTE_PGM_RSRC2` and the wave size are read out of the descriptor in the assembled
    object, so the `.amdhsa_*` directives take effect as they do under the HIP runtime: register counts, float
    rounding and denormal modes, IEEE mode, DX10 clamp. A test checks that an f32 rounding mode set there changes
    a sum on the hardware.
  - The descriptor's group segment becomes `COMPUTE_PGM_RSRC2.LDS_SIZE` (bits 15..23, 512-byte units). The
    assembler leaves that field 0, so the launcher programs it from the group segment, as the HIP runtime does;
    a kernel with LDS reads back what it wrote there (`tests/test_hw_oracle.py`, measured on gfx1036).
  - A descriptor that asks for something the launcher does not provide (scratch memory, more than the kernarg
    pointer in the user SGPRs, more than 65,536 bytes of LDS) is refused instead of being run differently.
  - The output buffer takes up to 16,384 words, as the Windows launcher does, so a 64x64 `image_fuzz` case (16,384
    words) runs here too.
  - Every submission gets command memory of its own. While one buffer was rewritten in place, the second kernel of
    a process ran the first one's commands, every time; `tests/test_hw_oracle.py` runs 300 distinct kernels to
    catch that.

How it works: each program becomes one kernel (`tools/hw_oracle/asm.py`: load the output pointer, run the program,
store `v[8:11]`), 500 kernels per code object. The code object needs an `.amdgpu_metadata` block; without it the HIP
runtime fails the load with error 209 (`hipErrorNoBinaryForGpu`), which looks like a wrong GPU. The launcher starts
one workgroup of one thread per kernel and pre-fills the buffer with `0xDEAD0000`, so a kernel that did not write
shows up as a mismatch. A program the assembler rejects is counted as a skip, not a pass.

What it can and cannot say:

- It says what **that GPU** does. A Radeon 880M is gfx1150 (RDNA3.5), not RDNA2: a match supports the reference, and
  a difference is first checked against the RDNA2 ISA text before it is called a reference bug. Only an RDNA2
  card settles RDNA2.
- Skipped on gfx11, and listed in the run's skip counts: `s_ff0_i32_b32` and `s_ff0_i32_b64` (GFX11 has no
  find-first-zero), and programs with two different 32-bit literals in one instruction (not encodable). Renamed
  opcodes (`s_andn2_*` to `s_and_not1_*`, `s_ff1_*` to `s_ctz_*`, `s_flbit_i32_b32` to `s_clz_i32_u32`, ...) are
  accepted by LLVM under their old names with the same behaviour and are run.
- Skipped on gfx12 as well: the twelve SOPK compares `s_cmpk_{eq,lg,gt,ge,lt,le}_{i32,u32}`, which AMD clang
  assembles for gfx1150 and rejects for gfx1201. They are about a third of the programs there (seed 1: 870 of 2000
  skipped in all, 1130 run).
- Only the scalar unit is observed, on one lane with EXEC = 1; nothing about wave64 or the other stages.

Result of the first runs (Radeon 880M, driver 26.11.2, AMD clang 23.0.0git): about 40,000 seeded programs
(1 to 10 instructions each, all opcodes in the table, seeds 1 to 3, 11, 21, 31, 41, 100, 200, with and without
`--bfe-any` and `--inline64`) gave 0 mismatches apart from one finding. `s_bfe_i32` with `offset + width > 32`:
the pseudocode taken literally (the old reference) sign-extends from bit `width - 1` of a zero-filled value, but
the hardware stops the field at bit 31 (`width = min(width, 32 - offset)`) and sign-extends from there. 4,000 random
control words, including bits 23 to 31 of the control word (ignored), fit that rule exactly. The translator already
did this (0 of 600 differ), so the reference was the one that was wrong, and it now follows the measured rule. The
rule rests on that gfx1150 measurement (4000 control words). LLVM's `constantFoldBFE` does the same for the VOP3
`v_bfe` nodes, but with a 5-bit width, so it is only analogous.

Result on RDNA2 (the integrated GPU of a Ryzen 9 7950X3D, Raphael, gfx1036; the Linux launcher, Linux 7.3.5,
libdrm 2.4.134, clang 23.1.5 from the distribution): 70,000 seeded programs, 0 mismatches.

| Run | Programs | Match | Mismatch | Skipped (two literals) |
|---|---|---|---|---|
| seeds 1 to 8 | 40,000 | 39,439 | 0 | 561 |
| seeds 1 to 4, `--bfe-any` | 20,000 | 19,727 | 0 | 273 |
| seeds 1 and 2, `--inline64` | 10,000 | 9,855 | 0 | 145 |

So the reference, the `s_bfe_i32` rule above included, holds on RDNA2 as measured, and GFX10's `s_ff0_i32_b32` and
`s_ff0_i32_b64`, which gfx1150 cannot run, are covered. As a control, a reference whose `s_min_i32` sets SCC for
equal operands too (`<=` for `<`) gave 2 mismatches in 5,000 programs, minimized to that instruction.

RDNA4 (Radeon RX 9070, gfx1201, driver 26.11.2, `rocm-sdk-core` 10.0.0 clang): 10,000 programs (seeds 1, 2, 3 and
500, 2000 each, and seed 500 with `--bfe-any`) gave 0 mismatches, so gfx1201 follows the same `s_bfe` rule. With
`--oracle translator,hw`, 200 of 200 programs matched (seed 1).

#### RDNA2 integer rows (`tools/int_hw_probe.py`, SHADER-137)

`tools/int_hw_probe.py` runs 87 hand-written cases across 19 integer groups (both wave32 and wave64) natively on
an AMD GPU through `tools/hw_oracle` and compares every output word against `tools/rdna2spirv.py` +
`tests/semantic/spirv_mini_interp.py` (with scalar ops run twice, SCC preset to 0 and 1, so SCC written vs kept is
observed directly).

Measured on an **AMD Radeon 680M (gfx1035, RDNA2)**, Windows 11 (`10.0.26300`), AMD driver `26.5.1`
(`32.0.21043.7012`, `amdhip64_6.dll`), conda-forge `clang 22.1.8` + `lld 23.1.6` (`--oracle both --wave both`):
**87 cases: 82 bit-exact matches, 5 known live translator divergences, 0 unexpected mismatches, 0 wave32/wave64
differences** (plus `tools/salu_diff.py --oracle hw,translator --bfe-any --programs 500 --seed 1`: 500/500 match).

- **Matched RDNA2 silicon in both wave32 and wave64 (82/87):** `s_absdiff_i32`, `s_mul_hi_i32/u32`, `s_lshl1..4_add_u32`
  (64-bit shift-out and add carry into SCC), `s_sext_i32_i8/i16`, `s_flbit_i32`, `s_cmov_b32`, `s_cmovk_i32`,
  `s_pack_ll/lh/hh_b32_b16`, `s_bfe_u32/i32` past bit 31, `s_brev_b64`, `s_bfe_u64/i64`, `s_bitset0/1_b64`,
  `s_flbit_i32_b64/i64`, `s_quadmask_b32/b64`, `s_bitreplicate_b64_b32`, all nine 64-bit scalar bitwise ops on scalar
  data, `s_movrels_b32/b64`, `v_add/sub/subrev_nc_u32` / `v_add/sub_nc_i32` / `v_add/sub_nc_u16` with `clamp` and
  `op_sel`, `v_mad_u64_u32` / `v_mad_i64_i32` (`sdst` bit 64 sign of exact 65-bit sum and inactive-EXEC masking),
  `v_sad_u8`/`hi_u8`/`u16`/`u32`, `v_msad_u8`, `v_lerp_u8`, `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`, `v_mqsad_u32_u8`,
  `v_mul_hi_u32_u24`/`i32_i24`, `v_lshrrev_b64`/`v_ashrrev_i64`, `v_mad_u16`/`i16` (`clamp`, `op_sel`, untouched half
  kept), `v_cvt_f16_u16/i16` and `v_cvt_u16/i16_f16` (`_e64` keeps high half; NaN → 0, ±inf saturates; note GFX10
  does not accept `op_sel` on these opcodes), `v_cvt_i32_f32` (control: all ±qNaN/±sNaN → 0), and
  `v_cvt_pknorm_i16/u16_f32` (±qNaN → 0, ±inf saturation, RNE ties at ±0.5).
- **Translator divergences found on gfx1035:** five (VOP3 `clamp` on the six carry-out add/sub ops ignored;
  `v_cvt_rpi_i32_f32` rounding `x + 0.5` in binary32; `v_cvt_flr/rpi_i32_f32` returning `0` for NaN instead of
  saturating by its sign). All are fixed (#1831, #1856), so `KNOWN_DIVERGENCES` is empty and all 87 cases
  match.

#### RDNA2 swizzled buffer addressing (`tools/buffer_swizzle_probe.py`, SHADER-343)

`tools/buffer_swizzle_probe.py` runs 22 hand-written MUBUF/MTBUF buffer-swizzle cases across 8 groups (both wave32
and wave64) natively on an AMD GPU through `tools/hw_oracle` and compares every output word against
`tools/rdna2spirv.py` + `tests/semantic/spirv_mini_interp.py`.

Measured on an **AMD Radeon(TM) Graphics (gfx1036, RDNA2)**, Windows AMD driver `32.0.21042.62`, ROCm clang
(`--oracle both --wave both`): **22 cases: 22 bit-exact matches, 0 known divergences, 0 unexpected mismatches,
0 unexpected wave32/wave64 diffs** (and the 64-thread `add_tid_wave_lane` case matches wave32 and wave64
individually, confirming wave-lane wrap at 32 vs 64).

Six silicon divergences from the earlier ISA-only implementation were found and fixed in `tools/rdna2spirv.py`:

- `idxen = 1` ignores `ADD_TID_ENABLE` completely (both address and bounds, linear and swizzled).
- `SWIZZLE_ENABLE = 1` with `idxen = 0` and `ADD_TID_ENABLE = 0` has no effect (linear address, mode 3 stays RAW).
- Multi-dword raw accesses step by `+4 * IS` only in scratch-swizzle mode (`SWIZZLE_ENABLE = 1`, `idxen = 0`,
  `ADD_TID_ENABLE = 1`), whereas indexed swizzle (`idxen = 1`) steps linearly by `+4` bytes even across `IS`
  boundaries.
- Scratch-swizzle mode suppresses typed formats > 4 bytes (MUBUF and MTBUF loads/stores) and 8-byte
  `buffer_atomic_*_x2` even in `OOB_SELECT 2` (while formats <= 4 bytes succeed); indexed swizzle executes them
  linearly.
- `OOB_SELECT 3` with active swizzle and `stride != 0` checks `(index + soffset) >= num_records` or
  `off + payload > stride`.
- `uint32(voff + inst_offset)` wraps at 32 bits before bounds checking, whereas RAW mode 3
  `uint64(off) + uint64(soffset)` does not wrap at 32 bits.

Two assembler facts matter when writing edge cases:

- An integer token in a float-operand position uses AMD's inline-constant encoding for values 0..64
  and -16..-1. `v_add_f32 v0, v1, 2` therefore means `v1 + 2.0`, not `v1 + bits(2)`. To feed a
  specific bit pattern, put it in a VGPR with `v_mov_b32` first.
- A literal outside the inline range is emitted as a 32-bit literal and printed as hex (for example
  `0x100`), so denormal patterns above the inline range can be written directly. The smallest
  denormal (`0x00000001`) cannot: load it with `v_mov_b32`.

The first test matrix should cover:

| Case | Program shape | What it pins |
|---|---|---|
| Inline constants | `v_mov_b32` with `1.0`, `-1`, `0x3f800000` | bit patterns, not just numeric equality |
| NaN | `inf + -inf`, `0 * inf`, NaN input through add/mul | canonical quiet NaN and propagation policy |
| Signed zero | `-0 + +0`, `0 - 0`, `-1 * 0` | sign of zero results |
| Denormals | smallest denormal + itself, `* 0.5`, `* 1.5`, max denormal + min denormal | gradual underflow and rounding |
| FMA | exact midpoint case (see below) | fused, single rounding |
| 16-bit | packed `0x3c003c00`, `v_and_b32 0xffff`, `v_lshrrev_b32 16` | packed halves survive moves and masks |
| `cndmask` | VCC lane mask 0b1010, both source values distinct | select direction and per-lane masking |
| Shifts | amount 0, 31, 32, 33; negative arithmetic shift | 5-bit amount mask and sign extension |
| EXEC low half | write only `exec_lo`, then a vector op | the high half of a wave64 mask is untouched |
| EXEC both halves | `exec_lo = 0xffff0000`, `exec_hi = 0` | inactive lanes unchanged |

The FMA case is the one a naive host implementation gets wrong. A product of two binary32 values can
land exactly on a binary32 midpoint, and a tiny third addend decides the rounding direction. Evaluating
`a * b + c` in host double and then rounding to binary32 can lose that addend. The prototype uses exact
rational arithmetic for FMA; a test pins the result so the implementation cannot regress to the
double-rounded answer.

#### `v_interp` ALU ops (`tools/interp_hw_probe.py`, SHADER-430)

`v_interp_*` also run in a compute wave: they read the attribute words from LDS at M0. On gfx1036 with M0 = 0,
attribute N takes 12 dwords from byte 48·N; channel c's P0 and P10 words are dwords 2c and 2c+1, and its P20 word is
dword 8+c (found by filling LDS with markers and reading them back with `v_interp_mov_f32`). So the tool writes
random parameter words there itself, packed f16 pairs and plain f32 values, runs every interp op through
`tools/hw_oracle` and checks each result against exact rational models. It also runs 30 hand-built f16 rounding ties
that tell one rounding of `p2_f16` from two.

Measured on the **Radeon iGPU of a Ryzen 7 9800X3D (gfx1036, RDNA2)**, AMD driver 32.0.13036.4, Windows 10 22H2,
`rocm-sdk-core` clang, seeds 1–8 × 4 kernels × 32 lanes. Every model matched, under both `--denorm32 3` (MODE keeps
f32 denormals) and `--denorm32 0` (MODE flushes them). That is 292,474 checks per mode plus the 30 ties.

| Op | Result on gfx1036 |
| --- | --- |
| `v_interp_mov_f32 p0` | the raw P0 word |
| `v_interp_p1_f32` | fma(P10, I, P0) with one rounding to f32. A product rounded first and then added differed in 723 of 4,096 f32 cases (seeds 1–4). A packed channel's words are read as raw f32 bits |
| `v_interp_p2_f32` | fma(P20, J, p1), one rounding |
| `v_interp_p1ll_f16` (`high`) | the chosen f16 halves widened exactly, then fma(h10, I, h0) with one rounding to **f32** |
| `v_interp_p2_f16` (`high`) | fma(h20, J, p1ll) rounded **once, to f16** (all 30 ties gave the single-rounding result, none the f32-then-f16 one). The high 16 bits of vdst are **kept** |
| f32 denormals | follow MODE: with `denorm_mode_32 = 0` a denormal P0, P10 or P20 word reads as 0 |

The translator wrote `p2_f16` as a full-register result with a zero high half. It now keeps the high half
(`tests/semantic/test_interp_f16_high.py`). This compute route does not reach the other half of SHADER-430: what the
SPI writes into LDS for a real draw, in particular whether a packed channel's P10/P20 are f32 differences of the raw
words or per-half f16 differences under `FP16_INTERP_MODE`. That needs a pixel-shader harness on AMD hardware. Not
covered either: `v_interp_p1lv_f16`, and f16 inf/NaN halves.

```powershell
python -I tools/interp_hw_probe.py                     # exit 0: all match; 77: no hardware oracle here
python -I tools/interp_hw_probe.py --denorm32 0 --seeds 1,2,3,4,5,6,7,8
```

## Harness design

`tools/shader_diff.py` owns the test table, the seed and the comparison, leaving
`cpu_ref.py` a pure interpreter (it is never modified by the runner). Per case:

- **Inputs.** Initial SGPRs/VGPRs (per lane), VCC, EXEC, and later buffer/image contents. Inputs are
  generated from a named seed (SplitMix64 or `random.Random(seed)`), so a failure is reproducible
  from the seed alone. Hand-pinned cases should also accept explicit values.
- **Outputs.** A declared list of observable registers, buffer ranges or image texels. Compare raw
  bits. For NaN, compare "both NaN" by default and record the payloads; for finite values, require
  exact equality.
- **First mismatch.** Report case name, seed, lane, register/buffer offset, expected bits, actual
  bits, and the RDNA2 instruction at the PC that last wrote the value. Stop at the first mismatch;
  the full diff is noise once one lane diverges.
- **Environment checks.** Query the device for f32 denormal preservation, f16 support and subgroup
  size before running cases that need them. A missing feature is a skip with a reason, never a pass.
- **Harness canaries.** Hand-written SPIR-V twins for a few cases, plus an intentionally wrong
  expected value, must fail. A harness that cannot fail is not a harness.
- **No silent skips.** Count `pass`, `mismatch`, `skip(reason)` and `harness-error`. A run whose
  mismatch count is zero but whose skip count is 90% is not evidence of broad correctness.

The existing `runtime/test` tools already expose the hard parts (device selection, push constants,
descriptor setup, readback). The differential runner should call them rather than grow a second
Vulkan stack.

### Integer VALU across the whole wave (`tools/wint_diff.py`)

The one-lane fuzzers cannot see which lanes an instruction writes or the lane masks it reads and makes.
`tools/wint_diff.py` runs seeded integer VALU programs on every lane of a wave32 or wave64 wave (`--wave 32|64|both`)
with EXEC rewritten between instructions: compares into SGPR masks and VCC (e64 and e32), `v_cmpx` into EXEC,
`v_cndmask_b32` on those masks, the carry-out and carry-in forms (`v_add/sub/subrev_co_u32`, `_co_ci_u32`, e64
and e32), `v_mad_u64_u32`/`v_mad_i64_i32` with a mask carry-out, 16-bit e64/VOP3 ops with `op_sel` writing one half,
data moved into an SGPR mask, and a set of 32-bit ops. The reference is a per-lane ISA model in the tool; the
translated side runs in `tests/semantic/spirv_wave_interp.py`. Shrinking, skips and exit codes are those of
`salu_diff.py`. Forms the translator is known to get wrong (VCC holding data read as a mask, #2206 for wave32 and
open for wave64; a mask read as VALU data, #293) are left out unless `--known`. `--oracle hw` runs each program as
one wave on an AMD GPU and compares every lane; on an RDNA2 iGPU (gfx1036) 1,000 programs in both wave sizes,
the known forms included, matched the reference.

```
python -I tools/wint_diff.py --programs 2000 --seed 1 --wave both
python -I tools/wint_diff.py --oracle hw --programs 500 --seed 1 --wave both
python -I tests/test_wint_diff.py
```

### LDS and `ds_*` on the GPU (`tools/lds_fuzz.py`, FUZZ-6)

LDS ops are workgroup-wide, so `tools/lds_fuzz.py` runs on the GPU rather than one lane in the
interpreter. Each seeded program is one 64-thread group (wave64, or two wave32s) with 4 KiB of LDS
that starts from a known pattern. It runs 4 to 12 random `ds_*` instructions, each followed by
`s_waitcnt` and `s_barrier`, then stores all 16 result VGPRs of every thread and the whole LDS.
`vk_compute_test` reads them back with `VKC_OUT_WORDS` / `VKC_DUMP`.

For the translator oracle, `lds_fuzz.py` and `valu_fuzz.py` create and retain a unique output folder per run. With
`--work DIR`, that folder is a new child of `DIR` (the printed work directory), rather than `DIR` itself; files from
older runs and other contents of `DIR` are left alone. Without `--work`, it is created in the system temporary folder.

- **Instructions.** Every read width (`b32` to `b128`, `u8`/`i8`/`u16`/`i16`, `u16_d16[_hi]`, `u8_d16[_hi]`, `i8_d16[_hi]`), the
  byte and short writes (also `d16_hi`), `read2`/`write2` with and without `st64`, the add, sub,
  min, max, and, or and xor atomics with and without return, `wrxchg`, `cmpst`, `ds_swizzle_b32`
  and `ds_bpermute_b32`.
- **Reference.** A model in the same file, written from the ISA and not from the translator.
- **Determinism.** Programs stay deterministic. Writes, returning atomics, `wrxchg` and `cmpst` use
  addresses no other lane touches in the same op. Only order-independent atomics without return
  share addresses. Every access is in bounds, because the translator wraps out-of-range indices
  where hardware returns 0.
- **Swizzle.** Offsets are the two forms the LLVM `swizzle()` macros encode, checked with
  `llvm-mc 23 -mcpu=gfx1013`: bit 15 = quad permute; otherwise and = bits 4:0, or = 9:5,
  xor = 14:10, within each 32-lane group.
- **EXEC.** Random EXEC masks are used on every op except `ds_swizzle`.
- **Reports.** A mismatch names the thread, the register or LDS word, and the op that last wrote it.
  `--no-exec KIND` drops one op kind's masks without changing the rest of the program, to separate
  a known finding from new ones.
- **Subgroup size.** Wave32 programs, and wave64 ones built with `--emulate`, ask `vk_compute_test` for
  32-lane subgroups (`VKC_SUBGROUP_SIZE=32`). Where the device can't require it but 32 is its default
  (MoltenVK), a probe shader checks the lanes first and the programs are compared (see `docs/testing.md`,
  "GPUs with 32-wide subgroups"). Where the device can't pin that size at all, `vk_compute_test`
  prints `compute subgroup size: 32 not available` and runs at the driver's choice, and the wave32
  program is a skip, not a mismatch: its words come from another wave size. Mesa llvmpipe (8 lanes)
  is such a device. Emulated wave64 uses no subgroup operations, so it is still compared there.
  `valu_fuzz.py` and `crosslane_fuzz.py` follow the same rule.

First results on an RX 6700 XT (RADV, Mesa 26.3.1), native wave64/wave32 and with
`--wave64-emulate` at subgroup 32:

- **One finding, already fixed by #253.** With wave32 `ds_bpermute` under EXEC, a source lane that
  is off returned its value instead of 0: 12 of 240 programs (seeds 1 to 3) failed, each time
  under a masked wave32 `ds_bpermute`, and with #253's translator all 480 programs (seeds 1 to 3,
  native and emulated) match.
- **Everything else matches.** With `--no-exec bpermute`, 1,040 programs match (seeds 1 to 8).
- **Not in the default mix.** `ds_permute_b32` is generated with `--permute` (below). The 8-bit d16 reads
  (`ds_read_u8_d16[_hi]`, `ds_read_i8_d16[_hi]`) are translated since SHADER-85 (`cs_w32_lds_u8_d16`) and the
  fuzzer generates them.

#### Hardware oracle for the LDS programs (`--oracle hw`)

`--oracle hw` runs each program natively on an AMD GPU through `tools/hw_oracle/` (setup: "Hardware oracle" under
FUZZ-4) and compares every stored word, 16 result registers per thread and all 1,024 LDS dwords, with the model;
`--oracle translator,hw` does both and prints what the translator did with each program the hardware disagrees
with. It tests the **model**, not the translator. Without the tools it prints `hardware oracle unavailable: ...` and
exits 77.

```
python -I tools/lds_fuzz.py --oracle hw --programs 2000 --seed 1
python -I tools/lds_fuzz.py --oracle hw --extended --programs 2000 --seed 1 --negative-control
python -I tools/lds_fuzz.py --oracle hw --programs 50 --seed 1 --wave 32 --only 1,18,20,24 --minimize
```

- **What runs.** The program's real group size (`--threads 32|64|128|256|mix`: one partial wave, one wave, or several
  waves joined by `s_barrier`), wave32 and native wave64 (`.amdhsa_wavefront_size32 0`), its EXEC masks (the entry
  EXEC is saved in `s10` and restored, so a partial wave keeps its missing lanes off), and the LDS size in the
  kernel descriptor and metadata (`.amdhsa_group_segment_fixed_size`; without it the group gets no LDS). Results go
  out with `global_store_b32` through the output pointer.
- **GFX11 spellings** (`hw_oracle/asm.py`, checked by assembling every mnemonic with AMD clang for gfx1030 and
  gfx1150): `ds_read_*` is `ds_load_*`, `ds_write_*` is `ds_store_*`, `ds_read2[st64]_b32` is
  `ds_load_2addr[_stride64]_b32`, `ds_write2[st64]_b32` is `ds_store_2addr[_stride64]_b32`, `ds_wrxchg_rtn_b32` is
  `ds_storexchg_rtn_b32`. The assembler still takes the old names on gfx1150 except `ds_cmpst_rtn_b32`.
  **`ds_cmpstore_rtn_b32` takes the value to store before the compare value**, the reverse of GFX10's
  `ds_cmpst_rtn_b32 vdst, vaddr, cmp, store`; llvm-mc encodes both spellings of the same text identically, so only
  the operand order says which. Unswapped, every program with a `cmpst` failed on gfx1150 (each minimized to that one
  op); swapped, none does. GFX12 takes the same order (measured on an RX 9070 XT, gfx1201: with 7 in the dword and
  data `v1 = 9`, `v2 = 7` it becomes 9, with `v1 = 7`, `v2 = 9` it stays 7, for `ds_cmpstore_rtn_b32` and
  `ds_cmpstore_b32`; gfx1036's `ds_cmpst` does the opposite), and with `--perturb cmpst_swap` 275 of 1,000 wave32
  programs mismatch there. The model keeps the GFX10 order. Nothing else the generator emits is missing on GFX11, so
  no other program is skipped for its semantics; an assembler rejection is counted as `rejected`, never as a pass.
- **GFX12** (`hw_oracle/asm.py` `sync_lines`): there is no `s_barrier` (a workgroup barrier is
  `s_barrier_signal -1` then `s_barrier_wait -1`), and the LDS wait is `s_wait_dscnt 0` instead of the legacy
  `s_waitcnt lgkmcnt(0)`. **A wave64 `ds_bpermute_b32` reaches all 64 lanes on GFX12**, while RDNA2 (and gfx1150)
  keep each 32-lane half separate: with every lane asking for lane `(l + 32) mod 64`, the RX 9070 XT returns lane
  `l ^ 32` and a gfx1036 iGPU returns lane `l`. The model describes RDNA2, so on gfx12xx a wave64 program with a
  `ds_bpermute` is skipped with that reason; under `--perturb bpermute_wave64_cross` all 1,000 wave64 programs of
  seed 1 match there. RX 9070 XT, AMD 26.11.2, `--programs 1000`, seeds 1 and 2: wave32 1,000/1,000 match; wave64
  537 and 506 match, 0 mismatch, the rest skipped for that reason. The gfx1036 iGPU gives 1,000/1,000 in both.
- **Wider op mix**, opt-in so that a seed keeps its programs: `--permute` (`ds_permute_b32`), `--swizzle-exec`
  (`ds_swizzle_b32` under EXEC masks), `--threads`, `--extended` (all three). `ds_permute_b32` and a masked
  `ds_swizzle_b32` carry a hypothesis (the model's reading of the ISA text); on the hardware a failing one is a
  mismatch like any other and counted in `hyp_only`, on the translator it is reported apart as `HYPOTHESIS MISMATCH`.
  The swizzle hypothesis was falsified at first ("an active lane reads its source lane's register whether or not
  that lane is on"): gfx1150 returns 0 when the source lane is off in EXEC, as `ds_bpermute` does. The model now
  says that.
- **Negative controls.** `--perturb NAME` judges the hardware against a deliberately wrong model (mismatches must
  appear); `--negative-control` re-judges the words of the run with all 16 (`max_i32` unsigned, `min_u32` signed,
  swapped `cmpst`, zero-extended `i8`/`i16`, d16 reads clearing the other half, `b16_d16_hi` shift, swapped
  `write2`, `st64` unit, bpermute reading off lanes or crossing the wave64 halves, reversed quad permute, swapped
  swizzle or/xor, permute writing no zero, permute lowest source winning a collision) and exits 1 if one goes
  unnoticed. `--minimize` drops ops while a program still differs, `--keep DIR` writes the native kernel of each
  mismatch, `--only I,J` runs chosen indices.

RDNA2 `ds_permute_b32` verification (`FUZZ-10`, `SHADER-133`): raw kernels on an AMD Radeon RX 6800 (`gfx1030`, wave32
and wave64) confirm that only active lanes send, the highest active source lane wins a collision, active destination
lanes nobody targets read `0`, inactive destinations keep their old values, and the two 32-lane halves stay separate
in wave64. On an AMD Radeon RX 6700 XT (`gfx1031`, RADV Mesa 26.3.1), 3,600 translator runs with colliding
`ds_permute_b32` multipliers (`mul` in `0..63`, seeds 1–3, 300 programs × wave32/wave64 native and `--emulate`)
matched the model bit for bit (`0` mismatches), and inverting the collision winner or zero-fill rules produced
mismatches in `39` and `56` of `200` seed-1 programs.

Results on a Radeon 880M (gfx1150, RDNA3.5, driver 26.11.2, AMD clang 23.0.0git), `--oracle hw`:

| Programs | Count (wave32 + wave64) | Mismatches |
| --- | --- | --- |
| default mix, seeds 1, 2, 3, 11, 21, 31, 41, 100, 200, 2,000 each | 36,000 | 0 |
| `--extended`, seeds 1, 2, 3, 11, 21, 31, 2,000 each (permute, masked swizzle, groups of 32 to 256) | 24,000 | 0 |
| seed 1, `--only` the twelve wave32 programs SHADER-105 names (1 18 20 23 24 26 29 30 32 33 34 38) | 12 | 0 |

Negative control (1,000 extended programs): each of the 15 wrong models mismatches on 126 to 547 programs, none goes
unnoticed. gfx1150 is RDNA3.5, not RDNA2; the RDNA2 runs follow.

Results on the integrated Radeon of a Ryzen 7 9800X3D (gfx1036, RDNA2, AMD driver 32.0.13036.4, Windows HIP launcher,
AMD clang 23.0.0git), `--oracle hw` on `main` `003e7812`:

| Programs | Count (wave32 + wave64) | Mismatches |
| --- | --- | --- |
| default mix, seeds 1, 2, 3, 2,000 each | 12,000 | 0 |
| `--extended`, seeds 1, 2, 3, 2,000 each (permute, masked swizzle, groups of 32 to 256) | 12,000 | 0 |

`hyp_confirmed` (programs with a hypothesis op that matched) in the extended runs: 1158/1185, 1183/1177 and 1151/1189
(wave32/wave64, seeds 1 to 3). So on RDNA2 too a masked `ds_swizzle_b32` whose source lane is off in EXEC reads 0,
and `ds_permute_b32` follows the model. Negative control (extended, seed 1): each of the 15 wrong models mismatches,
from 417 programs (`bpermute_inactive_reads`) to 1,123 (`write2_swapped`); none goes unnoticed.

With `--oracle translator,hw` on the same Radeon 880M (Vulkan, subgroup 64), seed 1, 100 programs per run: the
default mix matches the model on the translator, native and emulated (200/200 each). With `--extended` the
translator differs from the hardware in 22 of 177 programs (native and emulated alike), all in a `ds_swizzle_b32`
under an EXEC mask whose source lane is off: the translator hands the active lane the source lane's register, the
hardware gives 0 (the swizzle hypothesis above). So the translator differs from gfx1150 for a masked
`ds_swizzle_b32` with an off source lane; the gfx1036 hardware runs above show RDNA2 gives 0 as well (SHADER-207). Not
fixed here. Partial-wave programs (`--threads` 32
in wave64) are skipped for the translator, which cannot launch them. NVIDIA was not run.

**Other backends.** The Linux libdrm launcher of #1265 (`tools/hw_oracle/drm.py`) now runs these programs too. It
programs `COMPUTE_PGM_RSRC2.LDS_SIZE` from the descriptor's group segment (the assembler leaves that field 0, so the
4,096 bytes of LDS `lds_fuzz` asks for reach the hardware) and its output buffer takes 16,384 words like the Windows
launcher's, so a 64x64 `image_fuzz` case fits. On the integrated Radeon of a Ryzen 7 7700X (gfx1036, RDNA2, Linux
7.3.5-zen1-1-zen, libdrm 2.4.134, AMD clang 23.1.5):

| Run | Programs / cases | Mismatches |
| --- | --- | --- |
| `lds_fuzz.py --oracle hw --programs 500 --seed 1` (wave32 and wave64) | 1,000 | 0 |
| `image_fuzz.py --oracle hw --programs 120 --seed 1` (60 cases per wave; 8 to 64 per side, four 64x64 cases) | 120 | 0 |

`lds_fuzz --oracle hw` no longer exits 77 on Linux for LDS or the word count. The gfx1036 numbers earlier in this
section are still from the Windows HIP launcher; these two runs put the same model on the same architecture through
libdrm.

**Overlap with #789.** #789 adds `ds_permute_b32` to the default op mix of `lds_fuzz.py`; here it is opt-in
(`--permute`) so that existing seeds keep their programs. Whichever lands second reconciles the two.

**SHADER-105** (Intel `lds_fuzz` mismatches in DS atomics `max_i32`/`max_u32`, `ds_cmpst`, small reads, and
registers that kept their initial value): the model matches gfx1150 for every one of those families, under EXEC
masks, in wave32 and wave64, and in multi-wave groups. The Intel mismatches went away with #820's constant-EXEC fix
(50/50 native and 100/100 emulated for seeds 1 to 3). `ds_bpermute` (SHADER-31, #253) matches the model on gfx1150
too, including "an off source lane reads 0" in both wave sizes.

### Memory instructions (`tools/mem_diff.py`, FUZZ-5)

`cpu_ref.py` has no memory model, so memory instructions get their own runner. `tools/mem_diff.py`
carries a small reference model written from the RDNA2 addressing rules (V# base and stride, MUBUF
`soffset + offset + voffset + vindex * stride`, SMEM pointer/V# plus offsets, signed 12-bit `global_*`
offsets, 32-bit buffer atomics) and compares it with the translation run in `spirv_mini_interp`. Both
the exported words and the whole final memory are compared, so stores and atomics are checked too.

```powershell
python -I tools/mem_diff.py --programs 3000 --seed 1 [--family buffer_load] [--max-ops 4]
python -I tools/mem_diff.py --self-test
python -I tests/test_mem_diff.py
```

Families: `s_load`, `s_buffer_load`, `buffer_load`, `buffer_store`, `buffer_atomic`, `global`
(dword loads/stores, `ubyte`/`sbyte`/`ushort`/`sshort`). Every access is generated in bounds and
dword aligned (sub-dword loads at their natural alignment), with no format in word3: out-of-bounds
results, `OOB_SELECT`, swizzling and the `s_buffer_load` alignment truncation depend on descriptor
fields the translator does not model yet, so they need a GPU-confirmed rule before they are fuzzed.
`spirv_mini_interp` gained `OpSConvert` and the 32-bit atomics on PhysicalStorageBuffer memory for
this. Images have their own fuzzer (`tools/image_fuzz.py`, below).

The `buffer_format` family covers `buffer_load_format_*` on every format the translator decodes, with a
random DST_SEL of 0/1/X/Y/Z/W, and `tbuffer_load_format_*` on the five formats the 1.0.2.1 corpus uses
(`16_16_SNORM`, `8_8_8_8_UNORM`, `8_8_8_8_UINT`, `2_10_10_10_UNORM`, `16_16_16_16_UINT`, about 53,000
instructions). Generated MTBUF cases use a bound 32_UINT V# and take their data format from the
instruction. The reference decodes elements from the GFX10 format *name* in `tools/gfx10_formats.json`
(channel widths listed from the top bit). UNORM/SNORM values use correctly rounded division.
MUBUF repeats missing channels (*c* mod *n*) and DST_SEL 1 gives integer 1 for UINT/SINT and 1.0f
otherwise (SHADER-86, RX 6700). MTBUF loads return instruction-format channels in order, zero-fill
missing channels and return zero for an INVALID V# format (native gfx1036, wave32 and wave64).
Elements are naturally aligned. A half NaN widens to f32 with its payload shifted into the f32 mantissa
and its sign and quiet bit unchanged (`f16_to_f32`, bit-level so the Python version cannot change it);
the translator used to canonicalise it, which is the one class this fuzzer found
(`docs/shader_validation.md`, "Widening a 16-bit float NaN in a typed buffer load").

Over 5,000 programs the reference and translator match bit for bit; breaking the `16_16_SNORM` kind, the
`8_8_8_8_UINT` widths or the DST_SEL order in the translator makes 33, 37 and 191 of 300 programs fail.

A mutation check (inst offset dropped, V# stride narrowed, the SMEM pointer's high word dropped) makes
76, 45 and 42 of 150 programs fail, so the runner does catch address errors.

### Sum-of-absolute-differences family (`tools/sad_diff.py`)

The subset interpreter `cpu_ref.py` has no SAD ops, so the family gets its own runner, like the memory
instructions above. The family is seven opcodes — `v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16`, `v_sad_u32`,
`v_msad_u8` and the packed quad forms `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`. `tools/sad_diff.py`
carries a small reference written from the ISA pseudocode recorded in `docs/design.md`'s contract rows
(the "Sum of absolute differences" row hardware-verified on gfx1035 through SHADER-137's `int_hw_probe`
run; the `v_qsad` row bit-exact on an RTX 3060 in #623) and compares it with the translation run in
`spirv_mini_interp`, the four exported mrt0 words, bit for bit (everything here is integer, so a
mismatch can never be a rounding convention).

Programs are straight-line: seeds in v8..v15 and s4..s7, then family ops whose destinations stay inside
v8..v11 (the quad ops write pairs inside that), then `exp mrt0 v8, v9, v10, v11`, so every result is
observed and later ops chain on earlier ones. Every operand position takes every source kind the
encoding allows: VGPR, SGPR, a hex literal, an inline constant and `null` for the 32-bit sources; the
quad ops' 64-bit S0/S2 take VGPR/SGPR pairs, a hex literal (zero high word) or an inline constant
(sign-extended) — the 64-bit constant rule of the contract table, pinned both ways: a hex literal reads
the same as the register pair holding its word and a zero high word, while a negative inline constant
sign-extends, so the two constant forms differ in the high word.

Two modes run one after the other: a sweep over the whole source-kind space (every (S0, S1, S2) kind
combination, 125 per plain op, 80 per quad op, 785 programs) with values drawn from byte/half edge
words, and random seeded programs that widen the value coverage over the same kind space. Not
generated, on purpose: `clamp` (the contract rows say the translator rejects it; pinned as expected
rejections in `--self-test`), `op_sel`/`neg`/`abs` (the ISA does not define them for these opcodes),
`v_mqsad_u32_u8` (open PRs #1926/#2163 are on that op) and `null` as a 64-bit quad source (PR #2650's
subject).

```powershell
python -I tools/sad_diff.py --programs 2000 --seed 7
python -I tools/sad_diff.py --self-test
python -I tests/test_sad_diff.py
```

Across seeds 1, 2, 3, 7, 11 and 12 (the last pair with `--max-ops 12`) 12,810 programs match bit for
bit, zero mismatches and zero skips; the seed-7 headline run alone is 785 + 2,000 programs with about
1,050 to 1,175 cases per op. Two mutations of the translator's SAD rule — the per-byte `UMax`/`UMin`
pair replaced by a wrapped subtraction, and the `v_sad_hi_u8` shift dropped — make 36 and 17 of 60
programs fail (`--programs 60 --seed 2 --max-ops 5 --no-sweep`), so the runner does catch arithmetic
errors.

### Images on the GPU (`tools/image_fuzz.py`)

`tools/image_fuzz.py` starts on the image path, where the T# format, the view type and the binding layout
meet the driver. Each case assembles a pixel shader that reads one texel with `image_load` (a nonzero `dmask`, the
T# in `s[8:15]`), runs it through `runtime/test/vk_image_fuzz.cpp` against a synthetic texture of a chosen
format, and compares every RGBA32F pixel with an oracle that models the RDNA2 texel fetch for that format: a
fetch is a memory read, so there is no filter, border or LOD to model.

The comparison is bit exact (any NaN matches any NaN) with one exception. The translated shader fetches a UNORM
texel through a float view, so the 8-bit to float conversion is the driver's. RDNA2 returns the correctly rounded
c / 255. llvmpipe multiplies by an f32 1/255, which is one ulp above that for 126 of the 256 byte values (27 gives
0x3dd8d8da, not 0x3dd8d8d9), and a run stops at its first mismatch, so on llvmpipe no case after the first
`R8G8B8A8_UNORM` one ran. The translator oracle therefore accepts one ulp on a UNORM8 texel component, never on
another format or on a sentinel, and the summary says how many components needed it. `--oracle hw` stays bit exact.

Each run keeps its generated files in a unique output folder. With `--work DIR`, it is a new child of `DIR` and its
path is printed; existing contents and other runs are left alone. Without `--work`, it is in the system temporary folder.

```sh
cmake --build runtime/build --target vk_image_fuzz
python -I tools/image_fuzz.py --programs 8 --seed 1 [--show]
python -I tools/image_fuzz.py --self-test
python -m pytest tests/test_image_fuzz.py
```

Covered: 2D `image_load`, `R8G8B8A8_UNORM`, `R16G16B16A16_SFLOAT`, `R32G32B32A32_SFLOAT`, `R32_SFLOAT`,
and `R8G8B8A8_UINT`/`SINT`, all as sampled images, as the runtime binds a read-only texture (integer formats are
read through the translator's uint/sint aliases of the sampled array). The shader converts the loaded 8-bit integer values to float for
readback; every such value is exactly representable, including signed negatives. Wider integers are not covered.
Texture sizes divide the 64x64 draw (8/16/32/64), so every texel is read at least once. The mask cycle starts with
`0xf` and `0x5`, so the default eight programs already exercise packing across a gap. With 90 programs, each of the
six default formats exercises all 15 nonzero masks. Selected channels are packed into consecutive destination
registers, while unwritten registers keep initialized sentinel values.
The packing model follows the image VGPR usage description in AMD's
[RDNA 1.0 ISA guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna-shader-instruction-set-architecture.pdf);
the Vulkan test compares translated shaders on the host GPU, not native RDNA2 execution.

Devices: the harness needs `fragmentStoresAndAtomics`, and no without-format storage-image feature: a translated
`image_load` fetches every format from the sampled arrays, UINT/SINT ones through integer aliases (SHADER-151, the
"Typed image loads" row of `docs/design.md`). Until then it declared the uint/sint storage path even when the case
read a float format, so an Intel UHD 770 (Xe-LP), which lacks `shaderStorageImageReadWithoutFormat`, skipped every
case. Not covered yet: wider UINT/SINT formats, mips, 1D/2D-array/3D/cube,
`image_sample`/`gather` with filters and borders, and texel offsets with `VK_KHR_maintenance8`.

#### Hardware oracle for the image fetches (`--oracle hw`)

`--oracle hw` runs the same fetches natively on an AMD GPU through `tools/hw_oracle/` (setup: "Hardware oracle" under
FUZZ-4) and compares every result register of every texel, bit for bit, with the CPU model above;
`--oracle translator,hw` does both on the same textures and says, for each case the hardware disagrees with, what the
translator did. It tests the **model**: whether `image_load` on real hardware returns what the oracle says. Without the
tools it prints `hardware oracle unavailable: ...` and exits 77.

```
python -I tools/image_fuzz.py --oracle hw --programs 2400 --seed 1
python -I tools/image_fuzz.py --oracle hw --programs 180 --seed 1 --negative-control
python -I tools/image_fuzz.py --oracle translator,hw --programs 90 --seed 1
python -m pytest tests/test_image_fuzz_hw.py
```

- **What runs.** One raw compute kernel per case, wave32 and native wave64 alternating. The texture is linear
  (`SW_MODE` 0) device memory the kernel carries behind its own code, 256-byte aligned and found with `s_getpc_b64`, so
  the launchers need no input buffer; rows are padded to a 256-byte pitch and the padding is a pattern that is wrong to
  read (`0xCD`). The kernel builds the T# in `s[8:15]` (the base address from the PC, the other words are literals),
  runs `image_load v[8:..], v[2:3], s[8:15] dmask:.. dim:SQ_RSRC_IMG_2D unorm` for one texel per lane (1,024 lanes
  and a short loop for 64x64) and stores `v[8:11]` for each texel, so the sentinels of the registers `dmask` does not
  write are compared too. A texture that was not 256-byte aligned would make every word `0xBAADF00D`; the run reports
  that as an error instead of reading from a wrong base. The same texels, sizes (8 to 64 per side) and 15 masks as the
  translator run; a seed names the same textures for both.
- **The T# per family** (`hw_oracle/tsharp.py`, exact words in `tests/test_image_fuzz_hw.py`). Facts from Mesa's AMD
  register database and `ac_descriptors.c` and LLVM's `SIDefines.h` (MIT / Apache-2.0; own code, see
  [related_projects.md](related_projects.md)). GFX10 and GFX11 differ here, and GFX12 is a third layout that is not
  built (`--oracle hw` on gfx12xx exits 77 with that reason):

  | | GFX10 (gfx10xx, RDNA1/2) | GFX11 (gfx11xx, RDNA3 and RDNA3.5) |
  | --- | --- | --- |
  | WORD1 `FORMAT` | `[28:20]`, 9 bits | `[27:20]`, 8 bits |
  | `R8G8B8A8_UNORM`, `_UINT`, `_SINT` | 56, 60, 61 | 42, 46, 47 |
  | `R16G16B16A16_SFLOAT`, `R32G32B32A32_SFLOAT`, `R32_SFLOAT` | 71, 77, 22 | 57, 63, 22 |
  | WORD2 `RESOURCE_LEVEL` (bit 31) | 1 ("must be 1") | not there (0) |
  | `MAX_MIP` | WORD5 `[7:4]` on GFX10.3 | WORD1 `[19:16]` |
  | linear pitch | WORD4: `DEPTH` `[12:0]` and `PITCH_MSB` bit 13, GFX10.3 only | same |

  The pitch is always written explicitly (the row size rounded up to 256 bytes); GFX10.1 (gfx1010) has no such field
  and the builder leaves it zero. DST_SEL is X, Y, Z, W, except `R32_SFLOAT`, which reads (X, 0, 0, 1) as the model
  says. `WORD5` carries Mesa's `PERF_MOD` 4, the rest is zero.
- **Negative control.** `--negative-control` runs the cases and then re-judges the same hardware words with seven
  deliberately wrong models (UNORM8 `/256`, red and blue swapped, SINT8 read unsigned, `R32_SFLOAT` alpha 0, dmask
  components not packed, unwritten registers read as 0, texel coordinates transposed) and runs the kernels again with
  the T# `FORMAT` of the next format in the table. It exits 1 unless every one of them mismatches.

Results on a Radeon 880M (gfx1150, RDNA3.5, driver 26.11.2, AMD clang 23.0.0git):

| Run | Cases | Mismatches |
| --- | --- | --- |
| `--oracle hw`, seeds 1, 2, 3, 2,400 cases each (wave32 and wave64, all six formats, all 15 masks, 8 to 64 per side) | 7,200 | 0 |
| `--oracle translator,hw`, seed 1 (Vulkan on the same GPU, subgroup 64) | 12 | 0 (translator and hardware) |

Negative control (`--oracle hw --programs 180 --seed 1 --negative-control`): the unperturbed model mismatches on 0 of
180 cases; the seven wrong models are caught by 16 to 168 of them (`unorm_256` 30, `swap_rb` 144, `sint_unsigned` 30,
`alpha_zero` 16, `dmask_in_place` 132, `no_sentinels` 168, `transpose` 166), and a kernel built with the next format's
`FORMAT` code is caught on every format (24 to 30 of 30 cases each). So a mismatch would have shown.

Finding: none. The CPU model and the hardware agree word for word on all 7,200 cases, including NaN, infinity, signed
zero and denormal inputs (NaN payloads are compared as "any NaN", as in the translator run), the sentinel registers
`dmask` leaves alone, and the `R32_SFLOAT` (r, 0, 0, 1) fill. The translator run on 12 of the same cases agrees too; a
longer `translator,hw` run was not done (the Vulkan side takes about half a minute per case on this machine).

**RDNA3.5 is not RDNA2.** The gfx1150 T# and image unit are GFX11's; a match shows that the model is right for the
texel fetch of these formats on that hardware, which is not the console's. The GFX10 words have since run on RDNA2:

Results on the integrated Radeon of a Ryzen 7 9800X3D (gfx1036, RDNA2, GFX10 T# layout, AMD driver 32.0.13036.4,
Windows HIP launcher, AMD clang 23.0.0git), on `main` `003e7812`:

| Run | Cases | Mismatches |
| --- | --- | --- |
| `--oracle hw`, seeds 1, 2, 3, 2,400 cases each (wave32 and wave64, all six formats, all 15 masks, 8 to 64 per side) | 7,200 | 0 |

Negative control (`--oracle hw --programs 180 --seed 1 --negative-control`): the unperturbed model mismatches on 0 of
180 cases; the seven wrong models are caught by the same counts as on gfx1150 (`unorm_256` 30, `swap_rb` 144,
`sint_unsigned` 30, `alpha_zero` 16, `dmask_in_place` 132, `no_sentinels` 168, `transpose` 166), and a kernel built
with the next format's `FORMAT` code is caught on every format (24 to 30 of 30 cases each). So the GFX10 T# words of
`hw_oracle/tsharp.py` and the texel-fetch model hold on RDNA2 for these formats; `--oracle translator,hw` was not run
there. The Linux libdrm backend of #1265 (`tools/hw_oracle/drm.py`) now runs these cases too: `--oracle hw
--programs 120 --seed 1` on the same gfx1036 iGPU matched on 120 of 120 cases (60 per wave), four of them 64x64 and
therefore 16,384 words, with no rejections (Linux 7.3.5-zen1-1-zen, libdrm 2.4.134, AMD clang 23.1.5). It copies
only `.text`, which is where the texture lives, and the loaded code is 256-byte aligned.

### 32-bit integer VALU in every encoding (`tools/vint_diff.py`)

`tools/shader_diff.py` checks the common integer opcodes with VGPR sources in their e32 form. `tools/vint_diff.py`
adds the rest of the 32-bit integer VALU a compiler emits: the VOP1/VOP2 opcodes as e32 and e64, the VOP3-only ones
(`v_add3_u32`, `v_lshl_add_u32`, `v_alignbit/alignbyte/perm_b32`, `v_min3/max3/med3`, the 24- and 16-bit
multiplies with `op_sel`, `v_bcnt`, `v_bfm`, `v_ffbh/ffbl`, `v_cvt_pk_*16_*32`, ...) and VOPC e64 into a mask with
`v_cndmask_b32_e64`, with VGPR, SGPR, integer and float inline-constant and literal sources in every position the
encoding allows (one literal and two scalar values per VOP3). It runs one lane through
`tests/semantic/spirv_mini_interp.py` against its own ISA model, with the shrinking, skips and exit codes of
`salu_diff.py`; a compare's mask read as VALU data (#293) is left out unless `--known`. `--oracle hw` runs the same
programs on an AMD GPU; on an RDNA2 iGPU (gfx1036) 1,500 programs, the known form included, matched the reference. A 32-bit literal in a 16-bit source of `v_mad_u32_u16`/`v_mad_i32_i16` does not assemble, so literals there have 16 bits (zero-extended, measured).

```
python -I tools/vint_diff.py --programs 2000 --seed 1
python -I tools/vint_diff.py --oracle hw --programs 500 --seed 1
python -I tests/test_vint_diff.py
```

### 16-bit and memory ops on the GPU (`tools/valu_fuzz.py`, FUZZ-2, FUZZ-5)

`tools/valu_fuzz.py --family f16|mem` uses the same runner shape as `lds_fuzz.py`. It generates seeded
64-thread compute programs, translates them, runs them with `vk_compute_test`'s `VKC_DUMP`, and compares every
thread's 16 result VGPRs (plus the scratch memory for `mem`) bit for bit with a model of the instructions written in
the same file. The model doesn't import the translator. `tests/test_valu_fuzz.py` checks the models on the CPU:
the f16/f32 rounding against Python's IEEE packers, signed zeros, the partial-write rules and out-of-range
buffer accesses. If llvm-mc is present, it also checks that every family's
programs assemble for gfx1013.

- **`f16`**: f16 add/sub/mul/min/max/fma/fmac/med3, `v_cvt_f16_f32`/`v_cvt_f32_f16`/`v_cvt_pkrtz_f16_f32`,
  `v_pk_*` (f16 and shifts) with `op_sel`/`op_sel_hi`, 16-bit integer ops and `v_pack_b32_f16`. It only uses
  encodings `llvm-mc -mcpu=gfx1013` accepts: on GFX10 the VOP3 forms of the f16 VOP2 ops take no `op_sel`, and the
  16-bit integer ops exist only as `op_sel` VOP3. Which form keeps the other half follows the 16-bit row of
  `docs/design.md`. Results whose exact value is NaN, infinite or denormal, denormal inputs, and min/max of +0
  against -0 are marked uncertain and not compared.
- **`mem`**: `buffer_load/store_dword` to `dwordx4` with `offen`, `idxen`, `idxen offen` and offset-only addressing
  (offset-only loads use an SGPR `soffset` that is the slot of the wave's first thread), through V#s built in SGPRs.
  Each run draws a random OOB_SELECT, stride and `num_records` for them (any dword-aligned byte count, so accesses end
  mid-slot and straddle the end), and the model applies the per-mode rule of "Buffer bounds checking" in design.md per
  dword (modes 1 and 2 check once). `s_load_dword/x2/x4` and `s_buffer_load_dword/x2/x4` read what the stores wrote
  from the slot of the wave's first thread (a wave-uniform `soffset`, so no other wave races with the store); the
  `s_buffer_load` V# has its own stride and `num_records` and the model keeps design.md's bound
  (`num_records` bytes, or `num_records * stride`) and ignores OOB_SELECT. Byte/short buffer loads are still not
  generated (the translator rejects them).

`--skip-op OP` (an op name) sets a known finding aside without changing the rest of the program.
`--show SEED:INDEX:WAVE` prints a program and its model.

#### Hardware oracle (`--oracle hw`, `translator,hw`)

`--oracle hw` runs the same programs natively on the AMD GPU through `tools/hw_oracle/` (setup: "Hardware oracle"
above; exit 77 without it) and compares the 16 result VGPRs per thread (and the scratch memory for `mem`) with the
model, in wave32 and in native wave64, 64 threads, so it tests the **model** and not the translator. With
`translator,hw` both run on the same programs. `--negative-control` runs deliberately wrong models (six for `f16`,
eight for `mem`) and fails unless each one is caught.

```
python -I tools/valu_fuzz.py --family mem --oracle hw --programs 2000 --seed 1
python -I tools/valu_fuzz.py --family f16 --oracle hw --negative-control --programs 300
```

*What differs on GFX11 (gfx1150) and how it is handled.* The programs are the same ones the translator gets, so
every form has to mean the same on gfx1013 and gfx1150. Checked against Mesa ACO (`aco_ir.cpp`: `instr_is_16bit`,
`can_use_opsel`, `get_gfx11_true16_mask`), the `gfx11-rsrc.json` register description and by running them:

| Form | GFX10 (gfx1013) | GFX11 (gfx1150) | Handling |
|---|---|---|---|
| f16 VOP2 `_e32`/`_e64` (add, sub, mul, min, max, fmac), `v_cvt_f16_f32`, `v_cvt_f32_f16`, `v_cvt_pkrtz_f16_f32` | low half written, high half kept | same (ACO treats both generations alike; true16 bit 7 of a VGPR selects the half only for v128 and up, which the programs never use) | run |
| `v_fma_f16`, `v_med3_f16`, `v_add_nc_u16`, `v_sub_nc_u16` VOP3 with `op_sel` | `op_sel` source bits, destination bit = high half, other half kept | same | run |
| `v_mul_lo_u16`, `v_lshlrev/lshrrev/ashrrev_b16`, `v_max/min_u16/i16` VOP3 with `op_sel` | same fields | same fields and meaning, but LLVM's gfx11 assembler has no `op_sel` operand for these, and GFX11 renumbered the three shifts (`0x314..0x316` to `0x338..0x33a`) | run: with `op_sel` nonzero the instruction words are written out (`GFX11_VOP3_OPCODE`, checked against llvm-mc in the tests); all-zero `op_sel` uses the plain mnemonic |
| `v_pk_add/mul/max/min/fma_f16`, `v_pk_lshlrev/lshrrev_b16`, `v_pk_ashrrev_i16` with `op_sel`/`op_sel_hi` | per-half selection | same | run |
| `v_pack_b32_f16` | halves moved under IEEE mode 0; under IEEE mode 1 a **signaling NaN** half is quieted (`0x7c96` becomes `0x7e96`; measured on gfx1036, see the RDNA2 results below) | same, gfx1150 quiets it too (measured under IEEE mode 1 only) | run; a signaling-NaN half is "uncertain" in the model (the kernels run with IEEE mode 1) |
| f16 transcendentals (`v_exp/log/rcp/rsq/sqrt/sin/cos_f16`) | write the low half | same | not generated: covered by FUZZ-29 (#1310), which sweeps all 65,536 inputs on the hardware oracle |
| 16-bit integer VOP2 `_e32` forms (`v_add_u16_e32`, ...) | exist, define the whole register (design.md) | **do not exist** (only VOP3, with `op_sel`) | not generated, nothing to compare |
| `v_pk_fmac_f16`, `v_mad_f16`/`v_mac_f16`, `v_subrev_f16` | various | removed or not in this fuzzer | not generated |
| f16 VOP1/VOP2 with VGPR operands v128 and up | whole VGPR | bit 7 selects the high half of v(n-128) (true16) | not generated (the programs use v20..v35, `v0..v127` is what LLVM's fake16 encoding allows too) |
| V# word 1 / word 3 | STRIDE `[29:16]`, SWIZZLE_ENABLE bit 31; FORMAT 7 bits `[18:12]`, RESOURCE_LEVEL bit 24, OOB_SELECT `[29:28]` | STRIDE `[29:16]`, SWIZZLE_ENABLE `[31:30]`; FORMAT 6 bits `[17:12]`, no RESOURCE_LEVEL, OOB_SELECT `[29:28]`, TYPE `[31:30]` = 0 | the hardware kernels build the GFX11 word 3 (`word3()`), the translator programs the GFX10 one; same logical semantics (`--gfx10-word3` runs the old word on gfx1150: it still works, bit 24 is ignored) |
| `buffer_load_dword`, `s_load_dword`, ... | `dword` names | `b32` names | the hardware kernels use the family's spelling (LLVM accepts both) |

*Memory ordering the kernel needs on GFX11 (the model has none).* Measured with `--hw-sync` on 600 to 1,500
programs each: with no wait before an `s_load`/`s_buffer_load` that follows a store, 18 of 300 wave32 programs read stale
data; `s_waitcnt_vscnt null, 0` alone leaves 7 of 300, because an earlier scalar load can have left the line in the
scalar cache, which is not coherent with vector stores; `s_dcache_inv` alone leaves 69 of 1,500 (the store had not
left the pipe); both together give 0 mismatches over 20,000 programs, and adding `buffer_gl0_inv` and `buffer_gl1_inv`
changes nothing (a vector load after a store of the same thread needs neither). The default (`--hw-sync scalar`) is
vscnt wait plus `s_dcache_inv`; vector accesses only need `s_waitcnt vmcnt(0)`.

*Results (Radeon 880M, gfx1150, driver 26.11.2, AMD clang 23.0.0git).* Hardware, 5 seeds (101 to 105) x 2,000 programs x wave32 and native wave64 per family: **`mem` 20,000 of 20,000 match, `f16` 20,000 of 20,000 match, 0 mismatches, 0 assembler rejections** (plus earlier development runs of a further 15,000 `mem` and 12,000 `f16` programs; the one mismatch they produced, the `v_pack_b32_f16` NaN quieting above, is now modelled as uncertain). Negative controls (`--negative-control --programs 500`, 1,000 programs per row, unperturbed model: 0 mismatches): `f16` hi half zeroed 994, `op_sel` destination ignored 863, packed halves swapped 864, `v_min` as `v_max` 424, `v_pack_b32_f16` halves swapped 254, `op_sel` of the encoded 16-bit integer ops ignored 862; `mem` OOB_SELECT ignored 506, mode 0 without the stride check 54, index unchecked 364, soffset left out of the check 69, whole-access instead of per-dword 45, out-of-range loads keeping the old value 590, `s_buffer_load` unbounded 209, its stride ignored 81. Every perturbation is caught. Not run here: `translator,hw` for `mem` (see below); RDNA2 is in the next paragraph.

*Results on RDNA2 (gfx1036 integrated GPU, AMD Windows driver 32.0.21044.64, HIP backend, AMD clang 23.0.0git, `main` bb2e1842).*
- **Hardware, same protocol** (seeds 101 to 105 x 2,000 programs x wave32 and native wave64 per family): **`f16` 20,000 of 20,000 and `mem` 20,000 of 20,000 match, 0 mismatches, 0 assembler rejections.** On a gfx1030, ROCm 6.2's clang 19 rejected 3,768 of 4,000 `f16` programs (#1365); clang 23 rejects none.
- **Negative controls**, in the order above: `f16` 992, 865, 826, 430, 280 and 879 caught; `mem` 514, 57, 354, 60, 37, 580, 214 and 89 caught. The unperturbed model gives 0 mismatches.
- **`translator,hw` at seed 3** (100 programs, the same GPU through the AMD Vulkan driver): `mem` 200 of 200 match, `f16` 199 of 200. The one difference, `s3_22_w64`, is a driver mismatch of the #1506 kind. The hardware and `tests/semantic/spirv_mini_interp.py` both give the model's value for it, and the same emulated SPIR-V matches on NVIDIA.
- **`v_pack_b32_f16` with modifiers (`neg`, `abs`, `-|x|`) and signaling NaNs**, measured with 1,088 hand-written raw kernels in both wave sizes (wave32 and wave64, 100% parity) and both IEEE modes on gfx1036 (driver 32.0.21045.29003). The inputs covered normal floats (±1.0), zeros (±0.0), infinities (±inf), subnormals (±min_sub), quiet NaNs, and signaling-NaN halves, in either half with `op_sel`.
  - IEEE mode 1, the assembler's default: a signaling-NaN half is quieted (bit 9 set; sign bit cleared by `abs` and flipped by `neg`, e.g. `0xfc01` → `0x7e01` under `abs`, `0xfe01` under `-|x|`).
  - IEEE mode 0: the half is moved as pure bit operations (`& 0x7FFF` for `abs`, `^ 0x8000` for `neg`, `(x & 0x7FFF) ^ 0x8000` for `-|x|`), with `sNaN` payload and quiet bit preserved untouched.
  - Wave32 and wave64 produce identical results across all 1,088 kernels (0 differences).
  - The translator implements the IEEE mode 0 bit moves, which is exact on hardware.

*Findings.* The OOB_SELECT per-mode rules, the per-dword straddle rule and the `s_buffer_load` bound of design.md hold
on gfx1150 (they were measured on RDNA2 and, for the vector part, on this GPU before; the `s_buffer_load` bound and the
`idxen offen` and offset-only soffset forms are new on gfx1150). The one difference is `v_pack_b32_f16` quieting
signaling NaNs (only signaling NaN halves are not judged; a quiet NaN is compared). The default translator path of `mem` ran on the Radeon 880M (`VK_LOADER_DRIVERS_SELECT=*amd*`, `vk_compute_test` built with g++ as in the lab script): `--family mem --programs 200 --seed 1` gave 400 of 400 programs (native wave64 and wave32) matching the model, 0 not translated, 0 harness errors, so the translator agrees with the model on the OOB_SELECT, straddle, `s_load` and `s_buffer_load` rules there. On the RTX 5070 `vk_compute_test` loses the device (-4) for this and for the unchanged generator of `main`, so use the AMD device.

### Packed and 16-bit integer VALU (`tools/pk16_diff.py`)

`valu_fuzz.py --family f16` runs the 16-bit shifts, `v_add/sub_nc_u16`, `v_mul_lo_u16` and the 16-bit min/max on
the GPU with VGPR sources. `tools/pk16_diff.py` adds the packed integer ops (`v_pk_add/sub_u16/i16`,
`v_pk_mul_lo_u16`, `v_pk_min/max_u16/i16`, `v_pk_mad_u16/i16` and the packed shifts) with `op_sel`, `op_sel_hi` and
`clamp`, and the 16-bit VOP3 ops (`v_add/sub_nc_i16`, `v_min3/max3/med3_u16/i16`, `v_mad_u16/i16` next to the ones
above) with `op_sel` on every source and on the destination half and `clamp`. Sources are VGPRs, SGPRs, integer and
float inline constants and literals within the constant-bus limits. It runs one lane through
`tests/semantic/spirv_mini_interp.py` against its own ISA model, with the shrinking, skips and exit codes of
`salu_diff.py`. A constant whose high half a packed add/sub/mul/min/max/mad reads is left out unless `--known` (the
translator rejects it; #2224 changes that).

Measured on an RDNA2 iGPU (gfx1036) with `--oracle hw`: 2,300 programs, the `--known` forms included, match the
model. A 16-bit VOP3 op keeps the other half of its destination. A constant source is its 32-bit value split like a
register: `-16` is `0xfff0`/`0xffff`, `1.0` is `0x0000`/`0x3f80` (f32 bits, unlike the f16 ops), and a literal of a
16-bit VOP3 op has 16 bits (`0xffff` with `op_sel` reads 0). The VOP2-named 16-bit ops (`v_mul_lo_u16`,
`v_min/max_*16`, the shifts) only have a VOP3 encoding on GFX10, and the generator writes them with `_e64`, because
without encoding bytes the translator reads a bare mnemonic as the VOP2 write that zeroes the high half. For the same
reason, rules that depend on the encoding bytes, such as a literal's 12-byte VOP3 encoding (SHADER-208), are out of
its reach.

```
python -I tools/pk16_diff.py --programs 3000 --seed 11 --max-ops 8
python -I tools/pk16_diff.py --oracle hw --programs 500 --seed 1
python -I tests/test_pk16_diff.py
```

### Wave and cross-lane ops on the GPU (`tools/crosslane_fuzz.py`, FUZZ-7)

Cross-lane ops read other lanes, so like FUZZ-6 this runs on the GPU and not in the one-lane interpreter.
Each seeded program is one compute group of a wave32 or wave64 program. Group sizes cover one full wave, two or
three waves, and a partial last wave (20, 50, 40 or 100 threads). The program runs 24 to 28 random ops, then
every thread stores its six data VGPRs and eight data SGPRs, so each half of a wave reports its own copy of
every scalar. `vk_compute_test` reads them back with `VKC_OUT_WORDS` / `VKC_DUMP`.

- **Instructions.** `v_readlane_b32` and `v_writelane_b32` (constant and SGPR lane selects; in wave32 also
  selects of 32 and up), `v_readfirstlane_b32`, `v_cmp` into VCC or an SGPR mask, `v_cmpx`, `s_and_saveexec`
  and its restore, nested up to three deep, constant EXEC halves, `s_bcnt1`/`s_bcnt0`/`s_ff1` of a mask or
  EXEC, `v_mbcnt_lo/hi` on a mask, EXEC or plain data, `v_cndmask` on a mask, `s_and/or/xor/andn2` of masks,
  `v_permlane16_b32` / `v_permlanex16_b32`, and forward `s_cbranch_execz/vccz/vccnz` over one to four of these.
  With `--dpp`: a DPP16 or DPP8 src0 on `v_mov`/`v_not`/`v_add_nc_u32`/`v_xor`, with every DPP16 control,
  `row_mask`, `bank_mask`, `bound_ctrl` and `fi`. Masks read as plain data (`v_mov_b32 v, s_mask`) belong to
  #293 and are left out with `--no-maskdata`.
- **`--ext` (wave64 programs only).** Adds the ops the list above leaves out: carries (`v_add_co`, `v_sub_co`,
  `v_subrev_co_u32` and the `_ci` forms) through SGPR pairs and VCC; `ds_bpermute_b32`, `ds_permute_b32` (address
  permutations and colliding addresses) and `ds_swizzle_b32` (`QUAD_PERM`, `BITMASK_PERM`, `SWAP`, `REVERSE`,
  `BROADCAST`); `v_readlane`/`v_writelane` with an SGPR lane select of 64 and up; `s_bcnt0/1` and `s_ff0/1` of a
  mask, `s_and/or/xor/andn2_b64`, the ten `s_*_saveexec_b64` forms and `s_andn1/andn2_wrexec_b64`, `s_wqm_b64`,
  `s_cmp_eq/lg_u64` and `s_bitcmp0/1_b64`, each followed by an `s_cselect_b32` so that its SCC is stored;
  `s_cselect_b64` between masks, `s_mov_b64 exec, mask`, `s_bfm_b64` into a mask or EXEC; and a loop in LLVM's
  SI_LOOP shape (per-lane trip counts of 1 to 8, a break mask, `s_andn2_b64 exec`, `s_cbranch_execnz` back to the
  head) around one to four of the other ops. Without the flag, and for wave32 programs with it, a seed gives the
  same program as before. The model of every op was checked on an AMD Radeon RDNA2 iGPU (gfx1036, Ryzen 9
  9950X3D, raw wave64 kernels): `--ext --dpp --wave 64 --oracle hw --programs 100 --seed 21` matches 100/100,
  with groups of one to three waves and partial waves. What that run settled where the ISA guide is terse: the
  three DS ops need `s_waitcnt lgkmcnt(0)` before their result is used (without it a later VALU write to the same
  VGPR is overwritten when the result arrives; the generator emits the wait); `ds_bpermute_b32` and
  `ds_permute_b32` stay inside the lane's 32-lane half; with several senders `ds_permute_b32` keeps the highest
  lane's value; an EXEC-off source reads 0 in `ds_bpermute_b32` and in `ds_swizzle_b32`. The translator matches
  the same 100 programs on that GPU both natively (64-wide subgroup, `--w64-native`) and with
  `--wave64-emulate`, apart from 15 that reach a known open case and are filed `hyp_only`: an EXEC-off
  `v_permlane` source (#580) or `ds_swizzle_b32` source (#2780, tagged like the permlane case until it lands). An
  RTX 5080 (driver 616.92) gives 46 matches and 14 `hyp_only` of 60 emulated programs (`--seed 23`). Not
  generated: 32-bit ops on one half of a mask or of EXEC (#663, #3225) and carries with a DPP source (#3553).
- **Reference.** A 64-lane model in the same file, written from the ISA (rules in its docstring). Its unit
  test, `tests/test_crosslane_fuzz.py`, needs no GPU.
- **Defined behaviour only.** Lane selects, permlane and DPP never read a lane missing from a partial wave.
  EXEC constants only enable lanes that exist. Every mask register is zero-initialised, as compiled code does,
  so a branch that skips the `v_cmp` writing a mask leaves a defined mask after the join.
- **Hypotheses are reported apart.** `v_permlane16/x16` with op_sel FI = 0 reading a lane that is off in
  EXEC is modelled as the DPP "invalid source" case (vD kept). That is now measured, with the same result on both:
  @aslan2142 on RDNA2 and @AlxcBTW on RDNA3.5 (#374). FI = 1 returns the source lane's value, FI = 0 leaves the
  destination unwritten and FI = 0 with BOUND_CTRL gives 0; #580 (SHADER-74) makes the translator do that. The
  fuzzer's docstring (`tools/crosslane_fuzz.py`, since #750) already says it was measured, but `op_permlane` still
  tags the case `hyp` because the translator models it only with #580, so a program that hits it and mismatches
  is filed as `hyp_only` instead of `mismatch` (one that matches still counts as `match`). The model (vD kept)
  already matches the measurement; dropping the tag once #580 lands is a follow-up.
  `--permlane-fetch-inactive` models FI = 1 instead.
- **Modes.** Wave32 runs natively. Wave64 runs with `--wave64-emulate`; `--w64-native` adds the plain build,
  which on a 32-wide subgroup is the documented cross-half approximation and is expected to mismatch.
- **Hardware oracle (optional).** `--oracle hw` (or `translator,hw`) also runs every program natively on the AMD
  GPU through `tools/hw_oracle/` (setup: see the FUZZ-4 section above) and compares all stored words with the model;
  without the tools it prints `hardware oracle unavailable: ...` and exits 77. It tests the **model**, in wave32
  and in native wave64 (`.amdhsa_wavefront_size32 0`, wave64 code objects are assembled with
  `-mwavefrontsize64`), with the program's real group size (partial waves, two or more waves) and its EXEC setup,
  one thread per lane, each storing its 14 words with `global_store_b32`. Branch placeholders become labels. After
  every instruction the native kernel has `s_nop 7`, because the model has no hazards and the real unit wants wait
  states after some VALU writes of SGPRs, VCC and EXEC; `--no-hw-guard` leaves them out (a 4,000-program run
  without them matched too, so the generator does not reach a hazard on this GPU). With `translator,hw`, a program
  where the hardware differs from the model also prints what the translator did with it.
  `hyp_confirmed` counts programs with a hypothesis op that matched on the hardware, split into `hyp_permlane`
  (FI = 0 source lane off in EXEC keeps vD; the claim comes from the oracle measurements in #374 on RDNA2 and
  RDNA3.5, and #580 models it in the translator) and `hyp_dpp` (DPP8 source lane off in EXEC with fi = 0 reads 0;
  the claim is #238's, and nothing measured it before this oracle). On the hardware path a hypothesis that fails
  is a falsified claim: it counts as a mismatch (exit 1) and in `hyp_only`, and `--minimize` marks the
  hypothesis ops that remain. A startup probe checks that a wave64 kernel runs as wave64 (lane 32 reads 32) and a
  wave32 kernel as wave32, so a runtime that ignored the wave size would fail loudly. `--keep` writes the kernel
  body that ran.
  GFX11 differences looked at, none of which changes a result for what the generator emits: the assembler takes
  the same `row_share`/`row_xmask`/`dpp8`/`fi`/`bound_ctrl:1` and `v_permlane16/x16` spellings for gfx1030 and
  gfx1150; `row_newbcast`, `row_bcast` and `wave_shl` exist on neither (not generated); `v_cmpx_*_e32` writes EXEC
  only on both (the generator never uses the `_e64` form, whose GFX10 SDST the assembler no longer takes either);
  `v_mbcnt_*` is VOP3-only on both; `s_ff1_i32_*` is the GFX10 name of GFX11's `s_ctz_i32_*` (an accepted alias);
  `v_readlane_b32` with a lane select of 32 and up in wave32 uses select bits [4:0] on gfx1150 too. Nothing
  needed a skip, so `hw_skip_reason` returns none and the only skips are assembler rejections (none seen). That
  was gfx1150.
  On RDNA2 (gfx1036, through the Linux launcher of the FUZZ-4 section): 24,000 programs (seeds 1 to 6) match in
  both wave sizes, 12,000 each, 6,586 of them `hyp_permlane`; with `--dpp` (seeds 1 and 2) 8,000 more match,
  972 of them `hyp_dpp`. As a control, `--permlane-fetch-inactive` there mismatches in 239 of 1,000 programs. A
  GFX10 assembler has no `global_store_b32`, so `asm.raw_kernel_text` puts the GFX10 name (`global_store_dword`)
  back for `gfx10xx`; nothing else the generator emits needed a change.
  On a discrete RDNA2 card (RX 6700 XT, gfx1031, Windows, Adrenalin 26.8.1, through the HIP DLL of the driver;
  BUILD-162): with `--dpp`, seeds 1 to 8, 1,000 programs each (500 per wave size), all 8,000 match, 984 of them
  `hyp_dpp` and 1,964 `hyp_permlane`.
  On RDNA4 (RX 9070 XT, gfx1201, Windows, Adrenalin 26.11.2, `amdhip64_7.dll`, AMD clang 23.0.0git from the
  `rocm-sdk` wheels): the same eight runs, all 8,000 match, with the same 984 `hyp_dpp` and 1,964 `hyp_permlane`
  programs. gfx1201 is not RDNA2: the match supports the model and settles nothing about the console.
- **DPP sweep (`--dpp-sweep`, BUILD-162).** The random `--dpp` programs run full waves, four ops and mostly
  `row_mask`/`bank_mask` 0xF, so a rare control or mask pair can go untested for many seeds. `--dpp-sweep` replaces
  the random programs with one short program per case: an EXEC write, then one DPP op. It covers every DPP16 control
  (335: 256 `quad_perm`, `row_shl/shr/ror` 1 to 15, both mirrors, `row_share` and `row_xmask` 0 to 15) × `bound_ctrl`
  × `fi` × three EXEC patterns (all lanes; every other lane, so each source is invalid in some lanes; and bank 0 off
  in some rows, so whole 4-lane groups read inactive sources). It also covers all 256 `row_mask`/`bank_mask` pairs on
  `row_shr:1`, `quad_perm:[1,0,3,2]` and `row_share:5`, × `bound_ctrl`, under two EXEC patterns. And it covers 19
  DPP8 select sets (identity, reverse, swaps, the 8 broadcasts and the 7 rotations) × `fi` × the three EXEC patterns.
  That makes 7,206 cases per wave size. The op rotates through `v_mov`/`v_not`/`v_add_nc_u32`/`v_xor`, and every
  other group of four writes vD in place (`vD = src0`, as reductions do), so the model has to read every source
  before it writes. Each case carries its family label, and failures print it. The run ends with a table of cases
  and failures per family. The random draws of `--dpp` are unchanged: a test pins the program text of seed 77, and
  800 programs from before and after the change are identical.
  On the RX 6700 XT (gfx1031), `--dpp-sweep --oracle hw --wave both`: 14,412 cases, match 14,412, mismatch 0 in
  every family, 40 of them `hyp_dpp` (DPP8 with an inactive source and fi = 0 reads 0). This is the first measured
  check of every DPP16 control and mask pair on RDNA2 hardware, and it confirms the rules in the model's docstring
  (and `docs/design.md`): `bound_ctrl` with an invalid source, `fi` on DPP16 and DPP8 inactive sources, an
  unwritten lane keeping the old vD for non-mov ops, and `row_mask`/`bank_mask` in both wave sizes. In the same
  run (`--oracle translator,hw`), the translator matches all of them as well: 7,206 wave32 native and 7,206 wave64
  `--wave64-emulate`, with 0 rejected and 0 harness errors. One `vk_compute_test` runs per case, so the translator
  side takes about 90 minutes there; the hardware side takes about one.
  On the RX 9070 XT (gfx1201, the setup above), `--dpp-sweep --oracle hw --wave both`: 14,412 cases, match 14,412,
  mismatch 0 in every family, 40 of them `hyp_dpp`, in half a minute. The same DPP rules hold on RDNA4; the
  translator side was not run there.
  First results (Radeon 880M, driver 26.11.2): 24,000 programs on the hardware, half wave32 and half native
  wave64, seeds 11 to 16 (4,000 programs each), with and without `--dpp`, 12 to 60 ops, with and without
  `--no-maskdata` and `--no-hw-guard`: match 24,000, mismatch 0, hyp_only 0, rejected 0. 7,652 of them contained
  a permlane with an inactive source lane or a DPP8 with an inactive source lane (`hyp_confirmed`: 3,567 in
  wave32, 4,085 in wave64) and every one matched, so both hypotheses hold on gfx1150 in both wave sizes. RDNA2
  itself is not measured by this run.
  Against the translator (`translator,hw`, 200 programs, seed 7): every translator mismatch of this build had
  masks read as plain data (#293); with `--no-maskdata` it has 0 mismatches and only the known permlane
  `hyp_only` cases (#580). The hardware agrees with the model on both, so those are translator gaps, not model
  errors.
- **Reports.** The thread, wave, lane and register of the first differences. `--minimize` drops ops while the
  mismatch persists, and `--keep DIR` saves the programs. `--translator PATH` tests another checkout's
  translator (for example an open pull request).

First results on an RTX 4090 (subgroup 32, driver 621.14):

- **SHADER-52 (#407).** In wave32, `v_readlane`/`v_writelane` masked the lane select with 63, not 31. 23 of
  100 wave32 programs failed, each one minimized to a `v_writelane` with a lane select of 32 or more.
- **Two DPP bugs in #238** (SHADER-23, posted there). Non-mov DPP ops wrote `op(old vD, src1)` in lanes that
  should be left unwritten (invalid source without `bound_ctrl`, or a lane off in `row_mask`/`bank_mask`), and
  DPP16 ignored `fi:1`. Each was confirmed with a hand-written program.
- **Everything else matches.** On `main`, 1,000 programs (`--seed 8 --ops 28 --no-maskdata --no-w32-hi-lanes`, 500 wave32 native and 500 wave64 emulated) give 0 mismatches. 297 of them are `hyp_only` (permlane with FI = 0 and an inactive source), and with `--permlane-fetch-inactive` all 1,000 match. Native wave64 on the 32-wide card mismatches as documented: `s_bcnt1` of EXEC counts one half, and a `v_readlane` from the other half wraps.
- **Native wave64 on a 64-wide device: one driver fault (SHADER-705).** With `--wave 64 --w64-native --dpp --no-maskdata
  --seed 82001 --programs 600`, AMD's Windows driver on a gfx1036 integrated GPU (`32.0.21045.16010` and
  `32.0.21042.62`, two machines; found by @Vonsek, reduced by @zzEmmazz) gives one native mismatch:
  `cs_xl64_8200346406` (index 403) differs in 96 words, lanes 32-63 of each wave on the first driver and lanes 0-15
  and 32-47 on the second. The emulated build matches there.
  - **The same run under RADV** (Mesa 26.3.1, Linux) on another gfx1036 (a Ryzen 9 7950X3D's) and on an RX 7900 XTX:
    454 match, 0 mismatch, 146 `hyp_only`, natively and emulated. The program alone, and the 12 instructions it was
    reduced to, are exact natively on both devices: as 192 threads and as one 64-thread group, with the driver's
    default subgroup size and with `vk_compute_test --require-wave64`.
  - **Hardware:** `--oracle hw` on that gfx1036 matches the model in all 600 programs.
  - **So the native wave64 SPIR-V is right and that driver is not.** The wrong instruction is a `v_permlanex16_b32`
    that reads a register a DPP op with a partial `row_mask` has just written; equivalent rewrites of the shuffle
    index only move the wrong lanes (@zzEmmazz).
  - **In the game's shaders that shape does not occur.** Of 26,503, 2,293 use `v_permlanex16_b32` and 27 a DPP op with
    a partial `row_mask`. 6 have both, all wave32 compute, and in none does the shuffle read a register that such a
    DPP op wrote last (program order, branches ignored).
  - **Not checked:** the driver's generated code, so the fault is not located and other shapes may trigger it; the
    Windows driver with `--require-wave64` (the fuzzer's native run does not request a subgroup size); AMD's
    Windows driver on RDNA3 or newer.
  - `--wave64-emulate auto` stays native on that driver, because it can crash on emulated LDS code (#857).
- **Closed question (#374):** what hardware does for permlane with FI = 0 was measured on RDNA2 and RDNA3.5 (see above).

### Wave and cross-lane ops on the CPU (`tools/wave_diff.py`)

The same family as `crosslane_fuzz.py`, without a GPU or llvm-mc, so it can run wherever the translator tests run.
The reference is `cf_ref.WaveMachine` (FUZZ-8) extended in `wave_diff.WaveRef` with the instructions it leaves out:
`v_readlane`, `v_readfirstlane`, `v_writelane`, `ds_bpermute_b32` (per 32-lane half, an inactive source reads 0),
`v_permlane16/x16` (FI = 0 with an inactive source leaves vD, as measured above), `s_bcnt0/1` and `s_ff0/1`
(`_b32`/`_b64`, constants sign-extended to the operand width) and `s_not`. Each rule is written next to its instruction.
The translated side is `spirv_mini_interp.run_workgroup`, every lane of the wave in one subgroup of the wave's size, and
all 12 dwords every lane stores are compared (four VGPRs, two masks, VCC and the final EXEC as each lane's bit, four
scalars). Programs are straight-line wave32 and wave64 compute under random EXEC (`v_cmpx`, saveexec, constants).

By default the generator keeps lane masks and 64-bit data in separate registers and only issues `v_permlane` with every
lane on, so a default run is a regression check. Two known gaps on `main` are opt-in: `--masks-as-data` (#210, #293) and
`--permlane-partial` (SHADER-74, #580).

Results (`--max-ops 16`, half wave32, half wave64, nothing skipped): it found `s_and_b32 exec_lo, exec_lo, K` in
wave32 and `s_ff0_i32_b64` of a negative constant (#1409), both fixed on `main` since; on `main` 7b7d73f8, 600 of
600 match (seeds 401 to 403).

### 32-bit float VALU on the GPU (`tools/f32_fuzz.py`, FUZZ-3)

`--oracle hw` (or `translator,hw`) runs the same seeded VOP1/VOP2/VOP3 f32 programs natively on the AMD GPU and judges
the hardware's `v8` and VCC against `F32Model` bit for bit (NaN matches any NaN; an f16 NaN in a `v_cvt_f16_f32` /
`v_cvt_pkrtz_f16_f32` half too). Setup is the one in "Hardware oracle" above; without it the run prints
`hardware oracle unavailable: ...` and exits 77. It tests the **model**, on wave32 with one lane; with `translator,hw`
a hardware mismatch also prints what the translated SPIR-V did.

```
python -I tools/f32_fuzz.py --oracle hw --programs 5000 --seed 1            # table + informational hardware tables
python -I tools/f32_fuzz.py --oracle hw --negative-control                   # wrong models must be caught
python -I tools/f32_fuzz.py --oracle hw --perturb mul_rtz --programs 500     # one wrong model, mismatches must appear
```

- **Float MODE.** An assembled kernel's descriptor starts with IEEE 1, DX10_CLAMP 1, f32 denormals flushed
  (`.amdhsa_float_denorm_mode_32 0`), f16/f64 denormals kept (3), round to nearest (MODE `0x3c0`, read back with
  `s_getreg_b32`). The primary mode, `flush`, is the state inferred for the game's shaders (IEEE off, f32 denormals
  flushed, DX10_CLAMP on: MODE `0x1c0`). Three more are run for what the model does not judge: `keep` (f32 denormals
  kept, `0x1f0`), `no_dx10` (`0x0c0`), `ieee` (`0x3c0`). The run reads MODE back in a kernel for each and fails if a
  directive did not take. Rounding is always to nearest even. On gfx1170+ (GFX11.7, GFX12) IEEE and DX10_CLAMP are
  fixed and AMD clang rejects `.amdhsa_ieee_mode` / `.amdhsa_dx10_clamp` ("directive unsupported on gfx1170+"), so
  `asm.mode_skip_reason` reports a kernel that sets either as not runnable there and the runner never assembles it;
  the MODE probe then fails naming that reason instead of a whole chunk failing to assemble (FUZZ-43).
- **GFX11 against GFX10, handled explicitly** (`tools/hw_oracle/asm.py`, checked by assembling every opcode for gfx1013,
  gfx1010, gfx1030, gfx1100, gfx1150 and gfx1200):
  - *Skipped, never substituted:* `v_mad_f32`, `v_mac_f32`, `v_madak_f32` and `v_mad_legacy_f32` exist on GFX10.1 only
    (gfx1010/1011/1013, the console's family); gfx1030 and every GFX11 part reject them. The fused replacements
    (`v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_fma_dx9_zero_f32`) round once, so they are different instructions. The
    unfused mad semantics (and SHADER-118's flush) therefore stay unmeasured: no RDNA2 desktop part and no GFX11 part has
    them. Also not generated: `v_madmk_f32` (GFX10.1 only, like the mad opcodes), `v_min_legacy_f32` (in no GFX10 or GFX11
    part: llvm-mc rejects it for gfx1010, gfx1013, gfx1030 and gfx1150), `v_fma_legacy_f32`/`v_fmac_legacy_f32` (new in
    GFX10.3, not in gfx1013).
  - *Renamed, run:* `v_mul_legacy_f32` is `v_mul_dx9_zero_f32` in GFX11 (the DX9 rule `0 * x = 0`, inf and NaN included,
    held in every program) and `v_cvt_pkrtz_f16_f32` is `v_cvt_pk_rtz_f16_f32`; the assembler takes the old spelling.
  - *Same, with a syntax change:* GFX11's assembler wants `clamp` before `mul:N`/`div:2`; the native kernels print them in
    that order (the same bits). Opcode numbers differ between the families (`v_cmp_class_f32_e32` is `7d10` on gfx1013 and
    `7cfc` on gfx1150); only mnemonics are shared. omod and clamp exist in VOP3 on both; no op_sel is used for f32.
    The transcendental unit is separate on GFX11, so a `s_nop 7` follows the instruction in the native kernel
    (`--no-hw-guard` leaves it out).
  - *Not covered:* wave64 (the f32 ALU does not depend on it), `v_fma_mix*` and f16 arithmetic. The translated f64 VALU
    forms have a separate CPU differential harness in `tools/f64_diff.py` (below).
- **Programs.** The seeded random ones, plus `--probes N` (default 12) per op and kind: denormal operand, denormal
  result, NaN payloads, clamp of NaN/-0, invalid operations (inf - inf, 0 * inf, ...), and `cancel` (`z = -round(x * y)`,
  which separates fused from unfused). `v_min/max/min3/max3/med3_f32` are in the table for the hardware only, with the
  RDNA2 rules measured in #634.
- **Judged / not judged.** A program is *match* or *mismatch* where the model has an answer. Where it declines (denormal
  operand or result, clamp of NaN or -0, cube of NaN) the hardware still runs and is recorded in the informational tables,
  also under the other three modes. A *known deviation* is hardware behaviour already analysed and written down as a model
  alternative (`KNOWN_DEVIATIONS`); the hardware words must equal the alternative exactly, anything else is a mismatch.

Results (Radeon 880M, gfx1150, driver 26.11.2, AMD clang 23.0.0git), seeds 1 to 6, 5,000 random programs + 24 probes per
op and kind each, 46,967 programs in all: **30,577 match, 0 mismatch, 57 known deviation**, 5,417 skipped for GFX11,
5,903 denormal, 5,013 other not judged (clamp of NaN or -0, cube of NaN). "Match" accepts any NaN for any NaN: payload, sign and quiet bit are in the informational tables, not judged. Seed 1 by family:

| Family | Programs | Match | Mismatch | Known dev | GFX11 skip | Denormal | Other |
|---|---|---|---|---|---|---|---|
| add / sub / mul | 921 | 631 | 0 | 0 | 0 | 176 | 114 |
| DX9 legacy mul / mad | 453 | 133 | 0 | 0 | 239 | 51 | 30 |
| mad / mac / madak | 698 | 0 | 0 | 0 | 698 | 0 | 0 |
| fma / fmac / fmaak | 700 | 509 | 0 | 0 | 0 | 132 | 59 |
| int to f32 convert | 708 | 708 | 0 | 0 | 0 | 0 | 0 |
| f16 convert | 478 | 383 | 0 | 0 | 0 | 90 | 5 |
| `v_cvt_pk_u8_f32` | 183 | 130 | 0 | 0 | 0 | 53 | 0 |
| floor / ceil / trunc / rndne | 795 | 582 | 0 | 5 | 0 | 96 | 112 |
| rcp / rsq / sqrt / exp / log | 956 | 704 | 0 | 0 | 0 | 119 | 133 |
| cube | 786 | 435 | 0 | 1 | 0 | 104 | 246 |
| `v_cmp_class_f32` | 171 | 147 | 0 | 0 | 0 | 24 | 0 |
| min / max / med3 (hardware only) | 989 | 732 | 0 | 1 | 0 | 128 | 128 |

Negative control: `--negative-control` re-judges the same hardware words with a deliberately wrong model for six ops
(`v_add_f32` low bit flipped, `v_mul_f32` rounding toward zero, `v_fma_f32` unfused, `v_mul_legacy_f32` as IEEE multiply,
`v_rndne_f32` as floor, `v_min_f32` NaN rule changed); every one produced mismatches (1 to 112 per run; the `v_fma_f32` one only after the `cancel` probes were added, a plain random sweep missed it in seed 2;
the legacy multiply is the weakest, 1 to 4 in seeds 1 to 6, because `0 * inf` is rare in random programs; the `zero_inf` probes (a zero against inf, NaN or a normal number, DX9 multiply) were added afterwards and catch it at small sizes). The `cancel` probes (seeds 2, 7, 8) give 72 of 72 fused-FMA matches. Seeds 1 to 6 ran without them.

On RDNA2 (gfx1036, through the Linux launcher of the FUZZ-4 section, the same seeds 1 to 6): 38,858 programs,
**28,671 match, 0 mismatch, 56 known deviation**, 4,224 skipped (the four mad opcodes are GFX10.1-only), 3,097
denormal, 2,810 other not judged. The four MODE settings read back in the kernels as `0x1c0`, `0x1f0`, `0xc0` and
`0x3c0`, so the float mode of the kernel descriptor reaches the hardware through that launcher as well.
`--negative-control` catches all six wrong models there (3 to 25 mismatches each).

The same seeds through the Windows HIP launcher (`hw_oracle.c`) on the integrated GPU of a Ryzen 7 9800X3D (gfx1036;
Windows 10 22H2, AMD driver 32.0.13036.4, `amdhip64_6.dll`, AMD clang 23.0.0git from rocm-sdk-core 10.0.0, main at
41f44826) give the same counts in every column: 38,858 programs, **28,671 match, 0 mismatch, 56 known deviation**
(all `omod_neg_zero`), 4,224 skipped, 3,097 denormal, 2,810 other. `--negative-control` again catches all six (3 to
25 mismatches). Two launchers on two machines and two operating systems agree on the hardware words.

`v_rcp_iflag_f32` (SHADER-136; added to the rcp / rsq / sqrt / exp / log family after the run above, so it is not in
its counts) on the same gfx1036 through the Windows launcher, `--ops v_rcp_iflag_f32 --programs 1000 --probes 24`,
seeds 1 to 6: 6,552 programs, **5,971 match, 0 mismatch, 35 known deviation** (`omod_neg_zero`), 279 denormal, 267
other. It shares `v_rcp_f32`'s model, and the clamp and omod bits behave as on the other ops (summed over the six
seeds, per MODE setting flush / keep / no_dx10 / ieee): clamp makes a -0 result +0 in every mode (98 each) and a NaN
result +0 except with DX10_CLAMP off, where the NaN passes (169); omod is applied with IEEE off, including with f32
denormals kept (461 each; like the other transcendental ops), and ignored with IEEE on (461); a -0 result becomes +0
under omod (35 each, none with IEEE on); a signalling NaN passes unchanged with IEEE off and is quieted with IEEE on
(59); denormal operands read as zero and denormal results flush in every mode.

A discrete RDNA2 part (SHADER-353): an RX 6700 XT (gfx1031; AMD Windows driver 26.11.2, Vulkan 1.4.315, Windows 11
10.0.26200, `amdhip64_6.dll`, AMD clang 24.0.0git from rocm-sdk-core 10.2.0, MinGW gcc as the host compiler, main at
4eb6155d with this change), `--oracle hw --programs 5000`, seeds 1 to 6: 39,101 programs, **28,856 match, 0 mismatch, 66 known
deviation** (all `omod_neg_zero`), 4,056 skipped (the four GFX10.1-only mad opcodes), 3,201 denormal, 2,922 other. More
programs than on gfx1036, probably because `v_rcp_iflag_f32` and its probes are in the table now. MODE read back as `0x1c0`, `0x1f0`, `0xc0` and
`0x3c0`; `--negative-control` catches all six wrong models (5 to 23 mismatches). The omod table, summed over the six
seeds (flush / keep / no_dx10 / ieee), answers what the gfx1150 row of `docs/design.md` left open for RDNA2:

| omod, where it changes the result | flush | keep | no_dx10 | ieee |
|---|---|---|---|---|
| applied | 2,680 | 869 | 2,680 | 0 |
| applied, but a -0 result becomes +0 | 66 | 22 | 66 | 0 |
| ignored | 0 | 1,811 | 0 | 2,650 |
| -0 result kept (applied or ignored, same word) | 0 | 44 | 0 | 65 |
| other | 0 | 0 | 0 | 31 |

With denormals kept, the 869 + 22 applied are all `v_rcp/rcp_iflag/rsq/sqrt/exp/log_f32` and the four cube ops
(`OMOD_IGNORES_DENORM_MODE`), and the 1,811 ignored are all ALU ops. The 31 "other" with IEEE on are `v_min/max/min3/
max3/med3_f32` programs with a signalling NaN source, where the IEEE-off min/max rules of #634 do not apply; the omod
is not what differs there. Before SHADER-353, `omod_label` counted a -0 result that the omod cannot change as "applied",
which put 65 "applied" in the ieee column although no unambiguous program applied it there.

What the hardware answered where the model does not judge (gfx1150; the RDNA2 run above agrees where it is cited):

- **No model error found.** The model was right wherever it had an answer, in 30,577 programs; the min/max/med3
  rules measured on RDNA2 in #634 (one NaN gives the other operand, both NaN give src0, -0 < +0, med3 with a NaN is min3)
  also hold on gfx1150 for quiet and signalling NaNs with IEEE off.
- **On gfx1150 output modifiers turn a -0 result into +0** (`mul:2`, `mul:4`, `div:2`; 57 programs, every seed; a flushed
  denormal result, which is a -0, too). RDNA2 does the same: 56 programs in the same seeds on gfx1036. LLVM notes that omod mishandles signed zeros
  (`SIFoldOperands.cpp`) and D30209 describes -0 being forced to +0 before the modifier, which points the same way. The
  model and the translator both keep -0, so on that evidence the translator likely differs from the hardware
  (`translator,hw`: all 6 deviations of 2,000 programs are "matches the model, so it shares the model's answer"). Not
  fixed in this change; the finding is posted in the coordination issue (see #1006). A deviation is only accepted where
  the omod was active (not with IEEE on, not for ALU ops with f32 denormals kept).
- **omod only acts with MODE.IEEE = 0**, and for the ALU ops only with f32 denormals flushed: with `keep` the omod of
  add/mul/fma/min/max/cvt/rounding ops is ignored; the transcendental ops and the cube ops still apply it. LLVM says the
  same for both conditions (`SIFoldOperands.cpp`: "omod is ignored by hardware if IEEE bit is enabled"; "If output
  denormals are enabled, omod is ignored"). This supports
  the corpus inference (omod on 102,365 float instructions, so IEEE off and f32 denormals flushed in the game's shaders).
- **f32 denormals (SHADER-35's MODE question).** With `denorm_mode_32 = 0` a denormal operand is read as a zero of its sign
  and a denormal result is written as a zero of its sign, for add/sub/mul/fma/fmac/fmaak, min/max/med3, rounding ops and
  DX9 mul; with `3` they are exact. `v_rcp/rsq/sqrt/exp/log_f32` flush in both modes (as #634 measured on RDNA2).
  `v_cmp_class_f32` classifies a denormal as a denormal in both. `v_cubesc/cubetc` pass a denormal operand through.
- **clamp.** With DX10_CLAMP 1, clamp of a NaN gives +0; with 0 the NaN passes through. clamp of -0 gives +0 in every mode.
  The model still declines both.
- **NaN payloads.** With IEEE off a signalling NaN passes through add/mul/fma/rounding/transcendental ops unchanged
  (payload and sign); with IEEE on it is quieted (bit 22 set). With several NaN operands the first (lowest source) wins
  in add/mul/fma/fmaak/fmac; for min/max the rule above applies. `v_sub_f32` returns the NaN of src1 with its sign flipped
  (it computes src0 + -src1). Invalid operations (inf - inf, 0 * inf, sqrt/rsq/log of a negative) give `0xffc00000` in every
  mode and op, never `0x7fc00000`. `v_cvt_f16_f32` and `v_cvt_pkrtz_f16_f32` turn a NaN into an f16 NaN with the top 10
  payload bits and the quiet bit set (not the canonical `0x7e00` the model writes; payload not judged, quietness is).
  `v_cvt_f32_f16` keeps the f16 payload shifted left by 13 and does not quiet it (the model agrees).

### f32 misc ops and float compares (`tools/fmisc_diff.py`)

`tools/f32_fuzz.py` generates the f32 arithmetic. `tools/fmisc_diff.py` covers the f32 ops it leaves out:
`v_fract_f32`, `v_frexp_mant_f32`, `v_frexp_exp_i32_f32`, `v_ldexp_f32`, `v_cvt_u32_f32`, `v_cvt_i32_f32`,
`v_cvt_off_f32_i4` and `v_fmamk_f32`. It also covers the 16 `v_cmp_*_f32` conditions (e32 into VCC, e64 into an SGPR
mask or VCC) and `v_cmp_class_f32` e64, each read back with `v_cndmask_b32_e64`. Programs use VOP3 `neg`/`abs`,
`mul:2`/`mul:4`/`div:2` and `clamp`, with VGPR, SGPR, inline-constant and literal sources within the constant-bus
limits. It runs one lane through `tests/semantic/spirv_mini_interp.py` against its own exact-rational ISA model, with
the shrinking, skips and exit codes of `salu_diff.py`. Any NaN matches any NaN. It does not judge denormal operands
or results, a computed NaN read as an integer, the clamp of a NaN or an output modifier that gives a zero: those
programs are counted as skips with the reason.

`--oracle hw` runs the kernels under the MODE inferred for the game's shaders (IEEE off, f32 denormals flushed). With
the assembler's compute default (IEEE on) the hardware ignores `omod`: `v_fract_f32_e64 v4, -v16 mul:2` with v16 = 0.5
gives 0.5. On an RDNA2 iGPU (gfx1036), 1,800 programs gave 1,563 judged and 0 mismatches, so the model's rules hold
there. Those rules cover the `0x3f7fffff` cap of `v_fract_f32`, `fract(-0) = +0`, `frexp` of ±inf/NaN/±0, `ldexp`
overflow and underflow, saturating conversions, one rounding in `v_fmamk_f32`, unordered compares and the class bits.
Modifiers do not change whether a signalling NaN is signalling: `v_cmp_class_f32` with `-v`, `|v|` or `-|v|` on
`0x7f800001` still sets bit 0. The translator applies those modifiers with `OpFNegate`/`FAbs`, and in the
interpreter that quiets the NaN. That form is left out unless `--known` (`class-snan-mods`, posted as a finding).
`v_cmp_class_f32` without modifiers and `v_cvt_f32_ubyteN` are in `f32_fuzz.py`.

```
python -I tools/fmisc_diff.py --programs 3000 --seed 11 --max-ops 8
python -I tools/fmisc_diff.py --oracle hw --programs 500 --seed 1
python -I tools/fmisc_diff.py --oracle hw,vk --programs 600 --seed 11 --max-ops 8 [--wave 64]
python -I tests/test_fmisc_diff.py
```

`--oracle hw,vk` also runs the translated programs on a real Vulkan GPU (`tools/vk_fuzz_run.py`: llvm-mc for gfx1013,
`rdna2spirv.py`, `runtime/build/vk_compute_test`, `LEONIDA_VK_DEVICE`, `--build DIR`) as one wave of 32 or, with
`--wave 64`, 64 identical lanes (the lane masks as 64-bit pairs; the hardware kernels run in the same wave size), and
judges lane 0's four words against the hardware's, not the reference's: the interpreter cannot see a driver's denormal,
NaN and rounding rules. The seeds are computed from the buffer fill at run time (`--vk-constants` keeps them as
literals), because the AMD Windows driver (26.10.1) folds an ordered compare of a literal NaN to true in a wave32
pipeline (`v_cmp_lt_f32 vcc_lo, -10.0, NaN`, fine in wave64 and on the RTX 5080; with run-time operands it is right).
Differences are `denormal` (the reference declines the program for a denormal: f32 denormals are left to the driver),
`omod_neg_zero` (an output modifier made the hardware's -0 a +0, the translated -0 stayed; #1281) and `mismatch`,
which fails the run. Seeds 11 to 13, 600 programs each, up to 8 ops: Radeon gfx1036 wave32 597 match, 3 omod_neg_zero;
wave64 593 match, 3 denormal, 4 omod_neg_zero; RTX 5080 wave32 595 match, 3 omod_neg_zero, 2 denormal; no mismatch.

### Translated programs on a Vulkan GPU: `--oracle hw,vk` of `pk16_diff`, `vint_diff`, `wint_diff`, `sdwa_diff` and `fmisc_diff`

These fuzzers judge the translator in `spirv_mini_interp`, which cannot see what a driver does with the SPIR-V. `f32_fuzz`
has no Vulkan path (its GPU oracle is `--oracle hw`, above). With
`--oracle hw,vk` each program also runs through llvm-mc (gfx1013), `rdna2spirv.py` and `runtime/build/vk_compute_test`
(`tools/vk_fuzz_run.py`; `--build DIR`, `--work DIR`, `LEONIDA_VK_DEVICE`) as one wave, and its words are judged against
the hardware's words from a kernel run in the same wave size (`--wave 32` or `--wave 64`; wave64 needs native 64-wide
subgroups, so the Radeon and not the RTX 5080). `pk16_diff`, `vint_diff` and `fmisc_diff` run identical lanes and judge
lane 0, `sdwa_diff` too (its VCC seed differs per lane), `wint_diff` runs its per-lane programs in the wave size they were
generated for and judges every lane. The `v_mov_b32 vN, K` seeds of the lane-identical tools become `K ^ 0xEEEEEEEE`
xor the buffer fill vk_compute_test leaves in the output (`--vk-constants` keeps literals): the AMD Windows driver
(26.10.1) folds an ordered compare of a literal NaN to true in a wave32 pipeline, which says nothing about a shader given
a NaN at run time.

```
python -I tools/pk16_diff.py --oracle hw,vk --programs 500 --seed 201 --max-ops 8 [--wave 64]
python -I tools/vint_diff.py --oracle hw,vk --programs 500 --seed 203
python -I tools/sdwa_diff.py --oracle hw,vk --family f16 --programs 500 --seed 211
python -I tools/wint_diff.py --oracle hw,vk --wave both --programs 400 --seed 221
```

Fresh seeds, 400 to 500 programs each (AMD Radeon iGPU gfx1036, driver 26.10.1, and NVIDIA GeForce RTX 5080, driver
616.92; the f16 SDWA programs also count `nan`: the reference declines an f16 NaN, the hardware kernels run with IEEE on):

| tool | Radeon wave32 | Radeon wave64 | RTX 5080 wave32 |
|---|---|---|---|
| `pk16_diff` | 492 match, 8 mismatch | 492 match, 8 mismatch | 500 match |
| `vint_diff` | 500 match | 500 match | 500 match |
| `sdwa_diff --family int` | 500 match | 498 match, 2 mismatch | 500 match |
| `sdwa_diff --family f16` | 499 match, 1 nan | 498 match, 2 mismatch | 490 match, 10 nan |
| `wint_diff` | 398 match, 2 mismatch (wave32 and wave64 programs together) | | 400 match |

The pk16 mismatches were `v_med3_i16` with a constant operand (seven of eight on wave64, five of eight on wave32; the
rest `v_pk_*` or `v_min_i16`): the AMD driver returned a bound instead of the middle value for the translated
min/max/clamp median (fixed by the sum-minus-min-and-max form, GPU test `cs_med3_i16`). The others are Radeon-only and
not fixable in the translator as far as measured: `v_not_b32_sdwa v4, v3 dst_sel:WORD_0 dst_unused:UNUSED_PAD
src0_sel:BYTE_0` with v3 = 0xffff gives 0xffffff00 where the hardware gives 0x0000ff00 in a wave64 pipeline (the same
with the field extracts written as shifts and a mask), and `v_cvt_f32_f16_sdwa ... dst_sel:WORD_1` of an f16 denormal
differs in the low bits of the result.

### binary64 VALU on the CPU and Vulkan (`tools/f64_diff.py`, FUZZ-55)

`tools/f64_diff.py` covers the 47 binary64 instruction forms `tools/rdna2spirv.py` translates on main: four rounding
ops, add, f64-to-f32 conversion, reciprocal/reciprocal-square-root/square-root, division fixup and scaled FMA, both
frexp results, all 16 `v_cmp_*_f64` and 16 `v_cmpx_*_f64` predicates, and both class compares. It generates one-lane
programs with VGPR and SGPR pairs, float immediates, inline constants, 32-bit literals where the translator marks
them as upper-half f64 literals, and full-width literals in other instruction forms; it compares the four exported
dwords. Compare predicates are read through a mask-to-word operation, and `v_cmpx` cases also export the resulting
`EXEC` mask. Its reference decodes IEEE binary64 words and uses exact
`Fraction` arithmetic for finite addition and fused multiply-add; it does not import code from the translator or the
other fuzzer references. It models all four `MODE.FP_ROUND[3:2]` encodings for addition; the translator currently
rejects directed rounding (+infinity and -infinity), so those generated cases are counted skips. Any NaN matches any
NaN. Binary64 subnormal operands or results for `v_rcp_f64`, `v_rsq_f64` and `v_sqrt_f64`, and subnormal f32 results
from `v_cvt_f32_f64`, are counted skips because shader drivers may flush them to zero (FTZ).

`--sweep` (the default) tries each translated form, followed by `--programs N` seeded programs (`--seed S` and
`--max-ops N` control the campaign). `--no-sweep`, `--ops`, `--no-literals`, `--known`, `--all-failures`,
`--list-ops` and `--self-test` are available. The translator oracle executes generated SPIR-V with the local semantic
interpreter. The current CPU interpreters do not implement binary64 `OpFConvert` or `OpFDiv`; these are reported as
counted skips. Negative-input `Sqrt` and
`InverseSqrt` cases are also skipped because GLSL.std.450 leaves them undefined. `s_round_mode` encodings for directed
rounding that the translator rejects are counted as skips. The reference uses IEEE binary64 for reciprocal and square
root; native RDNA2 `v_rcp_f64`, `v_rsq_f64` and `v_sqrt_f64` have about 26 bits of precision (PROC-216), so this
oracle checks the translated SPIR-V contract, not the native instruction's approximation. Other unsupported interpreter
instructions are counted as skips with their reason. Vulkan's [SPIR-V environment](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html)
leaves the precision of these GLSL [extended instructions](https://registry.khronos.org/SPIR-V/specs/unified1/GLSL.std.450.html)
implementation-defined for binary64; the GPU oracle therefore counts `v_rsq_f64` and
`v_sqrt_f64` as unjudged skips instead of requiring bit-exact results. This was confirmed on an NVIDIA RTX 4070 with
`shaderFloat64`: `v_rsq_f64 3.25` produced matching high 32 bits but zero low 32 bits, unlike the exact reference.

Known divergence probes are excluded and counted by default, then reinserted with `--known`: binary64 output modifiers
(SHADER-721), binary64 `OpFNegate`/`FAbs` that the CPU interpreter rounds through binary32 (TEST-912, #4172),
driver-dependent NaN or signed-zero clamp results, and `v_div_fixup_f64` with a NaN `S0` and an ordinary finite
division. The latter follows the RDNA2 ISA reference (`NaN(S0) -> signed infinity`), while the current translator
preserves the NaN payload. Therefore `--known` can intentionally return 1 when this known mismatch is reproduced;
the minimized case is printed. `--oracle gpu` translates the same
fragment shader and runs it through `runtime/build/vk_harness`, comparing the raw RGBA32F output words. It assembles
the fragment module with `spirv-as` and the full-screen vertex shader with the Vulkan SDK's `glslc`; use `--exe PATH`
to select another `vk_harness`. It exits 77 if the executable, SDK tools, Vulkan device or required `shaderFloat64`
feature is unavailable. `--oracle hw` runs the same programs natively on an AMD GPU through `tools/hw_oracle` (one lane
of a wave32 compute kernel, as `tools/fmisc_diff.py` does) and compares the four words with the reference only, not with
the translator; `--hw-arch` picks the architecture to assemble for (default: detected). The campaign is one launch, and a
failing program is re-run alone while it is shrunk. The hardware runs every form, so the known-divergence exclusions
(translator limits) do not apply: output modifiers, clamp, negated/absolute operands and directed rounding are compared
too, and the kernels start in the game's inferred float MODE (IEEE off, DX10_CLAMP on). `v_rcp_f64`, `v_rsq_f64` and
`v_sqrt_f64` are counted skips there, because the native results are approximations (PROC-216) and the reference is exact;
an architecture that cannot set that MODE (gfx1170 and later) or a program the assembler rejects is a counted skip with the
runner's reason. It exits 77 only when the `tools/hw_oracle` tools or device are missing. The path is covered by
`tests/test_f64_diff.py` with a fake oracle (routing, `--hw-arch`, exit 77, skips, shrinking); **no AMD hardware run has
been done for it yet**.
Exit codes are 0 for no unexpected
mismatch, 1 for a mismatch (with instruction-removal shrinking), 2 for CLI/harness errors and 77 for unavailable
oracles. Skips always include a reason.

```
python -I tools/f64_diff.py --programs 300 --seed 1
python -I tools/f64_diff.py --known --programs 300 --seed 1
python -I tools/f64_diff.py --oracle gpu --programs 300 --seed 1
python -I tools/f64_diff.py --oracle hw --programs 300 --seed 1
python -I tools/f64_diff.py --self-test
python -I tests/test_f64_diff.py
```

**Vulkan check on an NVIDIA GeForce RTX 4070** (Windows 11, driver 621.01, Vulkan `shaderFloat64` enabled):

| Command | Match | Mismatch | Total | Skips |
|---|---:|---:|---:|---:|
| `--oracle gpu --programs 300 --seed 1` | 331 | 0 | 347 | 16 |
| `--oracle gpu --programs 200 --seed 2 --max-ops 4` | 218 | 0 | 247 | 29 |
| `--oracle gpu --ops v_cvt_f32_f64,v_rcp_f64,v_rsq_f64,v_sqrt_f64 --programs 80 --seed 1` | 39 | 0 | 84 | 45 |
| `--oracle gpu --ops v_cmpx_lt_f64,v_cmpx_eq_f64,v_cmpx_class_f64 --programs 40 --seed 1` | 43 | 0 | 43 | 0 |

The skips are counted by reason: subnormal FTZ, implementation-defined binary64 `InverseSqrt`/`Sqrt` precision,
non-finite scaled FMA, and translator-rejected directed rounding. The compared cases had no mismatches.

### Output modifiers of the binary64 and f16 division ops (`tools/f64_omod_probe.py`)

`tools/f64_omod_probe.py` measures what `mul:2`, `mul:4`, `div:2` and `clamp` do to six VOP3 results: the binary64
conversions `v_cvt_f64_i32` and `v_cvt_f64_u32`, `v_cvt_f32_f64` (an f32 destination), `v_frexp_mant_f64`,
`v_div_fixup_f64` and `v_div_fixup_f16`. Each case is measured plain and with each modifier in the same kernel launch,
so the probe compares the modifier's effect and not the semantics of the op. A hand-written kernel writes the whole
destination (both words for binary64, the low half only for the 16-bit form, as the ISA defines) and runs once in
wave32 and once in wave64, which must agree word for word. It needs `tools/hw_oracle` on a GFX10 AMD GPU: the kernels
set their own MODE from the descriptor, and a device of another family is reported as unavailable, since the gate was
measured on RDNA2. It reads no game files.

The five MODEs are `game_i0_d3` (the game's, and the translator's: IEEE 0, `denorm_mode_16_64 = 3`,
`denorm_mode_32 = 0`, DX10_CLAMP 1), `i0_d0`, `i1_d3`, `i1_d0` and `i0_d3_f32keep` (`denorm_mode_32 = 3`). The gate
turns out to be **per result precision, not per instruction family**: the f32 result of `v_cvt_f32_f64` follows
`denorm_mode_32` and scales whenever it is 0, while the f16 and binary64 results scale only with IEEE 0 and
`denorm_mode_16_64 = 0` and ignore the modifiers in the game's MODE. `clamp` applies in every MODE. That is why
`tools/rdna2spirv.py` keeps only `clamp` for these destinations, and the rows for these ops in `docs/design.md` carry
the measurement.

`--modes` takes a comma-separated subset of the five (all by default), `--waves` the wave sizes (`32,64` by default),
`--hw-arch` the architecture handed to `tools/hw_oracle`, and `--json` a path for the report: the device, the
architecture, the modes, the wave sizes, the modifier forms, the ops and the counts `scaling_gated_on`,
`scaling_gated_off`, `scaling_changed_value`, `clamp_changed_value`, `mismatches` and `wave_diffs`, with the first
cases of each. `--self-test` checks the model's own cases without a GPU. Exit 0 when every word matches the model and
the wave sizes agree, 1 on a mismatch or a wave difference (the first cases are printed), 2 on a CLI or harness error
and 77 when the oracle or a GFX10 device is missing. `tests/test_f64_omod_probe.py` covers the gate and the CLI with a
fake oracle.

```
python -I tools/f64_omod_probe.py                  # all five MODEs, wave32 and wave64
python -I tools/f64_omod_probe.py --waves 64 --json f64_omod.json
python -I tools/f64_omod_probe.py --self-test
python -I tests/test_f64_omod_probe.py
```

**Default-mode DX9 multiply regression.** `v_mul_legacy_f32` now explicitly flushes each f32 operand before the
DX9 `0 * anything => +0` override, and flushes the nonzero product before its VOP3 output modifier. The output is
flushed again before the active omod's `-0 => +0` rule. Previously it emitted an unguarded `OpFMul`, so on a Vulkan
driver that keeps f32 denormals, a subnormal input times a large number or an underflowing normal product yielded
nonzero results even though the inferred RDNA2 `MODE.denorm_mode_32 = 0` requires zero. The standalone semantic
test `test_valu_mul_legacy_denorm.py` checks the exact bit patterns and the two GPU fixtures
`ps_mul_legacy_denorm` / `ps_mul_legacy_denorm_mods` exercise signed input/output FTZ, legacy zero override, and
output scaling. This is a translation for the **inferred game MODE**, not a claim of correctly handling dynamic
denormal-preserving or IEEE modes; native gfx1013 measurements of these new fixtures were not available.


### f16 transcendentals on the hardware (`tools/f16_trans_sweep.py`, FUZZ-29)

The `16-bit results` row of `docs/design.md` listed `v_exp_f16`/`v_log_f16` and the NaN and denormal operands as
unverified. f16 has only 65,536 bit patterns, so instead of sampling, this tool runs every one of them through
`v_exp_f16`, `v_log_f16`, `v_rcp_f16`, `v_rsq_f16`, `v_sqrt_f16`, `v_sin_f16` and `v_cos_f16` on the hardware
oracle (setup above; exit 77 without it). Each op is 8 raw kernels of 1,024 threads. Each thread computes 8 inputs
and stores them packed two to a dword, which is 4,096 words per kernel; 8 kernels cover every input. The op runs in
wave32 and wave64 and under five MODEs, and MODE is read back with `s_getreg_b32` first:

| Mode | IEEE | f16 denormals | f16 rounding |
|---|---|---|---|
| `keep16` | 0 | kept (3) | nearest |
| `flush16` | 0 | flushed (0) | nearest |
| `ieee_keep16` | 1 | kept | nearest |
| `ieee_flush16` | 1 | flushed | nearest |
| `keep16_rtz` | 0 | kept | toward zero |

All modes have DX10_CLAMP 1 and f32 denormals flushed.

```
python -I tools/f16_trans_sweep.py                                  # all ops and modes, about 5 minutes
python -I tools/f16_trans_sweep.py --ops v_exp_f16 --modes keep16 --json out.json
python -I tools/f16_trans_sweep.py --self-test                      # the reference, no GPU
```

The reference is the exact result correctly rounded to f16, computed with `decimal` at 60 digits: for sin/cos that is
sin/cos(2πx), reduced in turns. That reference is the translator's rule (widen, the GLSL.std.450 function in f32,
round to nearest even) with an exact f32 function, the best a driver can do. Inputs where the translated SPIR-V is
undefined are tabulated, not judged. For each mode the tool counts:
- bit-exact results, 1 ulp and more than 1 ulp;
- denormal operands that give the hardware's own word for a zero of the same sign, and denormal results written as zero;
- NaN results, with signalling NaNs kept, quieted or changed.

It also counts the inputs on which each mode differs from `keep16`, and lists the word for 23 special inputs. For
sin/cos a second table compares the hardware with Sin/Cos of the f32 product 6.2831855 · x, by |x| band.

Result on an RX 6700 XT (gfx1031, Navi 22; the Linux launcher from #1265, Linux 7.3.5, amdgpu, clang 23.1.5).
wave32 and wave64 gave the same word for every input of every op in every mode. There was no GPU fault, timeout or
reset in the kernel log.

| Op (`keep16`) | Inputs that are not NaN | Bit-exact | 1 ulp | Undefined in SPIR-V |
|---|---|---|---|---|
| `v_exp_f16` | 63,490 | 63,490 | 0 | 0 |
| `v_log_f16` | 63,490 | 31,744 | 0 | 31,746 (≤ 0) |
| `v_rcp_f16` | 63,490 | 63,488 | 0 | 2 (±0) |
| `v_rsq_f16` | 63,490 | 31,744 | 0 | 31,746 (≤ 0) |
| `v_sqrt_f16` | 63,490 | 31,746 | 0 | 31,744 (< 0, -0 excluded) |
| `v_sin_f16` | 63,490 | 63,474 | 14 | 2 (±inf) |
| `v_cos_f16` | 63,490 | 63,484 | 4 | 2 (±inf) |

- **Correctly rounded.** The five non-trigonometric ops never differ from the correctly rounded result. sin/cos differ
  by 1 ulp on 14 and 4 inputs, all with |x| ≤ 1. For |x| > 1 every result is bit-exact, up to 65,504 turns: the f16
  ops have no ±256 domain limit.
- **The f32 product loses the turn.** Against Sin/Cos(f32 6.2831855 · x), which is what the translator computes before
  #594 is fixed, 16,366 sin and 2,036 cos inputs are more than 1 ulp off, up to 9,944 ulp. Nearly all of them have
  |x| > 1 (sin: 1,010 inputs at 1 < |x| ≤ 256 and 15,354 above 256). Reducing in turns first, as #1283 and #616
  propose, is what the hardware does; for f16 the reduction must not stop at ±256.
- **f16 denormals follow `denorm_mode_16_64`, for operands and results alike.** With 0, a denormal operand gives the
  word of a zero of its sign (`v_log_f16` -inf, `v_rcp_f16` and `v_rsq_f16` ±inf, `v_sqrt_f16` and `v_sin_f16` ±0).
  Denormal results become ±0 (831 for `v_exp_f16`, 4,094 for `v_rcp_f16`). With 3, both are kept. The f32
  transcendentals behave differently: they flush in both modes.
- **The f16 field of FP_ROUND is ignored.** Under round toward zero every op gives the same word as under round to
  nearest, on all 65,536 inputs.
- **IEEE only changes signalling NaNs.** With IEEE 0, all 1,022 f16 signalling NaNs pass through unchanged. With IEEE 1
  they are quieted (bit 9 set) with payload and sign kept. Quiet NaNs pass through unchanged in every mode. This is the
  f32 behaviour of the previous section.
- **Undefined in SPIR-V, defined on the hardware.** log(±0) = -inf; rcp(±0) = ±inf; rsq(+0) = +inf; rsq(-0) = -inf;
  sqrt(-0) = -0. log, rsq and sqrt of a negative (-inf included), and sin/cos(±inf), give `0xfe00`, the negative quiet
  NaN (`0xffc00000` in f32).
- **Controls.** `keep16_rtz` also judges against a round-toward-zero reference, so it shows what a wrong rounding
  model produces: about half the results 1 ulp off. The kernel layout is checked without a GPU: each of the 65,536
  inputs is computed once and stored in its own half (`tests/test_f16_trans_sweep.py`). On a GPU the test checks
  exp2 of integers and wave32 = wave64.

### d16 rounding of UNORM/SNORM buffer loads (`tools/d16_format_probe.py`, BUILD-161)

`docs/design.md` ("d16 and f16 format narrowing") narrows the f16 of a typed d16 buffer load the way it narrows an
image result: the exact binary32 decode, then integer round to nearest even (`Gen.f16_bits`). For buffer loads that
rounding was inferred, not measured. The probe measures it on a GFX10 GPU (RDNA1/RDNA2) through the hardware
oracle. Every value of UNORM8, SNORM8, UNORM10 (the x channel of format 50), UNORM16 and SNORM16 is loaded with
`buffer_load_format_d16_x`, `tbuffer_load_format_d16_x` and, for the f32 stage alone, `buffer_load_format_x`. Each
runs in wave32 and wave64.

```
python -I tools/d16_format_probe.py --json d16.json          # about 1,040 kernels of 1,024 threads; exit 77 without the oracle
python -I tools/d16_format_probe.py --self-test               # the rules against docs/design.md's #489 counts, no GPU
```

- **The kernels stage their own inputs.** Thread t stores the format bits of one value at dword t of the output
  buffer and waits for the store (`s_waitcnt_vscnt`). It then builds a GFX10 V# over the buffer: DST_SEL XYZW, the
  format, RESOURCE_LEVEL 1, OOB_SELECT 3, NUM_RECORDS the 4 KiB staged. It loads its element with `glc dlc` (past
  L0 and GL1) and stores the VGPR at dword 1024 + t. The host checks that every staged word reads back as written.
- **Rules compared**, each also with f16 denormal results flushed:
  - `f32_rte`, the translator's rule;
  - `f32_rtz`, which is `v_cvt_pkrtz_f16_f32` of the f32 decode, the #489 bug;
  - `rte` and `rtz`, which round the exact quotient once;
  - `mul_rte`, the f32 product v · f32(1/(2^n − 1)) rounded to nearest even.
- **What the formats can separate.** The 8- and 10-bit formats only separate nearest from toward zero: every other rule
  gives the same bits there. That covers 123 of the 256 UNORM8 values and 110 of the 256 SNORM8 values, the #489
  counts. Only four 16-bit values separate one rounding from two (`f32_rte` against `rte`): UNORM16 0xffaf and
  0xffef, SNORM16 0x7ff7 and 0x8009. The smallest UNORM16/SNORM16 values have quotients under 2^-14, so they show
  a flush of f16 denormals.
- **Controls:**
  - a 16_FLOAT d16 load must return its own bits (NaNs may be quieted);
  - `d16_x` must leave the high half of its VGPR as it was (`docs/design.md` measured this on gfx1036);
  - wave32 must equal wave64, and MUBUF must equal MTBUF.

  A failed control is exit 2; a wave or MUBUF/MTBUF difference is exit 1. Which rule the hardware follows is
  reported, not judged.

Measured on an RX 6700 (gfx1031, AMD Windows driver 32.0.21045.29003, ROCm wheel clang), the whole sweep in 36 s:

| Format | Values | d16 `rte` | d16 `f32_rte` | d16 `f32_rtz` | f32 `f32_rte` |
| --- | --- | --- | --- | --- | --- |
| UNORM8 | 256 | 256 | 256 | 133 | 256 |
| SNORM8 | 256 | 256 | 256 | 146 | 256 |
| UNORM10 | 1,024 | 1,024 | 1,024 | 514 | 1,024 |
| UNORM16 | 65,536 | 65,536 | 65,534 | 33,796 | 65,536 |
| SNORM16 | 65,536 | 65,536 | 65,534 | 34,820 | 65,536 |

- The d16 result is the exact quotient rounded once to f16, to nearest even, with f16 denormals kept. The two
  UNORM16 and two SNORM16 values that separate one rounding from two all came out 1 ulp below the f32-first result:
  0xffaf gave 0x3bfd (not 0x3bfe), 0xffef gave 0x3bff (not 0x3c00), 0x7ff7 gave 0x3bff, and 0x8009 gave 0xbbff.
  The translator now follows this (`Gen.d16_norm_f16`, `docs/design.md` "d16 and f16 format narrowing").
- The f32 result (`buffer_load_format_x`) is the correctly rounded quotient for every value. The reciprocal multiply
  matched only 130 of 256 UNORM8 values and 65,024 of 65,536 UNORM16 values.
- 16_FLOAT came back with its own bits for all 65,536 encodings, NaNs included. The high half of the d16_x VGPR was
  kept. MUBUF matched MTBUF, and wave32 matched wave64, for every value.

`tests/test_d16_format_probe.py` checks the rules against the #489 counts and the descriptor bits. It checks that
each value is staged once, and that the kernels assemble for gfx1031. A fake oracle plays the GPU to check the exit
codes. On a GFX10 GPU the test runs UNORM8 and checks the controls.

### Widening a 16-bit float NaN in a typed buffer load (`tools/f16_widen_probe.py`)

A typed buffer load of a half NaN had never been measured: `tools/mem_diff.py` decoded 16_FLOAT with
`struct`'s `'<e'` (payload kept from Python 3.14, canonical before) while the translator decoded it with
`UnpackHalf2x16`, whose result for a NaN is undefined and which the mini interpreter canonicalises to
0x7FC00000. The probe stages one half word with an untyped `buffer_store_dword`, waits, loads it with
`buffer_load_format_*` / `tbuffer_load_format_*`, waits, and stores the f32 lanes back.

```
python -I tools/f16_widen_probe.py --json f16_widen.json    # 380 kernels of one thread (190 per wave); exit 77 without the oracle
python -I tools/f16_widen_probe.py --self-test              # the three candidate rules, no GPU
```

- Formats 13 (`16_FLOAT`), 29 (`16_16_FLOAT`) and 71 (`16_16_16_16_FLOAT`), `buffer_` and `tbuffer_`
  forms, `_x`/`_xy`/`_xyzw`, wave32 and wave64. Every NaN word came back the same in every component of
  every format and form:

  | half | f32 | |
  | --- | --- | --- |
  | 0x7ddb | 0x7fbb6000 | sNaN, payload kept, quiet bit still 0 |
  | 0x7fe6 | 0x7ffcc000 | qNaN |
  | 0x7fbb | 0x7ff76000 | |
  | 0xffb2 | 0xfff64000 | negative, sign kept |
  | 0x7fcb | 0x7ff96000 | |
  | 0xff96 | 0xfff2c000 | |
  | 0x7c01 | 0x7f802000 | sNaN, smallest payload |
  | 0xfc01 | 0xff802000 | |
  | 0x3c00 | 0x3f800000 | 1.0 |
  | 0xc000 | 0xc0000000 | −2.0 |

- The rule is `sign | 0x7F800000 | (payload << 13)`: the sign is kept, the 10-bit payload moves into the
  f32 mantissa unchanged, and the quiet bit is **not** forced, so an sNaN stays signalling. The
  negative NaNs (for example 0xffb2 and 0xff96) keep their sign.
- `v_cvt_f32_f16` on the same halves (the instruction a driver lowers `UnpackHalf2x16` to) differs on an
  sNaN: 0x7ddb gives 0x7ffb6000 (quieted, payload kept) and 0x7c01 gives 0x7fc02000, while every qNaN is
  the same as the typed load. The translator therefore builds the NaN case from the bits with integer ops
  and uses `UnpackHalf2x16` only for the other halves, whose conversion is exact.
- Verdict: the hardware keeps the payload, so `mem_diff`'s Python 3.14 reference was right and the
  canonicalising translator was wrong. `mem_diff` now pins the rule in `f16_to_f32` (bit-level, Python
  version independent) and `tests/semantic/test_buffer_f16_widen_hw.py` pins the ten measured words for
  MUBUF and MTBUF, formats 13/29/71.

### `image_get_resinfo` and `image_get_lod` (`tools/image_query_probe.py`, SHADER-388)

`docs/design.md` ("Image atomics, size and LOD queries") took resinfo's depth word, the clamped LOD under a linear mip
filter and the order of u and v in the LOD query from the T# fields and the spec. The probe runs both instructions
natively on a GFX10 GPU through the hardware oracle. Neither reads memory, so a T# with no texture behind it is
enough. Each case runs in wave32 and wave64.

```
python -I tools/image_query_probe.py --json query.json      # 614 results; exit 77 without the oracle or on non-GFX10
python -I tools/image_query_probe.py --self-test             # the rules against values measured on gfx1036, no GPU
```

- **resinfo:** 2D and 2D-array T#s with WORD4 DEPTH 0, 5, the GFX10.3 pitch and 0x1FFF, BASE_LEVEL 0 to 2, levels
  inside and past LAST_LEVEL.
- **get_lod:** one 2x2 quad of lanes whose u steps across x and v across y. MIP_FILTER none, point and linear, with
  MIN_LOD/MAX_LOD pairs including MIN above MAX and MIN above the last level, LOD_BIAS 0, 1 and -0.5, at LODs that
  are exact in binary. Every lane of the quad must agree. A 64x4 image steps one axis at a time.

Measured on the Radeon Graphics of a Ryzen 9 9950X3D (gfx1036, AMD Windows driver 32.0.21045.16010, ROCm wheel
clang), every result matched the rules in the tool's docstring:

- The 2D depth word is 0, never DEPTH + 1. A 2D array's layer count is minified as max(1, layers >> (BASE_LEVEL +
  level)): 100 layers at level 3 gave 12.
- Past the last level, width, height and depth are 0, while the level count stays.
- Raw LOD = computed LOD + LOD_BIAS. The clamped LOD is the level that would be read: 0 with no mip filter, the
  clamp rounded half up with point, the clamp itself with linear.

`tests/test_image_query_probe.py` checks the rules against the measured values, the descriptor bits and the
kernels. A fake oracle plays the GPU to check the comparison and the exit codes; with a 2D depth of 1 it fails. On
a GFX10 GPU the test runs the whole probe.
### d16 image loads of 32-bit integer texels (`tools/image_d16_probe.py`, SHADER-387)

`docs/design.md` ("Typed image loads") saturates a 32-bit UINT or SINT texel to the 16-bit range of its signedness
when a d16 `image_load` packs it (Mesa sets `integer_dest_saturates` for AMD). Native gfx1036 had shown that on 2D
images; the probe measures it on every dimension the translator handles, through the hardware oracle on a GFX10 GPU.

```
python -I tools/image_d16_probe.py --json d16_image.json   # 16 kernels, one lane per texel; exit 77 without the oracle
python -I tools/image_d16_probe.py --self-test             # the rules, T# words and kernels, no GPU
```

- **Cases.** `SQ_RSRC_IMG_1D`, `2D`, `2D_ARRAY` (3 layers) and `3D` (depth 3), each with 32_UINT and 32_SINT
  (`dmask:0x1`) and 32_32_UINT and 32_32_SINT (`dmask:0x3`, both results in one VGPR), read with `image_load` and
  `image_load_mip` at mip 0. The texture sits in the kernel's code after `s_endpgm`, linear, with 256-byte rows (64
  R32 or 32 R32G32 texels), so the layout does not depend on how the T# states the pitch. The staged words start with the values that separate the rules (0x7fff, 0x8000, 0xffff, 0x10000,
  0x7fffffff, 0x80000000, -32769, -32768 and their neighbours), then a fixed mixing sequence.
- **Rules compared:** `trunc` (the low 16 bits), `sat_u` (min(w, 0xffff)) and `sat_s` (w clamped to
  [-32768, 32767]). Every pair differs on some staged value. A one-channel result also reports its high half, which
  starts as 0x5a5a.
- **Controls.** The same texel read without d16 must return the staged word, for x and, on two-channel formats, y:
  a wrong extent, layer or slice offset fails it (exit 2). On gfx1036, a T# with one layer fewer than a 2D
  array's three, a 3D depth of 1, or a 1D width of 32 failed 246 of 384, 246 of 384 and 31 of 64 controls.
  `image_load` and `image_load_mip` must agree (exit 1).

Measured on the integrated GPU of a Ryzen 7 7800X3D (gfx1036, AMD Windows driver 32.0.21045.29003, ROCm wheel
clang), wave32, every control read back:

| Dimension | Texels per format (one / two channels) | UINT | SINT | High half (one channel) |
| --- | --- | --- | --- | --- |
| 1D | 64 / 32 | `sat_u` on all | `sat_s` on all | kept |
| 2D | 128 / 64 | `sat_u` on all | `sat_s` on all | kept |
| 2D-array | 384 / 192 | `sat_u` on all | `sat_s` on all | kept |
| 3D | 384 / 192 | `sat_u` on all | `sat_s` on all | kept |

`image_load` and `image_load_mip` gave the same words on every texel. No result followed `trunc` where it differs
from saturation. The translator's `saturate_int16` follows this on every dimension; the translated 1D, 2D-array and
3D d16 loads have not been run on a GPU yet.

`tests/test_image_d16_probe.py` checks the rules, the T# fields and the kernels without a GPU, assembles them for
gfx1036 when AMD clang is there, and checks the exit codes with a fake oracle. On a GFX10 GPU it runs the probe.

### `v_fma_mix*` f16 rounding in nearest mode (`tools/fma_mix_probe.py`)

`docs/design.md` leaves one f16 question open. In round-to-nearest mode, does `v_fma_mixlo_f16` / `v_fma_mixhi_f16`
round the exact x · y + z once, straight to f16? Or does it round to f32 first and then narrow, which is what the
translator does (f32 `Fma`, then `f16_res`)? The RX 6800 word behind the toward-zero rule can't answer this. For
1 · 2 - 2^-30 both readings give 0x4000 in nearest mode, and toward zero two truncations equal one.

The two readings only differ when the f32 rounding lands exactly on an f16 tie, or moves a value off one. The probe has
seven such inputs, each one f16 ulp apart under the two rules, plus one control:
- a tie just above and just below, for both signs;
- the subnormal tie at 2^-25;
- the overflow tie at 65520 (inf against 65504);
- a tie made by the product rather than by z;
- the same with f16 operands (`op_sel_hi:[1,1,1]`).

Every side reports which rule its words follow:

```
python -I tools/fma_mix_probe.py                       # the translator, on the CPU interpreter
python -I tools/fma_mix_probe.py --oracle hw           # also the native instruction on an AMD GPU (hardware oracle)
python -I tools/fma_mix_probe.py --oracle vulkan --build runtime/build   # also the translated shader on this GPU
```

The `hw` side is the measurement. It runs one wave32 lane per input, under the MODE inferred for the game: IEEE 0,
DX10_CLAMP 1, f32 denormals flushed, f16 denormals kept, nearest. The `vulkan` side only shows that a driver runs the
translator's sequence as written. Its operands come in user SGPRs, so the driver can't fold them. Exit 0 means every
side follows the same rule. Exit 1 means they differ, or a side follows neither rule. Exit 77 means a requested side
is unavailable.

`tests/test_fma_mix_probe.py` checks the expected words against a derivation written out by hand. It also checks that
the translator follows the single rounding rule, and that the compute shader gives the pixel shader's words on the
interpreter. Hardware measurement on AMD RDNA2 confirmed that the hardware rounds once, straight to f16, so the translator
uses `Gen.fma_mix_odd` in nearest mode as well. Round-to-odd at f32 then narrows exactly, giving the single-rounded word on
all seven discriminating inputs.

Results:

| GPU | Driver | `translator` | `vulkan` | `hw` |
|---|---|---|---|---|
| NVIDIA GeForce RTX 5060 (Blackwell), Windows 11 | 621.24 | single | single, 8 of 8 words | not AMD |
| AMD Radeon iGPU (RDNA2, gfx1036), Windows 11, HIP backend | 32.0.21044.64 | single | single, 8 of 8 words | **single**, 7 of 7 ties; control `0x4100` |

RDNA2 rounds once. Every tie input gives the single-rounded word (A `0x3c01`, B `0x3c01`, C `0xbc01`, D `0x0001`,
E `0x7bff`, F `0x3c01`, G `0x3c01`), so the translator is one f16 ulp off on exactly these inputs (SHADER-135; the
translator is not changed here). On the RTX 5060, a translator copy changed to use
`fma_mix_odd` in nearest mode gave the single-rounded word for all seven inputs on the `vulkan` side as well.

Measurement on AMD RDNA2 (gfx1036) confirmed single rounding across all 7 discriminating cases (finding 7079901079).
Using `fma_mix_odd` produces the exact hardware single-rounded words in both nearest-even and toward-zero modes.
E `0x7bff`, F `0x3c01`, G `0x3c01`), which the translator produces across all 7 discriminating cases via
`fma_mix_odd` in both nearest-even and toward-zero modes (SHADER-297). On Vulkan, using `fma_mix_odd` matches the
exact single-rounded hardware words for all inputs.

### `v_cvt_pknorm_*` NaN and rounding on RDNA2 (`tools/pknorm_probe.py`, SHADER-339)

`docs/design.md` said `v_cvt_pknorm_i16_f32` / `v_cvt_pknorm_u16_f32` give 0 for a NaN, as `v_cvt_i32_f32` does,
without a hardware measurement. The probe puts each NaN pattern in S0 and in S1 next to a finite value: quiet
(`0x7fc00000`, `0xffc00000`), quiet with a payload (`0x7fc00001`, `0x7fffffff`, `0xffffffff`), signalling
(`0x7f800001`, `0xff800001`, `0x7fa00000`). Finite values are the semantic test's (±0, ±1, ±0.5, ±2, ±inf,
subnormals, steps of 1/32767 and 1/65535). It also has near-tie inputs: f32 values whose product with 32767 (or 65535)
rounds to exactly k + 0.5 in f32, while the exact product is just below or just above it. There the translator's
`OpFMul` then `RoundEven` and a single rounding of the exact product differ by one, in either direction.

```
python -I tools/pknorm_probe.py                 # the translator, on the CPU interpreter
python -I tools/pknorm_probe.py --oracle hw     # also the native instruction on an AMD GPU (hardware oracle)
```

The `hw` side runs one lane per case in wave32 and wave64, under five MODEs:
- the one inferred for the game (IEEE 0, DX10_CLAMP 1, f32 denormals flushed, f16/f64 kept, nearest);
- the same with IEEE 1;
- the same with DX10_CLAMP 0;
- the same with both IEEE 1 and DX10_CLAMP 0;
- the same with f32 denormals kept.

It reports the NaN halves that give 0, the words equal to the translator's, and the near-tie words equal to a single
rounding of the exact product. Exit 0 means every hardware word equals the translator's, 1 means one differs, 77
means the oracle is unavailable. `tests/test_pknorm_probe.py` checks four near-tie words against a derivation written
out by hand, and checks that the translator side rounds once there (since SHADER-363) and gives 0 for a NaN half.

Results (134 cases per run):

| GPU | Driver | NaN halves giving 0 | near ties: single / translator | other words equal |
|---|---|---|---|---|
| AMD Radeon iGPU (Ryzen 5 7600X, RDNA2, gfx1036), Windows 11, HIP backend, AMD clang (rocm-sdk-core 10.0.0) | 32.0.21045.29003 | 74 of 74 | 28 of 28 / 0 of 28 | 34 of 34 |

The ten runs (two wave sizes, five MODEs) give the same words. A NaN gives 0 on RDNA2, as the translator assumes,
whatever the IEEE, DX10_CLAMP and f32 denormal fields. The rounding does not match. The hardware rounds the exact
product once, to nearest even. Examples:
- `0x38400180` (exact 1.5 - 1.4e-9) gives 1, where the translator gives 2;
- `0x3c004101` (exact 256.5 + 1.5e-5) gives 257, where the translator gives 256.

SHADER-363 fixes the translator. It keeps the f32 product p and its exact error `Fma(x, s, -p)`. When p lies exactly
on k + 0.5, the error's sign gives floor(p) + 1 or floor(p), and a zero error keeps `RoundEven` (an exact tie). The
probe then exits 0 on the same iGPU: all 134 words equal the translator's in all ten runs (wave32 and wave64, five
MODEs), including the 28 near ties.

### Division fixup (`tools/div_fixup_diff.py`)

No other fuzzer generates `v_div_fixup_f16`, `v_div_fixup_f32` or `v_div_fixup_f64`. `tools/div_fixup_diff.py`
runs them through the translator and `tests/semantic/spirv_mini_interp.py`, against a model of the `docs/design.md`
rows that works on the operand bits and shares no code with the translator, and compares the four exported words bit
for bit. The model:

- `abs` then `neg` act on the sign bit of S0, S1 and S2, NaNs included.
- f32 reads an S1 or S2 denormal as a zero of its sign and keeps an S0 denormal. f16 and f64 keep every denormal.
- The first rule that holds gives the result: NaN(S2), NaN(S1), 0/0 or inf/inf (the negative quiet NaN), a zero
  numerator or an inf denominator (zero), a zero denominator or an inf numerator (inf), the exponent cutoff
  (`exp(S2) - exp(S1)` at most -151 on f32 and -1076 on f64, none on f16; zero), NaN(S0) (inf), S0's magnitude.
  The sign is sign(S1) ^ sign(S2).
- On f32, `mul:2`/`mul:4`/`div:2` scale the result, then `clamp` takes a NaN or any negative value to +0.
- On f16 and f64, `mul:2`/`mul:4`/`div:2` change nothing: the `docs/design.md` row (#6314, measured on gfx1036 with
  `tools/f64_omod_probe.py`) drops the scaling under `denorm_mode_16_64 = 3`, the game's `MODE`, while an f16 `clamp`
  still saturates. The measured inputs are normal values; the reference applies the row to every result.

`tests/test_div_fixup_diff.py` pins the model to the 73 gfx1036 rows of `tests/semantic/test_alu_div_fixup_f32.py`
and to the f16 and f64 rows of `test_alu_div_fixup.py`, the measured exponent cutoffs among them.

Sources are VGPRs, SGPRs, a literal (not on f64) and integer and float inline constants, within the GFX10 literal and
constant-bus limits. An f64 source is a VGPR pair (odd first registers too), an SGPR pair or an inline constant, and
`0.15915494309189532` is the binary64 1/(2 pi). Modifiers are `neg`/`abs`, `op_sel` on the f16 register sources and
the destination, `clamp` on f16 and f32, and the output multipliers on all three. Not generated: a literal on f64
(#3791 moves it to bits 63:32) and `clamp` on f64 (the translation needs Float64, which the interpreter does not run).

An f32 NaN that went through an output multiplier matches any NaN. Some programs are counted as not judged, with the
reason:
- an f64 exponent cutoff that depends on how a denormal's exponent counts (only normal operands were measured);
- an output multiplier on -0 without `clamp` (the hardware gives +0, not modelled), or on an f32 denormal;
- `clamp` of a positive f32 denormal, of an f16 -0 or of an f16 NaN (`FClamp` of a NaN is undefined);
- reading part of an f32 NaN whose payload is not modelled.

Stages:
1. A sweep of every encodable source-kind combination of each op.
2. An edge grid: special values as S1 and S2 under five S0 values, the exponent cutoff one below, at and one above
   (also against a NaN or inf S0, and on f64 with a denormal numerator), `abs`/`neg` on every kind of value, and
   the output modifiers on each rule's result (f16 multipliers in both destination halves, with and without `clamp`).
3. Random seeded programs, which read earlier results, f64 halves as f32 or f16 values included.

A translator skip exits 1, as a mismatch does.

```
python -I tools/div_fixup_diff.py --programs 3000 --seed 11 --max-ops 8
python -I tools/div_fixup_diff.py --ops v_div_fixup_f64 --programs 1000
python -I tests/test_div_fixup_diff.py
```

### `v_mad_f32`, `v_mac_f32`, `v_mad_legacy_f32` denormal flush on RDNA1 (`tools/mad_denorm_probe.py`, SHADER-340)

`docs/design.md` ("Denormal flush in `v_mad_f32`, `v_mac_f32`, `v_mad_legacy_f32`") had no hardware measurement: GFX10.3+
dropped these opcodes, and flushing only after `omod` (`div:2`/`mul:2`/`mul:4`) was an unverified assumption. The probe
covers 48 cases across six groups: subnormal operands in each source slot (`input`, 15), normal operands with a subnormal
product (`product`, 7), normal operands/products with a subnormal sum (`result`, 6), `omod`/`clamp` crossing the
subnormal threshold (`omod`, 8), `v_mad_legacy_f32`'s `0 * anything -> +0` with subnormals (`legacy`, 5), and controls
(`control`, 7). A `v_mul_f32(FLT_MIN, 0.5)` sanity kernel checks that `MODE.denorm32=3` actually keeps f32 denormals on
the device (`0x00000000` with `denorm32=0`, `0x00400000` with `denorm32=3`).

```
python -I tools/mad_denorm_probe.py                             # the translator, on the CPU interpreter
python -I tools/mad_denorm_probe.py --oracle hw                 # also the native instruction on an AMD GPU
```

Results (48 cases per run, wave32 and wave64 across 5 MODEs; `tests/test_mad_denorm_probe.py` checks the translator side):

| GPU | Driver | `game` / `dx10clamp0` / `denorm32_kept` | `ieee1` / `ieee1_dx10clamp0` (non-`omod` + `omod` ignored) |
|---|---|---|---|
| AMD Radeon RX 5600 XT (`gfx1010`, RDNA1), Windows, HIP backend (`amdhip64_6.dll`), AMD clang (`rocm-sdk-core 10.0.0`) | 32.0.21045.29003 | 48 of 48 (wave32 and wave64) | 48 of 48 (wave32 and wave64) |

Two findings from the RDNA1 measurement:
- **Subnormal inputs, products, sums, and legacy rules flush regardless of `MODE.denorm32`** (40 of 40 non-`omod` cases match bit-for-bit across all 5 MODEs, including `denorm32=3`). Under `MODE.IEEE=1`, hardware ignores `mul:2`/`mul:4`/`div:2` (`FLT_MIN div:2` gives `FLT_MIN`, as for other f32 VALU ops).
- **Hardware flushes both before and after `omod`.** Before SHADER-340, the translator flushed only after `clamp_mods`, so `(1.5*2^-126 - 2^-126) mul:2` gave `0x00800000` (`FLT_MIN`) and `mul:4` gave `0x01000000`, whereas the RX 5600 XT returns `+0` (`0x00000000`) in `game`, `dx10clamp0`, and `denorm32_kept`. `Gen.flush_f32_mods` now flushes the sum before `clamp_mods` and flushes again after it (`FLT_MIN div:2` → `+0`, `-FLT_MIN div:2` → `-0`), leaving non-modified `v_mad*`/`v_mac*` instructions byte-identical.

### f16 output modifiers, clamp and `v_fma_mix*` operands on RDNA2 (SHADER-135)

*Measured on an RDNA2 iGPU (gfx1036), AMD Windows driver 32.0.21044.64, HIP backend, AMD clang 23.0.0git, `main`
a0acd33d.* Raw kernels written by hand, one lane each. The MODE comes from the kernel descriptor (`ieee`,
`dx10_clamp`, `denorm_mode_32`, `denorm_mode_16_64`). Every program writes one word, starting from a destination of
`0x5a5a5a5a`, so a half the op must not write shows up. 1,385 programs assembled, each run in wave32 and wave64, and
the two wave sizes agree on every word. `tools/fma_mix_probe.py --oracle hw` (the section above) and
`tools/f16_trans_sweep.py --ops v_exp_f16,v_log_f16` ran on the same GPU.

- **An omod on an f16 result acts only with IEEE 0 and f16 denormals flushed (`denorm_mode_16_64 = 0`).** In that
  MODE `mul:2`, `mul:4` and `div:2` scale the results of `v_add_f16`, `v_mul_f16`, `v_fma_f16`, `v_fmac_f16_e64`,
  `v_cvt_f16_f32_e64`, `v_exp_f16_e64` and `v_rcp_f16_e64` (`v_fmac_f16_e64` 0.75 · 1 + -0 `mul:4` gives `0x4200`).
  With `denorm_mode_16_64 = 3` or IEEE 1 they leave the result as it is (`0x3a00`), and `denorm_mode_32` makes no
  difference. The f32 control (`v_add_f32_e64 mul:2`) follows `denorm_mode_32` instead, as on gfx1150 above. Unlike
  the f32 transcendentals there, `v_exp_f16` and `v_rcp_f16` obey the same rule as the ALU ops. When the omod acts, a
  -0 result becomes +0 (`v_fmac_f16` -0 · 1 + -0 `mul:2` gives `0x0000`), and it comes before the clamp
  (`v_cvt_f16_f32_e64` 0.75 `clamp mul:2` gives `0x3c00`, not `0x3e00`).
- **clamp** on `v_cvt_f16_f32_e64`, `v_fmac_f16_e64`, `v_fma_mixlo_f16` and `v_fma_mix_f32` clamps to [0, 1] in every
  MODE: -0.5 and -0 give +0, 3.0 and an overflow (1e5, 65504 · 2) give 1.0. A NaN gives +0 with DX10_CLAMP 1. With
  DX10_CLAMP 0 it stays the NaN the op gives without clamp. This is SHADER-186's f32 rule, here for the f16 results and
  the mix ops.
- **Constants in `v_fma_mix*` operands.** A float inline constant is the f32 constant where `op_sel_hi` makes the
  operand f32, and the f16 constant with a zero high half where it makes it f16. `1.0`, `-1.0`, `0.5`, `2.0` and
  `-4.0` give their value either way. `0.15915494` gives `0x3e22f983` as f32 and `0x3118` as f16 (`0x3e230000` once
  widened). With `op_sel_hi = 1` and `op_sel = 1`, every float constant reads +0.0. An integer inline constant is its
  32-bit integer. As an f32 operand, `1` and `64` are f32 denormals (+0 with `denorm_mode_32 = 0`) and `-1` is the NaN
  `0xffffffff`. As an f16 operand, `op_sel` picks a half: `1` gives `0x0001` low and +0 high, and `-1` gives `0xffff`
  in both. `op_sel` is ignored for an f32 operand. For gfx10, llvm-mc rejects a 32-bit literal in these ops ("invalid
  operand"). It encodes a literal equal to an f16 inline constant, such as `0x3c00`, as that constant (`0xf2`).
- **Mix-op denormals follow the mode of their own width.** An f16 operand (`op_sel_hi = 1`) and an f16 result
  (`mixlo`/`mixhi`) become a zero of their sign with `denorm_mode_16_64 = 0`, and are kept with 3, whatever
  `denorm_mode_32` says. An f32 denormal is flushed with `denorm_mode_32 = 0` and kept with 3.
- **Mix-op NaNs.** An f16 NaN operand widens with its payload (`0x7c01` → `0x7f802000`, `0xfe01` → `0xffc02000`).
  `v_fma_mix_f32` keeps a signalling NaN, f16 or f32, with IEEE 0, and quiets it with IEEE 1, payload kept
  (`0x7f800001` → `0x7fc00001`). `v_fma_mixlo_f16` quiets every NaN in every MODE and keeps the top payload bits
  (`0x7c01` → `0x7e01`, `0x7d55` → `0x7f55`, f32 `0x7fa00000` → `0x7f00`, f32 `0x7f800001` → `0x7e00`).
- **`v_exp_f16` and `v_log_f16` give gfx1031's counts exactly**, on all 65,536 inputs in the five MODEs of the f16
  transcendentals section: 0 ulp on every input that is not a NaN, with f16 denormals kept. With them flushed, 831
  `v_exp_f16` results and 1,023 `v_log_f16` operands become zero. Signalling NaNs are quieted only with IEEE 1, and
  wave32 = wave64.

**The translator against these words.** Each program was translated as a pixel shader and run on
`tests/semantic/spirv_mini_interp.py`, then compared with the hardware under the MODE the translator assumes for the
game (IEEE 0, DX10_CLAMP 1, f32 denormals flushed, f16 denormals kept). It gives the same word for the float
constants in every `op_sel`/`op_sel_hi` combination, for the integer constants as f16 halves, for clamp on values that
are not NaN, and for the f16 denormals. It differs in five ways, none fixed here:

1. **f16 omod.** The translator applies `mul:2`/`mul:4`/`div:2` to every f16 result. The hardware ignores them in this
   MODE (24 words: `v_add/mul/fma/fmac/exp/rcp_f16`). `v_cvt_f16_f32` drops its omod and clamp today, so its omod
   matches by accident; #718 would make it apply. The two assumptions can't both hold for the game. Either its shaders
   run with f16 denormals flushed, and keeping them is wrong, or every f16 omod in them does nothing. A corpus count
   of omod on f16 instructions would decide, since a compiler that emits one expects it to act (LLVM folds an omod only
   when "output denormals" are off, the `SIFoldOperands.cpp` comment quoted above). That count needs the game files.
2. **clamp of a NaN.** The interpreter refuses `FClamp` with a NaN operand, which GLSL.std.450 leaves undefined. The
   hardware gives +0 with DX10_CLAMP 1, for `v_fmac_f16`, `v_fma_mixlo_f16` and `v_fma_mix_f32` alike (SHADER-186).
3. **Mix-op rounding.** The hardware rounds once (the section above).
4. **f32 denormal operands of the mix ops.** They are not flushed with `denorm_mode_32 = 0`: `0x00000001`, `0x807fffff`
   and the integer constants `1` and `64` used as f32 give the denormal, where the hardware gives a zero of its sign
   (compare SHADER-118 for the mad ops).
5. **NaN payloads.** Where the hardware keeps the payload and, with IEEE 0, the signalling bit, the translator writes
   the canonical quiet NaN (`0x7fc00000`, `0x7e00`). For example, `v_fmac_f16` keeps `0x7c01`. This is the "not
   modelled yet" note of the f16 transcendentals row in `docs/design.md`, here for these ops too.

### Integer operations and scalar ALU verification (`tools/salu_diff.py`, gfx1036 follow-up to #1746)

As a follow-up to #1746 (SHADER-137), `tools/salu_diff.py` and `tests/test_salu_diff.py` were expanded to include 12 integer SALU operations (`s_absdiff_i32`, `s_mul_hi_i32`, `s_lshl1..4_add_u32`, `s_sext_i32_i8`, `s_sext_i32_i16`, `s_flbit_i32`, and `s_pack_ll/lh/hh_b32_b16`) and tested on native desktop RDNA2 silicon (`gfx1036`), complementing #1746's laptop `gfx1035` probe.

Hardware validation was executed via `tools/hw_oracle` and `tools/salu_diff.py` on native AMD RDNA2 silicon:

- **GPU**: AMD Radeon(TM) Graphics (`gfx1036`), AMD Windows driver `32.0.21045.29003`, Windows 11, `amdhip64_6.dll`, AMD clang 23.0.0git (`rocm-sdk-core 10.0.0`).
- **Second-silicon verification of #1746 (`tools/int_hw_probe.py`)**:
  - `python -I tools/int_hw_probe.py --oracle both --wave both`: **87 cases: 87 match, 0 known divergence, 0 unexpected mismatch, 0 wave32/64 diff** on native `gfx1036` (corroborating that the 5 divergences originally reported by #1746 are now fully resolved on `main`).
- **SALU operations (`s_absdiff_i32`, `s_mul_hi_i32`, `s_lshl1..4_add_u32`)**:
  - `python -I tools/salu_diff.py --ops s_absdiff_i32,s_mul_hi_i32,s_lshl1_add_u32,s_lshl2_add_u32,s_lshl3_add_u32,s_lshl4_add_u32 --programs 500 --oracle translator,hw`: **500 passed, 0 mismatches**.
  - Confirmed SCC preserved across operations with prior SCC=0 and SCC=1.
- **Sign-extension and pack (`s_sext_i32_i8`, `s_sext_i32_i16`, `s_flbit_i32`, `s_pack_ll/lh/hh_b32_b16`)**:
  - `python -I tools/salu_diff.py --ops s_sext_i32_i8,s_sext_i32_i16,s_flbit_i32,s_pack_ll_b32_b16,s_pack_lh_b32_b16,s_pack_hh_b32_b16 --programs 500 --oracle translator,hw`: **500 passed, 0 mismatches**.
  - `s_cmov_b32` verified on hardware under SCC=0 (destination preserved) and SCC=1 (destination loaded).
- **VALU product-sum (`v_mad_u64_u32`, `v_mad_i64_i32`)**:
  - 292 programs in wave32 across special and random 32/64-bit inputs: **292 passed, 0 mismatches**.
  - 192 programs in wave64 across special inputs: **192 passed, 0 mismatches**.
  - Carry-out and sign bits in sdst matched the 65-bit ISA reference bit-for-bit.
- **Integer saturation (`clamp`)**:
  - 567 programs tested across `v_add_nc_u32_e64`, `v_sub_nc_u32_e64`, `v_subrev_nc_u32_e64`, `v_add_nc_i32_e64`, `v_sub_nc_i32_e64`, `v_add_nc_u16_e64`, and `v_sub_nc_u16_e64`: **567 passed, 0 mismatches**.
  - 16-bit clamp confirmed preserving the high 16 bits of the destination VGPR.
- **EXEC lane masking**:
  - Multi-threaded kernel under alternating lane masks (`0xAAAAAAAA`, `0x55555555`) confirmed that inactive lanes write 0 in sdst mask words for `v_add_co_u32` and `v_mad_u64_u32`.
- **SAD family and lerp (`v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16`, `v_sad_u32`, `v_msad_u8`, `v_mqsad_u32_u8`, `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8`, `v_lerp_u8`)**:
  - Verified across edge byte patterns, 64-bit sliding windows, and zero masks: **0 mismatches** against ISA reference.
- **24-bit products and 64-bit shifts (`v_mul_hi_*24`, `v_lshrrev_b64`, `v_ashrrev_i64`)**:
  - Verified across signed/unsigned products, edge signs, and 0..63 shift counts: **0 mismatches**.

### Packed 16-bit integer operations on RDNA2 (SHADER-302)

*Measured on an RDNA2 iGPU (gfx1036), Windows 11, AMD Windows driver 32.0.21045.29003, HIP backend (`amdhip64_6.dll`), AMD clang 23.0.0git (`rocm-sdk-core 10.0.0`).*

380 raw compute kernels covering all 11 packed 16-bit integer instructions (`v_pk_add_u16`, `v_pk_sub_u16`, `v_pk_add_i16`, `v_pk_sub_i16`, `v_pk_mul_lo_u16`, `v_pk_mad_u16`, `v_pk_mad_i16`, `v_pk_max_u16`, `v_pk_min_u16`, `v_pk_max_i16`, `v_pk_min_i16`), tested in both wave32 and wave64:

- **Constant operands under `op_sel` / `op_sel_hi`:** Integer constants under VOP3P are uniformly treated as 32-bit dwords `C`. When `op_sel[i] == 0`, bits `[15:0]` are selected; when `op_sel[i] == 1`, bits `[31:16]` are selected. Positive inline constants zero-extend (`5` → `0x00000005`: lo=`0x0005`, hi=`0x0000`; `64` → `0x00000040`: lo=`0x0040`, hi=`0x0000`). Negative inline constants sign-extend (`-1` → `0xFFFFFFFF`: lo=`0xFFFF`, hi=`0xFFFF`; `-16` → `0xFFFFFFF0`: lo=`0xFFF0`, hi=`0xFFFF`). 32-bit literals preserve their full dword (`0x12345678`: lo=`0x5678`, hi=`0x1234`; `0x80017fff`: lo=`0x7fff`, hi=`0x8001`). Float inline tokens (`0.5`, `1.0`, `-2.0`, `4.0`) assemble as their 32-bit IEEE 754 bit patterns (`1.0` → `0x3f800000`: lo=`0x0000`, hi=`0x3f80`).
- **`clamp` saturation:** Unsigned add/mad saturate to `0xFFFF`; unsigned sub saturates underflow to `0x0000`; signed add/sub/mad saturate to `[-32768, 32767]`.
- **Silicon vs Translator parity:** 380 of 380 kernels match bit-for-bit between native hardware and `tools/rdna2spirv.py` lowered to SPIR-V (0 mismatches, 0 wave32/wave64 differences).
### Pixel-shader ISA on the hardware (`tools/ps_oracle.py`, EGEN-TOOL-PS-ORACLE)

`tools/hw_oracle` only launches compute kernels, so what only a pixel shader has (`v_interp_*` and the parameter
words the SPI writes into LDS, the barycentric VGPRs, exports and their formats, helper lanes) could not be
measured there. `tools/ps_oracle.py` runs hand-written RDNA2 pixel-shader ISA in a real draw on an AMD GPU, through
Vulkan's `VK_KHR_pipeline_binary`:

1. `vk_harness` builds a pipeline from a template vertex + pixel shader and writes its binary (`VKH_BINARY_OUT`).
   On the AMD Windows driver that is one PAL ELF: `.text` with `_amdgpu_gs_main` (the vertex stage) and
   `_amdgpu_ps_main`, and a `.note` with the PAL metadata in msgpack, from which the driver programs the register
   counts and the SPI setup (`spi_ps_input_ena/addr`, `spi_ps_input_cntl` per parameter, `spi_shader_col_format`).
2. The code under test is assembled with AMD clang (`LEONIDA_AMD_CLANG`, as for hw_oracle) and replaces
   `_amdgpu_ps_main`; `.text` grows if needed and the pixel stage's VGPR/SGPR counts are raised to the registers
   the code names. `--set PATH=JSON` changes a metadata entry as well, e.g.
   `.graphics_registers/.spi_ps_input_cntl/0/.fp16_interp_mode=true`.
3. `vk_harness` builds the pipeline from the patched binary (`VKH_BINARY_IN`), draws one full-screen triangle into
   an RGBA32F target and reads every pixel's four mrt0 dwords back bit for bit.

The template sets what the code finds when it starts, so `--show-template` prints the template's own ISA and SPI
metadata to copy the conventions from. With the built-in `interp2` template on gfx1036 the pixel stage is wave64,
the perspective I/J barycentrics are in v0/v1, M0 comes from s2 (`s_mov_b32 m0, s2`), and parameters 0 and 1 are
the vertex shader's locations 0 = (u, v, 0, 1) and 1 = (u, v, 0, 0), u and v going 0..2 across the screen.

```powershell
$env:LEONIDA_VK_DEVICE = "radeon"
python -I tools/ps_oracle.py --build <runtime build> --template interp2 --show-template
python -I tools/ps_oracle.py --build <runtime build> --template interp2 --asm body.s --size 4 1 [--json out.json]
```

Checked on an AMD Radeon iGPU (gfx1036, Windows driver 26.10.1): the driver accepts a patched binary (there is no
check over the code), the patched code runs as the pixel shader, and `v_interp_p1/p2_f32` of `attr0.x` at the four
pixel centres of a 4 x 1 target gives u = 0.125, 0.375, 0.625, 0.875, with `v_interp_mov_f32` reading the
parameter words P0 = 0 and P10 = 2.0 (`tests/test_ps_oracle.py`, which runs it on hardware when
`LEONIDA_PS_ORACLE_BUILD` names a runtime build). On other drivers the binary is not a single PAL ELF (the NVIDIA
driver returns two blobs) and the tool exits 77.

`--format` picks the color target (rgba32f by default; rgba16f/unorm/snorm/u/i, rg32f, r32f, rgba8*, rgb10a2unorm,
...), and the driver picks the export format for it as for a game pipeline; the readback is the target's texels as
stored. Measured on gfx1036 with the `const` template (EGEN-TOOL-PS-ORACLE-FORMATS):

| Target | `col_0_export_format` | What reaches the target |
|---|---|---|
| RGBA32F, RGBA32_UINT | 9 (32_ABGR) | the four exported dwords as they are; a channel exported as `off` is 0 |
| RG32F | 2 (32_GR) | R = the last enabled of x, z; G = the last enabled of y, w (all four enabled: z, w) |
| R32F | 1 (32_R) | the last enabled channel (all four enabled: w) |
| RGBA16F, RG16F, RGBA8_UNORM/SNORM, RGB10A2 | 4 (FP16_ABGR) | a `compr` export's halves as f16, converted by the CB; an f16 NaN half becomes `0xfe00` in an f16 target, 0 in a normalized one |
| RGBA16 UNORM/SNORM/UINT/SINT, RG16_UNORM | 5 / 6 / 7 / 8 | a `compr` export's 16-bit halves bit for bit (unorm16/snorm16/integer values, not f16) |
| RGBA8_UINT | 7 (UINT16_ABGR) | the 16-bit halves clamped to 255 |

A 16-bit export format takes a `compr` export; an uncompressed one into it is not meaningful (the target got the z
and w dwords). Limits: one draw of one triangle, mrt0 only; the vertex
stage is the template's, compiled by the driver; the SPI behaviour comes from the driver's metadata, so a `--set`
the hardware does not support is not refused here.

### Packed 16-bit integer operations on RDNA2 (SHADER-302)

*Measured on an RDNA2 iGPU (gfx1036), Windows 11, AMD Windows driver 32.0.21045.29003, HIP backend (`amdhip64_6.dll`), AMD clang 23.0.0git (`rocm-sdk-core 10.0.0`).*

380 raw compute kernels covering all 11 packed 16-bit integer instructions (`v_pk_add_u16`, `v_pk_sub_u16`, `v_pk_add_i16`, `v_pk_sub_i16`, `v_pk_mul_lo_u16`, `v_pk_mad_u16`, `v_pk_mad_i16`, `v_pk_max_u16`, `v_pk_min_u16`, `v_pk_max_i16`, `v_pk_min_i16`), tested in both wave32 and wave64:

- **Constant operands under `op_sel` / `op_sel_hi`:** Integer constants under VOP3P are uniformly treated as 32-bit dwords `C`. When `op_sel[i] == 0`, bits `[15:0]` are selected; when `op_sel[i] == 1`, bits `[31:16]` are selected. Positive inline constants zero-extend (`5` → `0x00000005`: lo=`0x0005`, hi=`0x0000`; `64` → `0x00000040`: lo=`0x0040`, hi=`0x0000`). Negative inline constants sign-extend (`-1` → `0xFFFFFFFF`: lo=`0xFFFF`, hi=`0xFFFF`; `-16` → `0xFFFFFFF0`: lo=`0xFFF0`, hi=`0xFFFF`). 32-bit literals preserve their full dword (`0x12345678`: lo=`0x5678`, hi=`0x1234`; `0x80017fff`: lo=`0x7fff`, hi=`0x8001`). Float inline tokens (`0.5`, `1.0`, `-2.0`, `4.0`) assemble as their 32-bit IEEE 754 bit patterns (`1.0` → `0x3f800000`: lo=`0x0000`, hi=`0x3f80`).
- **`clamp` saturation:** Unsigned add/mad saturate to `0xFFFF`; unsigned sub saturates underflow to `0x0000`; signed add/sub/mad saturate to `[-32768, 32767]`.
- **Silicon vs Translator parity:** 380 of 380 kernels match bit-for-bit between native hardware and `tools/rdna2spirv.py` lowered to SPIR-V (0 mismatches, 0 wave32/wave64 differences).
### Pixel-shader ISA on the hardware (`tools/ps_oracle.py`, EGEN-TOOL-PS-ORACLE)

`tools/hw_oracle` only launches compute kernels, so what only a pixel shader has (`v_interp_*` and the parameter
words the SPI writes into LDS, the barycentric VGPRs, exports and their formats, helper lanes) could not be
measured there. `tools/ps_oracle.py` runs hand-written RDNA2 pixel-shader ISA in a real draw on an AMD GPU, through
Vulkan's `VK_KHR_pipeline_binary`:

1. `vk_harness` builds a pipeline from a template vertex + pixel shader and writes its binary (`VKH_BINARY_OUT`).
   On the AMD Windows driver that is one PAL ELF: `.text` with `_amdgpu_gs_main` (the vertex stage) and
   `_amdgpu_ps_main`, and a `.note` with the PAL metadata in msgpack, from which the driver programs the register
   counts and the SPI setup (`spi_ps_input_ena/addr`, `spi_ps_input_cntl` per parameter, `spi_shader_col_format`).
2. The code under test is assembled with AMD clang (`LEONIDA_AMD_CLANG`, as for hw_oracle) and replaces
   `_amdgpu_ps_main`; `.text` grows if needed and the pixel stage's VGPR/SGPR counts are raised to the registers
   the code names. `--set PATH=JSON` changes a metadata entry as well, e.g.
   `.graphics_registers/.spi_ps_input_cntl/0/.fp16_interp_mode=true`.
3. `vk_harness` builds the pipeline from the patched binary (`VKH_BINARY_IN`), draws one full-screen triangle into
   an RGBA32F target and reads every pixel's four mrt0 dwords back bit for bit.

The template sets what the code finds when it starts, so `--show-template` prints the template's own ISA and SPI
metadata to copy the conventions from. With the built-in `interp2` template on gfx1036 the pixel stage is wave64,
the perspective I/J barycentrics are in v0/v1, M0 comes from s2 (`s_mov_b32 m0, s2`), and parameters 0 and 1 are
the vertex shader's locations 0 = (u, v, 0, 1) and 1 = (u, v, 0, 0), u and v going 0..2 across the screen.

```powershell
$env:LEONIDA_VK_DEVICE = "radeon"
python -I tools/ps_oracle.py --build <runtime build> --template interp2 --show-template
python -I tools/ps_oracle.py --build <runtime build> --template interp2 --asm body.s --size 4 1 [--json out.json]
```

Checked on an AMD Radeon iGPU (gfx1036, Windows driver 26.10.1): the driver accepts a patched binary (there is no
check over the code), the patched code runs as the pixel shader, and `v_interp_p1/p2_f32` of `attr0.x` at the four
pixel centres of a 4 x 1 target gives u = 0.125, 0.375, 0.625, 0.875, with `v_interp_mov_f32` reading the
parameter words P0 = 0 and P10 = 2.0 (`tests/test_ps_oracle.py`, which runs it on hardware when
`LEONIDA_PS_ORACLE_BUILD` names a runtime build). On other drivers the binary is not a single PAL ELF (the NVIDIA
driver returns two blobs) and the tool exits 77. Limits: one draw of one triangle, mrt0 as RGBA32F only; the vertex
stage is the template's, compiled by the driver; the SPI behaviour comes from the driver's metadata, so a `--set`
the hardware does not support is not refused here.

**On Linux (RADV).** Mesa's RADV returns the pipeline as one binary per stage, as `radv_shader_serialize` writes it
(Mesa 26.3 `radv_pipeline_cache.c`, structs in `radv_shader.h`): a type word, `struct ac_shader_config` (`num_sgprs`,
`num_vgprs`, ..., `rsrc1`), `struct radv_shader_info`, `total_size`, then `code_size`, `exec_size` and the other
sizes, then stats, code, IR and disassembly. The info struct's size changes between Mesa versions, so the tool finds
the fields from their values: the size block whose sizes add up to `total_size`, the pixel stage from the
`wave_size, ., stage = 4 (fragment), next_stage = none` bytes in the info, and `rsrc1` as the config word whose
VGPRS field is `(num_vgprs - 1) / (wave32 ? 8 : 4)` (`radv_postprocess_binary_config`) and whose FLOAT_MODE equals
`float_mode`. A layout where these do not agree is refused (exit 77), not guessed. The code is replaced, the sizes
are fixed, and `num_vgprs`/`num_sgprs` and rsrc1's VGPRS are raised; RADV builds an imported binary with
`radv_shader_create_uncached` (no cache lookup by key) and programs `SPI_SHADER_PGM_RSRC1_PS` from `rsrc1`. Two
limits: `--set` is refused (RADV keeps the SPI setup in its own structs, so a template has to ask for what it
needs), and only GFX10 (RDNA1/RDNA2) devices are taken, from the chip name RADV puts in the device name; GFX11 and
later interpolate with `lds_param_load` instead of `v_interp_*`, so e.g. an RX 9070 XT (`RADV GFX1201`) gets
`unsupported architecture` and exit 77.
### Packed 16-bit integer operations on RDNA2 (SHADER-302)

Mesa optimises varyings across stages, so a template's parameter layout can differ from the Windows driver's:
RADV moves `interp2`'s `in0 + in1` into the vertex shader and the pixel shader reads one parameter, attr0 =
(2u, 2v, 0, 1); `--show-template` shows the layout a run gets.

```sh
LEONIDA_VK_DEVICE=raphael python -I tools/ps_oracle.py --build <runtime build> --template interp2 --show-template
```

Checked on the Ryzen 7 9800X3D iGPU (`RADV RAPHAEL_MENDOCINO`, gfx1036, wave64 pixel stage, Mesa 26.3.1, CachyOS):
the Windows check's `attr0.x` code (with `v40`, past the template's 8 VGPRs) gives u = 0.25, 0.75, 1.25, 1.75,
P0 = 0 and P10 = 4.0 for the single parameter, and a second patch with another constant changes the result (nothing
is cached).

- **Constant operands under `op_sel` / `op_sel_hi`:** Integer constants under VOP3P are uniformly treated as 32-bit dwords `C`. When `op_sel[i] == 0`, bits `[15:0]` are selected; when `op_sel[i] == 1`, bits `[31:16]` are selected. Positive inline constants zero-extend (`5` → `0x00000005`: lo=`0x0005`, hi=`0x0000`; `64` → `0x00000040`: lo=`0x0040`, hi=`0x0000`). Negative inline constants sign-extend (`-1` → `0xFFFFFFFF`: lo=`0xFFFF`, hi=`0xFFFF`; `-16` → `0xFFFFFFF0`: lo=`0xFFF0`, hi=`0xFFFF`). 32-bit literals preserve their full dword (`0x12345678`: lo=`0x5678`, hi=`0x1234`; `0x80017fff`: lo=`0x7fff`, hi=`0x8001`). Float inline tokens (`0.5`, `1.0`, `-2.0`, `4.0`) assemble as their 32-bit IEEE 754 bit patterns (`1.0` → `0x3f800000`: lo=`0x0000`, hi=`0x3f80`).
- **`clamp` saturation:** Unsigned add/mad saturate to `0xFFFF`; unsigned sub saturates underflow to `0x0000`; signed add/sub/mad saturate to `[-32768, 32767]`.
- **Silicon vs Translator parity:** 380 of 380 kernels match bit-for-bit between native hardware and `tools/rdna2spirv.py` lowered to SPIR-V (0 mismatches, 0 wave32/wave64 differences).
#### Helper lanes at a triangle edge (`tools/ps_helper_probe.py`, SHADER-681)

What a pixel wave does with the helper lanes of a partly covered 2x2 quad, which the translator's EXEC model and
`s_wqm` lowering depend on (`docs/design.md` row "Whole-quad mode"). The probe draws one triangle, (-1, -1), (1.3, -1),
(-1, 0.9), into a 16 x 16 target, so its edge cuts quads; the varying is the vertex position, so each lane's
interpolated value is its own pixel centre. The template pixel shader takes a derivative, so the driver builds it as
a shader that needs quads. Seven wave64 programs replace the pixel shader. A helper is never written to the target,
so a covered lane of the same quad reads the helper's registers through DPP `quad_perm` and exports them.

```sh
LEONIDA_VK_DEVICE=raphael python -I tools/ps_helper_probe.py --build <runtime build> [--json out.json]
```

On the Ryzen 7 9800X3D iGPU (`RADV RAPHAEL_MENDOCINO`, gfx1036, Mesa 26.3.1): 137 covered pixels in three waves,
25 (covered lane, helper neighbour) pairs, and all 10 checks hold in three runs, with identical words except an
unwritten helper register:

| check | gfx1036 |
|---|---|
| entry EXEC | exactly the covered pixels: helpers start off |
| `s_wqm_b64 exec, exec` | turns on exactly the quads with a covered pixel |
| wave layout | quads packed from lane 0; a quad with no covered pixel is not in the wave |
| lane order in a quad | lane % 4 = (x & 1) + 2 (y & 1): top-left, top-right, bottom-left, bottom-right |
| a helper's `v_interp_p1/p2_f32` after `s_wqm` | its own pixel centre, extrapolated (error 1.64e-4, as covered) |
| a helper's v0/v1 without `s_wqm` | already its own pixel's I/J (to 7e-5; I + J > 1, outside the triangle) |
| a helper's VGPR without `s_wqm` | not written (an unwritten register, different in every run) |
| EXEC without `s_wqm` | stays the entry mask |
| DPP without `fi` from an inactive lane | leaves the destination unchanged |
| `exp mrt0 ... done vm` with helpers on in EXEC | writes no pixel outside the triangle |

The last row is what the driver's own code relies on: RADV compiles the template as `s_wqm_b64 exec, exec` at entry,
then the interpolation and an `exp mrt0 ... done vm` with the helpers still on. The programs are wave64; a wave32
pixel stage (`RADV_PERFTEST=pswave32`) and GFX11+ exit 77. Not measured: wave32, RDNA1/gfx1013 and the Windows driver.

### Packed 16-bit integer operations on RDNA2 (SHADER-302)

*Measured on an RDNA2 iGPU (gfx1036), Windows 11, AMD Windows driver 32.0.21045.29003, HIP backend (`amdhip64_6.dll`), AMD clang 23.0.0git (`rocm-sdk-core 10.0.0`).*

380 raw compute kernels covering all 11 packed 16-bit integer instructions (`v_pk_add_u16`, `v_pk_sub_u16`, `v_pk_add_i16`, `v_pk_sub_i16`, `v_pk_mul_lo_u16`, `v_pk_mad_u16`, `v_pk_mad_i16`, `v_pk_max_u16`, `v_pk_min_u16`, `v_pk_max_i16`, `v_pk_min_i16`), tested in both wave32 and wave64:

- **Constant operands under `op_sel` / `op_sel_hi`:** Integer constants under VOP3P are uniformly treated as 32-bit dwords `C`. When `op_sel[i] == 0`, bits `[15:0]` are selected; when `op_sel[i] == 1`, bits `[31:16]` are selected. Positive inline constants zero-extend (`5` → `0x00000005`: lo=`0x0005`, hi=`0x0000`; `64` → `0x00000040`: lo=`0x0040`, hi=`0x0000`). Negative inline constants sign-extend (`-1` → `0xFFFFFFFF`: lo=`0xFFFF`, hi=`0xFFFF`; `-16` → `0xFFFFFFF0`: lo=`0xFFF0`, hi=`0xFFFF`). 32-bit literals preserve their full dword (`0x12345678`: lo=`0x5678`, hi=`0x1234`; `0x80017fff`: lo=`0x7fff`, hi=`0x8001`). Float inline tokens (`0.5`, `1.0`, `-2.0`, `4.0`) assemble as their 32-bit IEEE 754 bit patterns (`1.0` → `0x3f800000`: lo=`0x0000`, hi=`0x3f80`).
- **`clamp` saturation:** Unsigned add/mad saturate to `0xFFFF`; unsigned sub saturates underflow to `0x0000`; signed add/sub/mad saturate to `[-32768, 32767]`.
- **Silicon vs Translator parity:** 380 of 380 kernels match bit-for-bit between native hardware and `tools/rdna2spirv.py` lowered to SPIR-V (0 mismatches, 0 wave32/wave64 differences).


### Real pixel shaders on two GPUs (`tools/ps_sweep.py`, `runtime/test/vk_ps_sweep.cpp`)

The oracles above run hand-written programs. This one draws translated corpus pixel shaders, to find what only real
shaders trigger. `ps_sweep.py sample` takes a fixed-seed stratified sample of the translated pixel shaders (strata:
needs barycentrics, descriptor bindings none / 0-3 only / any above 3, control-flow style; at least 10 per stratum,
the rest in proportion). `vk_ps_sweep` draws one of them with a generated vertex shader (a viewport-covering triangle
and a vec4 at every input location), the runtime's bindless set 0 with only its fallbacks, a zeroed 1 MiB heap, the
push-constant block and 32x32 `RGBA32F` targets cleared to -1. Everything is identical on every GPU, so the same
sample can be run on each and compared; `ps_sweep.py compare` classifies each shader: OK on both, differs beyond
tolerance (1e-3 absolute plus relative), NaN or all-zero on one GPU only, fails on one GPU, fails on both, timeout.
When both workers exit successfully, `compare` requires matching target counts (1–8) and exactly 16,384 bytes per
target in each raw file (32 × 32 pixels × 16 bytes). A missing file, a wrong byte length or an invalid or mismatched
count is reported as `invalid readback`, not as a GPU result. Failed and timed-out draws are classified by exit
code without reading either file. The size check establishes completeness; this command still compares float values
with tolerance, not byte-for-byte equality.
Variant `t` (`VKS_TEX=1`) replaces the fallback images with an 8x8 gradient (2D, 2D array, 1D, 3D) and a linear
sampler so sampled values are not all zero.

Warning: a few sampled shaders hang NVIDIA GPUs with the zeroed heap (see "Fails on one GPU, NVIDIA" below). The
`--timeout` of `ps_sweep.py run` only bounds the wait on the host side; the driver resets the GPU (TDR). On a GPU that
also drives the display the desktop freezes until each reset (a tester saw five in six minutes on an RTX 3060 around
shaders 120 to 310 of the 480). Run it on a GPU with no display attached, or expect a freeze of a few seconds per hang;
`--lo`/`--hi`/`--only` run part of the sample.

RTX 4080 SUPER and Intel UHD 770 (32.0.101.6129), Windows 11, 480 of 12,101 pixel shaders (seed 20261122):

| class | variant z (fallback images) | variant t (gradient textures) |
|---|---|---|
| OK on both | 262 | 260 |
| differs beyond tolerance | 0 | 1 |
| NaN or zero on one GPU only | 0 | 0 |
| fails on one GPU | 214 | 215 |
| fails on both | 0 | 0 |
| timeout (both) | 4 | 4 |

- **Fails on one GPU, Intel (210 in both variants):** the module needs `fragmentShaderBarycentric` (206) or
  `shaderResourceMinLod` (3) or both (1), and the UHD 770 has neither feature. The runtime's barycentric fallback is not
  part of this sweep. 5,298 of the 12,101 pixel shaders need barycentrics and 29 need MinLod.
- **Fails on one GPU, NVIDIA (4 in z, 5 in t):** `VK_ERROR_DEVICE_LOST` after a few seconds while the Intel GPU
  returns at once with every pixel unwritten. All are tier C shaders; in the one disassembled, a scalar loop reads a
  list from a buffer until an entry's low 16 bits are `0xffff`, and the zeroed heap has no terminator, so the loop
  never ends. A 10-line GLSL
  fragment shader with the same loop (`while ((heap[i] & 0xffff) != 0xffff) i++;` on a zeroed buffer reached through
  its device address) hangs the same way. The synthetic input, not the translation, is the cause; a real heap holds the
  terminator.
- **Timeout (4, both GPUs, 40 s):** large `dispatch loop` shaders; not root-caused.
- **Differs beyond tolerance (1, variant t):** a tier C shader (`wave64_graphics` C), 92 of 4,096 values of one target
  (about 2 % of the pixels), differences up to 0.06. It is identical with a constant texture and with a NEAREST sampler,
  so the cause is the two GPUs' linear-filter weights, not the translation.
- The 262 / 260 shaders that run on both agree to 4.9e-4 at most; 239 of the 260 in variant t are bit-identical.
  Of those, 107 write non-zero values and 153 write only zeros or nothing (a shader whose descriptors come from a
  zeroed heap mostly samples zero or discards). That is the limit of the synthetic inputs: a
  seeded heap was tried and abandoned, because a pointer read from float data is a wild address and takes NVIDIA's
  device down.

**With the fallbacks (`--bary-fallback` plus the MinLod fallback of #4642, same 480 shaders, variant z).** The same sample was
retranslated with both flags and drawn through `<name>.bary_gs.spv` where a shader reads raw barycentrics (`VKS_GS`,
`tools/bary_fallback_gs.py`). The Intel UHD 770 creates 440 of the 480 (265 before; the 207 that needed
`fragmentShaderBarycentric` or `shaderResourceMinLod` are gone), 32 time out (zeroed-heap loops, which the same
translation also hits on the RTX 4080 SUPER), and 8 are not translated (`computed I/J pair under --bary-fallback`,
#3898). Intel against NVIDIA on the same fallback translation: 434 OK on both, 0 differ, 4 NVIDIA device losses
(the unterminated loop above), 8 not translated, 34 timeouts on both. Intel with the fallbacks against NVIDIA with
the default translation: 373 OK, 53 differ by more than 0.1 and 8 are NaN on one side. The same 61 differ between the
two translations on NVIDIA alone, so it is the translation, not the GPU. **Cause (counted on the same 480 translated
again, `--bary-fallback` alone, without the MinLod flag of #4642, against the default):** 199 modules differ between the two translations, 190 that read a
16-bit interpolant (`v_interp_p1ll/p1lv/p2_f16`) and the 9 that read raw barycentrics; the other 273 translated under
both are byte-identical, so ordinary `v_interp_p1/p2/mov_f32` is not translated differently (the remaining 8 are not
translated under the fallback). `--bary-fallback` has no `PerVertexKHR` inputs, so a 16-bit interpolant is read as the
smooth f32 of its packed word (design.md, "16-bit interpolation"); `--f16-interp-fallback` is its companion, and
`make_game.py` turns it on with it wherever the device has no per-vertex inputs, as an Intel UHD has none. The sweep
used the first flag without the second, and its generated vertex shader wrote the decoded halves nowhere. The synthetic
`ps_interp_f16_high` drawn with `vs_f16_pack` agrees: with `--bary-fallback` alone its f16 halves are off by 0.984 on
both GPUs, with both flags and by default they are within 2e-3 on both, and `ps_interp_f32_mods` / `ps_interp_mov_mods`
(plain f32 and mov reads) are exact with `--bary-fallback`. `ps_sweep.py` now builds the vertex shader of a pixel
shader with `f16_halves` in its index entry so that the halves are decoded from the same words a default translation
reads per vertex. Not shown: the 61 individually (a 12-shader subset of the 190 agreed between the two translations on
the RTX 4080 SUPER; draws of these shaders take about 20 s each). Only 9 of the 480 read raw
barycentrics and need the geometry shader; all 9 agree on both GPUs.

## Coverage plan and cost

Start with straight-line scalar/vector ALU because it is deterministic, needs no descriptors and
covers the operations most shaders use. Then add, in order:

1. **EXEC/VCC and `cndmask`.** Per-lane masking and scalar state are where a one-lane-per-invocation
   translator differs most from hardware.
2. **16-bit arithmetic and partial writes.** `v_add/sub/mul/fma_f16`, `op_sel`, destination-half
   preservation, packed ops. The translator's contract calls out which forms keep the high half;
   those rules need value tests.
3. **Branches and loops.** Modelled control flow, then irreducible graphs and the dispatch-loop form.
   The existing control-flow unit tests check graph structure; differential tests check the values.
   `tools/cf_diff.py` (FUZZ-8) does this for whole waves: seeded programs built from the shapes
   compilers emit (saveexec if/else, `v_cmpx`, VCC- and SCC-voted branches, uniform and per-lane
   loops, two-entry cycles), run by `tools/cf_ref.py` (cpu_ref plus branches, compares and saveexec)
   and by `tests/semantic/spirv_wave_interp.py` (every invocation of a subgroup, with structured
   reconvergence), compared lane by lane in both the structured and the dispatch-loop form.
   The corpus-side counterpart is `tools/ab_control_flow.py`: it renders branchy pixel shaders
   translated both ways (real loops and conditionals, and `--dispatch-loop`) and compares the two
   images byte for byte. It needs a finished shaders stage and `runtime/build/vk_texture_test`, and
   writes the outcome of each shader to `outcomes.tsv`.
4. **LDS, global memory, buffers and images.** These need a memory model and descriptor setup; the
   existing `vk_compute_test`/`vk_store_test`/`vk_texture_test` programs show the shape.
5. **Wave operations.** Ballot, shuffles, DPP, `readlane`/`writelane`, `permlane`, `mbcnt`, and the
   wave64 split on 32-wide subgroups. These are the largest semantic gap and need a 64-lane model
   plus a subgroup-size-aware host. `tools/crosslane_fuzz.py` (FUZZ-7, above) is that model, run on
   the GPU.
6. **Pixel interpolation and exports.** Barycentrics, per-vertex inputs, compressed exports, MRT and
   depth. The existing GPU tests already pin several of these; a CPU model makes them differential.

Cost is dominated by the GPU dispatch and pipeline creation, not by the Python interpreter. A small
case is one `llvm-mc` run, one translation, one `spirv-as` + `spirv-val`, one pipeline and one
dispatch: seconds per case on a desktop GPU. The interpreter is milliseconds for the programs in this
subset. A suite of a few hundred cases is practical locally; running every corpus shader is not the
goal, because many need descriptors and data the harness cannot invent. The useful target is a
hand-picked suite that covers each instruction family and each known semantic trap, plus seeded
fuzzing over operand values and lane masks within that suite.

## What this cannot model

The prototype and the first harness stage deliberately do not cover:

- **Descriptors.** T#/V#, S# and `s_load`/`image_*`/`buffer_*` need realistic descriptor memory and
  binding setup. A wrong descriptor is indistinguishable from a wrong translation.
- **Interpolation.** `v_interp_p1/p2/mov` depend on rasterization state, barycentric conventions and
  the pixel center. The CPU model would need the same triangle and sample positions as the GPU.
- **Wave operations.** Ballot, shuffles, DPP, permlane and cross-half wave64 behaviour need a full
  64-lane model and a GPU whose subgroup size matches the wave. On a 32-wide subgroup host, a wave64
  program is not the same execution. The one-lane interpreter still can't run them; the GPU fuzzer
  `tools/crosslane_fuzz.py` covers them instead, with wave64 checked through `--wave64-emulate`.
- **Memory and LDS.** Address spaces, out-of-bounds rules, atomics, ordering and barriers. These are
  modelled later, with the same "unsupported, never guessed" rule.
- **Transcendentals and approximation.** `v_rcp_f32`, `v_rsq_f32`, `v_exp_f32`, `v_log_f32` and the
  trig instructions have implementation-defined accuracy. The other project handled a 5–14 ulp
  reciprocal difference by computing an envelope, not by demanding a bit match. A differential test
  should either use exact instructions or compare within a documented tolerance/envelope.
- **The MODE register and rounding modes.** The translator skips `s_setreg`; the CPU model assumes
  round-to-nearest and gradual underflow. Tests that depend on flush-to-zero or round-toward-zero are
  out of scope until the harness sets and checks those modes.
- **NaN payloads and signed-zero corner cases in min/max/clamp.** Vulkan leaves some of this
  implementation-defined. Compare NaNs by class, and pin only the cases the ISA defines.
- **Wave-size-dependent operand forms.** `v_cndmask_b32` takes a 32-bit SGPR mask in wave32 and an
  `s[lo:hi]` pair in wave64. The model follows the form printed in the listing; a listing assembled
  for the wrong wave size is a test bug, not a translator result.
- **Scheduling and hazards.** The model is sequential. It does not test wait counts, dependency
  stalls, or whether two instructions can overlap. Those are hardware timing properties, not
  arithmetic results.

Growing the model means adding one instruction family at a time, with a hand-written test and a
pinned expected value, and removing the `Unsupported` path for that family. A family that cannot be
modelled faithfully stays a reported skip; it is never approximated silently.

## Prior art

The technique is described in another project's write-ups; no code from them is copied.

- [Xbox Series shaders](https://github.com/realdody/americas-recomp/blob/xbox-series/docs/xbox/XBOX_SHADERS.md),
  sections "Execution harness" and "A CPU reference for small real shaders".
- [tools/xbox README](https://github.com/realdody/americas-recomp/blob/xbox-series/tools/xbox/README.md),
  which describes `exec` and the check against a Python reference.

Their reported numbers, as a scale reference only: 183 of 872 compute shaders of 3,000 bytes or less
gave byte-identical CPU/GPU results, with none differing and 687 skipped; for shaders up to 1,500
bytes the later run reported 242 matches, 1 mismatch (reciprocal precision), 2 harness failures and
435 skips; the pixel-shader run reported 157 bit-identical, 0 differing and 581 skips. The numbers
are context for the scale of the approach, not a result of this change.

The other project is GPL-2.0-only and this one is GPL-3.0-or-later. The design above is an
independent implementation from the ISA and from LLVM's disassembly; the links are for the technique
and the reported results, not for code.

## Deliberate limits of this change

This change adds the design, the CPU prototype and its unit tests. It does not modify
`tools/rdna2spirv.py`, does not add a GPU runner, and does not use game data. The prototype is useful
now because it pins the semantics of the first instruction families and gives the future harness a
reference that fails loudly on anything it does not understand.
