# Translator regression tests

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.

    python -I scripts/run_semantic_tests.py tests/semantic        # one file per core: 26 s instead of 253 s on 16 cores
    python -m unittest discover -s tests/semantic -v             # the same tests, one file after another

The runner starts `python -m unittest <file>` for each file, several at a time, prints the output of the files that
fail and one summary line; `-j N` sets how many run at once (`-j 1` is the discover command above) and `-v` prints
every file's output. Files run side by side, so a test writes only into its own temporary folder (`tempfile`), never
to a fixed path, and does not rely on another file having run first.

Every file here must call `tools/rdna2spirv.py` (directly, through `support.py`, or by running the tool) and assert
on what it emitted. A test that reimplements the instruction in Python passes whatever the translator does, so it is
not a regression test. `tests/test_semantic_translator_use.py` enforces this; the interpreter self-tests and the
files whose subject is hand-written SPIR-V are listed there with a reason. The guard checks that a file reaches the
translator, not that every assertion depends on it.

What they check (`tools/rdna2spirv.py` on pixel, vertex, compute and other stages, wave32 and wave64, mostly one
instruction family per file; the list below describes some of them, not every file):

- `test_ps_semantics.py` translates `ps_sdwa`, `ps_ops`, `ps_ops2`, `ps_ops3`, `ps_ops4` and `ps_legacy_mul` and
  *executes* the SPIR-V with `spirv_mini_interp.py`, then compares every `mrt0` word with the expected value. Each
  shader exports four 32-bit words, so six shaders give 24 asserted values. The expected words are written down by
  hand, with the ISA rules in the module docstring, and are cross-checked against plain-Python reference calculations.
- `test_value_semantics.py` does the same for the recent value-level fixes: scalar b16 shifts, packed b16 shifts
  with `op_sel`, `v_add_nc_u16`, `v_sin_f16`/`v_cos_f16`, `s_buffer_load_dwordx2` into VCC and flat
  `v_interp_mov` input next to an interpolated one. `ps_ops5`, `ps_ops6`, `ps_vcc_x2` and `ps_flat` give 16 more
  asserted values, and `ps_fma_literal` 4 more: which operand the literal K multiplies in
  `v_fmamk/fmaak/madmk/madak_f32`, and `ps_f16_vop2` (f16 VOP2 partial writes and `v_fmac_f16`, words measured on
  RDNA2 hardware) four more. The other `ps_f16_*` shaders use GLSL functions this interpreter does not model;
  `tests/test_f16_hw_words.py` runs all of them with `tests/spirv_interp.py`. `cs_bounds` runs as compute and asserts
  the four dwords in its buffer after in-bounds
  load/store and out-of-bounds load/store attempts (the V# has num_records 16). `ps_ops4` and `ps_legacy_mul`
  are also asserted there, duplicating the matching `test_ps_semantics.py` cases. As above, constants are
  written by hand and recomputed by plain-Python references checked against each other. `cs_fmt_1010102` runs
  as compute too and asserts 16 dwords: 10:10:10:2 words read through formats 50 and 44 (from the V# and from
  tbuffer's own format) and stored in both. `ps_scc_minmax` asserts SCC after `s_min`/`s_max` `_i32`/`_u32`:
  0 on a tie, and 1 on a pair (-1 against 1) that only the right signedness orders that way.
  `ps_vcmp_f32_nan` asserts 32 f32 VOPC results with a NaN input (ordered `ge`/`lg`/`o` false, the `n*`
  forms true, `v_cmp` into VCC and an SGPR, and `v_cmpx`), `ps_vcmp_64` 32 64-bit integer compares (`u64`/`i64`,
  equal high words compared on the unsigned low word), and `cs_w32_andn2_exec` (one 32-thread wave)
  `s_andn2_saveexec_b32` / `s_andn2_wrexec_b32`: the EXEC they leave, the old / new EXEC in D and SCC.
  `RawBufferTailBounds` runs one-thread compute programs written in the test around one `buffer_load` and
  asserts the words and the exact addresses loaded for the raw-buffer rule measured on an RX 6800 (per-dword
  `num_records` check on the exact byte offset); `MeasuredBoundsShaders` runs the four GPU tests of the SMEM and
  raw MUBUF rules (`cs_smem_bounds`, `cs_smem_stride_align`, `cs_smem_sload_neg`, `cs_buf_tail`).
- `test_buffer_oob_select.py` runs 33 hand-tabulated MUBUF cases through the translator and the interpreter, loads
  and stores, each with a literal and a run-time V#: the first-byte bound of every OOB_SELECT mode (the RDNA2 rules of
  docs/buffer_bounds.md, docs/design.md Buffer bounds checking).
- `test_buffer_swizzle.py` runs hand-worked MUBUF cases for swizzled V#s (SWIZZLE_ENABLE) and ADD_TID_ENABLE, the
  RDNA2 ISA's swizzled buffer_offset and its OOB_SELECT 3 rule (docs/design.md Swizzled buffer addressing): the
  exact addresses loaded and stored for dword, multi-dword and typed accesses, each with a literal and a run-time
  V#, and the GPU fixture `cs_buf_swizzle` on 64 threads.
- `test_tbuffer_store.py` runs the GPU test `cs_tbuffer_store` (13 `tbuffer_store_format_*` stores through V#s
  whose format differs from the instruction's, or is 0) and asserts the 52 words `check.py` expects, measured natively on a
  Radeon 680M; also the default `BUF_FMT_8_UNORM` byte, a format without an encoder staying `Unsupported`, and the
  store's data VGPRs counting as reads.
- `test_buffer_formats_all.py` runs `buffer_load_format_xyzw` (through the V# and through tbuffer) and `buffer_store_format_xyzw` of the 33 formats
  the SCALED, SINT, 10_11_11 / 11_11_10 FLOAT, 2_10_10_10 and 10_10_10_2 families added, with edge values, NaN, infinity and denormals, against
  `tools/buffer_format_fuzz.py`'s model, which the Radeon 880M reproduced for 1.33 million validated elements (docs/buffer_formats.md). Two stores of
  older formats that differ from the hardware are pinned as expected failures.
- `test_scalar_mem_smem.py` runs the code `Gen.scalar_load` emits with its own small evaluator (64-bit
  addresses, forward selection constructs) and asserts the SGPR values and the addresses read, including the
  measured SMEM truncation, `s_buffer_load` `NUM_RECORDS` bound and signed `s_load` immediate.
- `test_alu_movrel.py` runs hand-written listings for `v_movrels_b32`, `v_movreld_b32` and `v_movrelsd_b32`
  (10 cases): sentinels in v1..v8, M0 set by `s_mov_b32`, then the indexed register and its neighbours are
  exported, so a write must land only in `VGPR[D.addr + M0]`. A plain-Python register-file model checks the
  hand-written constants. `v_movrelsd_2_b32` is covered by `test_alu_movrelsd_2.py`.
- `test_andn2_exec_b64.py` runs `s_andn2_saveexec_b64` and `s_andn2_wrexec_b64` on all 64 lanes of a wave64
  (12 pairs of S0 and EXEC for each): every lane stores whether the instruction left it on, and lane 0 stores SCC
  and the population count and lowest lane of D, against the ISA's two lines written as integers. Results in the
  low half only, the high half only, both and neither are all in the table. It also runs the GPU test
  `cs_w64x_andn2_exec`.
- `test_saveexec_b64.py` does the same for `s_and_saveexec_b64`, `s_or_saveexec_b64`, `s_andn2_saveexec_b64` and
  `s_andn2_wrexec_b64` with S0 a compare result in VCC or in an SGPR pair (13 pairs of S0 and EXEC for each op and
  source), and runs the GPU test `cs_w64x_saveexec_b64`.
- `spirv_wave_interp.py` runs every invocation of one subgroup at once through structured control flow
  (split at a selection, rejoin at its merge, loops until no invocation is left) with the subgroup
  operations the translator emits. It is the SPIR-V side of `tools/cf_diff.py` (FUZZ-8);
  `test_spirv_wave_interp.py` pins its convergence rules on hand-written modules.
- `test_global_mem_offset.py` runs the address arithmetic `global_load*`/`global_store*` emit (its own small
  evaluator, since the interpreter refuses 64-bit integers and PhysicalStorageBuffer) and asserts the address
  accessed for the signed 12-bit instruction offset (-4, -2048, 2047, 0) in both addressing modes.
- `test_float_nan_fract.py` runs float min/max/min3/max3/med3 with a quiet NaN and `v_fract` next to 1.0 (f32 and
  f16) with inputs from memory, and the GPU fixtures `ps_fnan_med3`, `ps_fnan_fract`, `ps_fnan_med3_s2` and
  `ps_f16_fnan_med3`; the expected words of the first two were measured on an AMD Radeon RX 6800 (gfx1030), those of
  the other two follow the same measured med3 rule. A NaN result is asserted as "a NaN", not by its bits.
- `test_rule_gpu_fixtures.py` runs the GPU fixtures for the `docs/design.md` rules that used to be CPU-only (TEST-175:
  `v_cmp(x)_nlg`, `v_cvt_off_f32_i4`, the `v_fmac_f16` / `v_interp_p2/mov_f32` output modifiers and VOP3 modifiers on
  constants) through the interpreter, and checks that the words each `.asm` header derives are the ones `check.py`
  and `scripts/run_gpu_tests.py` expect. The two interp fixtures get the constants their vertex shader,
  `vs_interp_consts`, exports.
- `test_vcmpx_forms.py` runs the 14 `v_cmpx` forms that had no test (`eq`/`gt`/`le`/`lt`/`ngt`/`nle` `_f32`,
  `f_i32`, `t_u32`, `eq`/`gt`/`le` `_i64`, `ge`/`lt`/`ne` `_u64`) in both encodings: each on a table of
  operands (NaN, signed zeros, infinities; 64-bit values that only one word or the sign tells apart), where
  the pixel must survive exactly when Python's comparison of the operands holds, and on a 16-lane wave with
  one lane switched off beforehand, where the lanes left running and an untouched VCC are checked. It also
  runs the GPU test `ps_vcmpx_forms` and re-derives its four words.
- `test_image_bvh_tri.py` runs `image_bvh64_intersect_ray` on hand-built triangle nodes. It checks the vertex
  slots of each node type (TRI0–TRI3) and every selector of the barycentric remap byte at +60, against weights
  written by hand and a plain-Python solve (docs/design.md, Ray tracing row).
- `test_image_bvh_tri_hw.py` runs both forms of the instruction on the triangle nodes and rays measured natively on
  an RX 6800 (gfx1030) with `tools/bvh_tri_probe.py`'s kernels and compares the four words bit for bit: all 308
  single-lane rows (slots, both windings, misses, remap byte, t max and t sign, zero area, NaN and inf in the node
  and the ray, shared edges and vertices) and 84 lane results. The table is written from the raw outputs, not by
  hand.
- `test_interp.py` tests the interpreter against the SPIR-V / GLSL.std.450 definitions. The interpreter raises
  `UnsupportedConstruct` for anything it does not model instead of approximating it. It can seed Input variables,
  PushConstant words and a flat physical-address memory map for the tests above.
- `test_wave64_cross_half.py` translates the GPU tests `cs_w64x_readlane`, `cs_w64x_writelane` and
  `cs_w64x_bpermute` (no `--wave64-emulate`) and runs all 64 lanes in lockstep with
  `spirv_mini_interp.run_workgroup`. On one 64-wide subgroup the 24 words must equal the GPU check's values
  (`runtime/test/shaders/check.py`). On two 32-wide subgroups (NVIDIA without emulation) `ds_bpermute` stays
  exact, `v_writelane` writes stay exact, and a `v_readlane` from the other half is a shuffle from lane >= 32,
  undefined in SPIR-V (`UndefinedResult`); a "lane mod 32" model of issue #94 gives the words a 32-wide GPU
  shows (all 24 matched on an RTX 4070 SUPER, driver 610.88, without `--wave64-emulate`). Two SPIR-V sabotages (readlane lane & 31, bpermute without its half base) must change the
  64-wide result.
- `test_wave32_pixel_subgroup64.py` translates wave32 pixel shaders with lane-specific inputs in each physical half
  and runs them as one 64-lane subgroup and as two 32-lane subgroups. `v_readfirstlane`, `v_readlane`,
  `v_writelane`, `ds_bpermute`, `ds_permute`, ballots, and both votes must produce the same per-half results.
  The descriptor case makes the first quad of each half either not running (`run_workgroup(lanes=...)`) or helpers
  left out of subgroup operations (`helpers=...`), and binds the descriptor hash table and an image array
  (`resources=...`): each half must read its own image.
- `test_w64x_scalar_load_order.py` translates wave64 programs with `--wave64-emulate` and checks the emitted rounds:
  no case of the round loop's switch reads or writes buffer memory after a write, or writes after a read, so a
  `buffer_store` and a later `s_load` / `s_buffer_load` of the wave (and a scalar load and a later store) run in
  different rounds (#1587). Loads alone share a round.
- `test_wave64_subgroup_ops.py` runs eight more wave64 GPU tests the same way (`cs_w64x_ballot`,
  `_readfirstlane`, `_load_ballot_load`, `_partial`, `cs_wave64_ballot`, `cs_wm_bfm64`, `cs_wm_half_cmp`,
  `cs_wavemask`): on one 64-wide subgroup every word equals the GPU check; on two 32-wide subgroups the
  words pinned there are the ones an RTX 4070 SUPER (driver 610.88) wrote without `--wave64-emulate`. The
  cs_w64x tests with branches need llvm-mc encodings and are not covered.
- `test_alu_int_misc.py` translates one-instruction synthetic listings for `v_bfm_b32`, `v_and_or_b32`,
  `v_add_lshl_u32`, `v_lshl_or_b32`, `v_mad_i32_i24` and `v_mul_lo_u32` and checks the mrt0 word against
  hand-computed ISA values (masked shift counts, wrapping adds/products, 24-bit sign extension).
- `test_lds_subdword_wide.py` translates the GPU tests `cs_w32_lds_subdword` (`ds_write_b8`/`_b16` and their
  `_d16_hi` forms, `ds_read_u8`/`_i8`/`_u16`/`_i16`) and `cs_w32_lds_wide` (`ds_write`/`ds_read` `_b64`/`_b96`/`_b128`,
  `ds_read2st64_b32`) and runs 32 lanes in lockstep; the 16 words come from a byte-level LDS model of the ISA rules
  and match what an RTX 4070 SUPER (driver 610.88) wrote. Three SPIR-V sabotages (zero-extending `_i8`/`_i16`, no
  high-half shift for `_d16_hi`, `st64` offset in dwords) must change the result.
- `test_negative_controls.py` sabotages the translated SPIR-V text and requires the result to change.
- `test_mutations.py` sabotages a temporary copy of `tools/` (13 regressions across SDWA, min3/max3, `v_bfe_i32`,
  `v_pk_fma_f16`, `v_fma_f16 op_sel`, `v_subrev_co_ci_u32`, `v_cmpx_eq_u64`, `v_mul_i32_i24`, `v_ffbh_i32`,
  `v_pk_lshlrev_b16`, `v_sin_f16` and `v_mul_legacy_f32`) and requires `test_ps_semantics.py` to fail for the
  matching shader.
- To check a new test file the same way before sending it, run `python -I tools/semantic_mutants.py
  tests/semantic/test_x.py --op <op>` (or `--function NAME`, `--lines A-B`). It makes one-line mutants of the
  translator code for that op (swapped opcode and GLSL pairs, comparisons, small constants ±1, swapped `OpSelect`
  arms, a repeated operand) in a private copy of `tools/`, and lists the ones the file still passes. An op that is
  translated through a lookup table (`F1`, `LDS_ATOMICS`, ...) also gets its table entry swapped for the table's other
  values. A survivor
  usually means a case asserts on the file's own reference model rather than on translated output, or that its
  inputs can't tell the variants apart (all bytes of S0 >= S1, no bits overlapping, only lanes 0–3 checked). Some
  survivors are equivalent mutants, and in a branch shared by several ops the other ops' lines are listed too, so
  read each one before adding a case. `tests/test_semantic_mutants.py` tests the tool.
- `test_f32_valu.py` runs `ps_f32_mods` (`v_mac_f32` rounds the product before the add; VOP3 clamp and output
  modifiers on `v_cvt_f32_ubyte*`/`u32`/`f16` and `v_rcp_iflag_f32`), found by `tools/f32_fuzz.py` (FUZZ-3), and
  tests the interpreter's float operations against GLSL.std.450 (signed zeros, undefined inputs, a single-rounding
  `Fma`).
- `test_fixtures.py` checks `tests/semantic/fixtures/disasm/*.s` equals what `regen_fixtures.py` makes from
  `runtime/test/shaders/*.asm` (`python tests/semantic/regen_fixtures.py` rewrites them).
- `test_dpp_eager_subgroup.py` (SHADER-529) checks that no DPP ballot or shuffle is left where SPIRV-Cross
  (MoltenVK) would inline it into a short-circuiting `&&`/`||` or a ternary's arm, for 11 DPP16/DPP8 forms with
  and without a lane off in EXEC, using a model of SPIRV-Cross's single-use forwarding (its own tests cover the
  model). It also runs the same programs in `spirv_wave_interp.py` against a plain-Python DPP reference.

Limits: single lane (or lanes in lockstep, one instruction per lane per round, that exchange values only through
`OpGroupNonUniformShuffle`, `Ballot`, `Any`, `All` and `BitwiseOr` Reduce), control flow followed one branch at a time (the fragment-exit discard test, the compute
dynamic-bounds merge, loops of up to 10000 backward branches and calls of the module's own functions; no switches), no texture access. PushConstant words,
Input variables and a flat physical-address `{address: word}` map can be supplied per run. A shift by at least the bit width is an undefined value: arithmetic on it stays
undefined and any other use (store, branch or select condition, shuffle, output) raises. Values that pass
through a float op lose NaN payloads (Python floats), so expected output words avoid NaN bit patterns; a
signalling NaN read back through `OpBitcast` comes out quieted. `Fma` and `OpFmaKHR` (SPV_KHR_fma, `--fma khr`) are exact: the product and sum are rationals,
rounded once to binary32. `Exp2`/`Log2`/`Sqrt`/`InverseSqrt` are correctly rounded (Vulkan allows a few ULP), and
inputs GLSL.std.450 or SPIR-V leave undefined (`Log2` of x <= 0, `OpFDiv` by zero, `FMax` of NaN...) raise
`UnsupportedConstruct`. `Fract` is
computed exactly and rounded once to binary32. `NMin`/`NMax` give the other operand for a NaN, and they and
`FMin`/`FMax`/`FClamp` order -0 below +0 (the order RDNA2 hardware uses). The `.asm`
sources are written for `llvm-mc -mcpu=gfx1013`; these tests do not assemble them, so assembler syntax is not
checked here.
To add a shader: put the `.asm` in `runtime/test/shaders/`, add it to `SHADERS` in `regen_fixtures.py`, regenerate,
and add its expected words (with derivation) to the matching test file.
