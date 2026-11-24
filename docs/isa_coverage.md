# RDNA2 ISA coverage of the translator

`tools/isa_coverage.py` compares the instructions of AMD's machine-readable RDNA2 ISA specification with what
`tools/rdna2spirv.py` has a rule for. It reads no game data and needs no GPU; it is a count over the public ISA and
the translator's source.

## Getting the specification

AMD publishes the XML files on GPUOpen, [AMD machine-readable GPU ISA documentation](https://gpuopen.com/machine-readable-isa).
The zip is not in this repository (the RDNA2 file alone is 11 MB):

```sh
curl -L -o isa.zip https://gpuopen.com/download/machine-readable-isa/latest/
unzip isa.zip amdgpu_isa_rdna2.xml
```

The numbers below come from the package `AMD_GPU_MR_ISA_XML_2026_08_06.zip` (SHA-256
`82404f1126761b7877595b622afa7e1f311f2f41e89a3abe9aaf8ad045c082e2`), whose `amdgpu_isa_rdna2.xml` is "AMD RDNA 2",
schema 1.1.1, release date 2026-02-20, and carries `<License>MIT</License>`. AMD's reader for the same files is the
C++ [isa_spec_manager](https://github.com/GPUOpen-Tools/isa_spec_manager) (MIT); the schema is documented there in
`documentation/spec_documentation.md`. This tool parses the XML itself with `xml.etree`.

## Running it

```sh
python -I tools/isa_coverage.py --isa amdgpu_isa_rdna2.xml              # markdown report
python -I tools/isa_coverage.py --isa amdgpu_isa_rdna2.xml --list       # plus the unknown instructions per encoding
python -I tools/isa_coverage.py --isa amdgpu_isa_rdna2.xml --json
python -I tools/isa_coverage.py --isa amdgpu_isa_rdna2.xml --llvm-mc llvm-mc   # classify non-RDNA2 names with llvm-mc
```

`--isa` defaults to `$LEONIDA_RDNA2_ISA_XML`. With that variable set,
`python -I tests/test_isa_coverage.py` also checks the real translator against the real XML; without it those tests
skip and the rest runs on a small hand-written fixture in the same schema.

## What the classes mean

The translator dispatches on the opcode string with one long if/elif chain (`Gen._insn`), so the tool reads the
source with `ast` instead of running it:

| class | meaning |
|---|---|
| handled-exact | the name is compared with `op` (`==`, `in`) or is a key of a mnemonic table (`NOOPS`, `F2`, ...) |
| handled-family | no exact name, but it starts with a prefix `_insn` or a method it hands `op` to tests with `op.startswith`, or ends with a suffix one of their if/elif chains tests with `op.endswith` (alone, or beside a `<stem> in self.TABLE` guard, which limits the family to that table's stems: `s_*_saveexec_b32/b64` and `EXEC_LOGIC`) |
| refused | the name sits in a test whose whole body is `raise Unsupported(...)` |
| unknown | none of these: the final `raise Unsupported(op)` is what it reaches |

Limits. "Handled" means a rule exists, not that the rule is right; the GPU tests and `tests/semantic` do that.
A broad prefix such as `ds_` marks every `ds_*` instruction handled-family even where a sub-dispatch refuses one
form, so read handled-family as an upper bound; the instructions whose branch contains a `raise Unsupported` are
listed separately ("handled but with operand forms that raise"). Names built by string formatting are invisible.
Only the AMD XML defines the denominator: it omits a few gfx1013 instructions that LLVM accepts (below), and lists
some that a desktop part does not execute.

## Result (translator at `41f44826`, AMD XML of 2026-08-06)

Every new translator rule moves these numbers; rerun the tool for the current ones.

1,153 RDNA2 instructions: 436 handled-exact, 408 handled-family, 2 refused (`s_denorm_mode`, `s_round_mode`), 307
unknown. The unknown ones cluster in groups the corpus does not need or that are not written yet: the f64
transcendentals and `v_div_*`, FLAT and SCRATCH memory, the global atomics, the `image_gather4*` variants outside the
four `_l`/`_lz` forms, packed image formats (`*_pck`, `*_by2`), the 16-bit `v_pk_*` integer forms, `v_dot*`, and the
`*_saveexec` / `s_nand`/`s_nor`/`s_xnor` scalar forms. `python -I tools/isa_coverage.py --list` has the full list per
encoding.

Names the translator matches that are not in the RDNA2 XML (10), each checked with
`llvm-mc -arch=amdgcn -mcpu=gfx1013`:

| name | llvm-mc for gfx1013 | verdict |
|---|---|---|
| `v_add_u16`, `v_sub_u16` | other GPU only | internal: `v_add_nc_u16` / `v_sub_nc_u16` are re-dispatched under these names |
| `s_delay_alu` | other GPU only (RDNA3) | deliberate, in `NOOPS` |
| `v_mac_f32`, `v_mac_legacy_f32`, `v_mad_f32`, `v_mad_legacy_f32`, `v_madak_f32`, `v_madmk_f32`, `s_get_waveid_in_workgroup` | valid | real gfx1013 instructions that AMD's XML omits |

`tools/rdna2_decode.py` decodes raw words without LLVM from the opcode and operand tables in
`tools/rdna2_isa_tables.py`, which are derived from the same `amdgpu_isa_rdna2.xml`. The gfx1013 entries above
that the XML omits (`v_mac_f32`, `v_mad_f32`, `v_madak_f32`, `v_madmk_f32`, `s_get_waveid_in_workgroup`) are in
that table, and `v_mad_legacy_f32` is the name the decoder gives VOP3 opcode 0x140 on a gfx101* target. A name
added there belongs in the table above as well.

None is a typo that breaks translation, so there is nothing to fix in the translator. No name ends in a suffix
`base_op()` strips (`_e32`, `_e64`, `_sdwa`, `_dpp`), and every `startswith` prefix matches at least one RDNA2
instruction.

## Which rules the game reaches that hardware has not checked

Several rows of the translator contract in `docs/design.md` say part of a rule is not measured on hardware, or
not run on the shader corpus. `tools/op_reach.py` counts, over the disassembled corpus, the shaders and
instructions that use each instruction. It merges the `_e32`/`_e64`/`_sdwa`/`_dpp` encodings and counts only the
program up to a listing's last `s_endpgm`. The counts say whether 1.0.2.1 depends on the rule at all. They do
not settle the hardware question.

```sh
python -I tools/op_reach.py <work>/shaders/dxbc/disasm --index <work>/shaders/dxbc/index.json v_cos_f16 v_log_f16 ...
```

Measured on the 26,503 1.0.2.1 listings (`shaders.rpf`, LLVM gfx1013; 12,101 PS, 9,282 MS, 2,689 CS, 2,236 VS,
195 GS). The 2,035 decoded instructions after a listing's last `s_endpgm` are left out: they are data words, and
the only place `s_cmov_b32` appears (4 listings). The embedded programs (the BVH builder's 52 FSSM programs,
[embedded_shaders.md](embedded_shaders.md), and the 29 FidelityFX programs, [fspx.md](fspx.md)) are not counted,
and a later game build may differ.

**No 1.0.2.1 shader uses these instructions**, so on this build the unmeasured parts of their rows have no effect:

| `docs/design.md` row | instructions with 0 shaders |
|---|---|
| Sub-dword MUBUF loads | `buffer_load_ubyte`, `_sbyte`, `_ushort`, `_sshort` |
| 8/16-bit global stores and d16 loads | `global_store_byte`, `_short` and their `_d16_hi` forms; `global_load_{u,s}byte_d16`, `global_load_short_d16` and their `_hi` forms |
| `v_rcp_f64`, `v_rsq_f64`, `v_sqrt_f64` | all three |
| `ds_cmpst_f32`/`_rtn_f32`, `global_atomic_fcmpswap`, `buffer_atomic_fcmpswap` | all four |
| `ds_add_f32`, `ds_add_rtn_f32` | both |
| `image_atomic_{swap,inc,dec,fmin,fmax,fcmpswap}` | all six |
| Scalar select and pack | `s_cmov_b32` (program code), `s_cmovk_i32`, `s_cmov_b64`, `s_pack_ll/lh/hh_b32_b16` |
| Vector integer additions | `v_mul_hi_u32_u24`, `v_mul_hi_i32_i24`, `v_lshrrev_b64`, `v_ashrrev_i64`, `v_qsad_pk_u16_u8`, `v_mqsad_pk_u16_u8` |
| `v_div_fixup_f16`, `v_div_fixup_f64` | both |
| `v_frexp_mant_f16`, `v_frexp_exp_i16_f16`, `v_ldexp_f16` | all three |
| `v_frexp_mant_f64`, `v_frexp_exp_i32_f64` | both |
| `v_cvt_pknorm_i16_f32`, `v_cvt_pknorm_u16_f32` | both |
| `v_cvt_pknorm_i16_f16`, `v_cvt_pknorm_u16_f16` ("not run on the shader corpus") | both |
| `v_cmp_class_f16`/`_f64` and the `v_cmpx` forms ("and the shader corpus") | all four |
| `buffer_atomic_{and,or,xor,swap}_x2` | all four |
| `v_dot2c_i32_i16`, `v_dot8c_i32_i4` | both |

**These the game does reach**, so their unmeasured parts still matter. Shaders, then instructions in brackets:

| `docs/design.md` row | instruction: shaders (instructions), by stage |
|---|---|
| 16-bit interpolation | `v_interp_p1ll_f16` and `v_interp_p2_f16`: 4,776 PS (59,367 each) |
| `v_interp_mov_f32` p10 / p20 | 3,673 PS (6,926) |
| `v_cvt_f16_f32` output modifiers | `v_cvt_f16_f32`: 6,306 (44,496): 4,490 MS, 926 CS, 580 PS, 310 VS; the count is the instruction, not its modifiers |
| Known approximations (`v_fma_mix*` rounding twice; `v_cos/exp/log_f16`) | `v_fma_mix_f32` 9,620 (487,658), `v_fma_mixlo_f16` 6,155 (33,900), `v_fma_mixhi_f16` 842 (2,135); `v_exp_f16` 3 CS (12), `v_cos_f16` 1 CS (2), `v_log_f16` 0 |
| `v_cndmask_b32` source modifiers | `v_cndmask_b32`: 22,553 (1,490,241); the count is the instruction, not its modifiers |
| Image atomics, size and LOD queries | `image_get_resinfo` 407 (1,148: 381 PS, 26 CS), `image_get_lod` 653 PS (718) |
| Ray tracing (gfx1013 not measured) | `image_bvh_intersect_ray` 40 CS (638), `image_bvh64_intersect_ray` 24 CS (78) |
| `s_bfe_u32` / `s_bfe_i32` past bit 31 | `s_bfe_u32` 15,398 (90,156), `s_bfe_i32` 2,847 (6,040); whether an operand passes bit 31 is not counted |

A row's unmeasured part can be narrower than its instruction: a modifier, an operand range, an inactive lane or
a special value. Where the count is the instruction alone, the table says so, and the count is an upper bound for
that part. `tests/test_op_reach.py` checks the tool on hand-written listings.
