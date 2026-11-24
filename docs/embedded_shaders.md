# Shader containers embedded in the package's DLLs

**Status: measured** on the 1.0.2.1 package. This note covers the shader programs that ship outside
`shaders.rpf`: the DXBC containers inside the package's DLLs, what each one carries, and what the measurement
changes for the ray-tracing and upscaling paths. No container, shader or payload is reproduced here; the tables
hold counts, offsets, chunk names and module identifiers.

## 1. Why they matter

`tools/shader_carve.py` carves `shaders.rpf` only. Two DLLs in the same package carry compute programs of their
own, and `make_game.py`'s `shaders` stage translates both sets after the game's shaders (section 7 for the
builder, [fspx.md](fspx.md) for FidelityFX):

* `XboxBvhBuilder.dll`, Microsoft's XDXR acceleration-structure builder. Its constructor creates compute PSOs
  through device slot 62, and the builder is on the critical path for ray tracing.
* `amd_fidelityfx_upscaler_xbox.dll`, the FidelityFX 4.1.1 upscaler. Its programs are the `ACCC`/`FSPX` format
  covered in [fspx.md](fspx.md); this note only counts them.

`tools/pe_shader_containers.py` scans a binary for DXBC containers, validates the chunk table, reads the `ILDN`
module name and reports what each container carries. `--extract` writes the containers and an `index.json` the
translator reads; extracted shaders are game data and must stay in the user's work folder.

```sh
python -I tools/pe_shader_containers.py <dll> --json <work>/containers.json --extract <work>/embedded/dxbc
```

## 2. Inventory

The 16 DLLs and the package's root files were scanned first (`XboxBvhBuilder.dll` and `amd_fidelityfx_upscaler_xbox.dll`); `game_scarlett_master.exe` (RT-5) adds 10 more console containers:

| image | containers | bytes | min / median / max | chunks |
|---|---:|---:|---:|---|
| `XboxBvhBuilder.dll` | 52 | 674,404 | 4,072 / 7,454 / 74,508 | `SFI0 ISG1 OSG1 PSV0 RTS0 STAT ILDN HASH DXIL` |
| `amd_fidelityfx_upscaler_xbox.dll` | 29 | 289,656 | 1,764 / 5,176 / 42,616 | `SFI0 ISG1 OSG1 PSV0 RTS0 HASH ACCC` |
| other 14 DLLs | 0 | 0 | — | — |
| `game_scarlett_master.exe` | 10 | 9,520 | 924 / 936 / 992 | `SFI0 ISG1 OSG1 PSV0 HASH DXIL PRIV` |

The 29 `ACCC` containers hold `FSPX` payloads ([fspx.md](fspx.md)); the 52 `DXIL` containers are the subject of
this note. The FSR DLL has one rejected `DXBC` hit: the bytes are an immediate in its own magic check, and the
validator rejects it (the scan reports both counts).

### Retail executable (`game_scarlett_master.exe`, RT-5)

Scanning the retail executable (120,181,760 bytes, SHA-256 `e95020b2fa4a721db3df59fdecf75621730ac559d8835252f6c3b658d5a08993`)
with `tools/pe_shader_containers.py` finds **10** valid DXBC containers (9,520 bytes total, file offsets
`0x688cd3c`..`0x688fdf6`, sizes 924–992 bytes) and 1 rejected `DXBC` hit:

* **Stages (`PSV0`):** **1** vertex shader (`0x688cd3c`, 992 bytes) and **9** pixel shaders (924–980 bytes). Unlike the
  DLL containers above, none carries `ILDN`, `STAT`, `RTS0`, or LLVM bitcode (`has_bitcode = false`). All 10 carry
  console-only `XBCH` (`XDAT` tag `0x0d`, 307–368 bytes) and `NOBC` (tag `0x07`, 28 bytes), plus a 12-byte `PRIV` chunk.
* **Translation and GPU check:** `tools/pe_shader_containers.py --extract` previously hardcoded `"stage": "cs"` in
  `index.json`, causing `rdna2spirv.py` to reject all 10 with `CS group size`. With `stage` read from `PSV0`, all 10
  unpack via `tools/fssm_unpack.py` (10/10 `FSSM`, 1,245–4,176 bytes), disassemble cleanly via `tools/shader_disasm.py --backend python`
  (10/10 clean), translate via `tools/rdna2spirv.py` (10/10 translated, all wave64 tier A), and pass `vk_pipeline_check`
  on an NVIDIA GeForce RTX 5060 Laptop GPU (driver `617.14`, Windows 11): **10/10 modules and 10/10 pipelines OK** (1 VS, 9 PS, 0 failed, 0 skipped).

**Every one of the 52 containers carries two representations of its program:**

* an LLVM bitcode blob in the `DXIL` chunk (`BC C0 DE`, 1,820 to 70,276 bytes), and
* an `XDAT` directory with tag `0x0d` whose payload starts with `XBCH`: the console program. The same chunk also
  holds tag `0x07` (a second `DXIL` program header) and tag `0x12` (a 16-byte hash). The `STAT` chunk holds a
  second, different bitcode blob, not the same one.

### `STAT` and the `0x07` program header

Measured over all 52 builder containers in one local 1.0.2.1 package; no shader data is included here. The
bitcode spans below use the `DxilProgramHeader`/`DxilBitcodeHeader` offsets and sizes defined by
[DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler/blob/main/include/dxc/DxilContainer/DxilContainer.h).
All spans fit inside their enclosing parts, and all 104 `STAT` and outer `DXIL` bitcode modules were accepted by
`llvm-dis` 21.1.8. The header's size-in-dwords equals its enclosing part or record length in every case.

* **`STAT`:** 52/52 contain a `DXIL` program header and a bitcode module distinct from the outer `DXIL` bitcode.
  All have `dx.entryPoints` and `dx.resources` metadata, and none defines a function in the `llvm-dis` output;
  the outer modules define functions. DirectXShaderCompiler calls this the shader-statistics part and uses a valid
  `STAT` module for reflection (see its
  [container reflection implementation](https://github.com/microsoft/DirectXShaderCompiler/blob/main/lib/HLSL/DxilContainerReflection.cpp)).
  The metadata-only shape is consistent with that role; the builder's actual consumption of `STAT` is not measured.
* **XDAT tag `0x07`:** 52/52 contain a `DXIL` program header whose declared bitcode is byte-identical to that of
  the outer `DXIL` part. None contains the `NOBC` marker. The whole `0x07` record is not the same as the whole
  outer part, which also contains `XDAT`. This differs from the `NOBC` form documented for the game's shader
  corpus in [fssm_crosscheck.md](fssm_crosscheck.md); it does not change that separate measurement.

The bitcode is what a PC D3D12 compiler consumes; the runtime path is the `XBCH` record. The constructor passes
the **whole container** to the compute-PSO call: at `0x180021330` it loads `DAT_180025700` with size `0x1134`
(4,404 bytes, the size of container 0, `CopyAndCompactBVH`) and calls `FUN_18000c470`, whose vtable call at
`+0x1f0` (slot 62) receives that pointer, its size and two zero words. `FUN_18001d400` creates the
`XDXR GetBVHCompactedSize` PSO (0x14a4 bytes; measured below). `bvh_builder_test` measures **53** PSO
creations during construction over the **52** containers ([d3d12_vulkan.md](d3d12_vulkan.md) A.2), so one
container is created twice. It is **`GetBVHCompactedSize`** (5,284 bytes = 0x14a4): creations **#47 and #53** pass
byte-identical containers (RT-182, `LEONIDA_BVH_RECORD62=1`, which now prints each creation's module from its `ILDN`
part and says when a container repeats). The test now also prints each creation's call site and the name the builder
gives the returned PSO (RT-266, WARP, default constructor flag): the builder names every one of the 53 PSOs with
`SetName`, `XDXR <module>` (for example `XDXR InitPartitionedHeaders`; some modules are shortened, as
`XDXR CopyAndCompact`, or numbered, as `XDXR CalculateMortonCodes 0`/`1`). **#47 and #53 both get `XDXR
GetBVHCompactedSize`**, from the same call site in `FUN_18001d400` (return address `+0x1d503`, called through
`FUN_18000c470` at `+0xc4e4`); `FUN_18001d400` is called twice, once from `FUN_18000d850` (#47) and once from
`FUN_180021330` directly (#53). So the repeat is one PSO created twice under one name, not one container under two
names, and no creation is named `XDXR PostBuildInfoQuery` in this run (default flag, WARP; the flag-true
constructor was not run). The creation order, by module (it is not the table
order of section 3; `CopyAndCompactBVH` is #52, not the first):

| creations | modules, in the order of the `CreateComputePipelineState` calls |
|---|---|
| #1-#13 | `InitPartitionedHeaders`, `PT_WriteOfflinePartition`, `PT_WritePartitions`, `PT_WriteInstances`, `PT_LoadInstances`, `PT_BeginPass`, `ResetCounterValues`, `InitHeaders`, `CopyForBVHUpdate`, `ResetToIndexBuffer`, `CopyCounters`, `CalculateMortonCodes`, `CalculateMortonCodes64` |
| #14-#29 | `BitonicPreSortCS`, `BitonicOuterSortCS`, `BitonicInnerSortCS`, `BitonicPreSort64CS`, `BitonicOuterSort64CS`, `BitonicInnerSort64CS`, `RadixReset`, `RadixCount`, `PrefixSum64`, `PrefixSum128`, `PrefixSum256`, `PrefixSum512`, `PrefixSum1024`, `PrefixSum2048`, `PrefixSum4096`, `RadixAlloc` |
| #30-#46 | `RearrangeTriangles`, `RearrangeBVHs`, `LoadInstances`, `SplitInstances`, `LoadTriangles`, `LoadProceduralGeometry`, `ReloadTriangles`, `ReloadProcedurals`, `PT_ConstructBVH4`, `ConstructBVH4`, `RefitBVH4`, `BuildBVHSplits`, `BuildBVHSplits64`, `FindTreelets`, `ResetKDOP`, `BuildKDOP`, `BuildApexPointMap` |
| #47-#53 | `GetBVHCompactedSize`, `DecompressHeader`, `LeafRehydration`, `Box16Rehydration`, `Box32Rehydration`, `CopyAndCompactBVH`, `GetBVHCompactedSize` (again) |

The test also prints each creation's container with its `ILDN` module name and the first bytes of its digest
(RT-182): creations 1 to 51 are 51 distinct containers, `GetBVHCompactedSize` (5,284 bytes, 0x14a4) among them at 47;
creation 52 is `CopyAndCompactBVH` (4,404 bytes, 0x1134); creation 53 is `GetBVHCompactedSize` again, with the
same digest and size. So the 0x14a4-byte container `FUN_18001d400` passes is the `GetBVHCompactedSize` module, and
the PSO for it is created twice (under one name, `XDXR GetBVHCompactedSize`, per the SetName record above). Measured on 1.0.2.1 with an RTX 5080 (driver
621.01): all 53 creations return `S_OK`.

## 3. The 52 modules

The `ILDN` PDB path gives one module per container, all under
`X:\americas\src\dev\rage\3rdparty\Microsoft\XboxBvhBuilder\Build\Release\Output\XboxBvhBuilder\` (Microsoft's
component integrated into RAGE, not Rockstar shader code). Grouped by phase:

| phase | modules |
|---|---|
| sort (11) | `BitonicPreSortCS`, `BitonicPreSort64CS`, `BitonicInnerSortCS`, `BitonicInnerSort64CS`, `BitonicOuterSortCS`, `BitonicOuterSort64CS`, `RadixCount`, `RadixAlloc`, `RadixReset`, `CalculateMortonCodes`, `CalculateMortonCodes64` |
| prefix (7) | `PrefixSum64`, `PrefixSum128`, `PrefixSum256`, `PrefixSum512`, `PrefixSum1024`, `PrefixSum2048`, `PrefixSum4096` |
| kdop (3) | `BuildKDOP`, `ResetKDOP`, `BuildApexPointMap` |
| build (7) | `ConstructBVH4`, `BuildBVHSplits`, `BuildBVHSplits64`, `SplitInstances`, `FindTreelets`, `RearrangeBVHs`, `RearrangeTriangles` |
| refit (2) | `RefitBVH4`, `CopyForBVHUpdate` |
| load (6) | `LoadTriangles`, `ReloadTriangles`, `LoadInstances`, `LoadProceduralGeometry`, `ReloadProcedurals`, `ResetToIndexBuffer` |
| path tracing (6) | `PT_ConstructBVH4`, `PT_BeginPass`, `PT_LoadInstances`, `PT_WriteInstances`, `PT_WritePartitions`, `PT_WriteOfflinePartition` |
| rehydration (3) | `Box16Rehydration`, `Box32Rehydration`, `LeafRehydration` |
| headers/util (7) | `DecompressHeader`, `InitHeaders`, `InitPartitionedHeaders`, `GetBVHCompactedSize`, `CopyCounters`, `ResetCounterValues`, `CopyAndCompactBVH` |

## 4. They are console FSSM, and the existing translator already takes them

The `XBCH` record has the same shape as the game's own shaders: `XBCH` + stored/unpacked sizes + a zlib stream
that expands to `FSSM` ([fssm_crosscheck.md](fssm_crosscheck.md)). On all 52 containers:

| step | result |
|---|---|
| `tools/fssm_unpack.py` | **52/52** decompress to a valid `FSSM` blob (716 B to 101,320 B, median 2,200 B) |
| `tools/shader_disasm.py --mcpu gfx1013` | **52/52** disassemble; 42 with no invalid word, 10 with invalid words in the code |
| `tools/rdna2spirv.py` | **51** translate and pass `spirv-val` (`main` `49475796`) |

When this page was first written, 42 translated and 10 stopped at four rules. Three of those rules have landed since
(`427c1376`, from #1968), and one program still waits on a fifth rule:

| module(s) | blocker then | now |
|---|---|---|
| `BuildKDOP`, `LoadInstances`, `ReloadProcedurals`, `LoadProceduralGeometry`, `LoadTriangles` | `global_atomic_fmax` | translate |
| `PT_LoadInstances` | `global_atomic_fmin` | translates |
| `Box32Rehydration`, `ReloadTriangles` | `buffer_load_ushort` | translate |
| `LeafRehydration` | `tbuffer_load_format_x GFX10_FORMAT_8_UINT` | translates |
| `Box16Rehydration` | `buffer_load_ushort` | stops at `buffer_store_short` (#1864) |

The disassembly count is unchanged: the same 42 programs have no invalid word, and 10 have invalid words in the
code. All 10 of those translate; `Box16Rehydration` is one of the 42 clean ones.

On current `main`, 51 of the 52 programs translate and pass `spirv-val`. The four earlier blockers were
implemented in #1871 and #1968 (`427c1376`):

* `tbuffer_load_format_x GFX10_FORMAT_8_UINT` (#1871) unblocked `LeafRehydration`.
* `global_atomic_fmax` (#1968) unblocked `BuildKDOP`, `LoadInstances`, `ReloadProcedurals`, `LoadProceduralGeometry`, and `LoadTriangles`.
* `global_atomic_fmin` (#1968) unblocked `PT_LoadInstances`.
* `buffer_load_ushort` (#1968) unblocked `Box32Rehydration`, `ReloadTriangles`, and the load half of `Box16Rehydration`.

The single remaining untranslated program stops at one rule:

| module(s) | blocker |
|---|---|
| `Box16Rehydration` | `buffer_store_short` (#1864) |

Reproduction, after `--extract` above:

```sh
python -I tools/fssm_unpack.py <work>/embedded/dxbc
python -I tools/shader_disasm.py <work>/embedded/dxbc/fssm --out <work>/embedded/dxbc/disasm --mcpu gfx1013
python -I tools/rdna2spirv.py <work>/embedded/dxbc/disasm --out <work>/embedded/spirv
```

`make_game.py` runs the same four tools as a build step (section 7).

## 5. What this changes

* **The RT builder does not need a DXIL path for the console code.** Its programs are RDNA2 `FSSM`, and 51 of
  52 already pass `spirv-val` with the translator the project uses for the game's 26,503 shaders. The bitcode in
  the same container is a second representation (the one the PC-compiler experiment measured); it is not what
  the console runtime executes.
* **The missing rules were small, and only one is left.** `global_atomic_fmax`/`fmin` (6 programs),
  `buffer_load_ushort` (3) and `tbuffer_load_format_x` with format 8 (1) were ordinary translator rules, not a new
  compiler backend, and they are on `main`. The one left is `buffer_store_short` (#1864), for `Box16Rehydration`.
* **The rehydration programs are the ones closest to the shipped data.** `Box16Rehydration`,
  `Box32Rehydration` and `LeafRehydration` are exactly the programs that would turn the dehydrated BVH packs
  into runtime trees ([bvh.md](bvh.md) §3.1). `LeafRehydration` translates since #1871, `Box32Rehydration`
  translates since #1968 (`427c1376`), and only `Box16Rehydration` awaits #1864 (`buffer_store_short`).
* **What is still open.** None of the 52 has been executed on a GPU; translation and `spirv-val` do not prove the
  output. Which of them the game actually creates is a call-site question (RT-5). The module-to-pipeline order
  above comes from `ILDN`, not from a runtime trace. The `STAT` metadata-only shape is consistent with reflection,
  but whether the builder consumes it has not been measured.
* **They run on a GPU (TEST-594).** `tools/bvh_gpu_smoke.py` translates them from the package and runs each through
  `vk_compute_test`. Each program reads its root arguments from a table at `s[0:1]`, the heap base there, which the
  tool seeds with `VKC_HEAP_INIT`: every table word points past the table into zeros. **All 49 that aren't
  rehydration programs pass**: exit 0, no Khronos validation error, and the same heap bytes on a second run (AMD
  Radeon RX 9070, Windows 11, `--max-shared-memory 32768`, package 1.0.2.1). With these inputs only seven change any
  heap word (`InitHeaders`, `InitPartitionedHeaders`, `LoadInstances`, `LoadProceduralGeometry`, `LoadTriangles`,
  `PT_BeginPass`, `PT_WriteInstances`); the others leave early. So this alone shows that the translated programs
  run cleanly, not that their output is right. **Six are checked byte for byte** (`--checked`), the ones whose effect
  reads straight off their code. `RadixReset`, `ResetKDOP`, `ResetToIndexBuffer`, `CopyCounters` and
  `ResetCounterValues` get their root-table fields (an output address, a count of 20, a value or a source), and the
  tool compares all 20 entries they write and the bytes after them, which must stay untouched. `PrefixSum64` scans 64
  words in place into their exclusive prefix sum through five `ds_swizzle_b32` broadcast steps and stores the group's
  total; the input wraps past 2^32. All six match exactly. Negative controls: with the translator made to skip
  `v_add_nc_u32`, `ResetToIndexBuffer` fails (entries hold `base+4i` and zeros); with it skipping `ds_swizzle_b32`,
  `PrefixSum64` fails from element 2 on. So the check sees a wrong translation. Two things found on the way:
  * the wave64 bitonic sorts (`BitonicPreSort64CS`, `BitonicInnerSort64CS`) declare 32,780 bytes of workgroup memory
    with the translator's default cap, 12 over the 32 KiB AMD's Windows driver allows
    (`VUID-RuntimeSpirv-Workgroup-06530`). Their highest LDS offset is 8192, and `--max-shared-memory 32768` shrinks
    the array to what they use. That was before SHADER-579 moved the barrier-round counts into the LDS array: today
    both declare 32,768 bytes with the default cap, and section 7 has what the cap still changes;
  * with every table word 0 instead, the descriptors point at the table itself, and `RadixCount` then writes into the
    table that other waves are reading, so its output differs between runs. That is the test input, not the
    translator; the tool's default avoids it.
* **What is still open.** The other programs' outputs (the sorts, prefix sums, builders, refit and loaders) are not
  checked against a reference yet; the rehydration trio has one in TEST-376 (`tools/bvh_rehydrate.py`); `LeafRehydration` does not finish within 60 s with either table described
  above. Which of them the game actually creates is a call-site question (RT-5). The module-to-pipeline order
  above comes from `ILDN`, not from a runtime trace. The `STAT` bitcode and the `0x07 NOBC` record are not
  explained.

## 6. Tool and tests

`tools/bvh_gpu_smoke.py --game-dir <package> [--max-shared-memory N] [--checked] [--json report.json]` reproduces the
GPU run above: the extraction and translation of §4, then two dispatches per program, and with `--checked` the six
byte-exact cases. Its report holds module names, statuses
and counts only. `tests/test_bvh_gpu_smoke.py` checks the heap image, the word count and the judging of runs with a
fake `vk_compute_test`, and the checked cases' inputs and judging (a fake that does what each case describes passes,
one wrong byte fails), without a GPU or game files.

`tools/pe_shader_containers.py` is standard library only and reads the file, never runs it. It accepts a
container only when the chunk table is sorted, 4-byte aligned, entirely inside the declared total and its last
chunk ends exactly at that total; that is what rejects the FSR DLL's `DXBC` immediate. The synthetic tests in
`tests/test_pe_shader_containers.py` build containers by hand (no game data) and cover the accepted layout, a
bare magic, a truncated chunk, an `ACCC`/`FSPX` container, the `ILDN` name and the extraction index.

## 7. Build step and run-time lookup

**Build (D3D-557).** `make_game.py`'s `shaders` stage, after the game's own shaders and the FidelityFX step, runs
`bvh_shaders()` on the package's `XboxBvhBuilder.dll`. It is the recipe of section 4, with the extraction index
passed along so that each tool reads only what the one before it wrote:

```sh
python -I tools/pe_shader_containers.py <game>/XboxBvhBuilder.dll --extract <work>/shaders/bvh
python -I tools/fssm_unpack.py <work>/shaders/bvh --carve-index <work>/shaders/bvh/index.json
python -I tools/shader_disasm.py <work>/shaders/bvh/fssm --unpack-index <work>/shaders/bvh/fssm_index.json --out <work>/shaders/bvh/disasm --mcpu gfx1013
python -I tools/rdna2spirv.py <work>/shaders/bvh/disasm --out <work>/shaders/spirv/bvh --no-asm --max-shared-memory 32768
```

The last line also gets `--wave64-emulate` and the fallback flags of the build, like the game's shaders.

* **The key is the game shaders' key.** The programs are `FSSM` (section 4), so each index entry gets
  `fssm_sha256`, the SHA-256 of the blob `fssm_unpack.py` inflates from the container's `XBCH` record. That is what
  the runtime computes from the container the DLL passes to slot 62, as it does for a shader of `shaders.rpf`. No
  second key or name table was added. The 52 blobs are 52 different programs, and none of them is in a translated
  1.0.2.1 corpus of `shaders.rpf` (0 of 52 hashes among its 26,501 index entries), so the two indexes do not meet;
  if a program were in both, both entries would be candidates, the game's first.
* **One folder, redone as a whole.** The step is reused only when both indexes, every module the translation index
  names and a matching `bvh_build.json` (written last) are there. It runs again when the DLL, one of the three
  tools before the translator, the translator, `llvm-mc`, the wave64 mode, the flags or the memory cap change, and
  it never makes the game's shaders translate again. Before it runs, it removes the containers, blobs, listings and modules of the run before:
  `pe_shader_containers.py` never overwrites a container, so a second extraction into the same folder would write
  `<module>_NN.dxbc` beside the first and those would be translated as more programs. A package without the DLL
  skips the step.
* **A program without a rule is reported, not skipped.** There is no skip list. After the translation the step
  compares the extraction index with the translation index and logs the counts and the name of every program the
  runtime cannot find (no entry with an `fssm_sha256`, which includes an entry whose metadata failed). It logs them
  on every run, also when the step is reused. Today that is one line: 51 of 52, `Box16Rehydration`
  (`buffer_store_short`, #1864). The other 51 are published. When no program translates the step fails and keeps
  no index.
* **`--max-shared-memory 32768`.** The constructor creates a pipeline for every program and throws at the first
  failure, and the runtime refuses a module over the device's `maxComputeSharedMemorySize` (`E_NOTIMPL`,
  [design.md](design.md), "Workgroup memory limit"). 32,768 is what the AMD Windows and Intel drivers report.
  Measured on `main` `3faac557`, by `lds_bytes` and by comparing the modules byte for byte:

  | translation | default cap (65,536) | `--max-shared-memory 32768` |
  |---|---|---|
  | native | largest 32,768 B (`BitonicPreSort64CS`, `BitonicInnerSort64CS`, `RadixAlloc`), none over | all 51 modules identical |
  | `--wave64-emulate` | `RadixAlloc` 37,084 B, the others at most 32,768 B | `RadixAlloc` 22,236 B, the other 50 modules identical |

  So the cap changes one module, and only in an emulated build; the same 51 programs translate with and without
  it. Without it that build stops on a 32,768-byte device at creation #29: measured on the Radeon below with the
  emulated modules made at the default cap, `RadixAlloc` is refused (`declares 37084 bytes of workgroup memory`,
  `0x80004001`) after 28 creations.

**Run time.** `ShaderLibrary::load(dir)` (`runtime/src/shader_library.h`) reads `<dir>/index.json`, then
`<dir>/fspx/index.json` and `<dir>/bvh/index.json` when they are there. The builder's entries are named
`bvh/<name>`, the path of their module below `<dir>`, and go into the table of the game's shaders, so
`create_compute` finds them through the lookup it already had (`compute_program()` gives an `FSSM` program, looked up
by `(cs, fssm_sha256)`). A missing or unreadable `bvh/index.json` leaves the other indexes loaded, and the main
index is still required. `bvh_size()` counts the entries, which `shader_library_corpus_test` leaves out of the
entries a carved container must reach, like the FSPX ones.

**Measured: `LEONIDA_BVH_DEVICE=vulkan bvh_builder_test`** (package 1.0.2.1, `LEONIDA_BVH_RECORD62=1`,
`LEONIDA_SHADERS` on a folder holding this step's output and an empty main index, since only this step was run).
Before the step, on `main` `3faac557`, creation #1 failed on both GPUs: `0x80004005` with no library, `0x80070057`
with a translated corpus of 26,521 entries, none of them the builder's.

| GPU | translation | result |
|---|---|---|
| RTX 3070, NVIDIA 610.74 (subgroups 32..32, limit 49,152) | `--wave64-emulate` | 49 of 50 creations `S_OK`; #50 `Box16Rehydration` has no module, `E_INVALIDARG`, and the constructor throws `0x80070057` |
| same | native | the same 49 of 50; the runtime warns that this device cannot require 64-wide subgroups |
| Radeon integrated (Ryzen 7 9700X), AMD 32.0.21042.62 (subgroups 32..64, limit 32,768) | `--wave64-emulate` | 49 of 50 `S_OK`, stops at #50 the same way |
| same | native | 49 of 50 `S_OK`, stops at #50 the same way. That is the translator of `main` `561b65a8`; with the one of `3faac557`, 48 creations returned `S_OK` and #49 `LeafRehydration` did not come back from the driver's pipeline compile in 16 minutes (below) |

The rows above were measured before #1864 (`buffer_store_short`) was on `main`. With it (`main` `8575d36e`) the
step translates 52 of 52, and on the Radeon with the emulated modules all 53 creations return `S_OK` and
`bvh_builder_test` passes on the Vulkan device; the RTX 3070 and the native modules were not run again.

Before that, with #1864's translator change applied to a local copy of the tools (not part of this step), 52 of 52 translate
and pass `spirv-val`, and with the emulated modules all 53 creations return `S_OK` on both GPUs: the constructor
and the destructor return and the test passes. So with the emulated modules the one thing between this step and a
constructed builder on the Vulkan device is `buffer_store_short`. The PC-device run of the test (WARP) is
unchanged: 53 of 53.

**`LeafRehydration` on the AMD driver (SHADER-995).** The step was first measured with the translator of `main`
`3faac557`. `vk_pipeline_check` then created the same pipelines without the D3D12 layer: all 51 on the RTX 3070
(native and emulated) and on the Radeon (emulated), and on the Radeon with the native modules 50 of 51, while
`LeafRehydration`, a 1.66 MB wave64 module, was still compiling when the check was stopped after 15 minutes. Its
emulated module was created in 12 s on that driver, and its native module in 11 s on the RTX 3070.

The stop was that driver's compile time for the control-flow form of one module, not the lookup, native wave64 or
the module's size. The translator of `3faac557` emitted a structured tree deeper than 16 in the dispatch-loop form
(SHADER-299: one loop around an `OpSwitch` with a case per basic block), and this program's tree nests 18 deep
(`ReloadTriangles`' 17; found by lowering `--max-structured-depth` until the form changes). Since SHADER-427 the
bound is 64, so with `main` `561b65a8` all 51 native modules are structured (47) or barrier rounds (4), the native
row of the table above is that translation, and `vk_pipeline_check` creates all 51 native pipelines on the Radeon
(18.4 s on one thread, `LoadInstances` 8.6 s the slowest; the 51 emulated ones 26.9 s, `LeafRehydration` 11.6 s the
slowest). No build option is involved, and the step needed no change. @Jadiac5 measured the same translation on an
RX 9070 XT (RDNA4, AMD 32.0.32029.2010): 51 of 51 native and 51 of 51 emulated, the slowest module in 19 to 35 s
([their comment](https://github.com/ItsNotRexed/Leonida/pull/5680#issuecomment-7101091229)). Those are the
structured modules too, so the dispatch-loop module below has only been tried on the gfx1036 and the RTX 3070.

Measured on the Radeon (gfx1036, AMD 32.0.21042.62) with `vk_pipeline_check --jobs 1`, one module in the driver at
a time beside the one long run. That driver keeps compiled shaders on disk and gives back a module it has seen in
0.2 s, so every time is the first compile of a module, or of a copy with another generator word in its header:

| `LeafRehydration` module | size | shape | pipeline creation |
|---|---|---|---|
| dispatch loop, translator of `3faac557` | 1,658,968 B | 1,532 blocks: one loop around an `OpSwitch` with 581 cases | created after 2,039.5 s (34 minutes): 2,014 s of CPU time, one busy thread, the process at 1.2 GB after 5 minutes and 1.7 GB after 21. Slow, not hung |
| the same module, pipeline requiring 32-wide subgroups, or no size | | | not within 150 s either way |
| dispatch loop, current translator (`--dispatch-loop` or `--max-structured-depth 16`, the same bytes) | 1,192,148 B | 2,954 blocks, the same loop and switch | XXnot created within 10 minutes; the run was stopped there |
| structured, the default now | 1,238,360 B | 4,108 blocks, 1,763 selections, no loop, no switch | 2.9 s (3.6 s and 3.2 s for two more copies) |
| `--wave64-emulate`, translator of `3faac557` | 1,703,748 B | emulated wave64 structured | 12.6 s |
| `--wave64-emulate`, current translator | 1,358,436 B | emulated wave64 structured | 12.4 s |

What the time depends on, tried on the current translator's dispatch-loop module with a limit of 45 to 150 s per
compile:

* **Not the subgroup operations or the subgroup size.** All three native forms hold the same 590
  `OpGroupNonUniformAny` and no other subgroup operation. A copy with each of them replaced by an `OpLogicalOr` (a
  compile-time probe, not an equivalent module) did not finish in 120 s, and neither did the first module with
  32-wide or no required subgroup size.
* **Not the size of the switch by itself.** The other programs compile in the dispatch-loop form on that driver,
  2.5 to 5 times slower than structured: `LoadInstances` (628 cases, 5.35 MB) in 40.3 s against 8.6 s,
  `ReloadTriangles` (311) 3.6 s against 0.7 s, `ConstructBVH4` (206) 2.2 s against 0.6 s, `PT_ConstructBVH4` (186)
  1.9 s, `LoadTriangles` (185) 2.1 s, `SplitInstances` (153, 1.53 MB) 4.0 s against 1.2 s, `RefitBVH4` (125) 1.3 s,
  `FindTreelets` (89) 0.9 s, and the wave64 `Box32Rehydration` (47) 0.5 s and `RadixAlloc` (17) 0.4 s.
* **It is how much of this program the loop reaches.** With only the first K of the 581 cases left in the
  `OpSwitch` (the other blocks unreachable; a probe again): K = 64 0.8 s, 128 1.5 s, 256 3.6 s, 272 5.4 s, 280
  23.1 s, 284 24.5 s, 286 43.5 s, 287 42.9 s, and 288, 320, 384 and 512 not within 60 to 90 s. So past case 272 a
  few more blocks multiply the time. That looks like growth with the paths through the program rather than with
  its size, which these numbers suggest and do not show.
* **Nothing done to the module from outside changed that.** `spirv-opt -O` (1,350,016 B) and the local SSA
  rewrite with dead-code elimination (`--ssa-rewrite` after the two local eliminations and before
  `--eliminate-dead-code-aggressive`, 1,120,512 B) validate and did not finish in 120 s;
  `--eliminate-dead-branches --merge-blocks --cfg-cleanup` returns the module unchanged. Marking every load and
  store of the dispatch variables `%pc` and `%done` `Volatile`, which keeps the module equivalent, did not finish in
  150 s either (whether that driver honours the flag was not checked). `-Os`, `--merge-return` and
  `--scalar-replacement` were not run.

So the AMD compiler's time on this module comes from the dispatch-loop form of this one program: of the eleven
tried it is the only one that is both large and without a loop of its own (its structured module has no
`OpLoopMerge`), which is an observation, not an explanation, and which of the driver's passes takes the time was
not looked into. The default translation does not emit that form for any of these 51 programs, so the step keeps
what `wave64_settings` chooses for this driver (native, #857) and makes no exception for the builder. A module can
still come out in the dispatch-loop form elsewhere (a graph `cfg_structure` cannot make reducible, a tree deeper
than the bound, `--dispatch-loop`); [testing.md](testing.md), "Known problems", has the row for that.

Still open: nothing has dispatched one of these pipelines through the D3D12 layer, so this shows that the
constructor's pipelines are created, not that a build produces the right tree (section 5).

`tests/test_bvh_shaders_stage.py` runs the step on a synthetic DLL (hand-built containers around a made-up `FSSM`
blob, no game data): the key the extracted layout gives the translator, the tools and arguments, reuse and each
reason to run again, a second run that leaves no duplicate containers, the report of an untranslated program on
every run and the failure when nothing translates. `shader_library_test` covers the `bvh/` index (found by the
container's program hash, counted apart, a broken or missing index beside the others) and `d3d12_compute_test`
creates a pipeline through slot 62 whose only translation is in `<shaders>/bvh/index.json`.
