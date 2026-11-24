# GPU pipeline corpus check

`runtime/test/vk_pipeline_check.cpp` creates a `VkShaderModule` for every entry in the translator's
`index.json` and, for compute shaders, a real compute pipeline with the runtime's descriptor and push
constant contract. It is a driver-level check: it proves the modules and pipelines are accepted by
the Vulkan implementation, not that the shader results are correct. Value-level checks are in
`docs/shader_validation.md`.

Timing and memory figures in this page were measured with `MESA_SHADER_CACHE_DISABLE=true`. Without it, RADV's
on-disk shader cache (`~/.cache/mesa_shader_cache`, about 1.1 GB after a full corpus run) turns a repeated module
into a cache hit: 0.1 s and ~114 MB instead of a real compilation. That is what a resume run wants, but not what a
measurement should report.

The checker exits nonzero if a module fails to load, a pipeline fails to compile, or the requested final
cache save fails. It still finishes checking the other entries and attempts the final save so the next run
can reuse successful pipelines. Before creating a Vulkan shader module, it checks that the file has a
complete SPIR-V header, the SPIR-V magic word, and a whole number of 32-bit words. A rejected input
module counts as a failure. A rejected partner stub is unavailable, so the affected pipeline is skipped.
It also exits 2, before loading Vulkan, if `index.json` is missing, unreadable, empty or malformed, so a wrong
directory can't pass as an empty corpus. A valid `{}` is an intentionally empty index.

For an index entry with `lds_bytes`, the checker compares the reported workgroup memory with the selected GPU's
`maxComputeSharedMemorySize` for compute or `maxMeshSharedMemorySize` for supported mesh shaders. An entry above its
limit counts as a module failure before Vulkan receives the shader. Older indexes without this field still use the
normal driver check. Mesh pipelines on devices without mesh shader support still use the existing feature skip.

## Options

`vk_pipeline_check <spirv dir> [cache.bin] [options]`

| Option | Meaning |
|---|---|
| `cache.bin` | Optional positional pipeline cache. Loaded first if it exists, saved every 10 minutes and at the end, so a stopped run resumes. |
| `--jobs N` | Compile on N threads (default: half the hardware threads). |
| `--start I`, `--count N` | Check only the sorted index entries `[I, I+N)`. |
| `--stage all\|cs\|graphics` | Which stages to check. `--compute-only` and `--graphics-only` are aliases. |
| `--slow SEC` | Print pipelines slower than SEC (default 30); compiles past 60 s are named while running. |
| `--max-bytes N` | Skip `.spv` files larger than N bytes (0 = no limit); every skip is named. |
| `--large-module-bytes N` | Compile modules larger than N bytes alone (default 4 MiB; 0 disables isolation). |
| `--stats FILE` | Write the driver's per-shader statistics as CSV rows `name,stage,statistic,value`. Needs `VK_KHR_pipeline_executable_properties`; use it without `cache.bin`, because a cached pipeline may come without statistics. |
| `--cache-feedback` | Group successful pipeline creations by `VkPipelineCreationFeedback`: cache hits, no hit reported, and unavailable. |
| `--isolate` | POSIX only: check in child processes, `--isolate-batch N` modules at a time (default 256), so a crashing module is recorded as `MODULE CRASH` and the pass continues. Not compatible with `--stats`. |
| `--inflight-file F` | Keep the modules currently in the driver in F as `name<TAB>phase` lines. It is rewritten by the watchdog and when a large module enters or leaves, and emptied at the end; `make_game.py`'s pipelines stage reads it. |
| `--skip-file F` | Do not compile modules named one per line in F; each is reported as `MODULE CRASH` and counted as a failed pipeline. `make_game.py` writes F after a crash so the next run checks the rest. |

`--isolate-child` is internal to the parent/child protocol and is not meant to be passed directly. A module with a missing partner stub is skipped; without `--isolate`, a driver crash ends the run.

## Check user-defined interfaces between stages

`tools/shader_link_check.py` compares the binary SPIR-V declarations of the shaders you plan to put in the
same pipeline. Use it before driver compilation to locate missing varyings, overlapping components, type
mismatches and per-primitive mesh declarations. It reads local modules and needs only Python.

The declarations come from `tools/spirv_interface.py`, a standard-library reader of one module's interface
(the normalized `Location`/`Component` model, one array level and literal specialization defaults) with its
tests in `tests/test_spirv_interface.py`. It reflects declarations only, not instructions, device features or
resource bindings, so it does not replace `spirv-val`.

```powershell
python -I tools/shader_link_check.py check --vs <vertex.spv> --ps <pixel.spv>
python -I tools/shader_link_check.py check --vs <vertex.spv> --gs <geometry.spv> --ps <pixel.spv>
python -I tools/shader_link_check.py check --ms <mesh.spv> --ps <pixel.spv>
python -I tools/shader_link_check.py inspect <shader.spv>
```

Direct checks use `main` for every stage unless `--entry` sets another shared default.
Use a stage-specific override for modules with different entry names, for example
`check --vs vertex.spv --vs-entry Vertex --ps pixel.spv --ps-entry Fragment`.

To audit a batch, give it the shader names from your local translation directory in a JSON file. A stage can
use a string or a `shader` and `entry` object. Names refer to `<name>.spv` in `--shader-dir`. The manifest is read as
UTF-8 with or without the byte order mark, so a copy saved by Notepad or `Out-File -Encoding utf8` is taken as is.

```json
{
  "version": 1,
  "pipelines": [
    {"name": "two-stage", "vs": "vs_example", "ps": "ps_example"},
    {"name": "with-geometry", "vs": "vs_example", "gs": "gs_example",
     "ps": {"shader": "ps_other", "entry": "main"}}
  ]
}
```

```powershell
python -I tools/shader_link_check.py batch pairs.json --shader-dir <spirv> --quiet --report link-report.json
```

The report records per-pipeline results. It uses the Vulkan Location and Component model, including one
outer array level for geometry inputs, mesh outputs and fragment `PerVertexKHR` inputs. Add `--maintenance4`
when the actual device enables that feature and the pipeline relies on matching a wider output vector to a
narrower input vector. See [Vulkan interface matching][interfaces].

Exit codes are `0` for compatible interfaces, `1` for a mismatch and `2` for a bad input or an interface
the reader cannot interpret. Continue with the driver checker for shader instruction validation, resource
bindings, device limits and pipeline state. Shader pairs must come from the caller. Share aggregate counts
and error classes, keeping the local shader files and reports that identify them on your PC.

[interfaces]: https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-iointerfaces-matching

## Audit wave fidelity for the target GPU

The subgroup audit combines the translator's wave-width and partition-invariance metadata with
the target device's Vulkan stage capabilities. A device can require 64 lanes for a fragment
shader without being able to require 64 for a vertex or geometry shader. Audit the whole graphics
index or use the same manifest as the interface check.

```powershell
python -I tools/wave64_pipeline_audit.py <spirv>/index.json
python -I tools/wave64_pipeline_audit.py <spirv>/index.json --manifest pairs.json --strict --report wave64-summary.json
```

The audit checks the Vulkan device selected on this PC and respects `LEONIDA_VK_DEVICE`. To assess
another device, save `python -I tools/vk_subgroup_info.py --stage-caps` as UTF-8 JSON on that system and pass the file
with `--device-info`. The report counts each shader once and each pipeline by its worst stage. It
contains aggregate counts, capability masks and reason categories. Shader and pipeline names are
excluded from the report.

| Result | Meaning |
| --- | --- |
| `exact` | Fixed or explicitly requestable translated width, or workgroup exchange in compute/mesh |
| `partition-invariant` | A 32-wide graphics subgroup can run a tier A/B wave64 shader. Tier B atomic slot order can differ. |
| `default-width` | The reported device default matches, with no per-stage width guarantee |
| `approximate` | A tier C wave64 VS, PS or GS depends on cross-half behaviour on the 32-wide device |
| `unknown` | The index or device lacks the information for a decision |
| `unsupported` | The translated width, stage support or execution model does not fit the target |

In strict mode, exit 1 means at least one result other than `exact` or `partition-invariant`.
Invalid inputs or an unavailable Vulkan probe exit 2. This checks width fidelity from metadata.
Vulkan pipeline creation and drawing still need their own tests. See [Wave64 graphics in the
translator contract](design.md) for how the tiers are assigned.
## Test Mesa pipeline compilation without a physical GPU

On Linux, Mesa's DRM shims expose GPU profiles to its Vulkan drivers. This lets `vk_pipeline_check`
measure shader compile time and reproduce compiler failures on a machine without that hardware.
Readback and dispatch correctness still require a Vulkan device that executes commands.
See the [Mesa DRM shim guide](https://docs.mesa3d.org/ci/drm-shim.html).

From the Leonida root, build the checker and translate a synthetic fixture. The fixture build needs
`spirv-as`, `spirv-val` and LLVM 21 or later with the AMDGPU target. Set `LLVM_MC` if the assembler
has a versioned name.

```sh
cmake -S runtime -B runtime/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build --target vk_pipeline_check
fixtures=$(mktemp -d)
python3 -I runtime/test/shaders/build.py "$fixtures" cs_buf_fmt8
```

### AMD RADV

Build Mesa with `-Dvulkan-drivers=amd -Dtools=drm-shim`. Set `mesa_build` to that build directory
and `radv_icd` to the absolute path of the matching RADV Vulkan ICD manifest.

```sh
AMDGPU_GPU_ID=gfx1201 \
LD_PRELOAD="$mesa_build/src/amd/drm-shim/libamdgpu_noop_drm_shim.so" \
VK_DRIVER_FILES="$radv_icd" MESA_SHADER_CACHE_DISABLE=true \
  runtime/build/vk_pipeline_check "$fixtures/spirv" --jobs 1 --slow 0
```

The checker should select a device whose name includes `RADV GFX1201` and report one compute pipeline
compiled. For comparisons, use the same Mesa version, GPU profile and translation flags each time.
Run each comparison in a fresh process and leave the pipeline cache filename out.

Mesa 25.0.7 also supports a null device directly through the installed RADV ICD. On Debian 13
this runs the same compile check with the distro driver.

```sh
RADV_FORCE_FAMILY=GFX1201 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
MESA_SHADER_CACHE_DISABLE=true runtime/build/vk_pipeline_check "$fixtures/spirv" --jobs 1 --slow 0
```

### Intel ANV

Build the Intel shim with `-Dtools=drm-shim,intel` and use an ANV ICD from the same Mesa version.
When using a prebuilt driver, the shim can be built with `-Dvulkan-drivers= -Dgallium-drivers=`.
Set `intel_icd` to the absolute path of its ANV manifest and `intel_shim_build` to the shim's
Mesa build directory.

```sh
INTEL_STUB_GPU_PLATFORM=bmg \
LD_PRELOAD="$intel_shim_build/src/intel/tools/libintel_noop_drm_shim.so" \
VK_DRIVER_FILES="$intel_icd" MESA_SHADER_CACHE_DISABLE=true \
  runtime/build/vk_pipeline_check "$fixtures/spirv" --jobs 1 --slow 0
```

The example selects a Battlemage profile. Use `dg2` for an Alchemist profile or `adl` for Alder Lake.
On Mesa 26.1.6 all three compiled the example, including when it was rebuilt with
`LEONIDA_PC_TAIL_UBO=1` to move the push-constant tail to a uniform buffer.

## Corpus

The translated corpus is 26,503 entries: 2,236 vertex, 12,101 pixel, 2,689 compute, 9,282 mesh and
195 geometry shaders. A run was split into 12 contiguous index shards of about 2,209 entries each.
The shards were run with a per-shard pipeline cache; each saves a cache of roughly 190-203 MB.

| Shard | Modules | CS pipelines | Result | Cache | Wall |
|---|---:|---:|---|---:|---:|
| s2 | 2,207 | 208 | pass | 196,869,933 B | 2h21m35s |
| s3 | 2,209 | 200 | pass | 190,022,273 B | 1h50m54s |
| s4 | 2,210 | 213 | pass | 198,941,511 B | 2h01m55s |
| s5 | 2,209 | 248 | pass | 201,997,254 B | 2h01m09s |
| s6 | 2,209 | 213 | pass | 190,939,966 B | 2h00m05s |
| s7 | 2,209 | 229 | pass | 202,013,032 B | 1h59m27s |
| s8 | 2,210 | 227 | pass | 199,121,775 B | 2h00m11s |
| s9 | 2,210 | 240 | pass | 198,407,685 B | 2h00m27s |
| s10 | 2,210 | 218 | pass | 202,890,354 B | 2h01m58s |
| s11 | 2,209 | 221 | pass | 200,378,101 B | 2h01m42s |
| **total (10/12)** | **22,092** | **2,217** | **0 module / 0 pipeline failures** | **1,981,581,884 B** | |

Stage split over the ten shards: 2,217 compute, 163 geometry, 7,767 mesh, 10,088 pixel and 1,857
vertex modules. The ten shards cover **22,092 of 26,503 entries (83.4%)** and **2,217 of 2,689
compute pipelines (82.4%)**.

Two shards did not complete:

- **s0** was stopped after 1 h 57 m at 81% of its bytes, with no read progress for 27 minutes on a
  3.61 MB compute shader.
- **s1** was stopped at the same time at 83% of its bytes on a 2.52 MB pixel shader.

They account for the missing 4,411 modules (472 compute, 32 geometry, 1,515 mesh, 2,013 pixel, 379
vertex) and 472 compute pipelines. No summary was printed for either shard. This is a driver
compile-time limit, not a module or pipeline rejection: no `MODULE FAIL` or `PIPELINE FAIL` line
exists in any completed shard.

A later full-corpus build on different hardware (RTX 4060 Ti, driver 595) completed **26,503 of 26,503**
modules and pipelines with **0 failures** and a 3.81 GB cache; see the "Full-corpus pipeline build" row in
[`docs/design.md`](design.md). The two stalled shards above therefore record a hardware/driver-specific
compile-time stall, not a project-wide limit.

The recurring slow family at every shard tail was a set of 6.7-7.2 MB ray-tracing resolve compute
shaders. One observed single-module stall exceeded 27 minutes, read from the run's progress output.
The tool times each pipeline: `--slow SEC` prints `SLOW <stage> <name> (w<N>, <sec>s)` for a pipeline
that takes longer than SEC, and each stage's summary ends with
`compile time: <total>s total, <mean>s mean, slowest <name> (<sec>s)`. A run with `--slow 0` therefore
reports such a stall with its duration, not as a lower bound.

## Bounded guaranteed-finish sample

A separate bounded run covers eight 30-entry bins plus 12 large compute shaders: **252/252 modules
accepted** and **172/172 compute pipelines accepted**, 0 failures. Stage split: 172 compute, 20 each
geometry/mesh/pixel/vertex. Wave split: 97 wave64 / 75 wave32. The nine processes finished in about
5 minutes; their caches total 42,098,592 bytes. This is a sanity check on the same device, descriptor
layout and subgroup path, not a substitute for the 26,503-entry corpus.

A validation-layer run of the same 252 shaders (temporary instrumented copy, not committed) produced
**zero shader/module/pipeline messages**. The 9 errors and 36 warnings were loader noise from
overlay layers (a missing Epic JSON, three non-conforming layer names, one duplicate OBS hook). The
22,092 full-shard modules were not re-run through the validation layer; their result is driver
acceptance only.

## Limiting concurrent compilation of large modules

A SPIR-V module larger than 4 MiB is "large". [#884](https://github.com/ItsNotRexed/Leonida/issues/884)
showed why that matters: several compiles that each needed gigabytes of host RAM ran at the same time. Large
modules now share a memory budget (`runtime/test/large_module_gate.h`): each one reserves 512 bytes of RAM per
SPIR-V byte, and they enter the driver while their reservations fit three quarters of the memory available when
the run starts, up to `--jobs` of them. One large module always gets in when none is compiling, so a module
bigger than the whole budget still compiles, alone among the large ones. Small modules are never held back. On
Linux the budget is the smaller of `MemAvailable` and the room under the cgroup v2 `memory.max` of the process and
every cgroup above it (a container, or `systemd-run -p MemoryMax=...`); on Windows it is `ullAvailPhys`. Where
neither can be read, every large module compiles alone, as before. After each large module, glibc builds call
`malloc_trim(0)`: the driver's compiler frees its memory into the arena of the thread it ran on, and glibc kept it
there.

Measured on an RX 9070 XT (RADV GFX1201, Mesa 25.3.6, Mesa shader cache off), on 40 consecutive translated
compute modules of which 27 are over 4 MiB:

| Large modules | Threads | Wall time | Peak RAM |
|---|---|---|---|
| one at a time (`--large-parallel 1`, the old default) | 6 | 297 s | 7.1 GB |
| memory budget (default) | 6 | 83 s | 8.3 GB |
| memory budget (default) | 12 | 56 s | 15.1 GB |

One 5.7 MB ray-tracing compute module alone: 12.4 s and 1.96 GB peak (about 330 bytes per SPIR-V byte, hence the
512 estimate); a 0.36 MB one: 0.13 s and 94 MB. The process grew past one module's peak even one at a time because
of the arenas (16 modules, 6 threads: 6.2 GB; with `MALLOC_ARENA_MAX=1`: 2.1 GB).

- `--large-module-bytes N` changes the cutoff in bytes. `0` disables the gate.
- `--large-parallel N` caps how many large modules compile at once: `0` (default) by the budget, `1` alone with
  the other threads waiting (the behaviour before), `N` at most N without a budget.
- `--inflight-file` names every module in flight when a large one enters or leaves the driver, so with several
  large modules in flight it names all of them; the `pipelines` stage of `make_game.py` checks each alone.

## Skipping huge modules

`vk_pipeline_check` now accepts `--max-bytes N`:

```powershell
vk_pipeline_check <spirv dir> [cache.bin] --max-bytes 2000000 --count 100
```

Every `.spv` larger than `N` is reported as `BIG SKIP <name> (<bytes>, limit <N>)`, counted by stage,
and excluded from the module/pipeline totals; the summary prints how many entries were checked. On a
30-entry smoke test with `--max-bytes 20000`, 24 modules were skipped and the remaining 6 pixel
modules passed.

The report above therefore states the limit and the skip count; a later run on an RTX 4060 Ti / 595
completed the full 26,503-entry set without them (`docs/design.md`). `--max-bytes` remains a visibility
tool, not a pass: a skipped module has not been validated.

## Stack reserve (BUILD-1457)

A driver compiles on the thread that creates the shader module or the pipeline, and its compiler can recurse with
the shader's control-flow nesting. On Windows a thread that asks for no stack size gets the reserve written in the
executable's PE header, 1 MB by default. That is the checker's main thread and every `std::thread` worker.

`vk_pipeline_check` is linked with an **8 MB** reserve: `/STACK:8388608` in `runtime/CMakeLists.txt` and on the
checker's `cl` line in `scripts/build_runtime.ps1`. The CTest `vk_pipeline_check_stack` reads `SizeOfStackReserve`
back from the built file (`runtime/test/check_stack_reserve.cmake`; the program is not run), and
`tests/test_build_runtime_fallback.py` keeps the two numbers the same.

Measured on an AMD Radeon RX 9070 XT, AMD 26.11.2 (LLPC, Windows driver 32.0.32029.2010), each module alone with
no cache file (`--jobs 1 --start N --count 1`), on a 1.0.2.1 corpus translated at `aa2988cc` with native wave64.
The reserve was changed with `editbin /STACK` on copies of one binary. Counts only.

| Reserve | The 26 mesh modules that overflow at 1 MB | 28 structured compute modules that overflow at 1 MB |
|---|---|---|
| 1 MB | 26 exit 0xC00000FD (stack overflow) | 28 exit 0xC00000FD |
| 2 MB | 26 create their pipeline | 18 create theirs, 10 exit 0xC00000FD |
| 4 MB | 26 create their pipeline | 28 create their pipeline |
| 8 MB | 26 create their pipeline | 28 create their pipeline |

The 26 are what a corpus translated on `main` hits: 2.42 to 2.79 MB of SPIR-V each, among 846 mesh and vertex
modules of 1 MiB or more that were tried one by one at 1 MB (the other 820 pass). The 28 come from a sample of 60
compute modules, picked evenly by size, of the 781 that take the structured form once the structured depth bound
is 64 instead of 16 (SHADER-427): every sampled module from 4.76 MB up overflows 1 MB. So 2 MB is enough today and
not for the deeper trees. With the linked 8 MB, the same 26 and 28 create their pipelines, and with that binary's
reserve put back to 1 MB all 54 overflow again.

Not measured: other AMD GPU families, other vendors' Windows drivers, and a corpus translated with
`--wave64-emulate`.

**The runtime.** The D3D12 shim creates a pipeline on the thread that asks for it
(`runtime/shims/d3d12_vulkan_device.cpp`), so there the compiler runs on a game thread. The entry thread has 8 MB
from `run_entry` (`runtime/shims/process_shim.h`). A thread the game starts with `dwStackSize` 0, a thread-pool
thread and a `std::thread` of the shims get the reserve in `gta6rt.exe`'s header, because the game's `CreateThread`
is Windows' own. `gta6rt` is therefore linked with the same 8 MB (`GTA6RT_STACK` in `runtime/CMakeLists.txt`, and
the loader's `cl` line in `scripts/build_runtime.ps1`). The CTest `gta6rt_stack` reads the header, and
`run_tiny_game_stack` asks Windows from inside a program the loader runs: the entry thread and a thread created
with `dwStackSize` 0 both report 8 MB. Not covered: a thread the title creates with an explicit smaller size keeps
that size. Which threads the game creates pipelines on is not known without its executable.

## Build-stage batches (BUILD-1030)

`make_game.py` runs the pipelines stage in fresh checker processes over at most **500 index entries** each, on
Windows and POSIX. The ranges partition the same name-sorted index as `--start` / `--count`, including the final
short range. Each finished process saves to the same `pipeline_cache.bin`, and the next loads that checkpoint.
This bounds each invocation to a range that can be split further if concurrent heavy compiles exceed available
memory. A work folder with at most 500 entries keeps the single invocation. An older checker without the range
options keeps the older invocation; rebuild the runtime to get batching.

If a batch crashes, its in-flight modules are checked alone in fresh processes. A module that reproducibly crashes
is recorded as failed, and the rest is checked. If none crashes alone, the entire failed range is retried in
smaller processes, down to one entry; healthy modules are not added to the crash list. An unrecoverable crash,
an interrupted check or a setup error stops the stage. A validation failure in one batch remains a failure even
when subsequent batches succeed; it does not prevent those later batches from being checked.

The native checker returns 1 for initialization failures as well as validation failures. A completed shader pass
writes/clears its in-flight report even when validation fails. Exit 1 without a new report stops the stage instead
of repeating failed setup for the remaining batches; diagnostic probes use the same distinction. Each process
starts with a cleared report path, so an earlier process's report cannot hide an initialization failure.

An abnormal diagnostic-probe exit confirms a module crash only when that probe's fresh in-flight report names
the selected module. A missing, empty or unrelated report makes the result inconclusive and stops the stage
without blaming that module. The native report is periodic for small modules, so even a real fast crash can
leave no evidence; this conservative fallback does not guess its cause or add an unconfirmed crash-list entry.

The batch size limits the index range, **not total memory usage**. Concurrent heavy modules, loading a large
saved cache or compiling one large shader can still exceed a process's available memory; a range that
cannot complete even alone fails with a diagnostic rather than being counted as checked.

`tests/test_make_game_pipeline_batches.py` runs the real build-stage orchestration against scripted child processes:
bounded coverage and cache checkpoints, collective memory-crash recovery, a genuinely crashing module, validation
failures, interruptions, setup failures, inconclusive probes and compatibility with older checkers. It needs no
game files or GPU.

## Crash isolation (`--isolate`)

A driver that dies while compiling one module (for example the RADV `SIGFPE` in
[#1721](https://github.com/ItsNotRexed/Leonida/issues/1721)) ends the whole process, so the rest of the corpus
used to need that module excluded by hand with `--start/--count`. With `--isolate` the tool runs the check in
child processes on native Windows and POSIX:

```sh
vk_pipeline_check <spirv dir> cache.bin --isolate [--isolate-batch 256] [--jobs N ...]
```

- The parent opens no Vulkan device. It runs this executable, one child at a time, over `--isolate-batch`
  modules (default 256) of the same sorted, stage-filtered list, with the other options passed on (`--jobs` is
  the thread count inside each child). `--stats` is refused: children would overwrite one file.
- A child that is killed by a signal, or exits without finishing the pass after it started work, has its output
  discarded and its batch is bisected (re-run in halves) down to the module that kills it. That module is printed as
  `MODULE CRASH <name> (<stage>, driver crashed: SIGFPE)`, counted as a failed pipeline and listed at the end;
  the other modules are checked as usual and counted once. If both halves of a crashed batch run clean (a crash
  that needs several threads at once), the batch is reported as `MODULE CRASH [a, b) not reproduced ...` and
  counts as a failure. A child that fails before it starts work (no device, bad arguments) stops the run
  with exit status 1.
- Windows reports exception exit codes in hexadecimal, for example `driver exited with status 0xc0000005`.
- The pipeline cache: children run one after another and each seeds from and saves to the same `cache.bin`,
  like the chunked runs above; the save is atomic, a crashed child saves nothing, so the file stays valid and
  holds everything the finished children compiled. The parent prints its size at the end.
- Exit status: 0 only if no module failed or crashed and the cache saved; 1 otherwise. The summary is
  aggregated by the parent (modules, pipelines, skips, `--cache-feedback`); the per-stage wave and compile-time lines
  of a normal run are not repeated. Output of a batch appears when its child finishes.
- Each child loads the cache again, so a very large cache makes small batches slow; raise `--isolate-batch`.
  There is no timeout: a hung compile still hangs.

Test hook, not for normal use: `VK_PIPELINE_CHECK_SIMULATE_CRASH=<module name substring>[:SIGNAL]` makes the
process simulate a crash just before creating a matching module's pipeline. The default is `SIGFPE`.
On Windows, `SIGFPE` and `SIGSEGV` use the corresponding exception exit codes.
`tests/test_vk_pipeline_check_isolate.py` uses it with five hand-written compute
shaders to check `--isolate` without a crashing driver; it skips when the tool isn't built or there is no
Vulkan device.

## Application pipeline-cache feedback

With `--cache-feedback`, the checker groups successful pipeline creations by translated shader stage
and reports three counts. A **hit** means the driver set both the valid-feedback bit and the
application-cache-hit bit. **No hit reported** means feedback was valid and the hit bit was clear.
**Unavailable** means the driver returned no valid feedback, or feedback was unsupported.
The three counts sum to the successful pipelines for the stage. Skips and failures are counted separately.

A driver can also use its own caches. A clear hit bit does not establish a fresh compile, and a
reported hit may reuse data populated earlier in the same process. The seed-file byte count alone
cannot identify which pipelines benefited.

To compare runs, start with a new cache path and keep the device, index and flags identical, including
`--stats` if used. For example, from a scratch directory that contains no `sample.cache`, run the same
command twice, replacing `path/to/spirv` with the local translated-shader directory.

```sh
vk_pipeline_check path/to/spirv sample.cache --cache-feedback
vk_pipeline_check path/to/spirv sample.cache --cache-feedback
```

On Windows with an RTX 5090 (driver 621.14), a synthetic three-compute, one-vertex and one-pixel
sample reported zero application-cache hits on its first run and five on the seeded run. On Mesa
llvmpipe the same five pipelines succeeded both times and neither run reported a hit.

## Wave64-emulated corpus (`--wave64-emulate`)

A GPU that only runs 32-wide subgroups (NVIDIA, Intel) needs the corpus translated with
`--wave64-emulate`. Its cross-lane operations go through workgroup memory instead of subgroup operations.
Its modules are larger, and they behave differently under the driver. All runs below are on an
RTX 4060 Ti (NVIDIA 595 driver, 32 GB RAM).

**Translation.** All 26,503 shaders translate and pass `spirv-val` at 43d4703. The run used eight
`--shard` processes and took 80 minutes. The modules total 13.68 GB: median 3.3% larger than the
previous emulated translation (spirv4), 90th percentile 16% larger, with exact-semantics rules added
in between.

**One process per chunk.** A single `vk_pipeline_check --jobs 4` run over the emulated corpus reached
21,000 of 26,503 pipelines with no failures. Then the driver's compiler ran out of memory
(`LLVM ERROR: out of memory`). With 2 jobs, one process later grew past 100 GB of committed memory.
In those earlier runs, compiler allocations were reported to persist between pipelines; the original workaround
used one checker process per 300–500 entries. The build stage now uses at most 500 per process, with smaller
retries when needed (above).
Each process seeds from and saves to the same cache file, so the chunks resume. Since #903 the checker also
compiles modules over 4 MiB alone.

For BUILD-1030, [the reporter's later measurements](https://github.com/ItsNotRexed/Leonida/pull/3623#issuecomment-7085912874)
found no accumulation over a 1,500-entry range and cheap loading of the full cache; the large peaks came from
individual heavy modules. The recovery therefore splits the failed range until fewer such modules compile at
once, without treating the initial cache-growth hypothesis as established.

**The ScreenProbes shaders.** `indirect_diffuse_lighting`'s ScreenProbes_ActivateProbes,
ActivateProbesTarget and ActivateRays compute shaders were 60, 60 and 32 MB of SPIR-V. Their 3,281
typed-buffer loads each decoded every supported format inline. The exact UNORM/SNORM conversion added in
#755 made that about 900 instructions per load, which pushed the two largest past SPIR-V's id bound of
4,194,303 (#881). They were also the modules compiling when the driver ran out of memory. Since
dcd8ecf the run-time format decode is one function per module (`%fmt_decode_fn`):
ActivateProbes went from 6.16 M to 2.20 M instructions (41.7 MB), and all three now validate.

**Results.** With the pre-dcd8ecf translation (spirv4), entries 0–24,200 compiled with 0 module and
0 pipeline failures; two modules over 16 MiB were skipped in the chunked part. The spirv5 cache (all
26,503 entries, nothing skipped by size) is being built.

## Device

The reported runs used an NVIDIA GeForce RTX 5070 Ti (discrete, subgroup size 32 only), driver
621.24 / Windows 32.0.16.2124, Vulkan 1.4.357, loader 1.4.363, with 28 logical CPUs and 34 GB RAM.
The executable used the Vulkan SDK at `C:\VulkanSDK\1.4.363.0`; no source change was needed to build
or run it.

## What this does and does not prove

It proves, for the ten completed shards, that every translated module is accepted and every compute
pipeline is created with the runtime's resource contract. It does not prove:

- that any shader computes the right values (see `docs/shader_validation.md`);
- that the two incomplete shards are valid or invalid;
- that the graphics pipelines (vertex/pixel/geometry/mesh) execute or draw the right pixels: the tool
  creates them with `vkCreateGraphicsPipelines` (dynamic rendering) but never draws with them;
- that the game's descriptors, root signatures and push constants match the contract at run time
  (the missing executable still owns that).

No shader data, names or caches are committed. The caches live in the work directory and are inputs
to a later run, not part of the repository.
