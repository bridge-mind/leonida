# Running the GPU tests

The GPU tests are small hand-written RDNA2 programs in `runtime/test/shaders`. They are translated to SPIR-V and run
on your GPU, and the results are compared with what the RDNA2 hardware gives. They need no game files, and every
result helps: post it in [#166](https://github.com/ItsNotRexed/Leonida/issues/166), one comment per GPU and driver.
If your GPU is already covered there, testing an open pull request from the review queue in the [coordination
issue][claims] helps more.

This page collects the setup steps and the problems testers have run into so far.

[claims]: https://github.com/ItsNotRexed/Leonida/issues?q=is%3Aissue%20is%3Aopen%20in%3Atitle%20%22Coordinating%20who%20works%20on%20what%22

## What you need

- **Python 3.11+.** `scripts/run_gpu_tests.py` needs at least 3.10.
- **`llvm-mc` with the AMDGPU target** (LLVM 21+). The official LLVM Windows release has no AMDGPU target. On
  Windows, use one of:
  - `scripts\build_llvm_mc.ps1`, which builds a native `llvm-mc.exe` (needs git and the Visual Studio C++ tools);
  - conda-forge's `llvm-tools` package, whose Windows build includes the AMDGPU target;
  - `llvm-21` in WSL, with `LLVM_MC` set to the whole command:
    `LLVM_MC="wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"`.

  On Linux, the distribution's `llvm-mc` on `PATH` is enough, if it is 21 or newer. On macOS, Homebrew's `llvm` is
  keg-only, so `brew install llvm` doesn't put `llvm-mc` on `PATH`: the pytest suites find it in
  `/opt/homebrew/opt/llvm*/bin` (or `/usr/local/opt/llvm*/bin` on Intel), and the other tools need
  `LLVM_MC="$(brew --prefix llvm)/bin/llvm-mc"`.

  An older `llvm-mc` is not enough for everything. Ubuntu 24.04's default `llvm-18` (18.1.3) works for
  `valu_fuzz.py --family mem` and `lds_fuzz.py`, but rejects `ps_sdwa_f16_const`. LLVM 19+ assembles the stock CI
  test fixtures. However, full f16 fuzzing with `valu_fuzz.py --family f16` requires LLVM 21+, as LLVM 19 still
  rejects `op_sel` on the 16-bit integer VOP3 forms that the fuzzer generates
  (`v_ashrrev_i16_e64 ... op_sel:[0,0,0]`: "not a valid operand").

  On Windows, ROCm 7.2's bundled `llvm-mc.exe` reports LLVM 21.0.0git. It assembled the TEST-71 memory and
  LDS programs but rejected the 16-bit shift used by the f16 generator. To check an assembler directly in
  PowerShell, set `$env:LLVM_MC` to the path of its executable and run:

  ```powershell
  'v_lshlrev_b16_e64 v34, v27, v27 op_sel:[0,1,1]' | & $env:LLVM_MC -triple=amdgcn--amdpal -mcpu=gfx1013 -show-encoding
  ```

  If it reports `not a valid operand`, point `LLVM_MC` at a build that accepts this instruction. The Windows
  conda-forge `llvm-tools` 23.1.6 package assembled the same f16 programs.
- **SPIRV-Tools** (`spirv-as`, `spirv-val`): part of the Vulkan SDK, or a distribution package.
- **A C++20 compiler, CMake and the Vulkan headers** to build the test programs:
  - Windows: the full runtime needs Visual Studio 2022 with "Desktop development with C++" and Windows SDK
    10.0.26100 or newer. Install the Vulkan SDK for the backend headers. The standalone tools also compile with llvm-mingw and the Vulkan headers alone.
    Running the GPU tests needs a Vulkan loader and a compatible driver.
  - Linux: CMake 3.24+, GCC or Clang, and the Vulkan headers and loader. The headers should be recent enough
    for `VK_KHR_maintenance8`; older ones still build, but skip what needs it.
  - macOS: `brew install llvm molten-vk vulkan-headers vulkan-loader spirv-tools vulkan-tools`. The test programs
    load MoltenVK directly; set `LEONIDA_VULKAN_LOADER` to use the Vulkan loader instead.

## Build the test programs

Windows, from a "Developer PowerShell for VS 2022" or a plain shell:

```powershell
scripts\build_runtime.ps1            # CMake + Ninja into runtime\build
```

For the standalone Vulkan tools, put llvm-mingw on `PATH` and point CMake at the directory containing `vulkan/vulkan.h`:

```powershell
cmake -S runtime -B runtime/build/vulkan -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DLEONIDA_BUILD_WINDOWS_RUNTIME=OFF -DVulkan_INCLUDE_DIR=C:/Vulkan-Headers/include
cmake --build runtime/build/vulkan --parallel --target vk_harness vk_texture_test vk_sampler_test vk_compute_test vk_store_test vk_bindless_test
```

Pass `--build runtime/build/vulkan` to `scripts/run_gpu_tests.py`.

If you configure with CMake's Visual Studio generator instead, the programs end up in `runtime\build\Release` (or
`RelWithDebInfo`); pass that folder to `--build` below.

Linux and macOS:

```sh
cmake -S runtime -B runtime/build/linux
cmake --build runtime/build/linux --parallel
```

On macOS, add `-DVulkan_INCLUDE_DIR=/opt/homebrew/include` to the first command.

The test programs and `scripts/run_gpu_tests.py` should come from the same commit. Some checks tolerate an older
binary (`unpinned_subgroup_skip`), but mixing them can report false failures in the compute tests: 28 `cs_*`
failures on both an RTX 5070 Ti Laptop and a Radeon 610M with the `426023a` script and `48ad23f` binaries, where
either matching pair is clean. If you cannot build locally, the `runtime-windows` artifact of the Runtime workflow
belongs to the commit that built it; check that commit out as well.

### Without a compiler: a CI-built runtime

If you cannot build the test programs locally, the Runtime workflow publishes them as the `runtime-windows`
artifact of its own commit. It holds Windows executables, so use them on Windows or under Wine as in the
section below; GitHub expires the artifact after 7 days. Not every commit on `main` has one: the workflow runs
only when a commit touches what it builds or reads (the `paths` list in `.github/workflows/runtime.yml`), so a
commit that changes only docs, tests or other tools has no Runtime run. Take the most recent run under
Actions -> Runtime, and check out the commit it built, or start a run of the commit you want on your own fork with
"Run workflow" (`workflow_dispatch`). Download and unzip `runtime-windows`, and pass the unzipped folder to `--build`:

```sh
python -I scripts/run_gpu_tests.py --build <unzipped runtime-windows>
```

If you also have neither `llvm-mc` nor SPIRV-Tools (`spirv-as` / `spirv-val`) installed, `--disasm-dir` runs the
pre-disassembled subset in `tests/semantic/fixtures/disasm` and `--assembler python` (the default under `auto` when
`spirv-as` is absent) assembles SPIR-V with `tools/spirv_asm.py`:

```sh
python -I scripts/run_gpu_tests.py --build <unzipped runtime-windows> --disasm-dir tests/semantic/fixtures/disasm --assembler python
```

`build.py` and `check.py` take the same folder as their second argument:

```sh
python -I runtime/test/shaders/build.py <out dir>
python -I runtime/test/shaders/check.py <out dir> <unzipped runtime-windows>
```

The artifact belongs to the commit that built it, so run the scripts from that commit as well (see the note above).

#### Run one test from the Windows artifact

For a focused GPU check without building locally, extract the artifact and open
PowerShell in the directory containing the test executable. Keep the extracted
files together so the test can find any fixtures relative to that directory.
For example, to run the synthetic depth/stencil resolve test without game files:

```powershell
.\d3d12_depth_resolve_test.exe
Write-Output "Test exit code: $LASTEXITCODE"
```

Report the executable name, the workflow run and its commit, your GPU and driver,
the exit code, and the test summary in [#166](https://github.com/ItsNotRexed/Leonida/issues/166).
Label it as a single-test run so the result is easy to distinguish from a full suite run.

## Run them

The whole suite, assembled, translated and run in one go:

```sh
python -I scripts/run_gpu_tests.py --build runtime/build/linux     # Windows: the default runtime\build
```

`--filter <text>` runs only matching tests, and `--work <dir>` keeps the intermediate files.
Add `--require-pass` when verifying a specific path: an all-skipped run exits 1 instead of 0.
Skips still appear as `SKIP`, and individual skips are allowed if at least one test passes and none fail.

A GPU test program that runs longer than 120 seconds is killed and counted as a failure, so a hung driver does
not stall the suite. `LEONIDA_GPU_TEST_TIMEOUT` sets the limit in seconds for `check.py`, `run_gpu_tests.py` and
the `valu_fuzz.py` translator oracle; `0` turns the limit off.
**Rebuild the test programs after pulling.** The programs carry the V# and push-constant layout the translator expects. An executable from an older commit still loads and runs every shader, so a stale `vk_compute_test` does not fail loudly: when a newer translator starts enforcing a descriptor rule the old program does not set up (the raw V# in `pc[2..3]`, for one), every compute test reports the untouched `0xEEEEEEEE` fill pattern and the run looks like a translator regression. If all the compute tests in a run return that pattern, rebuild first (`cmake --build runtime/build/linux`, or `cmake --build runtime\build` on Windows) and run again before reading anything into the words.

The two-step form, which `CONTRIBUTING.md` describes:

```sh
python -I runtime/test/shaders/build.py <out dir>
python -I runtime/test/shaders/check.py <out dir> <folder with the test programs>
```

For the standalone `v_mullit_f32` readback, build the full-screen triangle and the
hand-written semantic fixture, then run 120 draws with runtime SGPR inputs.
The driver checks every pixel against an integer product rounded to 24 bits.
It reports the input words, pixel and channel on the first mismatch.

```sh
python -I runtime/test/shaders/build.py <out-dir> vs_ids ps_mullit
python -I runtime/test/shaders/mullit_readback.py <out-dir> <harness-directory>
```

For D3D12/runtime tests, `LEONIDA_VK_VALIDATION=1` enables `VK_LAYER_KHRONOS_validation`. When running with validation, set `DISABLE_RTSS_LAYER=1` and `DISABLE_VULKAN_OBS_CAPTURE=1` if MSI Afterburner (RivaTuner) or OBS is installed: their overlay layers can add unsupported swapchain usage, and in an elevated (administrator) terminal the loader ignores `VK_LOADER_LAYERS_DISABLE=~implicit~` (see "Known problems").

`vk_compute_test`, `vk_bindless_test` and `vk_image_fuzz` read `LEONIDA_VK_VALIDATION` too (`vk_compute_test`
through `runtime/test/vk_validation.h`): each validation message goes to stderr and a validation error makes the
program exit 1, so `LEONIDA_VK_VALIDATION=1 VK_KHRONOS_VALIDATION_GPUAV_ENABLE=1 python scripts/run_gpu_tests.py` fails
every compute test the layer objects to; the program's output ends with "N validation errors". The other shader test
programs (`vk_harness`, `vk_store_test`, `vk_texture_test`, `vk_sampler_test`) don't read it.
To check one, load the layer through the loader and run it by hand, for example
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_KHRONOS_VALIDATION_GPUAV_ENABLE=1 vk_harness vs.spv ps.spv out.raw`.
The layer prints its messages to stdout and the program's exit status doesn't change, and `run_gpu_tests.py` shows
only each test's last line, so read the program's own output. GPU-assisted validation (`..._GPUAV_ENABLE=1`)
instruments the translated shaders and reports what they actually do at run time: out-of-bounds buffer accesses, and
an image read through a view whose numeric type doesn't match the shader's (a T# FORMAT that disagrees with the
bound view). It runs on llvmpipe. `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=1` adds synchronization validation.

## The Windows runtime on Linux, under Wine or Proton

The Linux build above makes only the Vulkan programs. The loader (`gta6rt`), its shims and their tests are Windows
programs; `runtime/cmake/clang-cl-xwin.cmake` cross-compiles them on Linux with clang-cl and lld-link (LLVM 19+, but not 22.1.7 or 22.1.8 (see Known problems): the
MSVC STL that xwin lays out rejects older Clang; versioned names go in with `-DCMAKE_CXX_COMPILER=clang-cl-19` etc.)
against an MSVC CRT and Windows SDK that [xwin](https://github.com/Jake-Shadle/xwin) downloads from Microsoft
(running it means accepting Microsoft's licence), and `ctest` runs them through Wine:

```sh
xwin --accept-license --sdk-version 10.0.26100 splat --use-winsysroot-style --preserve-ms-arch-notation \
     --output ~/.local/share/xwin-sdk
cmake -S runtime -B runtime/build/windows --toolchain "$PWD/runtime/cmake/clang-cl-xwin.cmake" \
      -DXWIN_DIR="$HOME/.local/share/xwin-sdk"
cmake --build runtime/build/windows --parallel
WINEPREFIX="$HOME/.local/share/leonida-wine" ctest --test-dir runtime/build/windows
```

Use a recent xwin: 0.10.0 lays out SDK 10.0.26100 with the command above, while 0.6.5 stops with "unable to find
x86_64 headers for Win11SDK_10.0.26100".

On a machine without a display (a server, a container, WSL without WSLg), Wine has no graphics driver to create a
Vulkan instance with, so every Vulkan test fails with `vkCreateInstance failed (VkResult -3)` and, with
`WINEDEBUG=+vulkan`, "Failed to load Wine graphics driver supporting Vulkan". Run the tests under a virtual X server,
as CI does: `WINEPREFIX="$HOME/.local/share/leonida-wine" xvfb-run -a ctest --test-dir runtime/build/windows`
(packages `xvfb` and `xauth`, which `xvfb-run` needs). With Mesa installed, the tests then run on llvmpipe.

A virtual X server has no presentable surface, so presentation cannot be judged under it. With a real display (here
Wine 11.22 on a Wayland session, through XWayland) the same 123-test list gives 92 passed, 6 failed and 25 skipped,
and `vk_presenter_sync`, `xbox_present` and `dxgi_swapchain` pass, while they fail under `xvfb-run`. Use a real
compositor for those three. Six still failed with a real display: `device_proxy`, `d3d12_xs_root_signature`,
`xdxr_build`, `d3d12_xdxr_rt`, `d3d12_present` and `d3d12_compute`.

The build type defaults to `RelWithDebInfo`: xwin leaves out the debug CRT, so `Debug` needs `--include-debug-libs`.
`make_game.py`'s `runtime` stage makes this same build on Linux when the tools and the SDK are there (`--runtime-build
runtime/build/windows` takes a folder made by hand like a Windows build). CMake packages are
looked up only inside the SDK folder, so a Windows SDL3 for the GameInput shim is passed as `-DSDL3_DIR=<its
cmake folder>` (`CMAKE_PREFIX_PATH` is not searched here).

### Proton's wine

Proton's wine runs the suite against the D3D12 a Proton game gets: the prefix's vkd3d-proton and DXVK. Start any
program once with `proton run` so that it lays down the prefix libraries; the `STEAM_COMPAT_DATA_PATH` folder has to
exist first, because Proton opens `<data>/pfx.lock` in it:

```sh
mkdir -p ~/.local/share/leonida-proton
STEAM_COMPAT_DATA_PATH="$HOME/.local/share/leonida-proton" \
STEAM_COMPAT_CLIENT_INSTALL_PATH="$HOME/.local/share/Steam" \
  "$HOME/.local/share/Steam/steamapps/common/Proton 11.0/proton" run cmd /c exit 0
cmake -S runtime -B runtime/build/proton --toolchain "$PWD/runtime/cmake/clang-cl-xwin.cmake" \
      -DXWIN_DIR="$HOME/.local/share/xwin-sdk" \
      -DCMAKE_CROSSCOMPILING_EMULATOR="$HOME/.local/share/Steam/steamapps/common/Proton 11.0/files/bin/wine"
cmake --build runtime/build/proton --parallel 2
WINEPREFIX="$HOME/.local/share/leonida-proton/pfx" WINEDLLOVERRIDES="d3d12=n;d3d12core=n;dxgi=n" \
      WINEDEBUG=-all ctest --test-dir runtime/build/proton
```

`proton run` itself doesn't pass the program's output on, so the tests can't run through it. `WINEPREFIX` must be
the prefix the emulator made: a prefix from another wine version fails every test with "wine client error: version
mismatch". The tests need a real display here as well.

Measured on 2026-11-23 with Proton 11.0 (wine-11.0, vkd3d-proton 3.0) on an RX 7600 XT (radv): 155 tests
registered, 154 run, 127 passed, 20 skipped, one disabled and seven failed. The skips are the usual ones (game data,
`LEONIDA_FIRST_LIGHT_DIR`, no independent Vulkan queue for `d3d12_queue_wait`, and so on). All seven failures are in
the comparisons with the host D3D12, which is vkd3d-proton here:

- `d3d12_fence_timing`, `d3d12_private_data`, `d3d12_sampler2` and `d3d12_subresource_copy` disagree with
  vkd3d-proton on a return value or a field it writes (`GetCustomHeapProperties` of a custom heap type,
  `SetName(nullptr)`, unknown sampler flags, CPU-copy boxes).
- `d3d12_r9g9b9e5_uav` compares a clear with WARP, and Proton's dxgi answers `EnumWarpAdapter` with the first
  hardware adapter ("WARP not supported"), so the compared value is the hardware's.
- `d3d12_compute` and `d3d12_method_variants` crash with an access violation inside `d3d12core.dll` on their
  null-argument checks, not on device creation: `GetCachedBlob(nullptr)` and `CreateCommandQueue1(nullptr, ...)`
  are not validated by vkd3d-proton (filed upstream as vkd3d-proton issues 3512 and 3513; Wine's builtin d3d12
  crashes on the queue one too). `d3d12_compute` passes with only `WINEDLLOVERRIDES="dxgi=n"`, where the host is
  Wine's own d3d12 and the comparison skips.

The host's D3D12 here is Wine's vkd3d or vkd3d-proton, not Microsoft's runtime, and Proton's `EnumWarpAdapter`
returns the hardware adapter, so "WARP" is vkd3d-proton on the GPU. Most tests that compare with the host or WARP
skip that comparison under Wine and Proton (`runtime/test/host_wine.h`), and most of them still check the layer
against Microsoft's answers recorded in the test (`d3d12_subresource_copy` and `d3d12_private_data` keep theirs in
`d3d12_subresource_copy_answers.h` and `d3d12_private_data_answers.h`).
A few still compare with the host (`root_sig`, `device_proxy`, and some of `d3d12_vulkan_device`'s cases), so some
of those fail for that reason.

## GPUs with 32-wide subgroups (NVIDIA, Intel, Apple)

The game's shaders are mostly wave64. A GPU that can't run 64-wide compute subgroups skips the wave64 tests
(`cs_wavemask`, `cs_wave64_ballot`, `cs_wm_*`, `cs_w64x_*`, `ms_w64x_ballot`); that's expected, not a failure.

To test the wave64 emulation that makes them exact on these GPUs, set `LEONIDA_W64_EMULATE=1`, **for both
`build.py` and `check.py`**: `build.py` translates with emulation only when it's set. If you set it only for
`check.py`, the shaders are checked against the emulated expectations without being emulated, and every wave64 test
fails.

`ps_bitmask_perm` is a wave32 pixel shader, and a pixel shader's width can't be requested on every device. Intel
drivers report subgroups of 8..32 and may run it 8 or 16 wide; its second pattern reads lanes 16-31, which then
don't run. The test skips when the frame shows that (#1318), and the translation gives 0 for a source lane that
isn't running instead of an undefined value (#618). At 16 wide the first pattern stays inside the running lanes, so
it is still checked before the skip.
The wave32 tests whose words depend on 32 lanes (`cs_dpp*`, `cs_bpermute32`, `cs_exec_const`, `cs_wqm_w32`,
`cs_w32_*`, and in `check.py` every compute test the index records at 32) ask `vk_compute_test` for 32 lanes
(`VKC_SUBGROUP_SIZE=32`). MoltenVK can't require a subgroup size (`requiredSubgroupSizeStages` is 0, sizes 4..32),
but its default is 32 and the translator's SPIR-V 1.5 doesn't allow varying sizes, so Vulkan fixes the size at 32.
`vk_compute_test` checks that with a probe shader before the test (where the device has compute ballots): same
local size and flags as the test's `main`, every subgroup 32 lanes wide by its own lane numbers and ballot, and the
lanes in invocation order, which the translator assumes. It prints `compute subgroup size: 32 (observed: ...)` and
the test is judged. A wrong width fails the test with `observed wrong` and the first wrong invocation; 32-lane
subgroups in another order print `not available`, a skip. Measured
on an Apple M3 Max (MoltenVK 1.4.2 and 1.4.3): 17 workgroup shapes from 1 to 1024 invocations, partial last
subgroups, with and without full subgroups and register pressure, all 32 lanes in order. A device whose default
isn't the requested size still prints `not available`, and those tests skip.

## Wave64 pixel, vertex and geometry shaders (`w64_graphics.py`)

Graphics stages have no emulation: on a 32-wide subgroup a wave64 VS, PS or GS runs as two 32-lane halves, and the
translator marks each one tier A, B or C by whether that can change its results (`docs/design.md`, "Wave64 graphics:
partition invariance"). `runtime/test/shaders/w64_graphics.py` checks those tiers on a GPU by running each `ps_w64_*`,
`vs_w64_*` and `gs_w64_*` fixture at wave64 and at wave32: tier A and B must give the same words both ways, tier C
must differ. CI has no GPU, and the CPU wave interpreter (`tests/semantic/test_wave64_halves.py`) runs vertex shaders
only, so this comparison needs real hardware.

Vulkan can require a subgroup size for pixel shaders (`VKH_SUBGROUP_SIZE`) where the driver lists the fragment stage in
`requiredSubgroupSizeStages`, but not for vertex or geometry shaders, so those two use a RADV switch instead: on GFX10
and newer, RADV runs them as wave64 by default and as wave32 with `RADV_PERFTEST=gewave32`. The runner sets it itself
and checks the result with a probe fixture per stage (`vs_w64_c_lane_id`, `gs_w64_c_lane_id`): the largest lane id
must reach 32 at wave64 and stay below 32 at wave32. On other drivers (AMD Windows, NVIDIA, Intel) the probe sees no
change, and only the default-size checks run; tier C is then a `SKIP`, not a failure. Not every driver lists the
fragment stage either: on an RX 6800 (RDNA2) with AMD's Windows driver 26.11.2 the list is compute only, no size can
be requested for pixel shaders (they run at the 64-wide default), and the tier C pixel fixtures are a `SKIP` too
(2026-11-22, [d3d12_vulkan.md](d3d12_vulkan.md), "Subgroup size").

Build and run it on Linux with RADV, pinned to one GPU (`LEONIDA_VK_DEVICE`, see "Choosing the GPU"):

```sh
cmake -S runtime -B runtime/build/linux -G Ninja && cmake --build runtime/build/linux --target vk_harness
python3 -I runtime/test/shaders/build.py /tmp/w64g
LEONIDA_VK_DEVICE=9070 python3 -I runtime/test/shaders/w64_graphics.py /tmp/w64g runtime/build/linux
LEONIDA_VK_DEVICE=9070 python3 -I runtime/test/shaders/w64_graphics.py /tmp/w64g runtime/build/linux gs_w64_a_ring
```

Each line prints `ok`, a `SKIP` with its reason, or what differed. To add a fixture:

1. Write `runtime/test/shaders/{ps,vs,gs}_w64_<tier>_<name>.asm` with `; mattr: +wavefrontsize64,-wavefrontsize32`
   and a header that says what every result word is. Vertex and geometry fixtures store four words per vertex or
   per triangle to the heap record at `0x1000 + 16 * i`, and every word of a tier A or B fixture is a function of
   `i` alone, never of the lane (copy the store from `vs_w64_a_skip` or `gs_w64_a_ring`).
2. Add it to `FIXTURES` in `w64_graphics.py` with its tier and a function that gives the expected words for record
   `i` (`None` for a word that may depend on the wave size).
3. Add it to `EXPECT` in `tests/test_wave64_halves_fixtures.py` with the tier, reasons and idioms the translator
   should give it, and run `python -m pytest tests/test_wave64_halves_fixtures.py`.
4. Run `w64_graphics.py` on a RADV GPU and say which GPU and Mesa version in the pull request.
## GPUs with 128 bytes of push constants (AMD RDNA2 iGPUs on Windows, some Intel Arc drivers)

The translated shaders take 144 bytes of push constants: s0..s31, then the heap base and the code base. A device whose
`maxPushConstantsSize` is 128, Vulkan's minimum, can't take that, so the texture tests and every test drawn through
`vk_harness` skip there (`unsupported: need 144 push constant bytes, device has 128`); `vulkaninfo` shows the limit.
AMD's older Windows drivers report 128 on RDNA2 iGPUs (Radeon 680M on 22.40.84.06, 610M on 24.10.16.14, the Ryzen 5
7600's iGPU on 24.10.24); 26.11.2 reports 256 on a Radeon 610M, where these tests run without the variable below.
RADV reports more on the same chips.

To run them anyway, set `LEONIDA_PC_TAIL_UBO=1` **for both `build.py` and `check.py`** (or for
`run_gpu_tests.py`). The shaders are then translated with `--pc-tail-ubo`: s0..s31 stay in 128 push-constant bytes,
and the heap base and code base move to a uniform block at set 1 (`docs/design.md`, "Push-constant tail as a uniform
block"), which `vk_compute_test`, `vk_store_test`, `vk_harness`, `vk_texture_test`, `vk_sampler_test` and
`vk_bindless_test` bind (`runtime/test/vk_pc_tail.h`). If you set it
only for `check.py`, the shaders read the tail from push constants the programs no longer declare. `--mesh-compute`
(`LEONIDA_MESH_COMPUTE=1`) isn't combined with it yet: its vertex shader still reads the output address from push
constants, so those tests skip.

## GPUs with 128 bytes of push constants (AMD RDNA2 iGPUs on Windows, some Intel Arc drivers)

The translated shaders take 144 bytes of push constants: s0..s31, then the heap base and the code base. A device whose
`maxPushConstantsSize` is 128, Vulkan's minimum, can't take that, so the texture tests and every test drawn through
`vk_harness` skip there (`unsupported: need 144 push constant bytes, device has 128`); `vulkaninfo` shows the limit.
AMD's older Windows drivers report 128 on RDNA2 iGPUs (Radeon 680M on 22.40.84.06, 610M on 24.10.16.14, the Ryzen 5
7600's iGPU on 24.10.24); 26.11.2 reports 256 on a Radeon 610M, where these tests run without the variable below.
RADV reports more on the same chips.

To run them anyway, set `LEONIDA_PC_TAIL_UBO=1` **for both `build.py` and `check.py`** (or for
`run_gpu_tests.py`). The shaders are then translated with `--pc-tail-ubo`: s0..s31 stay in 128 push-constant bytes,
and the heap base and code base move to a uniform block at set 1 (`docs/design.md`, "Push-constant tail as a uniform
block"), which `vk_compute_test`, `vk_store_test`, `vk_harness`, `vk_texture_test`, `vk_sampler_test` and
`vk_bindless_test` bind (`runtime/test/vk_pc_tail.h`). If you set it
only for `check.py`, the shaders read the tail from push constants the programs no longer declare. `--mesh-compute`
(`LEONIDA_MESH_COMPUTE=1`) isn't combined with it yet: its vertex shader still reads the output address from push
constants, so those tests skip.

## GPUs without mesh shaders (NVIDIA Pascal and older)

Without `VK_EXT_mesh_shader`, the `ms_*` tests are skipped; that's expected, not a failure. To test the path that
runs them anyway, set `LEONIDA_MESH_COMPUTE=1` for `run_gpu_tests.py`, together with `LEONIDA_W64_EMULATE=1` on a
32-wide GPU (every mesh shader is wave64). The mesh shaders are then translated with `--mesh-compute`, dispatched as
compute and drawn by a generated vertex shader (`docs/design.md`, "Mesh shaders as compute"), so the device needs
`bufferDeviceAddress` instead, and `shaderInt64` for a module that declares `Int64`. The standalone tools
(`vk_harness`, `vk_compute_test`, `vk_texture_test`, `vk_store_test`, `vk_bindless_test`) read the modules'
capabilities first: they enable `shaderInt64` only for one that declares `Int64` (or `Int64Atomics`), and report 77
(SKIP, with the reason) when the device lacks it.

## Without a GPU: Mesa llvmpipe

Mesa's llvmpipe is a software Vulkan driver (`mesa-vulkan-drivers` on Debian and Ubuntu). CI runs the suite on it
for every pull request that touches the runtime, the scripts or a tool the jobs read (`.github/workflows/runtime.yml`
lists them), and you can do the same on Linux without a GPU:

```sh
sudo apt-get install -y libvulkan-dev mesa-vulkan-drivers vulkan-tools spirv-tools llvm-19
cmake -S runtime -B runtime/build/linux && cmake --build runtime/build/linux --parallel
python -I scripts/run_gpu_tests.py --build runtime/build/linux --llvm-mc /usr/lib/llvm-19/bin/llvm-mc
LEONIDA_W64_EMULATE=1 python -I scripts/run_gpu_tests.py --build runtime/build/linux \
    --llvm-mc /usr/lib/llvm-19/bin/llvm-mc --filter cs_
LEONIDA_BARY_FALLBACK=1 python -I scripts/run_gpu_tests.py --build runtime/build/linux \
    --llvm-mc /usr/lib/llvm-19/bin/llvm-mc
```

With Mesa 25.2 (llvmpipe, LLVM 20) the first run passes 52 of 88 and skips 36, the second passes 37 of 39 and
skips 2. The skips are expected: llvmpipe has 8-wide subgroups (so every wave64 test skips without emulation, and
the 32-lane tests `ps_bitmask_perm`, `cs_wqm_w32` and `cs_w32_lane_wrap` skip in both runs), no
`VK_KHR_fragment_shader_barycentric` (`ps_bary`, `ps_persp_bary`, `ps_uniform_branch`, `ps_scc_mask32`) and no
`VK_KHR_maintenance8` (`ps_texoff`). The third run draws those four barycentric tests through the fallback below
(with Mesa 26.0.8 it passes 123 of 189, the first run 120), and CI fails it unless every shader that got a fallback
geometry shader passes. With
`LEONIDA_W64_EMULATE=1` the mesh tests fail on llvmpipe, which is why the second run is limited to compute. This is
an llvmpipe bug, not the translator's (TEST-782, #3127): llvmpipe stores a mesh shader's per-primitive
`CullPrimitiveEXT` by bitcasting the boolean (`<N x i1>`) to a float vector, an invalid LLVM cast
(`emit_store_mesh_chan` in `lp_bld_nir_soa.c`, Mesa 26.3.1 and `main`). A constant cull flag is folded away before
that, so the native mesh tests pass; the emulated ones carry it through memory, so it reaches the cast. What the
cast becomes depends on the backend: on x86_64 at 256 bits, LLVM aborts ("LLVM ERROR: Cannot emit physreg copy
instruction"); at 128 bits (`LP_NATIVE_VECTOR_WIDTH=128`) some draw correctly and some draw nothing (with Mesa
26.0.3, `ms_tri`, `ms_w64x_ballot`, `ms_clip` and `ms_f16_pack` pass, while `ms_wave_info_s3` and `ms_v5_workgroup`
stay at the clear color); on arm64 (128 bits) `ms_tri`'s primitive comes out culled, so nothing is drawn. Storing a
constant `false` there instead makes the emulated `ms_tri` draw a frame byte-identical to RADV's (Mesa 26.3.1), and
`ms_tri`, `ms_wave_info_s3`, `ms_w64x_ballot` and `ms_v5_workgroup` byte-identical to NVIDIA's at both widths (Mesa
26.0.3, [measured by @TreXito](https://github.com/ItsNotRexed/Leonida/pull/3838#issuecomment-7086696759), with a 42-line reduced
reproducer). llvmpipe is not a substitute for a result from a
real GPU in [#166](https://github.com/ItsNotRexed/Leonida/issues/166): it catches regressions, not hardware
differences.

## Macs in CI

The `macos` job in `.github/workflows/runtime.yml` builds the runtime on GitHub's `macos-26` arm64 runner and runs
`ctest` and `run_gpu_tests.py` on MoltenVK; `macos-python` in `ci.yml` runs the Python suites. The runner is a VM
whose Metal device is "Apple Paravirtual device" (Metal family Mac2, not Apple7): MoltenVK runs on it with 32-wide
subgroups, so the wave64 and "32 lanes on request" tests skip there (about 75 of 190 in November 2026), and Mesa's
KosmicKrisp finds no device. `macos-15` can't build the runtime: Xcode 16's libc++ has no floating-point
`std::from_chars`. A test known to fail on MoltenVK is listed in the step's `EXPECTED_FAILURES` with the pull
request or coordination-issue finding that tracks it; it still runs, and `scripts/ci_expected_failures.py` fails the step
on any other failure. It takes `FAILED_LIST EXIT_STATUS [EXPECTED ...]`, answers `--help`, and reports a list it cannot
read or a non-numeric status as one line with exit code 2.

## GPUs without fragment barycentrics (NVIDIA Volta, Pascal and older)

`ps_bary`, `ps_persp_bary`, `ps_uniform_branch` and `ps_scc_mask32` read raw barycentrics, which need
`VK_KHR_fragment_shader_barycentric`. NVIDIA has it from Turing on; Volta (V100, Titan V) and older lack it, as do
llvmpipe and some other drivers. Where the device lacks it those four are skipped; that's expected, not a failure.
To test the fallback that runs them anyway, set `LEONIDA_BARY_FALLBACK=1`, **for both `build.py` and `check.py`**
(or for `run_gpu_tests.py`; CI does this on llvmpipe). The shaders are then translated with `--bary-fallback` and
drawn through a pass-through geometry shader that supplies the barycentrics (`docs/design.md`, "Barycentric
fallback"), so the device needs geometry shaders instead.

`ps_interp_f16_high` and `ms_f16_pack` read 16-bit interpolants as `PerVertexKHR` inputs, which need the extension
too and which `--bary-fallback` can't express. With `LEONIDA_F16_INTERP_FALLBACK=1` they read halves the producer
exports decoded instead (see Macs, below) and run with or without the bary fallback; `run_gpu_tests.py` turns that
on by itself where the device lacks the extension. Under the bary fallback without it they skip.

Intel Arc with Mesa ANV currently advertises the native barycentric feature but has a driver bug where translated
pixel shaders can read `BaryCoordKHR.i` as zero (SHADER-925). `make_game.py` detects ANV and automatically translates
with `--bary-fallback`; both entry points accept `LEONIDA_BARY_FALLBACK=1` to force it and
`LEONIDA_BARY_FALLBACK=0` to force native barycentrics. The CLI `--bary-fallback` flag also remains available for
game builds and other manually built fixtures.

AMD's Windows driver on gfx90c iGPUs (Renoir, Lucienne, Cezanne, Barcelo: Ryzen 4000/5000 mobile and 5000G, GCN 5)
also advertises the feature but returns a wrong third `BaryCoordKHR` component, about 0.99 where a small value is
expected (#5933: Vega 8, PCI 1002:15E7, driver 31.0.21921.13001). `ps_bary`, `ps_persp_bary` and `ps_ij_computed`
fail natively; through the fallback `ps_bary` and `ps_persp_bary` pass. `tools/vk_subgroup_info.py` reports
`barycentric_fallback` for these PCI IDs under `VK_DRIVER_ID_AMD_PROPRIETARY`, so `make_game.py` and
`run_gpu_tests.py` turn the fallback on by themselves there, with the same `LEONIDA_BARY_FALLBACK` override. RADV and
other GCN 5 GPUs are not covered: none has been measured.

## Macs: 16-bit interpolation without PerVertexKHR

`ps_interp_f16_high` (and `ms_f16_pack`) normally read 16-bit interpolants through `PerVertexKHR` inputs, which
MoltenVK can't convert to Metal ("PerVertexKHR decoration is not supported in MSL"). `run_gpu_tests.py` asks
`tools/vk_subgroup_info.py` whether the device can take them and, on MoltenVK, translates with
`--f16-interp-fallback` by itself: the vertex or mesh shader then also exports the decoded f16 halves
(`docs/design.md`, "16-bit interpolation without `PerVertexKHR`"). `LEONIDA_F16_INTERP_FALLBACK=1` or `0` forces it
on or off; `build.py` (for `check.py`) translates with the flag only when that variable is `1`. `ms_f16_pack` also runs
on a Mac with `LEONIDA_MESH_COMPUTE=1 LEONIDA_W64_EMULATE=1`.

## GPUs without VK_KHR_maintenance8 (NVIDIA Pascal and older)

`ps_texoff` and `ps_texoff_implicit` sample with a run-time texel offset, which needs `VK_KHR_maintenance8`;
without it they're skipped, which is expected, not a failure. To test the fallback, set `LEONIDA_TEXOFF_FALLBACK=1`
**for both `build.py` and `check.py`** (or for `run_gpu_tests.py`): the offset is then folded into the coordinate
(`docs/design.md`, "Texel offsets"), and the +16/-32 sample is judged on every device.

## Fused multiply-add: GLSL Fma or OpFmaKHR

`cs_fma_fused` checks that `v_fma_f32`, `v_fmac_f32`, `v_fmaak_f32`, `v_fmamk_f32` (and one `v_fma_f16`) round once, on
operands taken from user SGPRs so the driver cannot fold them. Run it with default `--fma glsl` (GLSL.std.450 `Fma`) or
with `LEONIDA_FMA=khr` **for `build.py`** (or for `run_gpu_tests.py`), which emits `OpFmaKHR`.
That needs `VK_KHR_shader_fma` with `shaderFmaFloat32`; the test programs enable it when the device has it and exit
77 (a SKIP) for a module that needs it on a device without it. The Vulkan specification allows a driver to split GLSL
`Fma` into two roundings, which is why `khr` exists: on AMD Windows (26.11.2) and NVIDIA (617.42) GLSL `Fma` stayed
fused, whereas **Mesa RADV (26.2.6-arch3.1, AMD Radeon RX 6800, Linux)** splits 5 of the 7 f32 ops under `--fma glsl`
(reusing the live `a * a` product in `v3`) and returns 8/8 fused exact under `LEONIDA_FMA=khr`. Under default
`--fma glsl`, `check.py` and `run_gpu_tests.py` therefore accept either the fused or the exact split word for each of
fused, whereas **Mesa RADV (Linux)** splits several of the 7 f32 ops under `--fma glsl` — the exact split count varies
by GPU, Mesa/ACO version, and execution modes (4 of 7 on RX 6800 / Mesa 26.2.6-arch3.1 and RX 7600 / Mesa 27.0.0-devel
with `SignedZeroInfNanPreserve 32`, 5 of 7 on RX 6800 before float controls, 7 of 7 on Radeon Pro V540 RDNA1 / Mesa
25.2.8) — while returning 8/8 fused exact under `LEONIDA_FMA=khr` where supported. Under default `--fma glsl`,
`check.py` and `run_gpu_tests.py` therefore accept either the fused or the exact split word independently for each of
the 7 f32 outputs while still requiring exact f16 round-to-odd (`0x00007351`); under `LEONIDA_FMA=khr`, all 8 words
must be strictly fused.
operands taken from user SGPRs so the driver cannot fold them. It passes with the default translation (`--fma glsl`,
GLSL.std.450 `Fma`) on a driver that keeps `Fma` fused, and with `LEONIDA_FMA=khr` **for `build.py`** (or for
`run_gpu_tests.py`), which emits `OpFmaKHR`. That needs `VK_KHR_shader_fma` with `shaderFmaFloat32`; the test programs
enable it when the device has it and exit 77 (a SKIP) for a module that needs it on a device without it. The Vulkan
specification allows a driver to split GLSL `Fma` into two roundings, which is why `khr` exists. The AMD Windows
(26.11.2) and NVIDIA (617.42) drivers we tried don't. Two do, failing `cs_fma_fused` with `glsl` in words 0, 2, 3 and 5
(`v_fma_f32`, `v_fmaak_f32`, `v_fmac_f32`), which get the split value:

- RADV (Mesa 26.3.1, RX 6700 XT, measured in #3004's review). It has `VK_KHR_shader_fma` since Mesa 26.2 (#1364), so
  `khr` fixes it, and so does `exact`.
- The Intel UHD Graphics driver (Xe-LP, Windows 101.7088). It has no `VK_KHR_shader_fma`, so only `exact` fixes it.

`LEONIDA_FMA=exact` (`--fma exact`) builds the single rounding from plain f32 operations and needs no extension; with
it `cs_fma_fused` passes on both of those, on an RTX 3050 Laptop (617.42), an RTX 5060 Laptop (621.01), an RTX 5070 Ti
(621.24) and a Radeon 780M (26.11.2).
Intel UHD Graphics (Xe-LP, Windows 101.7088) splits it too, words 0, 2, 3 and 5 (`v_fma_f32`, `v_fmaak_f32`,
`v_fmac_f32`), and has no `VK_KHR_shader_fma`, so `khr` can't be used there. For such a device,
`LEONIDA_FMA=exact` (`--fma exact`) builds the single rounding from plain f32 operations and needs no extension;
under it, as under `khr`, all 8 words must be strictly fused, and they are on that Intel UHD and on an RTX 3050
Laptop (617.42).

## Hardware oracle (optional, AMD)

`valu_fuzz.py`, `lds_fuzz.py`, `f32_fuzz.py` and `image_fuzz.py` take `--oracle hw`: the same programs run as native
compute kernels on an AMD GPU (`tools/hw_oracle/`; setup in [shader_validation.md](shader_validation.md), "Hardware
oracle"). It needs AMD clang with the amdgcn target (`LEONIDA_AMD_CLANG`), the AMD driver's HIP runtime and a host C
compiler on Windows, or the libdrm launcher on Linux. Without them a run prints `hardware oracle unavailable: ...` and
exits 77, so CI skips it. `image_fuzz.py --oracle hw` loads a linear texture through a T# built for the GPU's family
(GFX11 on gfx11xx, GFX10 on gfx103x; `tests/test_image_fuzz_hw.py` has the exact words) and compares every texel with
its CPU model; add `--negative-control` to check that wrong models are caught. A gfx1150 (RDNA3.5) is not RDNA2: a
match supports the model and does not prove RDNA2. Run these one at a time: a second job on the same GPU can hang it.

## Choosing the GPU

With more than one GPU, pick one through the Vulkan loader, for example
`VK_LOADER_DRIVERS_SELECT=*amd*` or `VK_LOADER_DRIVERS_SELECT=nv-vk*` (or `VK_DRIVER_FILES=<path to the ICD json>`).
`vulkaninfo --summary` shows which devices the loader then sees.

Two GPUs of one vendor share one driver, so the loader cannot tell them apart (an RTX 5070 Ti next to an RTX 3060,
say; without a choice the runtime and every test take the same one: a native driver over Microsoft's Dozen, discrete
over integrated, the first on a tie, `vulkan_device_rank` in `runtime/src/vulkan_device_filter.h`).
`LEONIDA_VK_DEVICE` names the GPU by part of its name, without case: `LEONIDA_VK_DEVICE=5070` or
`LEONIDA_VK_DEVICE=radeon`. The D3D12 tests (`vk_test_device.h`) and the layer's standalone device read it,
and a value that matches no device fails the run instead of testing
another GPU. For the host side of the comparison tests, `LEONIDA_HOST_ADAPTER` picks the D3D12 adapter the same way.
Unset, the host side is the adapter behind the layer's Vulkan device (`runtime/test/host_adapter.h`; `d3d12_buffer_tiles`
by the layer's `GetAdapterLuid()`), so on a PC with two GPUs the layer is compared with its own GPU's driver.
The GPU tools of `runtime/test/shaders` (`vk_compute_test`, `vk_harness`, ...) read it the same way through
`vk_device_pick.h`, and fail with a message naming the filter when it matches no device.

Two test settings make the layer's standalone device look like a GPU with fewer Vulkan queues, so D3D12 queues share
them (`docs/d3d12_vulkan.md` section 4.4) on any GPU: `LEONIDA_VK_MAX_QUEUES_PER_FAMILY=N` (1 to 16) creates at most
N Vulkan queues per queue family, and `LEONIDA_VK_GRAPHICS_FAMILY_ONLY=1` puts COMPUTE and COPY queues in the
graphics family, as on a GPU without async compute or transfer families. Both together give the shape of the Intel
UHD 770 on Windows and of lavapipe, one family with one queue; the `d3d12_queue_share_one_queue` and
`d3d12_queue_wait_one_queue` tests run that way, and any other test can be run so by hand. The device logs what it
applied; a queue count outside 1 to 16 is ignored, with a message. Only tests apply them: the device that
`gta6rt --run` creates for a game (`LEONIDA_D3D12=vulkan`) ignores both and prints `... is a test setting; the
game's device ignores it`, so leaving them set cannot slow a game down (`run_tiny_frame_queue_test_settings`).

## Known problems

| What you see | Cause | What to do |
|---|---|---|
| Under Wine 9.0 (Ubuntu 24.04's `wine`), a test or the runtime waits out every `std::condition_variable` timeout: `wait_registration_admission` ("native drainage complete"), `vk_presenter_sync` (`unblocked_during_present`), `frame_pacer`, `vk_fence_dispatcher_sync`, `context_concurrent_detach` | The STL's timed wait (`_Cnd_timedwait_for_unchecked`) is linked statically from `msvcprt.lib` and reads Windows' `_Mtx_t` layout, while `_Mtx_lock`/`_Mtx_unlock` come from Wine's builtin msvcp140, which used another layout until wine-10.1 (757bd7909fda); the wait never released the mutex (#3796) | Fixed in the build: `runtime/src/crt_timed_wait.cpp` routes the wait through msvcp140's own `_Cnd_timedwait` in every `/MD` binary, and `crt_timed_wait_test` checks it. A binary built before that fix still needs Wine 10.1 or newer (or Proton) |
| `ctest` passes but runs far fewer tests than expected (about 45 instead of 125 on Windows), no `d3d12_*` GPU tests, and `LEONIDA_D3D12=vulkan` does nothing | CMake found no Vulkan headers: `VULKAN_SDK` was not set in the environment that ran CMake (for example a user variable set after the terminal opened), and `CMakeCache.txt` keeps that failed lookup. The configure prints a `CMake Warning` "Vulkan headers not found" | Set `VULKAN_SDK` in that environment (or pass `-DLEONIDA_VULKAN_HEADERS_FETCH=ON`), delete `CMakeCache.txt` and configure again. `-DLEONIDA_REQUIRE_VULKAN=ON` makes this a configure error; CI and `scripts/build_runtime.ps1` (with `VULKAN_SDK` set) pass it |
| Windows refuses to start freshly built test programs ("An Application Control policy has blocked this file"), and the runner stops with a traceback | Smart App Control on Windows 11 | Turn Smart App Control off for the run |
| `tools/hw_oracle` works once, then stops with "hardware oracle unavailable: Windows Defender blocked the launcher ... (WinError 225)" (older runs: an `OSError: [WinError 225] ... contains a virus or potentially unwanted software` traceback) | Defender's machine-learning check flags the small unsigned `leonida_hw_oracle_<hash>.exe` that the runner builds in `%TEMP%` (seen as `Trojan:Win32/Bearfoos.A!ml`) and quarantines it; every later run rebuilds and loses it again | Add an exclusion for the build folder (`%TEMP%`) in Windows Security, Virus & threat protection, Manage settings, Exclusions, restore the file from Protection history if you want it back, and rerun. The program is built from `tools/hw_oracle/hw_oracle.c` and is a false positive |
| Validation errors without a VUID, or messages from a layer you didn't ask for | An implicit Vulkan layer from other software (Rockstar Social Club, ReShade, MangoHud, an overlay) | Set `VK_LOADER_LAYERS_DISABLE=~implicit~`. Overlays that hook the swapchain (RivaTuner `VK_LAYER_RTSS`, OBS `VK_LAYER_OBS_HOOK`) can stay loaded: see the next row |
| Validation errors that only your machine shows (e.g. `xbox_present`'s sRGB swapchain `STORAGE` message), with MSI Afterburner/RivaTuner or OBS installed | An overlay layer that hooks the swapchain: while RivaTuner Statistics Server is running, its `VK_LAYER_RTSS` adds `STORAGE` usage to it (OBS's `VK_LAYER_OBS_HOOK` injects `VK_IMAGE_USAGE_STORAGE_BIT` too). In an elevated (administrator) terminal the loader ignores `VK_LOADER_LAYERS_DISABLE` (`VK_LOADER_DEBUG=all` says so), so `~implicit~` leaves the layer in | Run the tests from a normal terminal, close the overlay application, or set the layer's own switch, which works elevated too: `DISABLE_RTSS_LAYER=1`, `DISABLE_VULKAN_OBS_CAPTURE=1` (OBS). Any other layer's switch is the `disable_environment` entry in its manifest (the JSON listed under `HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers` or `HKCU\...`). `VK_LOADER_DEBUG=layer` prints the layers that were actually inserted |
| GPU tests crash with an access violation (`0xC0000005`, exit `-1073741819`) right after device creation, with no layer message and validation off (e.g. `d3d12_method_variants` 9 of 10 runs, `xbox_present` 10 of 10, on `main` 561b65a83, RTX 4070 Ti, NVIDIA 616.56) | An implicit layer from an overlay or game tool you did not install for these tests. FPS Monitor's `fpsmonvk64.json` (listed under `HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers`) names itself `VK_LAYER_RENDERDOC_Capture`, so `DISABLE_VULKAN_OBS_CAPTURE=1` does not disable it | `VK_LOADER_DEBUG=layer` prints the layers that were actually inserted. That manifest's `disable_environment` is `DISABLE_LAYER=1`; `DISABLE_LAYER=1` or `VK_LOADER_LAYERS_DISABLE=~implicit~` gave 20 of 20 clean runs (a layer's own switch is its `disable_environment` entry, as the row above says) |
| `vk_sync_validation` fails with `LEONIDA_VK_VALIDATION=1` ("the write-after-write hazard was not reported") | Synchronization validation is off although the tests ask for it: an older layer, or a layer setting such as `VK_KHRONOS_VALIDATION_VALIDATE_SYNC=false` or a `vk_layer_settings.txt` that overrides it. Every other GPU test then reports "0 validation errors" without having been checked | Remove the override or update the Vulkan SDK, then rerun `ctest -R vk_sync_validation` until it passes |
| On RADV with Mesa 25.2 (Ubuntu 24.04's default; 25.2.8 measured) or 25.3 (from the source, not run), `cs_w32_salu64` fails with word 2 = `0` (want `0xFFFFFFFF`) and `cs_signed_zero_nan` fails with word 5 = `0` (want `1`). The `cs_signed_zero_nan` failure alone is also measured on RADV 25.0.7 (RX 7800 XT, Debian 13; @mrbean56, #4421) and on ANV 25.2.8 (Intel UHD 630; @theswiss24, #4571); llvmpipe on Mesa 26.0.8 (Ubuntu 26.04) still gives word 5 = `0`, which `check.py` does not judge there (`UNJUDGED_WORDS`) | Two Mesa bugs, not translator bugs (#166). `cs_w32_salu64`: ACO's `combine_add_sub_b2i` reads a temporary without counting the use, so dead-code elimination removes it (`ACO ERROR: ... never defined` with `MESA_SHADER_CACHE_DISABLE=true`); `RADV_DEBUG=llvm` passes. `cs_signed_zero_nan`: NIR's range analysis (`fmul_is_a_number`) calls `0 * x` a number when `x` may be infinite, so the `x != x` check folds to false despite `SignedZeroInfNanPreserve`. The same NIR code is behind llvmpipe's word 5 (#664) and ANV's, since both drivers run NIR; `cs_w32_salu64` is ACO's, so only RADV has it | Expected with these Mesa versions. From the source, the ACO function is gone from Mesa 26.0 and the NIR code was replaced in 26.1 (neither run); RADV from Mesa 26.3.1 passes both on an 8060S (gfx1151) |
| Every test crashes inside `libnvidia-eglcore.so` | NVIDIA's 535 server driver (Ubuntu 24.04's default) | Use driver 615 or newer |
| `build.py`: 0 translated, reason `.text` | An old `build.py` with LLVM 19 | Update to current `main` |
| `vk_harness` doesn't compile: `VK_KHR_maintenance8` names unknown | Vulkan headers older than the SDK the code was written against | Update to current `main` (the code now builds without it), or use current Khronos Vulkan-Headers; conda-forge's package is too old |
| `ps_bary` and `ps_uniform_branch` fail by about 0.9 | The GPU runs pixel shaders wave64 without fragment barycentrics (GCN3 and RDNA1 on RADV), or an old AMD Windows driver (22.40) | Current `main` skips them on such devices; on Windows, update the driver |
| `ps_bary`, `ps_persp_bary` and `ps_ij_computed` read the third barycentric as about 0.99 on a gfx90c iGPU (Vega 8, AMD Windows driver 31.0.21921.13001; #5933) | The driver's native `BaryCoordKHR` is wrong there: the same tests pass through `--bary-fallback`, and natively on RADV (Radeon 8060S) | `tools/vk_subgroup_info.py` turns the fallback on by itself for gfx90c under AMD's driver, and `make_game.py`, `run_gpu_tests.py`, `build.py` and `check.py` follow it when `LEONIDA_BARY_FALLBACK` is unset; `=1` forces the fallback and `=0` forces native barycentrics, where they still fail |
| `cs_w32_salu64` gives eight wrong words on RDNA1 under RADV Mesa 25.2.8 (`0x1E6A2C48 0x1234 0 0x100 47 48 0xFFFF8000 0xFFFFFFFE`, where word 2 should be `0xFFFFFFFF`) | Only the high word of the `s_bfe_i64` result is wrong. The same words appear on four pinned V540s (Navi 12, PCI `1002:7360`) and on a clean upstream/main worktree, and three different valid SPIR-V sign-extension formulations give the same result while the CPU interpreter matches, so this is the driver version rather than the translator (SHADER-276) | Expected with Mesa 25.2.8; the fixture passes on NVIDIA and on AMD's Windows driver. Try a newer Mesa before treating it as a translator bug; do not weaken the fixture's expectation |
| `d3d12_typeless_rtv` fails its WARP comparison on an AMD desktop iGPU (TEST-591) | [Reported on the Ryzen 5 7600X iGPU with Windows 10 build 19045 and AMD 23.40.02](https://github.com/ItsNotRexed/Leonida/issues/166#issuecomment-7083403389). The exact cause is still open. | Retest on a newer Windows and AMD driver combination. Passing results were reported on the [Ryzen 9 7900 iGPU with AMD 25.10.30.02](https://github.com/ItsNotRexed/Leonida/issues/3232#issuecomment-7085263408), a [gfx1036 iGPU with AMD 26.5.2](https://github.com/ItsNotRexed/Leonida/issues/2891#issuecomment-7084023801), and the [Ryzen 7 9800X3D iGPU with driver 32.0.21045.29003](https://github.com/ItsNotRexed/Leonida/pull/3440). |
| `ps_interp_f16_high` fails on a Mac: `PerVertexKHR decoration is not supported in MSL`, then `vkCreateGraphicsPipelines` -3 | MoltenVK can't convert `PerVertexKHR` inputs to Metal | Current `main` translates it with `--f16-interp-fallback` on MoltenVK (see "Macs: 16-bit interpolation") |
| `ps_vcmpx_forms` fails on a Mac: word 1 reads `0xf05e8e16`, want `0xf05e8a12` | MoltenVK's fast-math makes a compare of a NaN with itself true (SHADER-234) | Current `main`'s test programs turn it off (`MVK_CONFIG_FAST_MATH_ENABLED=0`, unless you set the variable yourself); rebuild them |
| On a Mac, a translated shader's GPU time is 1.5 to 16 times what it would be without the precise-arithmetic overhead | MoltenVK's SPIRV-Cross turns every `NoContraction` float add, subtract and multiply into a call to an `[[clang::optnone]]` helper that Metal does not inline. MoltenVK 1.4.2 and `main` up to at least `5189f8de` are affected (SHADER-234, [design notes](design.md#translator-status-and-runtime-contract)) | Expected with these MoltenVK versions, and the output is still exact. Before you time shaders on a Mac, build MoltenVK with SPIRV-Cross `98cbb7c0` or newer in `ExternalRevisions/SPIRV-Cross_repo_revision` (KhronosGroup/MoltenVK#2917 moves it to `57e4efcc`), and point `LEONIDA_VULKAN_LOADER` at its `libMoltenVK.dylib` |
| `d3d12_descriptor_test` reports that the set had room for fewer than 5001 2D views and 2100 samplers, and the device logs that the bindless set holds fewer samplers than a full shader-visible sampler heap | The device's descriptor limits, or its `vkGetDescriptorSetLayoutSupport`, allow no bigger bindless set ([d3d12_vulkan.md](d3d12_vulkan.md) section 5). MoltenVK's answer is ignored, since it rejects every layout of `maxPerSetDescriptors` (1212) or more although it runs the full set | Expected where the limits are low. The test still checks the last index of each array and that the next view and sampler are refused |
| `cs_dpp_ops` fails on a Mac with MoltenVK 1.4.2: `out[0..7] = 100 100 11 ...`, want `100 10 11 ...` | MoltenVK 1.4.2's SPIRV-Cross moves the DPP lowering's subgroup ballot and shuffle into a short-circuiting `&&` and `?:`, so only some lanes run them and lane 1 sees its source lane 0 as inactive. SPIRV-Cross `5c372ffb` (in MoltenVK `main`) keeps them out of `?:`; a ballot on the right of `&&` is still moved there in SPIRV-Cross `57e4efcc` | Expected with MoltenVK 1.4.2; it passes with MoltenVK `main` (point `LEONIDA_VULKAN_LOADER` at its `libMoltenVK.dylib`) and on KosmicKrisp |
| `check.py`'s `cs_fma_fused` fails on a Mac with MoltenVK 1.4.2: word 7 reads `15360` (`0x3c00`), want `29521` (`0x7351`) | The f16 multiply-add (round to odd). The MSL of 1.4.2 and of MoltenVK `main` differ only in SPIRV-Cross's precise-arithmetic helpers (`[[clang::optnone]]` and `fma(l, r, 0)` against `math_mode(safe)` and `fma(l, r, -0.0)`, the SHADER-182 / SHADER-234 change) | Expected with MoltenVK 1.4.2; it passes with MoltenVK `main` |
| `gs_tri` and `ms_tri` fail on a Mac | MoltenVK has no geometry stage and no `VK_EXT_mesh_shader` | Current `main` reports them as skipped |
| With `LEONIDA_W64_EMULATE=1` on llvmpipe, the `ms_*` tests fail. Measured with Mesa 26.3.1: on x86_64 at 256-bit vectors (the default there) all six abort with "LLVM ERROR: Cannot emit physreg copy instruction"; on arm64 (128-bit) `ms_tri` draws nothing (the frame stays at the clear color); on x86_64 at 128 bits (`LP_NATIVE_VECTOR_WIDTH=128`) `ms_tri` happened to draw correctly, but with Mesa 26.0.3 `ms_wave_info_s3` and `ms_v5_workgroup` draw nothing there (4 of 6 pass, [measured by @TreXito](https://github.com/ItsNotRexed/Leonida/pull/3838#issuecomment-7086696759)) ([measured runs](https://github.com/ItsNotRexed/Leonida/issues/3127#issuecomment-7086469679)) | An llvmpipe bug: a mesh shader's runtime `CullPrimitiveEXT` is stored through an invalid `bitcast <N x i1> to <N x float>` (TEST-782, #3127; see "Without a GPU: Mesa llvmpipe") | Run the mesh tests on llvmpipe without `LEONIDA_W64_EMULATE`, or add `LEONIDA_MESH_COMPUTE=1` (the compute fallback): with it all 6 `ms_*` pass emulated on llvmpipe (x86_64, 256-bit) |
| On a Mac, `valu_fuzz.py --family f16` reads `+0` where the model wants `-0`, from a multiply whose exact product is -0 (`v_mul_f16`, `v_pk_mul_f16`) or from an FMA whose product and addend are both -0 (`v_fma_f16`, `v_fmac_f16`, `v_pk_fma_f16`) | MoltenVK computes a `NoContraction` multiply as `fma(l, r, +0)`, which drops the sign. MoltenVK 1.4.2 and `main` up to at least `5189f8de` are affected (SHADER-182, [design notes](design.md#translator-status-and-runtime-contract)) | Expected with these MoltenVK versions. To run the rest of the fuzzer clean, add `--skip-op` for each of the five ops. To see them match, install Homebrew's `mesa` and use its KosmicKrisp driver: `LEONIDA_VULKAN_LOADER=/opt/homebrew/opt/vulkan-loader/lib/libvulkan.1.dylib VK_DRIVER_FILES=/opt/homebrew/share/vulkan/icd.d/kosmickrisp_mesa_icd.aarch64.json python3 -I tools/valu_fuzz.py --family f16 --build runtime/build/linux`. Or build MoltenVK with SPIRV-Cross `300681bb` or newer in `ExternalRevisions/SPIRV-Cross_repo_revision`, and point `LEONIDA_VULKAN_LOADER` at its `libMoltenVK.dylib` |
| A pixel shader exporting non-canonical f32 NaNs reads back `0x7fc00000` from an RGBA32F color attachment | KosmicKrisp (Mesa 26.3.1), reproduced on Apple M3 Max, M2 Ultra and M1 Pro ([#3125](https://github.com/ItsNotRexed/Leonida/issues/3125)): `nir_to_msl` emits `(NAN)` for `float`-typed NaN constants (`MESA_KK_DEBUG=msl`); MoltenVK 1.4.2 preserved both tested payloads on the same hardware | `ps_nan_payload` allows the measured canonical word only for the NaN channels and only when `vk_harness` reports `driver: KosmicKrisp`. Finite channels and other drivers still require exact words. Integer or storage-buffer readback still canonicalizes constants that NIR infers as `float` (see the `cs_bvh` row below). |
| `cs_bvh` and `cs_bvh32` fail on a Mac with KosmicKrisp: `out[3]` reads `0x7fc00000`, want `0xffffffff` | KosmicKrisp's `nir_to_msl` compiler (Mesa 26.3.1, measured on Apple M2 Ultra, macOS 27.3), not the translator or the buffer format. The triangle branch writes bitcast `float` words and the box branch writes `OpSelect %uint` with `0xffffffff` (`-1`) into the same VGPRs; NIR erases `OpBitcast`, type inference marks the shared register and `bcsel` as `float`, and `nir_to_msl` prints any `float`-typed 32-bit constant in the IEEE-754 NaN range (`0x7f800001..0x7fffffff`, `0xff800001..0xffffffff`) as `(NAN)` (`0x7fc00000`) | Expected with KosmicKrisp in Mesa 26.3.1; MoltenVK 1.4.2 and llvmpipe (Mesa 26.3.1) return `0xffffffff` on the same machine. Do not weaken `vk_compute_test`'s expected BVH words |
| CMake stops with "needs the Windows SDK 10.0.26100 or newer", or MSVC reports `'Desc_1_2': is not a member of 'D3D12_VERSIONED_ROOT_SIGNATURE_DESC'` | A Windows SDK older than 10.0.26100 (10.0.22621 is the common one), which has no version 1.2 root signatures | `winget install Microsoft.WindowsSDK.10.0.26100`, or add it in the Visual Studio Installer, then delete `runtime/build` and configure again |
| `ctest` lists `fsr_dispatch` and `bvh_builder` as failed (`Not Run`) after clang-cl hits an internal compiler error in `fsr_dispatch_test.cpp` and `bvh_builder_test.cpp` | clang-cl 22.1.8 on Fedora (Linux cross-build) and Windows, plus 22.1.7 on Windows | Use clang-cl 23.1.6 (both files compile on Windows) |
| `d3d12_compute`, `d3d12_draw` or another test that writes a shader library under the temp directory fails now and then (a PSO creation `line N: SUCCEEDED(...)`, or exit 3 with no message) but passes when run alone | Two copies running at once (`ctest -j`, or two suites sharing one Wine prefix or `TMPDIR`) used the same `leonida_<test>` folder under the temp directory, and one copy deleted the shader modules the other was loading: 6 of 40 runs failed with four copies of `d3d12_compute` at once on a Mac under Wine | Current `main` gives each process its own folder (`<temp>/leonida_<test>_<pid>`) |
| A Windows CI artifact run under Wine by hand: `d3d12_compute` and the other tests that read `LEONIDA_TEST_*_SPV` end with exit 3 and no message | The `.spv` paths are compiled in as `D:/a/Leonida/Leonida/runtime/build/generated/...`; the prefix has no `D:` drive and the test isn't run from the artifact folder | Run it from the artifact folder (the tests then find `generated/<name>.spv` there) or through `scripts/run_wine_tests.py`, which maps that drive for the run; current `main` names the file it couldn't copy (`cannot copy ...: File not found`) |
| Linux cross-build: CMake can't find `clang-cl-19` (or another versioned `clang-cl-<N>`) although `clang-19` and `lld-19` are installed | On Debian and Ubuntu, the versioned `clang-cl-<N>` link is in the `clang-tools-<N>` package, not in `clang-<N>` (`/usr/lib/llvm-<N>/bin/clang-cl` comes from it too); `lld-link-<N>`, `llvm-lib`, `llvm-rc` and `llvm-mt` come from `lld-<N>` and `llvm-<N>` | `sudo apt-get install clang-tools-19 lld-19 llvm-19` (Ubuntu 24.04 has 19 and 20), then pass the versioned names as in [the cross-build](#the-windows-runtime-on-linux-under-wine-or-proton) |
| Under Ubuntu's Wine 9.0, `wineboot --init` (and every Wine program after it) stops at once with `free(): invalid pointer` or a segmentation fault, before printing anything else; the prefix holds only a `wineserver` file | That Wine opens `/run/user/<uid>` for its server directory, whether `XDG_RUNTIME_DIR` is set or not, and aborts when the folder doesn't exist. The login manager makes it for a desktop session; a container or a shell started without a login session (a root shell in Docker, a CI runner image) has none | Create it before starting Wine: `mkdir -p /run/user/$(id -u) && chmod 700 /run/user/$(id -u)` (setting `XDG_RUNTIME_DIR` alone doesn't help), delete the half-made prefix, and run `wineboot --init` again |
| `run_gpu_tests.py` fails on a `dict \| None` annotation | Python older than 3.10 | Use Python 3.11+ |
| `vulkaninfo` lists `Microsoft Direct3D12 (<gpu>)` next to the real GPU and the tests run on it | The Windows "OpenCL™, OpenGL® and Vulkan® Compatibility Pack" installs Mesa's Dozen, a Vulkan-on-D3D12 layer that reports `DISCRETE_GPU` like the native card | Current `main` prefers a native ICD over Microsoft's vendor (`0x1414`) in the runtime and every harness; set `LEONIDA_VK_DEVICE=<part of the real GPU name>` to select explicitly. See #915. |
| A GPU test that uses `image_load` (or lighting built from real shaders) reads zeros from its image; @Silliana saw every texel except pixel (0,0) come back zero in their own layer (#603, not reproduced here) | In this runtime the shader turns the T# words into an index through the descriptor hash table (`runtime/src/descriptor_table.h`: a miss gives index 0), and index 0 is a 1×1 image of zeros (`runtime/src/vulkan_bindless.h`), so a hand-built or zeroed T# that was never registered reads zeros even at (0,0). The translator does not compare coordinates with the T# size, and `cs_imgload.asm` only uses its T# words as a hash key | Register the descriptor through the runtime (for D3D12, create it with `CreateShaderResourceView`) so its T# hits the table, with the image's real width and height (stored as size − 1, see `runtime/src/gfx10_descriptors.h`) |
| `cs_sample_lz_int_d16`, `ps_sample_implicit_sint`, `ps_ssharp`, `ps_tex`, `ps_texture_mip0`, `ps_texture_sample` and `ps_texture_sample_d16` crash (exit -11 or 139) on an Intel GPU on Linux, while NVIDIA and llvmpipe pass them (TEST-399) | A bug in Mesa ANV's shader compiler: compiling these texture-sampling shaders, `nir_opt_gcm` (`nir_opt_gcm.c:430`, `gcm_schedule_early_src`) dereferences a phi source an earlier pass has freed. The SPIR-V passes `spirv-val`. Seen with Mesa 26.1.6 (Arrow Lake) and 26.2.7 (Iris Xe, ADL GT2); Mesa 26.3.1 and `main` (bb98606) pass the whole suite on the same Iris Xe | Use Mesa 26.3.1 or newer. To check a newer Mesa without installing it, build its Intel Vulkan driver and point `VK_DRIVER_FILES` at its `src/intel/vulkan/intel_devenv_icd.x86_64.json` |
| A test exits with `3221225477` (`0xC0000005`, an access violation) on AMD, only when `VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump` is set (`d3d12_sparse_queue_family_contract` sets it itself) | The Vulkan SDK's API dump layer (1.4.363.0) crashes the test process now and then: on an RX 9060 XT (RDNA4, Windows driver 32.0.32029.2010) `d3d12_buffer_tiles` and `d3d12_queue_wait` crashed in 1 and 4 of 8 runs in one session and in 3 and 1 of 30 in another; the layer's `detailed` and `pre_dump` settings make no difference. Not seen in 24 runs of those tests without the layer, nor in 8 runs each of the SDK's `vkcube` and `vulkaninfo` with it, so it is not simply the layer on that driver | Run the test again. Do not read a crash under the layer as a failure of the test: run the same program without the layer to check its result, and use `LEONIDA_VK_VALIDATION=1` (the Khronos layer) for validation messages |
| On an AMD integrated GPU (gfx1036) with AMD's Windows driver (`32.0.21042.62` and `32.0.21045.16010` measured), `crosslane_fuzz.py --wave 64 --w64-native --dpp --no-maskdata --seed 82001 --programs 600` reports one native mismatch: `cs_xl64_8200346406` differs in 96 words (32 lanes of each of its three waves), while `--wave64-emulate` and the hardware oracle match | AMD's Windows driver, not the translator (SHADER-705; the driver's generated code was not inspected). The same native wave64 SPIR-V is exact under RADV (Mesa 26.3.1) on another gfx1036 and on an RX 7900 XTX, and in the CPU interpreter. The wrong instruction is a `v_permlanex16_b32` that reads a register a DPP op with a partial `row_mask` has just written ([shader_validation.md](shader_validation.md), FUZZ-7) | Nothing to change in a build: no shader of the game has that shape (0 of 26,503; the two instructions meet in 6 shaders, all wave32 compute). `--wave64-emulate auto` stays native on this driver because of #857, and the emulated translation of the program matches there |
| With `--wave64-emulate` (or `LEONIDA_W64_EMULATE=1`) on an NVIDIA RTX 5080 with Windows driver 32.0.16.1692 (616.92), `crosslane_fuzz.py --seed 91001 --dpp --no-maskdata` reports a hard mismatch for `cs_xl64_9100373886`: after `s_xor_b64 s[16:17], vcc, vcc` (0 in every lane), the second of two nested `v_cndmask_b32_e64 … s[16:17]` returns the old register instead of the lane id | An NVIDIA driver miscompile, not the translator (SHADER-720). The same SPIR-V (spirv-val clean) matches the model on an RTX 3060 with 616.64, an RTX 5060 Ti (Blackwell) with 621.14, an RTX 5070 Ti with 621.24 and an RTX 5080 with the Linux 580.65.06 driver. Hand edits narrow it to a value selected through the `%active` gate of a store; on the Linux 580 driver a similar pattern, a subgroup shuffle selected by an `%active` that is false, still fails in three wave32 `v_permlane16`/`v_permlanex16` programs of the same seed (#3796, #3918) | Update the NVIDIA driver to 621.14 or newer on Windows. If a crosslane mismatch only shows on one driver, check another driver or GPU before changing the translator |
| `lds_fuzz.py --wave 64` (native, not `--emulate`) reports a `ds_bpermute_b32` mismatch on an AMD GPU on Windows, e.g. `cs_ldsfuzz_s7031_496_w64` (seed 7031): small values like `0x14` where the model wants another lane's data, while `--oracle hw` and `--emulate` on another GPU agree with the model | The AMD Windows driver (32.0.21042.62, gfx1036) miscompiles a native wave64 compute shader with two subgroup shuffles; a reduced SPIR-V module is in #4613 | Not a translator regression: check the same seed with `--oracle hw` and with `--emulate`. Native wave64 results on that driver need that cross-check |
| On AMD's Windows driver a compute pipeline creation takes tens of minutes: `vk_pipeline_check` prints `still compiling <name> after ...s` with one busy thread, or a compute PSO creation through the D3D12 layer seems never to return | The driver's compiler on one module in the dispatch-loop form (`"control_flow": "dispatch loop"` in the translated `index.json`). Measured on a Radeon gfx1036 iGPU, driver 32.0.21042.62 (SHADER-995): `XboxBvhBuilder.dll`'s `LeafRehydration` in that form (a 581-case switch in a loop) was created after 34 minutes of one busy thread (XXTESTING), while its structured module is created in 2.9 s and an RTX 3070 (NVIDIA 610.74) created the dispatch-loop one in 11 s. It is not every such module: ten other programs of that DLL compile in that form on the Radeon in 0.4 to 40 s ([embedded_shaders.md](embedded_shaders.md), section 7). Not tried on another AMD GPU or driver branch | Translate with current `main`. Since SHADER-427 only a tree nested deeper than 64, a graph that cannot be made reducible or `--dispatch-loop` gives that form, and none of that DLL's programs has it; the shaders stage translates again by itself when the translator changed. A wave64 compute shader that still has it comes out in another form with `--wave64-emulate on` (12.4 s for this program on that driver), which that driver can crash on for other programs (#857) |

### Signed zero in a float storage image (SHADER-605)

Build `vk_store_test` as above and put `spirv-as` and `spirv-val` on PATH. Run this standalone
probe from the repository root, passing the path to the executable you built:

```powershell
python -I runtime/test/shaders/probe_signed_zero.py runtime/build/vulkan/vk_store_test.exe
```

The probe assembles and validates two hand-written Vulkan 1.2 compute modules. Each requests
`SignedZeroInfNanPreserve 32` and executes three constant `OpImageWrite` instructions into
a `VK_FORMAT_R32G32B32A32_SFLOAT` image. Only the SPIR-V image format changes (`Unknown` or
`Rgba32f`). The harness copies the images back, issues a transfer-write to host-read barrier,
waits for completion, and compares all 768 words, including 756 prefilled guard words.
A bit mismatch prints the word index and returns exit code 1. Both variants run even if the
first comparison fails.

On a GeForce RTX 5070 with Windows driver 616.56, both formats matched 765/768 words.
In the first texel, `(-0, +0, +0, +0)`, word 0 read back as positive zero. The second texel,
`(-0, 1, +0, +0)`, retained every bit. In the third, `(-0, -0, +0, +0)`, words 8 and 9 both
read back as positive zero. All guards matched. This reproduces the discrepancy independently
of RDNA translation. Independent runs of the same probe reproduced those three mismatches on
a Linux RTX 5090 (595.99.02). On Windows, a Ryzen 7 9800X3D Radeon iGPU (25.10.30.02),
an RTX 3080 (616.64) and an RTX 3060 (621.14) preserved all 768 words in both formats.
The device reports are in [PR #3260](https://github.com/ItsNotRexed/Leonida/pull/3260).

### Which known problems the game's shaders reach

Several rows of the table above are tied to particular instructions. A GPU test or fuzzer that hits one does
not by itself mean the game does. The counts below say how many of the 26,503 1.0.2.1 shaders (`shaders.rpf`,
LLVM gfx1013) contain those instructions, by stage and by the wave size their FSSM header gives. They are
counts of instructions over the program up to each listing's last `s_endpgm`. They are not reproductions: a
shader that contains an instruction does not have to take the exact path a row describes.

```sh
python -I tools/op_reach.py <work>/shaders/dxbc/disasm --index <work>/shaders/dxbc/index.json \
    --fssm <work>/shaders/dxbc/fssm s_bfe_i64 '*_dpp' v_permlanex16_b32 ds_bpermute_b32 v_mul_f16 ...
```

The corpus has 2,662 wave32 shaders (1,535 CS, 1,127 PS) and 23,841 wave64 shaders (every VS, GS and MS, 1,154
CS, 10,974 PS).

| Row | What the game's shaders contain (1.0.2.1) |
|---|---|
| `cs_w32_salu64` on RDNA1 / RADV 25.2.8 (the high word of `s_bfe_i64`) | **No shader uses `s_bfe_i64`** (nor `s_bfe_u64`). The fixture's failure does not reach a game shader. |
| `lds_fuzz.py --wave 64` `ds_bpermute_b32` on AMD's Windows driver | **No wave64 shader uses `ds_bpermute_b32`**: it occurs in 5 wave32 CS (74 instructions), and `ds_permute_b32` in none. |
| `crosslane_fuzz.py --wave 64` on an AMD iGPU (a `v_permlanex16_b32` after a partial-`row_mask` DPP) | DPP in 4,313 shaders, 2,496 of them wave64 (2,364 PS, 113 CS, 17 MS, 2 VS). `v_permlanex16_b32` in 2,293 shaders, 561 of them wave64. Whether any of them pairs the two with a partial `row_mask` is not counted, so these are upper bounds. |
| `cs_dpp_ops` on MoltenVK 1.4.2 | DPP (any `*_dpp`) in 4,313 shaders (66,354 instructions): 1,817 wave32, 2,496 wave64. |
| `valu_fuzz.py --family f16` signed zero on MoltenVK | `v_mul_f16` 124 shaders, `v_fma_f16` 124, `v_fmac_f16` 130, `v_pk_mul_f16` 55, `v_pk_fma_f16` 47, mostly PS (113–115 for the scalar forms). |
| `ps_interp_f16_high` on MoltenVK (16-bit interpolation) | `v_interp_p1ll_f16` / `v_interp_p2_f16` in 4,776 PS (679 wave32, 4,097 wave64), 59,367 instructions each. The count is the instruction, not the `high` operand. |
| `gs_tri` and `ms_tri` on a Mac | 195 GS and 9,282 MS, all wave64. |

## Posting your result

Use the template at the top of [#166](https://github.com/ItsNotRexed/Leonida/issues/166): GPU, driver, subgroup
range, OS, the commit you built, the tool versions and the counts each script printed. Post counts and pass/fail
lines only, never shaders or logs from the game. A failure that isn't in the table above gets its own issue, linked
from your entry.

The CPU, GPU, VRAM and RAM fields come from `tools/pc_checker.py`, which compares this PC with the Xbox Series X
reference specs (or `--profile 2` for the community 1080p Low estimate). Run it in text mode; the default form opens a
window:

```powershell
python -I tools/pc_checker.py --cli --check    # spec lines, then exit 0 good, 1 weak, 2 unknown
python -I tools/pc_checker.py --cli --check --json
```

`--cpu`, `--gpu` and `--ram` override the detected values when a VM or a remote session hides them. Its scores are
rough estimates made for this tool, not benchmarks, so put the detected model names in the #166 entry, not the score.
`python -I tests/test_pc_checker.py` checks the parsing and the comparison with the window mocked.
