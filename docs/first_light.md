# Drawing model UVs and textures with First Light

**Status: measured** on the 1.0.2.1 package and an NVIDIA RTX 2060 SUPER (driver 620.73.04, Vulkan 1.4.357,
CachyOS). The untextured fixture draws a model from the package with a shader pair chosen to display its UVs.
With `--dds`, the default pixel shader samples the supplied texture at those UVs. Section 6 gives the textured
draw recipe. The image stays in the user's work folder.

## 1. What was drawn

The model is one static drawable exported from `data.rpf` by `rpf9dump models`: one mesh, 4,213 vertices and
7,968 triangles, with positions and UVs and no bones or fragments. Its OBJ is the only `data.rpf` export with
those counts and has SHA-256
`a7178cf4b7a00c9df838fb072c59e2c048ceed52306620d859faae4d3991d6b9`; `manifest.json` records it. The tool that
assembles it, `tools/first_light.py`, centres and fits X/Y to NDC and flattens Z (the chosen VS applies no
matrix), so the draw is an orthographic projection of the model's own geometry.

The untextured fixture uses two containers from `shaders.rpf`, carved and translated by the repository's tools.

| stage | FSSM SHA-256 | what it does |
|---|---|---|
| VS | `128867d3233a97ba5b203c313a24ab8a77df20c2a308091a5595bb9f669677f4` | forwards POSITION and TEXCOORD from the vertex fetch to clip space and a varying |
| PS | `ed4147a5f7514947c2eb8d79255bd80f72761ac1ba2205dd811d3e050cf1b495` | writes the interpolated TEXCOORD as the colour; no bindings, no texture |

The model's authored shader assignment can be checked separately. In the model export manifest for the OBJ
above, `MeshShaders[0]` selects material slot 0, the only slot. Looking up its `Materials[0].Shader` in the
native shader database yields one effect with one technique and four compute-only passes. The VS and PS
in the table are the fixture's chosen pair. The model's use during gameplay remains unverified.
See the [effect lookup tool](../tools/sgd5_effects.py) and the
[Windows reproduction of the UV draw](https://github.com/ItsNotRexed/Leonida/pull/2893#issuecomment-7084947193).

Both disassemble with no invalid word, translate, and pass `spirv-val`; `tools/shader_link_check.py check` reports
the pair compatible. At 128×128 the draw covers **12,804 of 16,384 pixels (0.781)** and the centre pixel is
`(0.498, 0.496, 0.0, 1.0)`: the model is the disc its geometry projects to, and the colour is its own UV map. The
test derives the expected centre UV from the mesh on the CPU (`0.4978, 0.4965`), so the GPU pixel has to match
the geometry, not just be non-clear; it also requires the `[0, 0]` corner to keep the clear value and bounds the
coverage at 0.5–0.99, which an all-opaque image cannot pass. Four extra probes where U and V differ by more than
0.15 catch a swapped-U/V image even when the centre is symmetric (this model's centre has U ≈ V). The
`OracleTests` run those checks on synthetic images without a GPU or a package: the accepted fixture, an
all-opaque image, a swapped UV and a short buffer. The same `check_image` function backs the tool's `--check`, so
the command line and the test agree. The PPM the harness writes is a real image of the model; it stays in the
user's folder.

## 2. Why it matters

Until now the D3D12 draw tests (M1, M1.5, M2, M3) were met with hand-written RDNA2 or GLSL shaders and synthetic
geometry, and the pipeline check compiles the game's 26,503 shaders without ever drawing with one
([pipeline_check.md](pipeline_check.md)). This closes the "shaders of the game itself" gap for the native Vulkan
path: a real model, a real translated VS/PS pair, a vertex buffer, a pipeline and a readback, end to end, on a
consumer GPU. It does not need the executable and it does not use a synthetic stand-in for either side.

The pieces it proves are the ones the D3D12 layer also relies on: ISG register → Vulkan vertex input location,
the 144-byte push-constant block, the translator's SPIR-V for a graphics pair, and `vk_harness`'s device,
pipeline and readback code. The same pair also draws through the port's D3D12 layer (section 5).

## 3. Reproduce it

```sh
git submodule update --init
dotnet build tools/rpf9dump/rpf9dump.csproj -c Release
DOTNET_ROLL_FORWARD=Major tools/rpf9dump/bin/Release/net8.0/rpf9dump models \
    "<package>/data.rpf" ~/leonida-work/assets

python -I tools/first_light.py --game "<package>" --work ~/leonida-work --list
python -I tools/first_light.py --game "<package>" --work ~/leonida-work \
    --obj ~/leonida-work/assets/data/models/<model>.obj --no-fit --run
# <model>.obj: 4,213 vertices, 7,968 triangles; the manifest records its SHA-256
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out python -I tests/test_first_light.py -v
python -I tools/first_light.py --check --out ~/leonida-work/first_light/out
```

`--check` on an existing folder re-runs the harness through the same fresh path and `check_image` as the test; a
harness that exits 0 without writing, an all-opaque image and a swapped-U/V image all exit 1. The test has
twenty-five cases: the two package-gated draw/manifest checks, the `--check` command, four oracle fixtures, the
mesh depth fit, a preview with non-finite readback channels, the stage files of two shaders with the same name, two
OBJ parsing checks, seven checks of the textured path (section 6) and six CPU checks of `--camera` (section 7).

`first_light.py` carves `shaders.rpf` if the work folder does not have it, translates only the chosen pair, converts
the OBJ to the harness's mesh file (`u32 vertex_count, u32 index_count`, `(x,y,z,u,v) f32` vertices, `u32`
indices), writes `manifest.json` with the counts and FSSM hashes, and runs `vk_harness`. `--vs` and `--ps` take
other names from the carve index; `--no-fit` keeps a model that is already in NDC (the sky dome is). The test is
package-gated: without `LEONIDA_FIRST_LIGHT_DIR` it skips, and it writes its image to that folder.

Each run stages its pair in `<work>/first_light/dxbc` and translates it into `<work>/first_light/spirv`. Both folders
are removed at the start of a run, so they hold that run's two shaders and nothing an earlier run selected; do not
keep files there. The output folder (`first_light/out` or `--out`) and `<work>/shaders` are not emptied. If either
folder is a symlink or a junction, or a file in it cannot be removed, the run stops and names it.
With `--keep-z` or `--no-fit`, a triangle can cross the near or far clip plane. For the
passthrough VS, the CPU predicts a UV only where the interpolated depth is between 0 and 1,
inclusive. If several triangles cover a probe there, the last one drawn supplies its UV.

## 4. The harness mode

`runtime/test/vk_harness.cpp` gained two environment options, inert when unset (the existing three-vertex and
mesh-shader paths are unchanged): `VKH_DRAW_MESH=<file>` with `VKH_VERTEX_LAYOUT=0:3f,1:2f` reads the binary mesh,
creates a vertex and an index buffer, and draws it with `vkCmdDrawIndexed`. The layout lists `location:components`
and only float attributes. The first-light pair above uses no other harness option: the VS reads no user SGPRs and
the PS has no bindings.

## 5. Through the D3D12 layer

`runtime/test/d3d12_first_light_test.cpp` (ctest `d3d12_first_light`) draws the same model and pair through the
Xbox-ABI device proxy over the Vulkan device, natively on Windows. `CreateGraphicsPipelineState` gets the game's
own containers and finds both stages by the FSSM hash inside them, in the index `first_light.py` writes next to
the SPIR-V (`vs.dxbc`, `ps.dxbc`, `index.json`); the input layout (POSITION float3, TEXCOORD float2) is matched
to the VS signature, the mesh goes through DEFAULT vertex and index buffers, and one `DrawIndexedInstanced` draws
it into a cleared R32G32B32A32 target read back with `CopyTextureRegion`.

D3D12 clip space is y-up and the layer flips the viewport, so `vk_harness` row r is row 127 − r here. On an RTX
3060 (Windows 11, driver 616.64, Vulkan SDK 1.4.363.1) the draw covers the same **12,804 of 16,384 pixels**, the
centre pixel is `(0.49756, 0.49634, 0.0, 1.0)` and the corner keeps the clear (−1). Against `vk_harness`'s
`image.raw`, 16,383 pixels are bit-exact and one is one f16 step apart (0.74219 here, 0.74170 there): the PS
exports with `v_cvt_pkrtz_f16_f32`, which rounds toward zero, so an f32 interpolant a few ulps lower under the
opposite viewport direction drops one step. `vk_harness` drawing the mesh mirrored in y reproduces the D3D12
image bit for bit. The test allows that one step and nothing else; it is clean under `LEONIDA_VK_VALIDATION=1`.

```sh
python -I tools/first_light.py ... --no-fit --run          # as in section 3; writes vs.dxbc, ps.dxbc, index.json
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out ctest --test-dir <build> -R d3d12_first_light -V
```

Without `LEONIDA_FIRST_LIGHT_DIR` the test prints SKIP and exits 77.

## 6. With a texture (`TEST-524`)

Passing `--dds <exported.dds>` to `tools/first_light.py` switches the default pixel shader to a translated
single-sample game pixel shader (`PS_Copy_3ea4b38a_Wrapped`, FSSM SHA-256
`19e2e7612505b9d3b19e6607b47e09dcdde02a2c9987a63e834d9586f217090f`, wave32, `inputs=[0]`, `outputs=[0]`,
`shader_link_check.py check`: compatible with `VS_main_170e32e6_Wrapped`). `first_light.py` decodes the DDS
(for example a BCPack-decoded TTPF albedo) with `tools/dds_png.py`, writes `texture.rgba` (`u32 width, u32 height` +
`R8G8B8A8_UNORM` texels) into the output folder, and passes `VKH_TEX_RGBA`, `VKH_TSHARP_VA` (`0x100`),
`VKH_SSHARP` (the shader's literal sampler `0x92, 0xc00000, 0x14500000, 0`, bilinear + clamp-to-edge) and
`VKH_SGPRS` (`s9 = 0x100`) to `vk_harness`.

`vk_harness` enables `bufferDeviceAddress`, `runtimeDescriptorArray` and `descriptorIndexing` only when
`VKH_TEX_RGBA` is set (inert otherwise), writes the 8-dword `T#` at `VKH_TSHARP_VA` in the 64 KiB heap, binds
set 0 (`binding 0` 2D images with a 1×1 magenta fallback at index 0 and the decoded texture at index 1; `binding 1`
samplers with the decoded `S#` at index 1; `binding 2` the `DescriptorTable` hash buffer), and draws the mesh.

The CPU oracle (`check_image`) samples the decoded `R8G8B8A8_UNORM` texture bilinearly (clamp-to-edge) at
`centre_uv` and each `uv_probes` entry and requires the GPU readback to match within `0.02` (rejecting the
magenta unbound-texture fallback and wrong-colour probes). Verified on Windows 11 with an NVIDIA GeForce RTX 3060
(driver 621.24, Vulkan 1.4.357): **12,804/16,384 (0.781)** covered. Without `LEONIDA_FIRST_LIGHT_DIR`,
`python -I tests/test_first_light.py -v` runs **46 tests: 42 passed, 4 skipped** (the 3 package-gated
`FirstLightTests` require `LEONIDA_FIRST_LIGHT_DIR`, and `test_harness_rejects_wrapped_tsharp_va_overflow` needs a
built `vk_harness`; the 4 `OracleTests` and the other 11 `TexturedOracleTests` run without a package). With
`LEONIDA_FIRST_LIGHT_DIR` set against a local package run (and `vk_harness` built) none of the 46 is skipped.

```sh
python -I tools/first_light.py --game "<package>" --work ~/leonida-work \
    --obj ~/leonida-work/assets/data/models/<model>.obj \
    --dds ~/leonida-work/assets/ttpf/<pool>/<texture>.dds --no-fit --run
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out python -I tests/test_first_light.py -v
```

## 7. A camera (TEST-523)

`first_light.py --camera` draws the same model in perspective with a game VS that applies a world matrix and a
view-projection matrix of its own, read from two constant buffers. The PS is the one from section 1:

| stage | FSSM SHA-256 | what it does |
|---|---|---|
| VS | `e909a9e5a0096e00bc2b621ae3bdc96ac904f3047aed3917835774cbccb09716` | POSITION × world × view-projection to clip space (computed w); TEXCOORD to a varying |
| PS | `ed4147a5f7514947c2eb8d79255bd80f72761ac1ba2205dd811d3e050cf1b495` | as in section 1 |

The VS's scalar loads give its constant layout. The 32-bit address of a descriptor table is in s9 (the high half is
0). The V# at table + 0x20 points at the world rows: X, Y, Z and translation, one float4 each at 0x00–0x30, of
which the VS reads xyz. The V# at table + 0x60 points at a constant buffer whose view-projection rows sit at
0x160–0x19f. Both matrices multiply a row vector: `world = [x y z 1] W`, `clip = [world 1] VP`. The view-projection
block is common in the game: on 1.0.2.1, all 9 translated VS modules with exactly POSITION and TEXCOORD inputs and
a clip w that is not the constant 1.0 read 16 dwords at 0x160 of a constant buffer, as do 123 of the 254 VS
modules with a POSITION input and a computed w, and 1,266 of all 2,236 VS modules. That it is the view-projection
everywhere is a hypothesis; for this VS the arithmetic shows it.

The mesh keeps the model's own coordinates. The tool builds a world matrix that centres the model and scales its
bounding sphere to radius 1. The camera sits 3 radii away, `--yaw` degrees around Z (the game's up axis) from −Y
and `--pitch` degrees above the XY plane, with a vertical `--fov` (defaults 30, 25 and 50). The projection is
Vulkan's, with the near and far planes just outside the sphere, so depth clips no vertex. The tool writes the two
matrices, the two V#s (raw buffers, stride 0, sized to what the VS reads) and the table into `vk_harness`'s heap
(`VKH_HEAP`) and the table address into s9 (`VKH_SGPRS`). The manifest's `camera` block keeps both, so `--check`
and the test draw with the same constants. The prediction projects the same vertices with the same float32
matrices on the CPU and interpolates UV perspective-correctly, dividing each screen-space weight by its vertex's
clip w. The output goes to `<work>/first_light/camera`, leaving the orthographic folder for section 5.

On an Apple M4 (macOS, MoltenVK 1.4.2), with the section 1 model and the default camera at 128×128:

* The draw covers **6,851 of 16,384 pixels (0.418)**.
* The centre pixel matches the predicted UV `(0.8100, 0.3096)`, and so do the four U ≠ V probes, within the
  oracle's 2e-3. The corner keeps the clear value.
* On a full 32×32 grid of pixel centres, the predicted coverage agrees on all 1,024. The UV agrees within 2e-3 on
  all 428 covered ones, with the worst at 5.05e-4.
* The same draw without the constants covers nothing and is rejected. With the camera turned 2° in yaw, the
  centre pixel is off and the check is rejected too.
* The pair passes `spirv-val`, and `shader_link_check.py check` reports it compatible.

The model's triangles are small, so perspective-correct and affine UV differ by at most 1e-4 at the probes. The
image therefore does not tell the two apart; `CameraTests` in `tests/test_first_light.py` checks the
perspective-correct weights on one large triangle instead. Those CPU tests also read the constants back the way
the VS's loads do (table, V#, num_records, rows) and compare the result with the matrices the prediction uses.

```sh
python -I tools/first_light.py --game "<package>" --work ~/leonida-work \
    --obj ~/leonida-work/assets/data/models/<model>.obj --camera --run   # coverage window 0.2-0.99
LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/camera python -I tests/test_first_light.py -v
```

## 8. Limits and next

* The camera draw (section 7) runs in `vk_harness` only. Through the D3D12 layer, s9 and the table it points at
  have to come from root arguments, which `d3d12_first_light_test` does not set up yet.
* `--camera` and `--dds` don't combine: the camera VS's table and the textured PS's T# both take s9 of the shared
  push constants and heap VA 0x100, so `first_light.py` refuses the pair. A textured perspective draw needs a
  second constant layout.
* One model and one draw. The `levels.rpf` windows have world props with POSITION/NORMAL/COLOUR/UV layouts that
  match the `sga_dr_*`/`Rage_Transform*` families by signature; pairing one of those is the natural next image.
* The game still cannot boot: the executable is not in the package (RT-5), so no call site selects these programs.
