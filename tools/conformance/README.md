# Conformance harness: does the Vulkan-backed D3D12 device render what D3D12 renders?

The shim in `runtime/` reimplements D3D12 on Vulkan. Its device, resource, view, sampler,
copy, clear and barrier paths are covered by `runtime/test/`, but nothing today checks the
thing that matters most: **that a draw produces the same pixels as the host's D3D12.**

This harness defines a small set of deterministic scenes, renders them with native D3D12 on
the same machine, and checks every pixel that the scene claims. When the shim's draw path
lands, the same scenes go through it and the two image sets are diffed.

## Layout

```
tools/conformance/
  spec/scenes.h                  the contract: scene kinds, geometry, state, expected pixels
  spec/shaders/*.hlsl            the two shader pairs, compiled by dxc
  native/                        native D3D12 reference renderer (Windows, D3D12 + WIC)
  compare.py                     check a render against the contract, diff two renders
  COMPARE.md                     compare.py's modes, exit codes and limits
```

## The scenes

Seven scenes, each chosen to make one failure visible at an exact coordinate:

| scene | proves |
|---|---|
| `clear` | render target creation, clear value, readback |
| `fullscreen` | vertex shader runs, pipeline binds, one triangle covers the target |
| `depth_order` | depth test rejects a later draw that is behind (green survives over red only because of it) |
| `scissor` | scissor clips at `{48,32,176,96}`, right/bottom exclusive |
| `blend` | 50% source over the clear colour, `SRC_ALPHA`/`INV_SRC_ALPHA` |
| `indexed` | the index buffer is read; vertices are stored out of draw order |

The contract states the baseline pipeline state every scene inherits (cull NONE,
`TRIANGLELIST`, sample mask, viewport, clip-space z in 0..1 with `w = 1`, row 0 is the top).
Anything an implementer would otherwise have to guess is written down, because two guesses
that disagree produce two different images and neither is wrong on its own terms.

## Run it

```powershell
# 1. build and render the reference images
powershell -NoProfile -ExecutionPolicy Bypass -File tools\conformance\native\scripts\build.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\conformance\native\scripts\run.ps1

# 2. check every declared pixel
python tools\conformance\compare.py expect --manifest tools\conformance\native\out\run1\manifest.json

# 3. prove the scenes are not vacuous: corrupt three pieces of state and require three failures
powershell -NoProfile -ExecutionPolicy Bypass -File tools\conformance\native\scripts\negative-control.ps1
```

Requirements: Windows with the D3D12 runtime, Visual Studio 2022 (C++ x64), Windows SDK
10.0.26100 (`dxc.exe`), Python 3.10+ for `compare.py` (standard library only; no pip
installs). The renderer needs a D3D12 device with feature level 11_0 — it runs on the
Microsoft Basic Render Driver as well as on hardware.

## Status today

Against the shim as of `2e5c0f2`, the driver in the pull request reports:

```
clear          OK       3/3 pixel checks matched (row pitch 1024, readback 262144 bytes)
fullscreen     BLOCKED  ID3D12Device::CreateRootSignature (E_NOTIMPL 0x80004001)
depth_order    BLOCKED  ID3D12Device::CreateRootSignature (E_NOTIMPL 0x80004001)
scissor        BLOCKED  ID3D12Device::CreateRootSignature (E_NOTIMPL 0x80004001)
blend          BLOCKED  ID3D12Device::CreateRootSignature (E_NOTIMPL 0x80004001)
indexed        BLOCKED  ID3D12Device::CreateRootSignature (E_NOTIMPL 0x80004001)
```

`clear` already proves the whole resource → clear → barrier → copy → readback chain through
the Vulkan device. The five drawn scenes stop at the first call the shim genuinely lacks, so
each one becomes a progress marker as the draw path lands: `CreateRootSignature` →
`CreateGraphicsPipelineState` → the `SetPipelineState` / root-binding group → viewport and
scissor → vertex/index buffers → `OMSetRenderTargets` → `DrawInstanced` /
`DrawIndexedInstanced`.

## The shader format question

The reference renderer compiles HLSL to **DXIL** because native D3D12 requires it. The
shim's documented contract is **SPIR-V**, because the game's shaders reach it from the RDNA2
translator. A PSO implementation that understands only one of them cannot serve both.

The cheapest resolution, when the draw path lands: compile each conformance shader twice
(`dxc -T vs_6_6` and `dxc -spirv -T vs_6_6`) and let each renderer take the blob it wants.
One workload, one set of expected pixels, no DXIL translator in the shim.

## Why the scenes are trustworthy

Three independent checks, all reproducible:

1. **The renderer verifies itself.** It reads every declared pixel back from the GPU *and*
   from the PNG decoded off disk, prints one line per check, and exits non-zero on any
   mismatch. 18 checks, 18 passes, on both the NVIDIA and AMD adapters.
2. **The negative control asserts its own result.** It disables the depth test, the blend
   state and the scissor rect in a throwaway copy, then requires exactly `depth_order`,
   `blend` and `scissor` to fail — each at the pixel that names the state. If a
   `String.Replace` silently stops matching, the script throws instead of reporting success.
3. **Determinism is measured**, not assumed: two runs produce byte-identical PNGs.

## Extending it

Add a `Scene` to `spec/scenes.h` with its expected pixels. The renderer picks it up through
`SceneKind`; `/we4062` makes an unhandled enumerator a build error, so a new kind cannot
silently render as a clear-only scene. Add a matching row to the shim driver's switch when
the draw path exists.
