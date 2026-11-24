# Multi-GPU validation on identical cards

A small box with several identical GPUs is useful for **parallel sharding**, not for inventing several independent
hardware results. This page records the setup used for the eight-card RDNA1/RADV farm and the rules that kept its
numbers meaningful. Counts and pass/fail lines only: no shader corpus, game files, keys or private paths belong
here or in any report generated from it.

## Hardware used

Eight AMD Radeon Pro V540 cards, Navi 12 / gfx1011 (RDNA1, PCI `1002:7360`, subsystem `1002:1a34`, VBIOS
`113-D3020100-103`), on bare metal (no VM passthrough), each bound to `amdgpu` and reachable through a render node.
RADV Mesa 25.2.8, Vulkan device 1.4.318, loader 1.3.275, subgroup range 32..64. Ubuntu 24.04.4, kernel
7.0.0-31-generic, GCC 13.3.0, Python 3.12.3, CMake 3.28.3, llvm-mc 21.1.8 with the AMDGPU target, SPIRV-Tools
2026.1 (2025.1 is too old for the `FMAKHR` capability used by `LEONIDA_FMA=khr`).

| PCI BDF | Render node | Shard label |
|---|---|---|
| `0000:1c:00.0` | `renderD128` | gpu0 |
| `0000:1f:00.0` | `renderD129` | gpu1 |
| `0000:55:00.0` | `renderD130` | gpu2 |
| `0000:58:00.0` | `renderD131` | gpu3 |
| `0000:8e:00.0` | `renderD132` | gpu4 |
| `0000:91:00.0` | `renderD133` | gpu5 |
| `0000:c7:00.0` | `renderD134` | gpu6 |
| `0000:ca:00.0` | `renderD135` | gpu7 |

## Pinning one device per process

Use the PCI BDF, never an index, and exclude software drivers explicitly:

```sh
lspci -Dn -d 1002:7360
DRI_PRIME='pci-0000_1c:00.0!' VK_LOADER_DRIVERS_SELECT='*radeon*' vulkaninfo --summary
```

For every BDF in the table, that check must show **exactly one** RADV NAVI12 device. If it shows none or several,
stop and report the blocker; do not substitute llvmpipe. The ICD file on this box is `radeon_icd.json`, so
`VK_LOADER_DRIVERS_SELECT='*radv*'` matches nothing and makes `vkCreateInstance` fail with `-9`.

## Shard discipline

- Split work into **disjoint** units: different seeds for the fuzzers, different `--start`/`--count` ranges for a
  translated index, or different test names. Eight processes running the same full suite are eight copies of one
  result, not a validation farm.
- Run one process per card, keep logs per card, and aggregate counts afterwards.
- Each PR verification used a clean worktree at the PR head and, where a base was named, a second worktree at that
  base with its own freshly built programs. Mixing a script from one commit with binaries from another produces
  false failures (see "Known problems" in [testing.md](testing.md)).
- When a shard fails, rerun the *same* shard on two or three other cards. Failing everywhere points to a driver or
  corpus issue; failing on one card points to that card. Re-run single failing tests in isolation before calling a
  result reproducible.
- A transient single-card anomaly must be reported as such. On this box `0000:ca:00.0` intermittently produced a
  few mismatching pixels in different fragment tests on different full-suite runs, while each failing test passed
  3/3 in isolation and the other seven cards were clean. It was never counted as a PR regression.

## Results already collected

These are aggregate results from this box; the full per-command counts are in the linked #166 entries, not in this
page. No game-derived data was used for any of them.

- Single-card RDNA1/RADV baseline suite and TEST-71 fuzz matrix: [#166 entry](https://github.com/ItsNotRexed/Leonida/issues/166#issuecomment-7078373329).
- Eight-card fallback matrix and extended fuzz campaign (seeds 4–11, 200 programs, both waves): [#166 entry](https://github.com/ItsNotRexed/Leonida/issues/166#issuecomment-7079130604).
- Native MUBUF `OOB_SELECT` measurement on gfx1011 with `tools/hw_oracle` and `tools/buffer_bounds_probe.py`: 12 cases
  at wave32 and wave64 (24 groups), all following the measured RDNA2/GFX11 column, 8 disagreements with Mesa's
  GFX10 column, 0 wave32/wave64 differences (SHADER-341; results in [buffer_bounds.md](buffer_bounds.md)).

Two failures recur on these cards at the tested commits and were confirmed on clean base worktrees, so they are
driver-version limitations rather than regressions from the PRs tested here: `cs_w32_salu64` (wrong high word of
an `s_bfe_i64` field) and `cs_signed_zero_nan`. The latter is the signed-zero class tracked in #1203. A real
regression was found on PR #2715: two wave64 pixel fixtures returned the clear value on all three cards tested while
the PR's base passed them.

## Using a translated corpus

If a private translated shader directory containing `index.json` and `stubs/` is available, shard the *sorted* index
with `vk_pipeline_check --start`/`--count` and one pipeline cache per shard. Record ok/skipped/failed by stage and
generic error category only: never post shader names, contents or private paths. RDNA1 legitimately skips mesh,
barycentric and maintenance8 shaders; report those separately from passes. If no such directory exists, do not run
this phase and do not report pipeline counts as if you had.
