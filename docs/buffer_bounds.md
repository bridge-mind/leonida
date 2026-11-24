# Buffer bounds: V# `OOB_SELECT`, measured on hardware

Which buffer accesses the hardware treats as out of range, per the V# `OOB_SELECT` field (word 3 bits [29:28]),
measured natively on AMD GPUs (SHADER-116). The translator reads this field and picks the bounds check by mode
(SHADER-110); the contract row is "Buffer bounds checking" in
[design.md](design.md). Out of range, a load returns 0 and a store or atomic does not touch memory.

## Measured rules

`N` is `NUM_RECORDS` (V# word 2), `index` the VGPR index (`idxen`), `offset` the VGPR byte offset (`offen`) and
`soffset` the scalar offset. Every access below is a 4-byte `buffer_load_dword` / `buffer_store_dword`.

| `OOB_SELECT` | Out of range on RDNA2 (gfx1036) when | Mesa GFX10 column | Mesa GFX11 column |
|---|---|---|---|
| 0 structured with offset | `index >= N` or `offset + 4 > stride` | `offset >= stride`: differs | same |
| 1 structured | `index >= N` | same | same |
| 2 disabled | `N == 0` | same | same |
| 3 raw | `offset + soffset + 4 > N`; `index * stride` moves the address but is not checked | `offset >= N`: differs | `offset + 4 > N`, without `soffset` |

- **RDNA2 follows Mesa's GFX11 column**, not the GFX10 one: of 732 cases, 729 match GFX11 and 696 match GFX10.
  Every GFX10 miss is an access that only partly fits, for example a raw buffer with `N = 3` read at offset 0.
- **In raw mode `soffset` is part of the checked offset.** The 3 cases that match neither column are raw reads at
  offset 8 with `soffset` 8 and `N = 18`: 8 + 8 + 4 > 18, out of range on hardware.
- Loads and stores agree in every case.
- In-range accesses at a dword-aligned address read and wrote the expected dword (0 exceptions). An address that is
  not a multiple of 4 (stride 18 with `index > 0`) is rounded by the hardware; where such an access lands was not
  part of this measurement.
- @AlxcBTW measured RDNA3.5 (Radeon 880M, gfx1150) through the HIP runtime and found it matches the GFX11 column
  ([finding](https://github.com/ItsNotRexed/Leonida/issues/374#issuecomment-7072793165)). The RDNA2 result here
  agrees with it, plus the `soffset` rule.

**RDNA1 measurement (SHADER-341):** the console target is gfx1013 (GFX10.1); gfx1036 is GFX10.3, and the
RDNA2 measurement above shows Mesa's GFX10 column is wrong for GFX10.3. A second native measurement through
`tools/hw_oracle` (Linux libdrm launcher, AMD clang 18) on an **AMD Radeon Pro V540, Navi 12 / gfx1011, RDNA1**
(`1002:7360`, bare metal, RADV Mesa 25.2.8, Ubuntu 24.04) found the same: `tools/buffer_bounds_probe.py` ran 12
one-lane cases for a 4-byte `buffer_load_dword` at wave32 and wave64 (24 groups, 0 wave32/64 differences, 0
launcher errors). Every case followed the RDNA2 / Mesa GFX11 column and **8 of 24 disagreed with Mesa's GFX10
column**. The witnesses include mode 0 stride 16 at offset 13 (out of range on hardware, in range by the GFX10
column), mode 3 `NUM_RECORDS` 18 at offset 16 (out of range), mode 3 `NUM_RECORDS` 3 at offset 0 (out of range),
and mode 3 offset 8 with `soffset` 8 and `NUM_RECORDS` 18 (out of range, confirming that `soffset` is part of the
RAW check). Swizzled V#s and unaligned addresses were not covered. This makes GFX10.1 follow GFX10.3 here in the
cases tried, but gfx1013 itself is still unmeasured: the card used was gfx1011 (Navi 12), not the console part.

**Open:** measure the same table on gfx1013 (PS5) itself when a card with that ID is available, and extend the
probe to swizzled V#s and unaligned accesses.

## How it was measured

Ryzen 7 9800X3D integrated GPU (gfx1036, RDNA2), AMD driver 3679.0 (PAL, LC), Windows 11. A standalone OpenCL
host (not in this repository) loads the driver's `OpenCL.dll`, and each case is a one-lane kernel whose inline
assembly runs a single `buffer_load_dword` or `buffer_store_dword` with a V# built by hand:

- word 0–1: the buffer's address, `STRIDE` in word 1 [29:16];
- word 2: `NUM_RECORDS`;
- word 3: `DST_SEL` x/y/z/w, `FORMAT` = GFX10 `32_FLOAT` (22), `RESOURCE_LEVEL` = 1, `OOB_SELECT` = 0–3.

The buffer holds a pattern (dword `i` = `0xA0000000 | i`), so a non-zero load names the dword it read, and a store
writes a marker whose position is found by comparing the whole buffer afterwards. The cases are modes 0–3 × stride
0, 16, 18 × `N` 0, 3, 18, 48 × `offen` (offsets 0, 12, 16, 44, 48), `idxen` (indices 0, 2, 3, 4) and
`idxen offen` (six index/offset pairs), plus raw cases with `soffset` 8: 732 in all, each a load and a store.
Three runs gave identical results. Each case was compared with both Mesa columns as listed in @AlxcBTW's finding
(Mesa `src/amd/common/ac_descriptors.c`).
