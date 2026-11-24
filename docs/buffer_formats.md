# Typed buffer formats: every GFX10 code a V# or MTBUF can name

What `buffer_load_format_*`, `buffer_store_format_*` and `tbuffer_load_format_*` do for each GFX10 buffer data format
(V# word 3 `FORMAT` [18:12], or the `format:` modifier of an MTBUF instruction), measured on a Radeon 880M and on a real RDNA2
(gfx1036) and held against the translator. The contract row is "Typed buffer formats" in [design.md](design.md); the code is
`Gen.FORMATS`, `Gen.decode_format` and `Gen.encode_format` in `tools/rdna2spirv.py`; the oracle is
`tools/buffer_format_fuzz.py`.

**Where it was measured.** Locally only on the Radeon 880M (gfx1150, RDNA3.5, GFX11), which numbers its formats differently
(8_8_8_8_UNORM is 56 on GFX10 and 42 on GFX11), so every format there is mapped by name (`GFX11_CODES` in the tool, taken from
what the AMD assembler encodes for `format:[BUF_FMT_<name>]`). @AwkGrep256 then ran the same tool on a real **RDNA2, gfx1036**
(Ryzen 7 9800X3D iGPU, Linux DRM backend of #1265, native GFX10 numbers; see the comment on
[#2022](https://github.com/ItsNotRexed/Leonida/pull/2022)): 1,330,469 elements of the 63 formats GFX11 has with 0 mismatches,
and 199,144 elements of the 14 codes only GFX10 has with 0 mismatches. The 880M and gfx1036 agree on everything below. The
console's own gfx1013 (GFX10.1) is the same generation but not the same chip, and was not measured; nor were the d16 forms.

## Which codes there are

GFX10's `FORMAT` is 7 bits, so a buffer can carry codes 1..77 (the table in `tools/gfx10_formats.json` goes to 292
because images use 9 bits; 128 and up are image-only: sRGB, BC, depth, 5_6_5, 1_5_5_5, 5_5_5_1, 4_4_4_4, 5_9_9_9, 6E4).
There are no 5_6_5-style formats for buffers. **All 77 are in `Gen.FORMATS`**: every f32/u32/i32 code, the 8/16-bit and
8_8/16_16/8_8_8_8/16_16_16_16 families in all of UNORM, SNORM, USCALED, SSCALED, UINT, SINT (and FLOAT for 16-bit), the
10_11_11 (30..36) and 11_11_10 (37..43) families (only the FLOAT ones exist on GFX11) and the 10_10_10_2 (44..49) and
2_10_10_10 (50..55) families in all six numeric formats. Codes outside the table (0, 78..127) decode to 0 and store nothing.

The 14 codes GFX11 lacks (30..35, 37..42, 46, 47) could not be measured on the 880M; the layout and conversions of the
model were validated on gfx1036. The GFX11-only look of a code number is never relied on: the translator reads the GFX10
numbering, as the console does.

## Names and bit order

Names list the component widths from the top bit down, so the **last** number is `x` in bit 0: `2_10_10_10` (50) is
R10G10B10A2 with x in bits 9:0 and a 2-bit alpha on top, `10_10_10_2` (44) has a 2-bit x. `10_11_11_FLOAT` (36) is x =
11 bits [10:0], y = 11 bits [21:11], z = 10 bits [31:22]; `11_11_10_FLOAT` (43) is x = 10 bits, y and z 11 bits each.
Both GPUs confirmed all of these.

## Loads (measured)

Every result word of an `xyzw` load of all formats the GPU has, through a V# (`DST_SEL` = XYZW) and through tbuffer's own format
(with a V# that names 32_FLOAT), equals the model below on both GPUs. Per format: every value of every field up to 12 bits (and all
65,536 half floats), edge values (0, 1, max, signed min and its neighbours), random values above that, and random bytes
past the element.

| Format class | Result |
|---|---|
| UNORM | correctly rounded f32 of x / (2ⁿ−1) |
| SNORM | correctly rounded x / (2ⁿ⁻¹−1); the most negative code is −1.0 |
| USCALED / SSCALED | the integer, as a float, **not** divided (exact: at most 16 bits) |
| UINT / SINT | the zero-extended / sign-extended integer, in the integer register |
| 16_FLOAT | the half as f32; a NaN keeps its sign and payload (`0x7C01` gives `0x7F802000`, **not quieted**) |
| 10_11_11 / 11_11_10 FLOAT | unsigned float, 5 exponent bits, bias 15, 6 (11-bit field) or 5 (10-bit field) mantissa bits, no sign. Exponent 0: m·2⁻²⁰ (m·2⁻¹⁹ for 5 bits), a denormal that is **kept**. Exponent 31: infinity, or a NaN whose payload is the mantissa shifted up (`0x7C1` gives `0x7F820000`), unquieted |
| channels the format lacks | through a V#: component *c* mod *n* repeats (the translator's existing rule, SHADER-86, now also measured on gfx1150); through tbuffer: 0 (the instruction names `X000`, `XY00`, `XYZ0` or `XYZW` by the format's component count, and the V# `DST_SEL` is ignored) |

## Stores (measured)

`buffer_store_format_xyzw` and `_xy` (and the tbuffer forms) of every format into a buffer pre-filled with `0xDEAD0000`, the
whole 16 bytes compared, so a byte written outside the element shows up.

| Format class | Result |
|---|---|
| UINT | saturates to the field width (min(x, 2ⁿ−1)) |
| SINT | saturates to the signed range of the field |
| UNORM | clamp to [0, 1], then the **exact** product x·(2ⁿ−1) rounded to nearest even; a NaN stores 0 |
| SNORM | clamp to [−1, 1], then the exact product x·(2ⁿ⁻¹−1) rounded to nearest even, **except the 2-bit field** (10_10_10_2 and 2_10_10_10 SNORM), where a half rounds away from zero (0.5 gives 1, −0.5 gives −1; nearest even would give 0) |
| **USCALED / SSCALED** | **converted as UNORM / SNORM**: storing 200.0 into 8_USCALED writes 255, storing 0.5 into 8_SSCALED writes 64. The SCALED formats are for loads; a store does not undo the "not divided" |
| 16_FLOAT, 10_11_11 / 11_11_10 FLOAT | **round toward zero** (not nearest): 0.49999997 gives `0x37FF`, not `0x3800`. A value too large for the format gives the **largest finite** value, not infinity: 65520.0 and 1e30 give `0x7BFF` for halves; only a real +infinity stores infinity. Denormals are produced. The unsigned floats store 0 for every negative number, −0 and −infinity |
| NaN into a float format | stays a NaN: the sign (halves only) and the top mantissa bits of the payload, or 1 when those are all zero (`0x7F800001` gives `0x7C01`) |
| a format with two components | **`buffer_store_format_*` through a V# with the identity selectors (XYZW) takes channels (x, w), not (x, y)**: `8_8_UINT` of (0x11, 0x22, 0x33, 0x44) writes 0x11, 0x44, and `buffer_store_format_xy` into it writes 0x11 and 0 (w is not supplied). **With the selectors the runtime gives a two-component format (X, Y, 0, 1, `gfx10_descriptors.h`) it takes (x, y)**, as the narrow-integer fixtures measured on gfx1036 (`test_buffer_format_narrow_int.py`). On the 880M the second component is y exactly when DST_SEL_Y is Y and DST_SEL_Z is a constant (XY01, XY0W, XY00), and w for the other thirteen selector sets tried (XYZW, XYZ1, XYZ0, X001, X0ZW, ...). **`tbuffer_store_format_*` takes (x, y)** whatever the V# says. 1-, 3- and 4-component formats take x, xyz and xyzw (`_xy` into them: x, y, then 0). The translator follows this rule (`Gen.format_encode_call`); d16 forms and `_xyz` into a two-component format were not measured. Selectors with a swapped pair, e.g. YXZW, reorder the channels in ways that are not modelled |

`DST_SEL` also reorders a store's channels on the 880M in a way that is not a plain swizzle (`xzwy` stores `zxyw`); only
the identity and the B/R swap (`zyxw`, the 8_8_8_8 and 2_10_10_10 UNORM case the translator already handles) are modelled.

## What ran

`python -I tools/buffer_format_fuzz.py --oracle hw --budget 40 --seed 7` (loads and stores, V# and tbuffer, stores as `_xyzw` and
`_xy` with a different value in every register, and for the two-component formats once more with the runtime's X,Y,0,1 selectors):

- **Radeon 880M (gfx1150):** 1,390,400 validated elements, 0 mismatches (63 formats; the table below).
- **Radeon (gfx1036, RDNA2), by @AwkGrep256 on the older tool that counted the (x, w) routing apart:** 1,330,469 validated
  and 26,523 known-difference elements, 0 mismatches, i.e. 1,356,992 elements all equal to this model; and
  199,144 elements of the 14 GFX10-only codes (`--formats 30,31,32,33,34,35,37,38,39,40,41,42,46,47`), 0 mismatches.
- `--negative-control` shows the harness is not vacuous (a kernel whose V# names another format differs on all but one or
  two elements of five formats). `--self-test` checks the model without a GPU.

| Code | Format | Translator | V# loads | tbuffer loads | V# stores (xyzw and xy) | tbuffer stores (xyzw and xy) |
|---|---|---|---|---|---|---|
| 1 | `8_UNORM` | before | 338 | 338 | 3,024 | 3,024 |
| 2 | `8_SNORM` | before | 338 | 338 | 3,024 | 3,024 |
| 3 | `8_USCALED` | added | 338 | 338 | 1,864 | 1,864 |
| 4 | `8_SSCALED` | added | 338 | 338 | 1,864 | 1,864 |
| 5 | `8_UINT` | added | 338 | 338 | 1,688 | 1,688 |
| 6 | `8_SINT` | added | 338 | 338 | 1,688 | 1,688 |
| 7 | `16_UNORM` | before | 606 | 606 | 3,024 | 3,024 |
| 8 | `16_SNORM` | before | 606 | 606 | 3,024 | 3,024 |
| 9 | `16_USCALED` | added | 606 | 606 | 1,864 | 1,864 |
| 10 | `16_SSCALED` | added | 606 | 606 | 1,864 | 1,864 |
| 11 | `16_UINT` | added | 606 | 606 | 1,688 | 1,688 |
| 12 | `16_SINT` | added | 606 | 606 | 1,688 | 1,688 |
| 13 | `16_FLOAT` | before | 65,618 | 65,618 | 1,864 | 1,864 |
| 14 | `8_8_UNORM` | before | 594 | 594 | 3,024 | 3,024 |
| 15 | `8_8_SNORM` | before | 594 | 594 | 3,024 | 3,024 |
| 16 | `8_8_USCALED` | added | 594 | 594 | 1,864 | 1,864 |
| 17 | `8_8_SSCALED` | added | 594 | 594 | 1,864 | 1,864 |
| 18 | `8_8_UINT` | added | 594 | 594 | 1,688 | 1,688 |
| 19 | `8_8_SINT` | added | 594 | 594 | 1,688 | 1,688 |
| 20 | `32_UINT` | before | 357 | 357 | 1,688 | 1,688 |
| 21 | `32_SINT` | before | 357 | 357 | 1,688 | 1,688 |
| 22 | `32_FLOAT` | before | 357 | 357 | 1,864 | 1,864 |
| 23 | `16_16_UNORM` | before | 1,130 | 1,130 | 3,024 | 3,024 |
| 24 | `16_16_SNORM` | before | 1,130 | 1,130 | 3,024 | 3,024 |
| 25 | `16_16_USCALED` | added | 1,130 | 1,130 | 1,864 | 1,864 |
| 26 | `16_16_SSCALED` | added | 1,130 | 1,130 | 1,864 | 1,864 |
| 27 | `16_16_UINT` | before | 1,130 | 1,130 | 1,688 | 1,688 |
| 28 | `16_16_SINT` | added | 1,130 | 1,130 | 1,688 | 1,688 |
| 29 | `16_16_FLOAT` | before | 131,154 | 131,154 | 1,864 | 1,864 |
| 36 | `10_11_11_FLOAT` | added | 5,202 | 5,202 | 1,864 | 1,864 |
| 43 | `11_11_10_FLOAT` | added | 5,202 | 5,202 | 1,864 | 1,864 |
| 44 | `10_10_10_2_UNORM` | before | 3,158 | 3,158 | 3,224 | 3,224 |
| 45 | `10_10_10_2_SNORM` | added | 3,158 | 3,158 | 3,224 | 3,224 |
| 48 | `10_10_10_2_UINT` | added | 3,158 | 3,158 | 1,688 | 1,688 |
| 49 | `10_10_10_2_SINT` | added | 3,158 | 3,158 | 1,688 | 1,688 |
| 50 | `2_10_10_10_UNORM` | before | 3,158 | 3,158 | 3,224 | 3,224 |
| 51 | `2_10_10_10_SNORM` | added | 3,158 | 3,158 | 3,224 | 3,224 |
| 52 | `2_10_10_10_USCALED` | added | 3,158 | 3,158 | 1,864 | 1,864 |
| 53 | `2_10_10_10_SSCALED` | added | 3,158 | 3,158 | 1,864 | 1,864 |
| 54 | `2_10_10_10_UINT` | added | 3,158 | 3,158 | 1,688 | 1,688 |
| 55 | `2_10_10_10_SINT` | added | 3,158 | 3,158 | 1,688 | 1,688 |
| 56 | `8_8_8_8_UNORM` | before | 1,106 | 1,106 | 3,024 | 3,024 |
| 57 | `8_8_8_8_SNORM` | before | 1,106 | 1,106 | 3,024 | 3,024 |
| 58 | `8_8_8_8_USCALED` | added | 1,106 | 1,106 | 1,864 | 1,864 |
| 59 | `8_8_8_8_SSCALED` | added | 1,106 | 1,106 | 1,864 | 1,864 |
| 60 | `8_8_8_8_UINT` | before | 1,106 | 1,106 | 1,688 | 1,688 |
| 61 | `8_8_8_8_SINT` | added | 1,106 | 1,106 | 1,688 | 1,688 |
| 62 | `32_32_UINT` | before | 632 | 632 | 1,688 | 1,688 |
| 63 | `32_32_SINT` | before | 632 | 632 | 1,688 | 1,688 |
| 64 | `32_32_FLOAT` | before | 632 | 632 | 1,864 | 1,864 |
| 65 | `16_16_16_16_UNORM` | before | 2,178 | 2,178 | 3,024 | 3,024 |
| 66 | `16_16_16_16_SNORM` | before | 2,178 | 2,178 | 3,024 | 3,024 |
| 67 | `16_16_16_16_USCALED` | added | 2,178 | 2,178 | 1,864 | 1,864 |
| 68 | `16_16_16_16_SSCALED` | added | 2,178 | 2,178 | 1,864 | 1,864 |
| 69 | `16_16_16_16_UINT` | before | 2,178 | 2,178 | 1,688 | 1,688 |
| 70 | `16_16_16_16_SINT` | added | 2,178 | 2,178 | 1,688 | 1,688 |
| 71 | `16_16_16_16_FLOAT` | before | 262,226 | 262,226 | 1,864 | 1,864 |
| 72 | `32_32_32_UINT` | before | 907 | 907 | 1,688 | 1,688 |
| 73 | `32_32_32_SINT` | before | 907 | 907 | 1,688 | 1,688 |
| 74 | `32_32_32_FLOAT` | before | 907 | 907 | 1,864 | 1,864 |
| 75 | `32_32_32_32_UINT` | before | 1,182 | 1,182 | 1,688 | 1,688 |
| 76 | `32_32_32_32_SINT` | before | 1,182 | 1,182 | 1,688 | 1,688 |
| 77 | `32_32_32_32_FLOAT` | before | 1,182 | 1,182 | 1,864 | 1,864 |

"before" means the format was in `Gen.FORMATS` before the SCALED / SINT / small-float change. (Codes 5, 6, 11, 12, 18, 19, 28,
61 and 70, the narrow UINT and SINT ones, were added to main's table by a parallel change, measured on gfx1036, while that branch
was open; they are marked "added" here because the 880M run covers them too and the table entries are identical.) Codes
30..35, 37..42, 46 and 47 have no GFX11 form and are not in this table; gfx1036 ran them.

## Where the translator stood, and what changed

Decode of every older format matched the 880M word for word from the start. **Three places in the older stores differed
from the hardware; they follow it now** (each pinned in `tests/semantic/test_buffer_formats_all.py`, `StoresOfTheOlderFormats`,
and in the GPU fixtures), on the evidence of both GPUs (gfx1036 by @AwkGrep256, #2022):

1. **16_FLOAT, 16_16_FLOAT and 16_16_16_16_FLOAT stores** rounded to nearest even and overflowed to infinity; the hardware
   rounds toward zero and saturates to ±65504 (0.49999997 gives `0x37FF`, 65520.0 and 1e30 give `0x7BFF`), a NaN stays a NaN
   and keeps its sign. This is the f16 *format* store; the d16 *data* rounding of BUILD-161 (#1759) is not touched. In a
   200-row sample of edge-heavy inputs 89 of 200 (16_FLOAT), 132 and 174 rows had differed.
2. **UNORM and SNORM stores** multiplied in f32 and rounded that, which rounds twice (0.3·65535 is 19660.5008 exactly and
   19660.5 in f32, a tie that went to 19660; the hardware stores 19661). They now round the exact product once, by integer
   arithmetic (`Gen.encode_norm_bits`), for every NORM and SCALED format.
3. **A two-component format stored through a V#** took (x, y) always. With identity selectors (XYZW) hardware takes (x, w) (`_xy`: (x, 0)); with the runtime's XY01 it takes (x, y), which is what the game's descriptors have, so the translator now chooses by DST_SEL_Y and DST_SEL_Z. tbuffer stores keep (x, y). **The first reading of this finding, "(x, w) on RDNA2", came from a probe that only used XYZW; the narrow-integer fixtures, measured on gfx1036 with XY01, showed it does not hold for the runtime's selectors.**

The 12 codes that were missing (30..35, 37..42) are in the table now.

## Not measured

The console's own gfx1013; `d16` loads and stores of the new kinds and of the two-component routing (they use the same
d16 path as the existing formats: SINT saturates signed, UINT unsigned; BUILD-161 owns d16 rounding); `_xyz` stores into a
two-component format; formats on unaligned addresses beyond what the existing tests cover; swizzled V#s.
