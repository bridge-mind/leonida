# What game shaders expect from their inputs

Some game shaders give a black, speckled or hung result when one input value is wrong, and that looks like a translator
or layer bug when it isn't. This page lists those inputs per shader, so a wrong picture or a device-lost has a
documented first check. Each section says what was reported, by whom, and what was measured in the shader to confirm
it.

Measured on the local package's disassembly (`work/shaders/dxbc/disasm`) at `16282101`. Offsets, bit positions and
counts only; no shader text.

## Bloom and exposure

Reported by @Silliana in [#3235](https://github.com/ItsNotRexed/Leonida/issues/3235). The observations below were
checked against the local disassembly. Offsets are bytes into the constant buffer read by the named shader.

### A black bloom chain (`PS_DownSampleBloomFilterKaris`)

- Check the RGB tint at `+0x00`, `+0x04` and `+0x08` first. The shader multiplies the filtered colour by these
  values before export. An all-zero tint makes bloom black even with a valid source. The fourth value at `+0x0C`
  feeds the exported alpha.
- The sample loop has a limit of 13. Its luminance-dependent weight has the form `1 / (1 + k × luminance)`,
  with scale `k` at `+0x1C`. Setting it to zero removes that dependence for finite inputs. The spatial filter
  and output tint still apply.

### Flat, wrong or slowly changing exposure (`PS_CalculateExposure`)

- The mode dword at `+0x410` selects the exposure calculation. A nonzero value takes the path that computes
  logarithmic exposure from the measured luminance. Check this before diagnosing the automatic exposure path.
- On the compensation path, a nonzero dword at `+0x40C` selects the fixed value at `+0x408`. Zero selects a
  sample from a 1D compensation curve. Check that the curve is bound when using the texture path.
- Check the previous-exposure texture when the pass changes slowly. The adaptation code subtracts its sample
  from the target exposure and can blend back toward the target. It tests `+0x3B0` against zero and compares
  a value derived from the exposure delta against the threshold at `+0x3B4`. For a single-frame comparison,
  the issue report suggests seeding the previous exposure with the target.

### Luminance downsample and bloom upsample

All five sampling sites in `PS_LuminanceSourceDownsample` contribute to the export. All nine in
`PS_UpSampleBloomFilterBlend` do too. Check the source resolution, sampling and reduction before comparing
the luminance pyramid with a whole-frame average.
When inspecting the upsample in isolation, check in the pipeline state how its output blends with the previous level.

## Studio-shot lighting (`PS_STUDIOSHOT_PBR_MATERIAL_STANDARD_LIGHTING`)

Reported by @Silliana in [#1892](https://github.com/ItsNotRexed/Leonida/issues/1892). The corpus has one program
with this name. It shades with up to eight lights, read from a buffer of 512-byte light records. Colour and
intensity come from a separate table of light profiles.

### First check for a device-lost or a dark frame

1. **The dword at record +0x4C.** Its top byte is a loop count, so it must hold an integer and never a float (see
   [Light-shaping planes](#light-shaping-planes)). #1892 reports a GPU hang when a float sits here. That report gives
   the dword as +0x40; the count is measured at +0x4C (+0x40 is a different field, described below).
2. **The profile intensity.** A light whose profile intensity is 0 or less is skipped entirely. So an all-black light
   doesn't mean the light maths is broken.
3. **The shadow-strength byte at record +0x73.** At 0 the shadow map is still sampled, but the result does nothing.

### Light loop

- The loop always runs eight times (light 0 to 7); the light count doesn't come from a constant. An unused slot is
  switched off through its profile: point it at a profile with intensity 0.
- Light *i*'s record starts at byte *i* × 512 (0x200) of the light buffer.
- All 61 image samples and all 27 cube-coordinate sequences in the program are inside this loop. That matches #1892:
  the shader has no ambient term, and its 2D and cube samples are per-light cookies and shadows. Indirect light comes
  from the separate probe system (`indirect_light_probes` / `CS_ApplyProbeIrradiance`), not from this shader.

### Light record (512 bytes)

The fields below are the ones confirmed. The rest of the record is read but not decoded here.

| Offset | Type | Use |
|---|---|---|
| +0x10 | 3 × f32 | Light position; the shader subtracts the world position of the pixel from it. |
| +0x20 | u32 | Bits 0–8: profile index (9 bits) into the profile table. Bits 9–17 and 18–21 are two more fields, not decoded. |
| +0x28 | u32 | Flags. Bit 24 enables the light-shaping planes; bit 16 takes part in the shadow-map gate; bits 30–31 are the light type (0–3), which selects how +0x3C is used; bits 17, 20, 21, 26 and 27 are tested, not decoded. |
| +0x3C | 2 × f16 | Light shape size: the light's dimensions, used differently per light type. See [Light shape size](#light-shape-size). |
| +0x40 | u32 | Low 24 bits non-zero turns on one per-light factor; at zero that factor is 1.0, so the light still shades. The top byte isn't read on the paths checked. |
| +0x4C | u32 | Top byte: number of light-shaping planes (0–255). See below. |
| +0x5F | u8 | Read as byte / 255; at 0 one block of the light's shading is skipped. |
| +0x73 | u8 | Shadow strength, read as byte / 255 right before the shadow-map block. |
| +0xB0 | planes | Light-shaping planes, 32 bytes apart. |

#### Light-shaping planes

When flag bit 24 at +0x28 is set and the dword at +0x4C is at least `0x01000000`, the shader loops over *n* planes,
where *n* is that dword's top byte. Plane *k* is four f32 (normal xyz, then w) at record +0xB0 + 0x20 × *k*. For each
plane it computes `saturate(dot(normal, pixel − light) + w)`, squares it and multiplies it into a per-light factor that starts at 1.0.

Only 11 planes fit in the record (the last at +0x1F0). A larger count reads on into the next light's record, and past the
end of the buffer after the last light. A float in this dword has its exponent in the top byte: 1.0f (`0x3F800000`)
gives 63 planes. The loop itself ends after at most 255 iterations. If a device-lost still shows up there, check this
dword and the buffer's bounds first.

#### Light shape size

+0x3C holds two f16. How they are used depends on the light type (bits 30–31 at +0x28):

| Type | Low half | High half | Source |
|---|---|---|---|
| 0 | Sphere radius: clamped against the reflection ray to pick the closest point on the light. Brightness grows with radius², since intensity is radiance. | No effect. | Tested by @Silliana [on the pull request](https://github.com/ItsNotRexed/Leonida/pull/3157): radius 60 → 5 dimmed the light about 134× (radius² predicts 144×). |
| not decoded | Half-size along one axis of a rectangle. | Half-size along the other axis. | Shader: the light's position is moved by ± each half along its axis, and points are clamped to those bounds. |
| not decoded | — | A length along the light's axis (clamped to ± half of it), also squared and compared against a distance. Probably a tube light's length (@Silliana's hypothesis, not tested). | Shader. |

The low half also sets one blend factor. No range-style falloff was found: neither half divides a distance, and the
early reading of the high half as a range is dropped.

A value that's too big makes the light too large. On a sphere light, the brightness grows with the square of the
radius. That gives scattered single-pixel specular blowups that look like a translator bug, for example when a range
is written into both halves (reported by @Silliana).

#### Correction to #1892

#1892 puts two checks on one dword at +0x40: a top-byte loop count, and low 24 bits that must be non-zero or the light
is skipped. The shader splits these across two dwords:

- The count is the top byte of the dword at **+0x4C**, gated by flag bit 24 at +0x28. On that path the low 24 bits
  aren't tested.
- The low-24-bit test is on the dword at **+0x40**, and at zero it disables one factor, not the whole light.

The light is skipped only through a profile intensity of 0 or less.

### Profile table

- It is a separate buffer from the light records, made of 16-byte entries.
- The light record's profile index (bits 0–8 of +0x20) selects the entry.
- The entry's fourth dword (+12) is the intensity, as f32. When it's 0 or less the light is skipped before the rest of
  its record is read. #1892 gives the first three dwords as the RGB colour; the shader loads the whole entry later in the
  light's shading.

## ACES LUT bakes (`postfx_aces`)

Reported by @Silliana in [#1896](https://github.com/ItsNotRexed/Leonida/issues/1896). The corpus has ten bake
programs: `CS_BakeACESLUT`, `CS_BakeACESWithGamutCompressLUT`, and the `TonemapOnly` versions of both, each with a
choice of shaper (Linear, Log, PQ or QuasiLog; the `TonemapOnly` bakes exist only with QuasiLog). Each one writes
an RGBA 3D LUT through a storage image (`image_store`, 3D). In the corpus the program names have lost the `A` of
`ACES` (`CS_BakeCESLUT_…`) everywhere except the gamut-compress bakes.

### First check for an identity-looking or flat LUT

1. **Which spline you filled.** The bakes read their tone curve from a spline in the constant buffer. In the
   `TonemapOnly` bakes, the dword at **+0x118** chooses between two of them (see [Two splines](#two-splines)):
   non-zero uses the one at +0x70, zero uses the one at +0x00. Filling one and pointing the switch at the other
   gives a flat or meaningless curve.
2. **The dword at +0x13C.** It is the switch #1896 describes, but it doesn't turn the curve off: every path
   evaluates the spline. At zero the spline is applied to R, G and B separately (three evaluations). When it is
   non-zero, the shader also evaluates it on AP1 luminance (0.2722, 0.6741, 0.0537), for four evaluations. In the
   non-`TonemapOnly` bakes the dword at +0x11C then blends the luminance result with the per-channel one.
3. **The quasi-log addressing**, below, if you read the LUT back.

### Quasi-log addressing

The four QuasiLog bakes turn each LUT index *i* into a linear value as #1896 gives it:

`linear = 0.18 · 2^(14 · (i · s − 0.41838)) − 0.0031`

The shader's constants are −0.418377, 14.0, 0.18 and −0.0031050. The scale *s* is the first dword of a second
constant buffer and multiplies the thread index; #1896's formula makes it 1/(N − 1) for an N-entry axis. Mid grey
(linear 0.18) falls at *i* · *s* ≈ 0.4201, which is index 13.0 on a 32-entry axis. #1896 reports that mid grey comes
out at about 0.103. That needs the bake to run and isn't measured here. The Linear, Log and PQ bakes don't contain
these constants.

### Two splines

The RRT's own spline (the reference `SegmentedSplineParams_c5`) isn't an input: its coefficients are a constant
table inside the program, read relative to the code (`s_getpc_b64`), with the thresholds log10 0.18 = −0.7447 and
log10(0.18 · 2^18) = 4.673 as literals. The constant buffer holds the output tone curve, packed at 4 bytes per value
(not the 16-byte stride an HLSL array would get).

**At +0x00, all ten bakes: reference order** (`SegmentedSplineParams_c9`, 8 knots per half):

| Offset | Values | Use |
|---|---|---|
| +0x00 | 10 × f32 | Low coefficients |
| +0x28 | 10 × f32 | High coefficients |
| +0x50 | 2 × f32 | Min point (x, y) |
| +0x58 | 2 × f32 | Mid point (x, y) |
| +0x60 | 2 × f32 | Max point (x, y) |
| +0x68 | f32 | Low slope |
| +0x6C | f32 | High slope |

**At +0x70, the two `TonemapOnly` bakes only: points first.** This is the layout #1896 describes: three (x, y,
slope) points come first, and the coefficient blocks follow them.

| Offset | Values | Use |
|---|---|---|
| +0x70 | 3 × f32 | Min point (x, y, slope) |
| +0x7C | 3 × f32 | Mid point (x, y, slope) |
| +0x88 | 3 × f32 | Max point (x, y, slope) |
| +0x94 | 6 × f32 | Low coefficients |
| +0xAC | 6 × f32 | High coefficients |

The x of each point is the value the shader compares log10(input) against, so the 12-byte stride is measured. The
role of the other two values in each triplet is #1896's reading. The non-`TonemapOnly` bakes don't read +0x70 to
+0xC3; their next values start at +0xC4.

#### Correction to #1896

#1896 says the bakes use a non-reference order. That is true only of the `TonemapOnly` bakes' second spline at
+0x70. The spline at +0x00, which every bake reads (the only one outside the `TonemapOnly` bakes), is in the reference
CTL order. The RRT spline isn't in the constant buffer at all. The +0x13C switch picks between a per-channel curve and
one that also uses luminance; it doesn't decide whether the curve applies.

## Sky passes (`sky_scattering`, `scattering_transmittance_system`)

Reported by @Silliana in [#1897](https://github.com/ItsNotRexed/Leonida/issues/1897). This is a physically based
atmosphere: two lookup tables are built at runtime and the sky passes sample them. Neither table is a texture on
disk.

### Order

| Pass | Samples | Writes |
|---|---|---|
| `sky_scattering:CS_Transmittance` | nothing | one 2D storage image |
| `sky_scattering:CS_MultipleScattering` | one 2D texture | one 2D storage image (of two, see below) |
| `scattering_transmittance_system:CS_SkyAtmosphereLUT` | two 2D textures | three 2D storage images |
| `sky_scattering:CS_SkyView` | two 2D textures | two 2D storage images (of four, see below) |

`CS_Transmittance` reads no texture, so it goes first. `CS_MultipleScattering` samples one texture, the
transmittance table #1897 names, so it runs second and not side by side with `CS_Transmittance`. `CS_SkyView` and
`CS_SkyAtmosphereLUT` each sample two textures, the two tables, and run after both. Every sample is `image_sample_lz`
(mip 0). Which texture is bound to which table comes from #1897, not from the program. If a sky pass runs before the
tables are built, it reads whatever is in them, and the black or noisy sky looks like a translator bug.

### Transmittance addresses texel corners

`CS_Transmittance` turns its integer texel index into table coordinates as x × 1/256 and y × 1/64, with no +0.5,
so each texel holds the value at its corner, not its centre (pointed out by @Silliana in
[#3217](https://github.com/ItsNotRexed/Leonida/pull/3217#issuecomment-7084678180)). A reference table built at texel
centres differs from it slightly, most visibly near the ground. That difference is expected and isn't a translator bug.

### The dword at +0xD0: output format, not an enable

#1897 reports a SkyView dword that skips the stores when it is zero. In `CS_SkyView` and `CS_MultipleScattering`
the dword at **+0xD0** of the pass's constant buffer is compared as a float with 0.0. Both outcomes store, but to
different descriptors:

| +0xD0 | `CS_SkyView` writes | `CS_MultipleScattering` writes |
|---|---|---|
| non-zero | RGB9E5 bits to the images at table +0x40 and +0x80 | RGB9E5 bits to the image at table +0x40 |
| zero | float RGB (alpha 0) to the images at table +0x60 and +0xA0 | float RGB (alpha 1.0) to the image at table +0x60 |

So a zero dword gives a black sky without an error if only the RGB9E5 images are bound: the stores go to the other
pair of descriptors. Because the test is a float compare, write a float such as 1.0f there. An integer 1 is a
denormal and compares equal to 0.0 if denormals are flushed. This is a hypothesis that depends on the program's float
mode, which isn't checked here.

### RGB9E5 output

On the non-zero path both programs pack RGB9E5 themselves: they clamp each channel to [0, 65408], take the shared
exponent from the largest, put R, G and B in bits 0–8, 9–17 and 18–26 and the exponent in 27–31. They then write the
same 32-bit word to all four components of the storage image. The data is useless read back as floats, as #1897
says. The layer's UAV path for `R9G9B9E5_SHAREDEXP` already covers this: on a GPU without native RGB9E5 storage it
binds an `R32_UINT` storage view ([`d3d12_vulkan.md`](d3d12_vulkan.md), from #1375 and #1388).
`CS_Transmittance` and `CS_SkyAtmosphereLUT` contain no RGB9E5 packing.

### The shared atmosphere constant buffer

Reported by @Silliana in [#3990](https://github.com/ItsNotRexed/Leonida/issues/3990): the four passes read one
atmosphere struct, each through its own buffer descriptor fetched from the resource table. Confirmed in the
disassembly of the 1.0.2.1 package (llvm-mc 23.1.5, gfx1013): the 26,503-program corpus has exactly one program for
each of the four names, and each reads the offsets below through a single descriptor. The three `sky_scattering`
passes fetch that descriptor from table offset **+0x220**; `CS_SkyAtmosphereLUT` fetches it from **+0xC60**. So when
one pass looks right and another looks wrong, check the slot binding before the layout. The +0xD0 output-format
dword above sits in this same buffer.

Everything is in km (#3990). Field names are #3990's reading; the offsets and the "consumed by" column are measured.
A pass can load a dword as part of a wider load without using it, so "consumed" means an instruction reads the
register before it is overwritten.

| Offset | Type | Field (#3990) | Consumed by |
|---|---|---|---|
| +0x20 | f32 | Planet radius | all four |
| +0x30 | f32 | Rayleigh scale height | none (loaded, see below) |
| +0x34 | 3 × f32 | Rayleigh scattering RGB | all four |
| +0x40 | f32 | Mie scale height | none (loaded, see below) |
| +0x44 | 3 × f32 | Mie scattering RGB | all four |
| +0x50 | f32 | Top of the atmosphere | all four |
| +0x60 | 3 × f32 | Mie absorption RGB | all four |
| +0x6C | f32 | Mie g (phase asymmetry) | `CS_SkyView` only |
| +0x70 | f32 | Ozone layer width | all four |
| +0x74 | 2 × f32 | Ozone below the layer: constant, slope | all four |
| +0x7C | 2 × f32 | Ozone above the layer: constant, slope | all four |
| +0x84 | 3 × f32 | Ozone absorption RGB | all four |
| +0x90 | 3 × f32 | Sun direction | `CS_SkyView` only |
| +0xA0 | 3 × f32 | Moon direction | `CS_SkyView` only |

The two scale-height dwords are where the disassembly adds to #3990: every pass loads them inside a wide scalar
load, but in all four programs the registers holding +0x30 and +0x40 are overwritten or never touched again before
any instruction reads them. Only the RGB coefficients next to them are consumed. A wrong value in those two dwords
cannot by itself explain a bad sky from these passes.

The ozone block matches #3990's two-layer linear reading: the dword at +0x70 is compared per lane with a float
(`v_cmp_gt_f32`) in the ozone block, the constant/slope pairs feed `v_fma_f32`, and the absorption RGB feeds the
per-channel accumulate. `CS_SkyView` reads the sun direction at +0x90 and a second direction at +0xA0,
which #3990 reads as the moon; its second output is the sky lit by that direction. The fourth dword of each
direction (+0x9C, +0xAC) is not consumed.

Some neighbouring dwords are read but not decoded here: +0x24 to +0x2C (next to the planet radius), +0x54 to +0x5C
(`CS_SkyView`), and +0x04 to +0x13 (`CS_SkyView` and `CS_SkyAtmosphereLUT`).

Sanity checks from #3990 (GPU experiments, not repeated here): zeroing the Rayleigh RGB makes the zenith go basically
white, zeroing the ozone absorption mostly bumps green, and raising Mie g brightens the sky toward the sun. With
Earth-like values `CS_Transmittance` lands within about 0.01 of a reference transmittance table, which makes it the
cheapest pass to sanity-check a translation.
