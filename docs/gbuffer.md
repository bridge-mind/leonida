# Deferred G-buffer targets and declared output signatures (`PROC-146`)

Reported by @Silliana in [#1891](https://github.com/ItsNotRexed/Leonida/issues/1891) for `PS_DeferredTextured` and
checked here across all **2,989 `*Deferred*` pixel shaders** (and all **1,703 six-target pixel shaders**) in the
1.0.2.1 `shaders.rpf` corpus. Counts, target masks, numeric constants and format requirements only; no shader text or
disassembly.
checked here across the `*Deferred*` pixel shaders and six-output pixel shaders in the 1.0.2.1 `shaders.rpf` corpus.
Counts, target masks, numeric constants and format requirements only; no shader text or disassembly.

## 1. Corpus census of deferred pixel-shader output signatures (`OSG1` / `PSV0`)

Every DXBC container in `shaders.rpf` carries an `OSG1` (and `PSV0`) output signature table with `{register,
semantic_index, component_type, mask, min_precision}` per declared output element. (The FSSM header dword at `+0x0FC`
holds `5` and `6` on the two sampled `PS_DeferredTextured` blobs below, but holds `16` and `18` on two stock vehicle
deferred passes, so target counts must be read from `OSG1` rather than FSSM `+0x0FC`.)

Depending on whether container names are matched case-insensitively on raw RPF entry names (`2,989` pixel shaders) or
case-sensitively on `Deferred` in the carved `.dxbc` file stems (`3,004` pixel shaders, including `807`
`PS_DeferredTextured` variants: 380 five-target, 371 six-target and 56 three-target):

| `OSG1` outputs declared | `*Deferred*` PS (entry match / `.dxbc` stem match) | `PS_DeferredTextured` count | Corpus-wide PS (entry / `.dxbc` stem) | `OSG1` component type on all colour targets | Per-target `OSG1` write masks (`SV_Target0..N-1`) |
|---|---:|---:|---:|---|---|
| 6 colour (`SV_Target0..5`) | 953 / 949 | 371 | 955 / 955 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0x3, 0xF` (`SV_Target4` mask is 2-component `RG`) |
| 6 colour (`SV_Target0..5`) | 737 / 735 | 0 | 747 / 747 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0xF, 0xF` (all six `RGBA`) |
| 5 colour + `SV_Depth` | 0 / 0 | 0 | 1 / 0 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0x3` + `SV_Depth` (`0x1`) |
| 5 colour (`SV_Target0..4`) | 1,160 / 1,160 | 380 | 1,168 / 1,166 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0x3` (`SV_Target4` mask is 2-component `RG`) |
| 5 colour (`SV_Target0..4`) | 2 / 2 | 0 | 20 / 20 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0xF` |
| 3 colour (`SV_Target0..2`) | 125 / 125 | 56 | 248 / 248 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0x3` (`SV_Target2` mask is 2-component `RG`) |
| other (`0..4` outputs) | 12 / 33 | 0 | — | `float32` (3 single-target `uint32`) | depth-only, discard, or reduced passes |

The two six-colour-target rows total **1,702** pixel shaders corpus-wide (`955 + 747`; plus 1 five-colour-target +
`SV_Depth` shader with 6 `OSG1` rows = 1,703 six-row `OSG1` tables). In the compiled RDNA2 code across the `*Deferred*`
pixel shaders, **16,306 of 16,306 (100%)** `exp mrtN` instructions use compressed half-float export (`compr`,
`EXP_FORMAT_FP16_ABGR`), fed by `v_cvt_pkrtz_f16_f32`.

## 2. G-buffer target layout (`mrt0..mrt5`)

Comparing #1891 against the 5-target (`mrt0..mrt4`) and 6-target (`mrt0..mrt5`) `PS_DeferredTextured` programs (note
that the RTV format column below is **inferred from the shader-side encoding and #1891**, not declared by `OSG1` or a
captured PSO):

| Target | `OSG1` mask | Inferred RTV format class (from shader encoding / #1891) | Channel layout and shader-side encoding in `PS_DeferredTextured` |
|---|---|---|---|
| `mrt0` (`SV_Target0`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Albedo + alpha/material byte:** `rgb` holds base colour/albedo; in the sampled 6-target variant `a` packs a byte extracted from an integer register (`ubyte0`) and scaled by `1.0 / 255.0` (`0x3b808081`). |
| `mrt1` (`SV_Target1`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Scalar in `r` + 24-bit octahedral world normal split across `g`, `b`, `a`:** see section 3 below. |
| `mrt2` (`SV_Target2`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Square-root encoded channels:** producer applies hardware square root (`sqrt(x)` or `sqrt(1.0 - sqrt(x))`) before `v_cvt_pkrtz_f16_f32` export; #1891 notes that the lighting consumer squares `mrt2` back (tentatively specular / $F_0$ material terms, pending lighting-consumer verification). |
| `mrt3` (`SV_Target3`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Flags + byte IDs:** `x` and `y` carry clamped `[0, 1]` values biased by a dither term (`scale = 2.0 / 255.0` = `0x3c008081`, `offset = -1.0 / 255.0` = `0xbb808081`); `z` and `w` carry two 8-bit integer IDs extracted with byte-to-float conversion (`ubyte0`, `ubyte1`) and multiplied by `1.0 / 255.0` (`0x3b808081`). |
| `mrt4` (`SV_Target4`) | `0x3` (`RG`, 2,113 / 2,109 deferred PS) or `0xF` (`RGBA`, 739 / 737 deferred PS) | Float format (e.g. `R16G16_FLOAT`) | **Motion vectors:** unscaled signed float screen-space motion offset (`xy`; in the sampled 5- and 6-target `PS_DeferredTextured` disassembly with `mask = 0x3`, exported as `exp mrt4, vN, vN, off, off compr`). |
| `mrt5` (`SV_Target5`) | `0xF` (`RGBA`, 1,690 / 1,684 deferred PS) | `R8G8B8A8_UNORM` | **Sixth deferred target** on the 6-target permutations (`rgb` material terms + clamped alpha). |

## 3. The 24-bit octahedral normal in `mrt1.gba` and why binding `f16` targets breaks lighting
semantic_index, component_type, mask, min_precision}` per declared output element. (The FSSM header dword at `+0x0FC`
holds `5` and `6` on the two sampled `PS_DeferredTextured` blobs below, but holds `16` and `18` on two stock vehicle
deferred passes, so target counts must be read from `OSG1` rather than FSSM `+0x0FC`.)

Depending on whether container names are matched case-insensitively on raw RPF entry names (`2,989` pixel shaders) or
case-sensitively on `Deferred` in the carved `.dxbc` file stems (`3,004` pixel shaders, including `807`
`PS_DeferredTextured` variants: 380 five-target, 371 six-target and 56 three-target):

| `OSG1` outputs declared | `*Deferred*` PS (entry match / `.dxbc` stem match) | `PS_DeferredTextured` count | Corpus-wide PS (entry / `.dxbc` stem) | `OSG1` component type on all colour targets | Per-target `OSG1` write masks (`SV_Target0..N-1`) |
|---|---:|---:|---:|---|---|
| 6 colour (`SV_Target0..5`) | 953 / 949 | 371 | 955 / 955 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0x3, 0xF` (`SV_Target4` mask is 2-component `RG`) |
| 6 colour (`SV_Target0..5`) | 737 / 735 | 0 | 747 / 747 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0xF, 0xF` (all six `RGBA`) |
| 5 colour + `SV_Depth` | 0 / 0 | 0 | 1 / 0 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0x3` + `SV_Depth` (`0x1`) |
| 5 colour (`SV_Target0..4`) | 1,160 / 1,160 | 380 | 1,168 / 1,166 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0x3` (`SV_Target4` mask is 2-component `RG`) |
| 5 colour (`SV_Target0..4`) | 2 / 2 | 0 | 20 / 20 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0xF, 0xF, 0xF` |
| 3 colour (`SV_Target0..2`) | 125 / 125 | 56 | 248 / 248 | `float32` (`ctype = 3`, `prec = 0`) | `0xF, 0xF, 0x3` (`SV_Target2` mask is 2-component `RG`) |
| other (`0..4` outputs) | 12 / 33 | 0 | — | `float32` (3 single-target `uint32`) | depth-only, discard, or reduced passes |

The two six-colour-target rows total **1,702** pixel shaders corpus-wide (`955 + 747`; plus 1 five-colour-target +
`SV_Depth` shader with 6 `OSG1` rows = 1,703 six-row `OSG1` tables). In the compiled RDNA2 code across the `*Deferred*`
pixel shaders, **16,306 of 16,306 (100%)** `exp mrtN` instructions use compressed half-float export (`compr`,
`EXP_FORMAT_FP16_ABGR`), fed by `v_cvt_pkrtz_f16_f32`.

## 2. G-buffer target layout (`mrt0..mrt5`)

Comparing #1891 against the 5-target (`mrt0..mrt4`) and 6-target (`mrt0..mrt5`) `PS_DeferredTextured` programs (note
that the RTV format column below is **inferred from the shader-side encoding and #1891**, not declared by `OSG1` or a
captured PSO):

| Target | `OSG1` mask | Inferred RTV format class (from shader encoding / #1891) | Channel layout and shader-side encoding in `PS_DeferredTextured` |
|---|---|---|---|
| `mrt0` (`SV_Target0`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Albedo + alpha/material byte:** `rgb` holds base colour/albedo; in the sampled 6-target variant `a` packs a byte extracted from an integer register (`ubyte0`) and scaled by `1.0 / 255.0` (`0x3b808081`). |
| `mrt1` (`SV_Target1`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Scalar in `r` + 24-bit octahedral world normal split across `g`, `b`, `a`:** see section 3 below. |
| `mrt2` (`SV_Target2`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Square-root encoded channels:** producer applies hardware square root (`sqrt(x)` or `sqrt(1.0 - sqrt(x))`) before `v_cvt_pkrtz_f16_f32` export; #1891 notes that the lighting consumer squares `mrt2` back (tentatively specular / $F_0$ material terms, pending lighting-consumer verification). |
| `mrt3` (`SV_Target3`) | `0xF` (`RGBA`) | `R8G8B8A8_UNORM` | **Flags + byte IDs:** `x` and `y` carry clamped `[0, 1]` values biased by a dither term (`scale = 2.0 / 255.0` = `0x3c008081`, `offset = -1.0 / 255.0` = `0xbb808081`); `z` and `w` carry two 8-bit integer IDs extracted with byte-to-float conversion (`ubyte0`, `ubyte1`) and multiplied by `1.0 / 255.0` (`0x3b808081`). |
| `mrt4` (`SV_Target4`) | `0x3` (`RG`, 2,113 / 2,109 deferred PS) or `0xF` (`RGBA`, 739 / 737 deferred PS) | Float format (e.g. `R16G16_FLOAT`) | **Motion vectors:** unscaled signed float screen-space motion offset (`xy`; in the sampled 5- and 6-target `PS_DeferredTextured` disassembly with `mask = 0x3`, exported as `exp mrt4, vN, vN, off, off compr`). |
| `mrt5` (`SV_Target5`) | `0xF` (`RGBA`, 1,690 / 1,684 deferred PS) | `R8G8B8A8_UNORM` | **Sixth deferred target** on the 6-target permutations (`rgb` material terms + clamped alpha). |

## 3. The 24-bit octahedral normal in `mrt1.gba` and `UNORM8` vs `FLOAT16` storage

`PS_DeferredTextured` does not store a 3-channel `(nx, ny, nz)` normal in `mrt1`. Instead:

1. The unit world normal `(nx, ny, nz)` is divided by its $L_1$ norm $|nx| + |ny| + |nz|$ and octahedral-folded across
   the $nz \le 0$ hemisphere into $(u, v) \in [-1, 1]^2$.
2. Both coordinates are quantized to 12-bit unsigned integers $q_u, q_v \in [0, 4095]$ with the scale constant
   `2047.5` (`0x44fff000`):
   $$q_u = \operatorname{uint32}(u \cdot 2047.5 + 2047.5), \qquad q_v = \operatorname{uint32}(v \cdot 2047.5 + 2047.5)$$
3. The two 12-bit integers are combined into one 24-bit word $q = (q_v \ll 12) \mid q_u$ (`q_u` in bits `[11:0]`,
   `q_v` in bits `[23:12]`).
4. The three bytes of $q$ (`ubyte0 = q[7:0]`, `ubyte1 = q[15:8]`, `ubyte2 = q[23:16]`) are converted to `f32`,
   multiplied by `1.0 / 255.0` (`0x3b808081`), rounded toward zero to `f16` (`v_cvt_pkrtz_f16_f32`), and exported to
   `mrt1.g`, `mrt1.b`, and `mrt1.a`.

### Arithmetic across all 256 byte values and consequence for the runtime

- **Declared signatures do not separate `UNORM8` from `FLOAT16`:** because `OSG1` and `PSV0` declare `float32` on all
  six `SV_Target0..5` outputs and the RDNA2 code uses `exp mrtN ... compr` (`EXP_FORMAT_FP16_ABGR`) for all of them,
  neither the DXBC signature nor the FSSM export opcode indicates that `mrt0..mrt3` (and `mrt5`) are `R8G8B8A8_UNORM`
  while `mrt4` is a float motion target.
- **Why binding `R16G16B16A16_FLOAT` to `mrt1` or `mrt3` corrupts the packed bytes:** `v_cvt_pkrtz_f16_f32` rounds
  `byte * (1.0 / 255.0)` **toward zero** (`RTZ`), producing an `f16` value slightly *below* `byte / 255.0` for most byte
  values `1..254`. When `mrt1` is bound as `R8G8B8A8_UNORM`, the render-target output stage quantizes with
  $\lfloor x \cdot 255.0 + 0.5 \rfloor$ and restores the exact byte `0..255`. When `mrt1` is bound as an `f16` float
  target and the lighting pass reconstructs the bytes by multiplying by `255.0`, the RTZ truncation drops 1 LSB on
  `ubyte1` (bits `[15:8]`, which straddles the 12-bit split between $q_u$ and $q_v$) and `ubyte2` (bits `[23:16]`),
  shifting the reconstructed normal by whole octahedral steps and producing speckled lighting (#1891).
- The runtime must always bind the PSO's declared `RTVFormats[]` (`R8G8B8A8_UNORM` for the byte-packed G-buffer targets
  and a float format for `mrt4`) and must never infer `R16G16B16A16_FLOAT` for all G-buffer attachments from `OSG1` or
  `exp ... compr`.
### Arithmetic across all 256 byte values and consequence for the runtime

- **Declared signatures do not separate `UNORM8` from `FLOAT16`:** because `OSG1` and `PSV0` declare `float32` on all
  six `SV_Target0..5` outputs and the RDNA2 code uses `exp mrtN ... compr` (`EXP_FORMAT_FP16_ABGR`) for all of them,
  neither the DXBC signature nor the FSSM export opcode indicates which G-buffer targets are `R8G8B8A8_UNORM` and which
  target (`mrt4`) is a float motion target.
- **`UNORM8` vs `SFLOAT16` round-trip across `b = 0..255`:** `v_cvt_pkrtz_f16_f32(float32(b) * 0x3b808081)` rounds
  **toward zero** (`RTZ`) to binary16, placing all **254 intermediate bytes (`b = 1..254`)** slightly *below* `b / 255.0`
  (with a maximum shortfall of `0.1240234375` of a byte before integer conversion):
  - When the attachment is `R8G8B8A8_UNORM`, the render-target output stage quantizes with nearest-integer rounding
    ($\lfloor x \cdot 255.0 + 0.5 \rfloor$) and stores **256 of 256** bytes exactly in every channel.
  - When the attachment is `R16G16B16A16_SFLOAT`, the stored binary16 RTZ value is kept verbatim. How a downstream
    lighting shader recovers each byte from `x * 255.0` determines whether a 1-LSB error occurs: **nearest-integer
    rounding** recovers **256 of 256** bytes, whereas **integer truncation** (`uint32(x * 255.0)`) recovers only
    **2 of 256** bytes (`0` and `255`) and yields `b - 1` on all **254** intermediate values `1..254` across all three
    byte positions (`ubyte0`, `ubyte1`, `ubyte2`). When `ubyte1` (bits `[15:8]`, straddling the 12-bit boundary between
    $q_u$ and $q_v$) or `ubyte2` (bits `[23:16]`) drops by 1 under a truncating consumer, the reassembled 24-bit
    octahedral normal shifts by large steps (#1891; actual lighting-consumer decode and PSO formats remain to be
    verified against the lighting pass).
- The runtime must therefore bind the PSO's declared `RTVFormats[]` rather than inferring `R16G16B16A16_FLOAT` for all
  G-buffer attachments from `OSG1` or `exp ... compr`.
