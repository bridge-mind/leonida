# Calibrated TTPF export

The Python exporter can replay a locally measured reconstruction profile for 2D BC1, BC3, BC4, BC5, BC6H and BC7 textures and 2D arrays. A single-entry export accepts `--calibration PROFILE.json`. The assets stage discovers `<work>/bcpack-calibration.json`. The profile is data. The compiled library is loaded from this crate's release build directory.

From the repository root, build with Rust/Cargo and run in the Python environment where NumPy is installed:

```text
python -m pip install numpy
cargo build --release --locked --offline --manifest-path tools/bcpack_calibration/Cargo.toml
python -IB tests/test_calibrated_texture.py -v
python -I tools/ttpf_mips.py export path/to/pool.ttpf 0 output.dds --calibration path/to/profile.json
```

The synthetic test constructs its own BC1 pool and profile, runs the CLI and compares the full mip payload with known blocks. A real profile requires the caller's calibrated tables. The exporter does not derive the tables automatically. A matching file hash verifies reproduction of a previous fitted export. Original pixel fidelity needs independent reference pixels.

## Faster strict decoding

Once built, the same library also speeds up the default strict path. Its `unpack_model` export (`src/model.rs`) is a line-by-line port of the lane-array model in `tools/bcpack.py` (`_decode_lanes_raw`, virtual zero bytes included), and `tools/bcpack.py` passes it its own probability tables. It therefore accepts exactly the streams the Python model accepts, with the same symbols. It is about 30 times faster than the Python model on synthetic chunks, and NumPy is not needed. When the library is not built, or a stream is refused, Python decodes the stream as before and raises its usual error. `python -I tests/test_bcpack_native.py` checks that the two agree, and `LEONIDA_BCPACK_NATIVE=0` turns the native path off.

For BC1, BC3, BC4 and BC5 chunks, `decode_blocks` (`src/blocks.rs`) also rebuilds the blocks in the same call. It takes only the strict path: a chunk with a non-zero lane 1, or one that references a built-in history entry that is not confirmed, is decoded by Python, so errors and provisional entries do not change. On synthetic 16,384-block chunks, it is about 18 times faster than Python end to end for BC1, 38 times for BC4 and, measured the same way on a Windows PC, about 47 times for BC3 and 45 times for BC5.

The `unpack` export above runs `third_party/bcpack-rs`, which follows the hardware RTL. It is faster but refuses some streams the model accepts, for example every `q = 0` fragment whose omitted trailing zero byte comes before a symbol. That is why the strict path does not use it.

## Automatic asset export

Run the ordinary build or `python -I make_game.py --game <package> --work <work> --stages assets`. The assets stage runs the C# exporter on each new pool and revisits existing pool manifests. It attempts full mip chains for missing and partial BCn entries whose allocation includes BCPack. With no profile, it uses the Python strict decoder. For calibrated 2D textures and arrays, install NumPy, build the library as above and place a matching profile at `<work>/bcpack-calibration.json` before running the stage.

The DDS files and their manifest are under `<work>/assets/ttpf/<pool>/` or `<work>/assets/ttpf_pack/<pack>/<pool>/`. The manifest keeps the asset key for model lookup and records the new filename, mip count and decoder. A failed reconstruction keeps the previous file reference and records the reason in `BCPackLimitation`. The stage publishes each DDS before replacing the pool manifest. A completed pass skips decoding on the next build. Adding or changing the profile triggers another pass.

## Version 1 profile

The top-level JSON object contains `"version": 1` and the sections used by the requested texture:

- `bc1.endpoints` holds 256 rows of six RGB565 components indexed by predictor age. Negative values mark unknown components. `bc1.selectors` contains four rows of 256 unsigned 32-bit masks in output bit order. The synthetic test generates both fields.
- `bc3.endpoints` maps negative history offsets to eight RGB-alpha components in endpoint order. Offset -512 is the zero-distance predictor. `color` and `alpha` map quadrant and pair ranks to packed selector bits.
- `bc4` maps profile names to dictionaries of history-slot numbers and two byte-valued endpoints. BC5 uses the decoder's built-in tables and the compatibility endpoint ordering.
- `bc6h` uses the existing external BC6H history section.
- `bc7.labels` has 1,024 rows of mode, partition, rotation and index selection indexed by the two signature symbols. `anchors` contains 128 unsigned 16-bit masks. Each entry of `seeds` is a 256 × 56 × 2 array indexed by age and field with left-aligned byte value and known-bit count. Tables are named `base`, `secondary`, `early` and `refined`. Optional fitted endpoint adjustments are keyed by SHA-256 of the contiguous lane-symbol matrix followed by little-endian signed 64-bit unresolved block indices. Each adjustment gives age, endpoint field and fitted byte value.
- Optional `entries` maps pool basename and zero-based index to a profile name (default `base`). Optional `compatibility` maps a compressed BC1 stream's SHA-256 to a list of flattened symbol positions. Rotate the symbols at those positions left once to reproduce a known historical feeder ordering.

Cargo builds the entropy decoder from the two upstream modules included in `third_party/bcpack-rs`. With the Rust toolchain installed, this step works offline with an empty Cargo cache. The [source and license notes](../../third_party/bcpack-rs/README.md) identify the upstream revision. The Python reader handles streams for which that reader does not report full consumption. The fitted BC7 reconstruction tracks precision while decoding blocks in archive order. Visible blocks must resolve before the DDS is written.
