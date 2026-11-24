# BCPack entropy decoder

Leonida uses `rans::decode_slice` to unpack BCPack lane symbols. The two Rust modules in `src/rans.rs` and `src/tables.rs` are copied verbatim from [Ahmet-Dedeler/bcpack-rs at 0a887de1edc9c22a425605091f1a8365fdf74dbb](https://github.com/Ahmet-Dedeler/bcpack-rs/tree/0a887de1edc9c22a425605091f1a8365fdf74dbb). The local Cargo manifest and module root expose these files as a standard-library-only crate.

The upstream crate is licensed under MIT or Apache-2.0. Both license texts are included here. The rANS decoder and model tables also carry Microsoft's MIT notice. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

To update, copy both modules, both license files and the third-party notices from the same upstream revision. Update the revision here, then build the calibration library offline and run its synthetic DDS tests as described in the [calibrated export guide](../../tools/bcpack_calibration/README.md).
