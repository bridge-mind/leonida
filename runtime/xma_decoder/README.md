# XMA2 decoder C ABI

This crate exposes a C ABI around the separately licensed xma Rust package pinned in Cargo.toml. It decodes one mono or stereo stream from a non-empty buffer of complete 2048-byte XMA2 packets into interleaved signed 16-bit PCM.

Each decoder handle serializes decode calls. The returned PCM block owns its samples and stays valid until leonida_xma2_pcm_destroy, even after its decoder is destroyed. Do not destroy a decoder concurrently with a decode call or destroy a handle more than once.

The tests build packets in memory; they do not open or contain audio fixtures. The synthetic C++ smoke test verifies the exported ABI and ownership boundary.

This PR adds the decoder boundary only; the Windows XAudio2 engine/source-voice adapter is a follow-up stage.

Run:

```powershell
cargo fmt --manifest-path runtime/xma_decoder/Cargo.toml -- --check
cargo test --manifest-path runtime/xma_decoder/Cargo.toml --locked
cargo build --manifest-path runtime/xma_decoder/Cargo.toml --locked
```

The C++ smoke test is run by the xma2-decoder workflow with MSVC. It links ws2_32, userenv, and ntdll for Rust's Windows standard library.
