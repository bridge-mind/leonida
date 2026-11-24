// Helpers for forwarding Xbox root-signature descriptions to the PC d3d12.dll.
//
// The measured Xbox descriptions carry type bits in the high byte of D3D12_ROOT_SIGNATURE_FLAGS (observed value
// 0x03000000). The PC serializer rejects those bits, so the shims pass a shallow copy with bits 24-31 cleared and
// every other flag bit preserved. The pointed-to parameter/sampler arrays are not modified.
#pragma once

#include <d3d12.h>
#include <cstdint>
#include <vector>

namespace rt::d3d12_xs {

constexpr uint32_t kXboxRootSignatureFlagHighByte = 0xFF000000u;

// Always copies `in` to `out`. Returns true for the known 1.0/1.1/1.2 descriptor versions and clears the high flag
// byte in the copy. Returns false for an unknown version, leaving the copy byte-for-byte identical to `in` so the
// caller can forward it unchanged and log the unsupported version.
bool copy_versioned_root_signature_desc_clear_xbox_flags(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out);

// Windows 10's built-in d3d12.dll serializes root signatures up to version 1.1: it rejects every 1.2 description,
// even an empty one, with E_INVALIDARG, and reports HighestVersion 1.1. Version 1.2 differs from 1.1 only in
// D3D12_STATIC_SAMPLER_DESC1::Flags, so a 1.2 description whose static samplers all have Flags == 0 is the same
// root signature written as 1.1. This rewrites it that way: `out` becomes a 1.1 description that shares `in`'s
// parameters and points its samplers at `samplers` (resized here; keep it alive while `out` is used). Returns false,
// with `out` and `samplers` untouched, when `in` is not version 1.2 or a static sampler has non-zero Flags (a UINT
// border colour or non-normalized coordinates, which 1.1 can't express).
bool downgrade_root_signature_1_2_to_1_1(const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& in,
                                         std::vector<D3D12_STATIC_SAMPLER_DESC>& samplers,
                                         D3D12_VERSIONED_ROOT_SIGNATURE_DESC& out);

// Legacy D3D12SerializeRootSignature descriptor (version 1.0): copies and clears the high flag byte.
void copy_root_signature_desc_clear_xbox_flags(const D3D12_ROOT_SIGNATURE_DESC& in, D3D12_ROOT_SIGNATURE_DESC& out);

// D3D12CreateRootSignatureDeserializer path: the host deserializer accepts a correctly hashed blob whose RTS0
// flags carry the Xbox high byte and hands the byte back, and CreateRootSignature then refuses it (E_FAIL on WARP
// and an RTX 5070). So copy the blob, clear bits 24-31 of the RTS0 flags, recompute the DXBC hash the host
// deserializer checks (a stale one is E_INVALIDARG), and return true with the copy in `out`. Returns false (and
// leaves `out` empty) when no rewrite is needed: no RTS0 chunk, no Xbox high byte, or a malformed/truncated header.
bool copy_rts0_blob_clear_xbox_flags(const void* data, size_t size, std::vector<uint8_t>& out);

// The DXBC container hash of `blob` (bytes 20..size), as stored at bytes 4..19. `size` must be at least 20.
void dxbc_checksum(const uint8_t* blob, size_t size, uint32_t out[4]);

}  // namespace rt::d3d12_xs
