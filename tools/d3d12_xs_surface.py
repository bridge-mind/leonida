#!/usr/bin/env python3
"""The measured Xbox Series ``d3d12_xs.dll`` export surface, with provenance.

This is the machine-checkable half of ``docs/d3d12_xs_surface.md``. It does not talk to a game, a
vendor DLL or the network. The data is the surface SCARAB measured from the console runtime
(``p-portela/SCARAB`` @ d304fa2, GPL-3.0-only), plus the 28 D3D12 IIDs, which are also Windows SDK
facts. See the doc for the per-item evidence and for the claims that did not hold up.

Checks that always run:
  * 116 slots, ordinals 1..116 exactly once, 23 functions + 93 GUID data;
  * every standard GUID in the table is byte-identical to the Windows SDK, when the SDK is found;
  * GUIDs are valid RFC 4122 version 4, unique, and the 10 console GUIDs are not in the SDK set;
  * the 88 IID ordinals 28..115 are alphabetical, except IID_ID3D12ConsDmaCommandList pinned at 100;
  * the four unresolved console command-list IIDs are not assigned values or ordinals in the local tables.

Optional checks:
  * ``--scarab PATH`` cross-checks the table against a SCARAB checkout (values, ordinals, catalogue).
  * ``--require-sdk`` turns a missing SDK into a failure instead of a skip.

Usage:
  python tools/d3d12_xs_surface.py --check
  python tools/d3d12_xs_surface.py --check --sdk
  python tools/d3d12_xs_surface.py --check --scarab C:\\path\\to\\scarab-ref
  python tools/d3d12_xs_surface.py --markdown
"""
# SPDX-License-Identifier: GPL-3.0-or-later
# Original code. Facts sourced from SCARAB (GPL-3.0-only) and the Windows SDK; no SCARAB code copied.
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

SCARAB_COMMIT = "d304fa26c0292e2f3ba76f23b15c9f30734bec02"

# (ordinal, export, kind). Kind: standard | console. Names are SCARAB's published spellings; the
# five D3D12Cons* names are internal aliases, not verified wire names (doc section 2.3).
FUNCTION_EXPORTS = [
    (1, "D3D12ZeroMemoryX", "console"),
    (2, "PixGetPixieEntryPoints", "console"),
    (3, "D3D10CreateBlob", "console"),
    (4, "D3D12ConfigureVirtualMemoryX", "console"),
    (5, "D3D12CreateDevice", "standard"),
    (6, "D3D12CreateRootSignatureDeserializer", "standard"),
    (7, "D3D12CreateVersionedRootSignatureDeserializer", "standard"),
    (8, "D3D12EnableExperimentalFeatures", "standard"),
    (9, "D3D12GetDebugInterface", "standard"),
    (10, "D3D12SerializeRootSignature", "standard"),
    (11, "D3D12SerializeVersionedRootSignature", "standard"),
    (12, "D3D12UploadCustomMicrocodeX", "console"),
    (13, "D3D12ConsCreateDevice", "console"),
    (14, "D3D12ConsEnableMemoryProfiling", "console"),
    (15, "D3D12ConsGetProcessDebugFlags", "console"),
    (16, "D3D12ConsSetProcessDebugFlags", "console"),
    (17, "D3D12ConsUnloadDriverModule", "console"),
    (18, "D3DFlushEntireCpuCache", "console"),
    (19, "DXGIXGetFrameStatistics", "console"),
    (20, "DXGIXPresentArray", "console"),
    (21, "DXGIXSetFrameNotification", "console"),
    (22, "DXGIXSetVLineNotification", "console"),
    (116, "InternalDriverCallX", "console"),
]

# Ordinals 23..115: 93 GUID data exports. Evidence is "documented" when iid_console.inc carries the
# ordinal, "derived" when it comes from the alphabetical rule anchored by IID_ID3D12Debug @76.
# Kind is debug | blob | d3d11 | d3d12 | console. The 10 console GUIDs are listed in CONSOLE_IIDS.
DATA_EXPORTS = [
    (23, "DXGI_DEBUG_ALL", "debug", "derived"),
    (24, "DXGI_DEBUG_APP", "debug", "derived"),
    (25, "DXGI_DEBUG_D3D11", "debug", "derived"),
    (26, "DXGI_DEBUG_DX", "debug", "derived"),
    (27, "DXGI_DEBUG_DXGI", "debug", "derived"),
    (28, "IID_ID3D10Blob", "blob", "derived"),
    (29, "IID_ID3D11Asynchronous", "d3d11", "documented"),
    (30, "IID_ID3D11BlendState", "d3d11", "documented"),
    (31, "IID_ID3D11BlendState1", "d3d11", "documented"),
    (32, "IID_ID3D11Buffer", "d3d11", "documented"),
    (33, "IID_ID3D11ClassInstance", "d3d11", "documented"),
    (34, "IID_ID3D11ClassLinkage", "d3d11", "documented"),
    (35, "IID_ID3D11CommandList", "d3d11", "documented"),
    (36, "IID_ID3D11ComputeShader", "d3d11", "documented"),
    (37, "IID_ID3D11Counter", "d3d11", "documented"),
    (38, "IID_ID3D11CounterSampleX", "console", "documented"),
    (39, "IID_ID3D11CounterSetX", "console", "documented"),
    (40, "IID_ID3D11DepthStencilState", "d3d11", "documented"),
    (41, "IID_ID3D11DepthStencilView", "d3d11", "documented"),
    (42, "IID_ID3D11Device", "d3d11", "documented"),
    (43, "IID_ID3D11Device1", "d3d11", "documented"),
    (44, "IID_ID3D11DeviceChild", "d3d11", "documented"),
    (45, "IID_ID3D11DeviceContext", "d3d11", "documented"),
    (46, "IID_ID3D11DeviceContext1", "d3d11", "documented"),
    (47, "IID_ID3D11DeviceContextX", "console", "documented"),
    (48, "IID_ID3D11DeviceX", "console", "documented"),
    (49, "IID_ID3D11DmaEngineContextX", "console", "documented"),
    (50, "IID_ID3D11DomainShader", "d3d11", "documented"),
    (51, "IID_ID3D11GeometryShader", "d3d11", "documented"),
    (52, "IID_ID3D11HullShader", "d3d11", "documented"),
    (53, "IID_ID3D11InputLayout", "d3d11", "documented"),
    (54, "IID_ID3D11PerformanceContextX", "console", "documented"),
    (55, "IID_ID3D11PerformanceDeviceX", "console", "documented"),
    (56, "IID_ID3D11PixelShader", "d3d11", "documented"),
    (57, "IID_ID3D11Predicate", "d3d11", "documented"),
    (58, "IID_ID3D11Query", "d3d11", "documented"),
    (59, "IID_ID3D11RasterizerState", "d3d11", "documented"),
    (60, "IID_ID3D11RasterizerState1", "d3d11", "documented"),
    (61, "IID_ID3D11RenderTargetView", "d3d11", "documented"),
    (62, "IID_ID3D11Resource", "d3d11", "documented"),
    (63, "IID_ID3D11SamplerState", "d3d11", "documented"),
    (64, "IID_ID3D11ShaderResourceView", "d3d11", "documented"),
    (65, "IID_ID3D11Texture1D", "d3d11", "documented"),
    (66, "IID_ID3D11Texture2D", "d3d11", "documented"),
    (67, "IID_ID3D11Texture3D", "d3d11", "documented"),
    (68, "IID_ID3D11UnorderedAccessView", "d3d11", "documented"),
    (69, "IID_ID3D11UserDefinedAnnotationX", "console", "documented"),
    (70, "IID_ID3D11VertexShader", "d3d11", "documented"),
    (71, "IID_ID3D11View", "d3d11", "documented"),
    (72, "IID_ID3D12CommandAllocator", "d3d12", "derived"),
    (73, "IID_ID3D12CommandList", "d3d12", "derived"),
    (74, "IID_ID3D12CommandQueue", "d3d12", "derived"),
    (75, "IID_ID3D12CommandSignature", "d3d12", "derived"),
    (76, "IID_ID3D12Debug", "d3d12", "derived"),
    (77, "IID_ID3D12DescriptorHeap", "d3d12", "derived"),
    (78, "IID_ID3D12Device", "d3d12", "derived"),
    (79, "IID_ID3D12Device1", "d3d12", "derived"),
    (80, "IID_ID3D12Device2", "d3d12", "derived"),
    (81, "IID_ID3D12Device3", "d3d12", "derived"),
    (82, "IID_ID3D12Device4", "d3d12", "derived"),
    (83, "IID_ID3D12DeviceChild", "d3d12", "derived"),
    (84, "IID_ID3D12Fence", "d3d12", "derived"),
    (85, "IID_ID3D12Fence1", "d3d12", "derived"),
    (86, "IID_ID3D12GraphicsCommandList", "d3d12", "derived"),
    (87, "IID_ID3D12GraphicsCommandList1", "d3d12", "derived"),
    (88, "IID_ID3D12GraphicsCommandList2", "d3d12", "derived"),
    (89, "IID_ID3D12Heap", "d3d12", "derived"),
    (90, "IID_ID3D12Object", "d3d12", "derived"),
    (91, "IID_ID3D12Pageable", "d3d12", "derived"),
    (92, "IID_ID3D12PipelineLibrary", "d3d12", "derived"),
    (93, "IID_ID3D12PipelineLibrary1", "d3d12", "derived"),
    (94, "IID_ID3D12PipelineState", "d3d12", "derived"),
    (95, "IID_ID3D12Resource", "d3d12", "derived"),
    (96, "IID_ID3D12RootSignature", "d3d12", "derived"),
    (97, "IID_ID3D12RootSignatureDeserializer", "d3d12", "derived"),
    (98, "IID_ID3D12Tools", "d3d12", "derived"),
    (99, "IID_ID3D12VersionedRootSignatureDeserializer", "d3d12", "derived"),
    (100, "IID_ID3D12ConsDmaCommandList", "console", "documented"),
    (101, "IID_ID3DDeviceContextState", "d3d11", "documented"),
    (102, "IID_IDXGIAdapter", "d3d11", "documented"),
    (103, "IID_IDXGIAdapter1", "d3d11", "documented"),
    (104, "IID_IDXGIAdapter2", "d3d11", "documented"),
    (105, "IID_IDXGIDevice", "d3d11", "documented"),
    (106, "IID_IDXGIDevice1", "d3d11", "documented"),
    (107, "IID_IDXGIDevice2", "d3d11", "documented"),
    (108, "IID_IDXGIDeviceSubObject", "d3d11", "documented"),
    (109, "IID_IDXGIFactory", "d3d11", "documented"),
    (110, "IID_IDXGIFactory1", "d3d11", "documented"),
    (111, "IID_IDXGIFactory2", "d3d11", "documented"),
    (112, "IID_IDXGIObject", "d3d11", "documented"),
    (113, "IID_IDXGISwapChain", "d3d11", "documented"),
    (114, "IID_IDXGISwapChain1", "d3d11", "documented"),
    (115, "IID_IGraphicsUnknown", "console", "documented"),
]

# The 83 standard GUIDs. Values match the Windows SDK 10.0.26100.0 headers (doc section 3).
STANDARD_GUIDS = {
    "DXGI_DEBUG_ALL": (0xE48AE283, 0xDA80, 0x490B, 0x87, 0xE6, 0x43, 0xE9, 0xA9, 0xCF, 0xDA, 0x08),
    "DXGI_DEBUG_APP": (0x06CD6E01, 0x4219, 0x4EBD, 0x87, 0x09, 0x27, 0xED, 0x23, 0x36, 0x0C, 0x62),
    "DXGI_DEBUG_D3D11": (0x4B99317B, 0xAC39, 0x4AA6, 0xBB, 0x0B, 0xBA, 0xA0, 0x47, 0x84, 0x79, 0x8F),
    "DXGI_DEBUG_DX": (0x35CDD7FC, 0x13B2, 0x421D, 0xA5, 0xD7, 0x7E, 0x44, 0x51, 0x28, 0x7D, 0x64),
    "DXGI_DEBUG_DXGI": (0x25CDDAA4, 0xB1C6, 0x47E1, 0xAC, 0x3E, 0x98, 0x87, 0x5B, 0x5A, 0x2E, 0x2A),
    "IID_ID3D10Blob": (0x8BA5FB08, 0x5195, 0x40E2, 0xAC, 0x58, 0x0D, 0x98, 0x9C, 0x3A, 0x01, 0x02),
    "IID_ID3D11Asynchronous": (0x4B35D0CD, 0x1E15, 0x4258, 0x9C, 0x98, 0x1B, 0x13, 0x33, 0xF6, 0xDD, 0x3B),
    "IID_ID3D11BlendState": (0x75B68FAA, 0x347D, 0x4159, 0x8F, 0x45, 0xA0, 0x64, 0x0F, 0x01, 0xCD, 0x9A),
    "IID_ID3D11BlendState1": (0xCC86FABE, 0xDA55, 0x401D, 0x85, 0xE7, 0xE3, 0xC9, 0xDE, 0x28, 0x77, 0xE9),
    "IID_ID3D11Buffer": (0x48570B85, 0xD1EE, 0x4FCD, 0xA2, 0x50, 0xEB, 0x35, 0x07, 0x22, 0xB0, 0x37),
    "IID_ID3D11ClassInstance": (0xA6CD7FAA, 0xB0B7, 0x4A2F, 0x94, 0x36, 0x86, 0x62, 0xA6, 0x57, 0x97, 0xCB),
    "IID_ID3D11ClassLinkage": (0xDDF57CBA, 0x9543, 0x46E4, 0xA1, 0x2B, 0xF2, 0x07, 0xA0, 0xFE, 0x7F, 0xED),
    "IID_ID3D11CommandList": (0xA24BC4D1, 0x769E, 0x43F7, 0x80, 0x13, 0x98, 0xFF, 0x56, 0x6C, 0x18, 0xE2),
    "IID_ID3D11ComputeShader": (0x4F5B196E, 0xC2BD, 0x495E, 0xBD, 0x01, 0x1F, 0xDE, 0xD3, 0x8E, 0x49, 0x69),
    "IID_ID3D11Counter": (0x6E8C49FB, 0xA371, 0x4770, 0xB4, 0x40, 0x29, 0x08, 0x60, 0x22, 0xB7, 0x41),
    "IID_ID3D11DepthStencilState": (0x03823EFB, 0x8D8F, 0x4E1C, 0x9A, 0xA2, 0xF6, 0x4B, 0xB2, 0xCB, 0xFD, 0xF1),
    "IID_ID3D11DepthStencilView": (0x9FDAC92A, 0x1876, 0x48C3, 0xAF, 0xAD, 0x25, 0xB9, 0x4F, 0x84, 0xA9, 0xB6),
    "IID_ID3D11Device": (0xDB6F6DDB, 0xAC77, 0x4E88, 0x82, 0x53, 0x81, 0x9D, 0xF9, 0xBB, 0xF1, 0x40),
    "IID_ID3D11Device1": (0xA04BFB29, 0x08EF, 0x43D6, 0xA4, 0x9C, 0xA9, 0xBD, 0xBD, 0xCB, 0xE6, 0x86),
    "IID_ID3D11DeviceChild": (0x1841E5C8, 0x16B0, 0x489B, 0xBC, 0xC8, 0x44, 0xCF, 0xB0, 0xD5, 0xDE, 0xAE),
    "IID_ID3D11DeviceContext": (0xC0BFA96C, 0xE089, 0x44FB, 0x8E, 0xAF, 0x26, 0xF8, 0x79, 0x61, 0x90, 0xDA),
    "IID_ID3D11DeviceContext1": (0xBB2C6FAA, 0xB5FB, 0x4082, 0x8E, 0x6B, 0x38, 0x8B, 0x8C, 0xFA, 0x90, 0xE1),
    "IID_ID3D11DomainShader": (0xF582C508, 0x0F36, 0x490C, 0x99, 0x77, 0x31, 0xEE, 0xCE, 0x26, 0x8C, 0xFA),
    "IID_ID3D11GeometryShader": (0x38325B96, 0xEFFB, 0x4022, 0xBA, 0x02, 0x2E, 0x79, 0x5B, 0x70, 0x27, 0x5C),
    "IID_ID3D11HullShader": (0x8E5C6061, 0x628A, 0x4C8E, 0x82, 0x64, 0xBB, 0xE4, 0x5C, 0xB3, 0xD5, 0xDD),
    "IID_ID3D11InputLayout": (0xE4819DDC, 0x4CF0, 0x4025, 0xBD, 0x26, 0x5D, 0xE8, 0x2A, 0x3E, 0x07, 0xB7),
    "IID_ID3D11PixelShader": (0xEA82E40D, 0x51DC, 0x4F33, 0x93, 0xD4, 0xDB, 0x7C, 0x91, 0x25, 0xAE, 0x8C),
    "IID_ID3D11Predicate": (0x9EB576DD, 0x9F77, 0x4D86, 0x81, 0xAA, 0x8B, 0xAB, 0x5F, 0xE4, 0x90, 0xE2),
    "IID_ID3D11Query": (0xD6C00747, 0x87B7, 0x425E, 0xB8, 0x4D, 0x44, 0xD1, 0x08, 0x56, 0x0A, 0xFD),
    "IID_ID3D11RasterizerState": (0x9BB4AB81, 0xAB1A, 0x4D8F, 0xB5, 0x06, 0xFC, 0x04, 0x20, 0x0B, 0x6E, 0xE7),
    "IID_ID3D11RasterizerState1": (0x1217D7A6, 0x5039, 0x418C, 0xB0, 0x42, 0x9C, 0xBE, 0x25, 0x6A, 0xFD, 0x6E),
    "IID_ID3D11RenderTargetView": (0xDFDBA067, 0x0B8D, 0x4865, 0x87, 0x5B, 0xD7, 0xB4, 0x51, 0x6C, 0xC1, 0x64),
    "IID_ID3D11Resource": (0xDC8E63F3, 0xD12B, 0x4952, 0xB4, 0x7B, 0x5E, 0x45, 0x02, 0x6A, 0x86, 0x2D),
    "IID_ID3D11SamplerState": (0xDA6FEA51, 0x564C, 0x4487, 0x98, 0x10, 0xF0, 0xD0, 0xF9, 0xB4, 0xE3, 0xA5),
    "IID_ID3D11ShaderResourceView": (0xB0E06FE0, 0x8192, 0x4E1A, 0xB1, 0xCA, 0x36, 0xD7, 0x41, 0x47, 0x10, 0xB2),
    "IID_ID3D11Texture1D": (0xF8FB5C27, 0xC6B3, 0x4F75, 0xA4, 0xC8, 0x43, 0x9A, 0xF2, 0xEF, 0x56, 0x4C),
    "IID_ID3D11Texture2D": (0x6F15AAF2, 0xD208, 0x4E89, 0x9A, 0xB4, 0x48, 0x95, 0x35, 0xD3, 0x4F, 0x9C),
    "IID_ID3D11Texture3D": (0x037E866E, 0xF56D, 0x4357, 0xA8, 0xAF, 0x9D, 0xAB, 0xBE, 0x6E, 0x25, 0x0E),
    "IID_ID3D11UnorderedAccessView": (0x28ACF509, 0x7F5C, 0x48F6, 0x86, 0x11, 0xF3, 0x16, 0x01, 0x0A, 0x63, 0x80),
    "IID_ID3D11VertexShader": (0x3B301D64, 0xD678, 0x4289, 0x88, 0x97, 0x22, 0xF8, 0x92, 0x8B, 0x72, 0xF3),
    "IID_ID3D11View": (0x839D1216, 0xBB2E, 0x412B, 0xB7, 0xF4, 0xA9, 0xDB, 0xEB, 0xE0, 0x8E, 0xD1),
    "IID_ID3D12CommandAllocator": (0x6102DEE4, 0xAF59, 0x4B09, 0xB9, 0x99, 0xB4, 0x4D, 0x73, 0xF0, 0x9B, 0x24),
    "IID_ID3D12CommandList": (0x7116D91C, 0xE7E4, 0x47CE, 0xB8, 0xC6, 0xEC, 0x81, 0x68, 0xF4, 0x37, 0xE5),
    "IID_ID3D12CommandQueue": (0x0EC870A6, 0x5D7E, 0x4C22, 0x8C, 0xFC, 0x5B, 0xAA, 0xE0, 0x76, 0x16, 0xED),
    "IID_ID3D12CommandSignature": (0xC36A797C, 0xEC80, 0x4F0A, 0x89, 0x85, 0xA7, 0xB2, 0x47, 0x50, 0x82, 0xD1),
    "IID_ID3D12Debug": (0x344488B7, 0x6846, 0x474B, 0xB9, 0x89, 0xF0, 0x27, 0x44, 0x82, 0x45, 0xE0),
    "IID_ID3D12DescriptorHeap": (0x8EFB471D, 0x616C, 0x4F49, 0x90, 0xF7, 0x12, 0x7B, 0xB7, 0x63, 0xFA, 0x51),
    "IID_ID3D12Device": (0x189819F1, 0x1DB6, 0x4B57, 0xBE, 0x54, 0x18, 0x21, 0x33, 0x9B, 0x85, 0xF7),
    "IID_ID3D12Device1": (0x77ACCE80, 0x638E, 0x4E65, 0x88, 0x95, 0xC1, 0xF2, 0x33, 0x86, 0x86, 0x3E),
    "IID_ID3D12Device2": (0x30BAA41E, 0xB15B, 0x475C, 0xA0, 0xBB, 0x1A, 0xF5, 0xC5, 0xB6, 0x43, 0x28),
    "IID_ID3D12Device3": (0x81DADC15, 0x2BAD, 0x4392, 0x93, 0xC5, 0x10, 0x13, 0x45, 0xC4, 0xAA, 0x98),
    "IID_ID3D12Device4": (0xE865DF17, 0xA9EE, 0x46F9, 0xA4, 0x63, 0x30, 0x98, 0x31, 0x5A, 0xA2, 0xE5),
    "IID_ID3D12DeviceChild": (0x905DB94B, 0xA00C, 0x4140, 0x9D, 0xF5, 0x2B, 0x64, 0xCA, 0x9E, 0xA3, 0x57),
    "IID_ID3D12Fence": (0x0A753DCF, 0xC4D8, 0x4B91, 0xAD, 0xF6, 0xBE, 0x5A, 0x60, 0xD9, 0x5A, 0x76),
    "IID_ID3D12Fence1": (0x433685FE, 0xE22B, 0x4CA0, 0xA8, 0xDB, 0xB5, 0xB4, 0xF4, 0xDD, 0x0E, 0x4A),
    "IID_ID3D12GraphicsCommandList": (0x5B160D0F, 0xAC1B, 0x4185, 0x8B, 0xA8, 0xB3, 0xAE, 0x42, 0xA5, 0xA4, 0x55),
    "IID_ID3D12GraphicsCommandList1": (0x553103FB, 0x1FE7, 0x4557, 0xBB, 0x38, 0x94, 0x6D, 0x7D, 0x0E, 0x7C, 0xA7),
    "IID_ID3D12GraphicsCommandList2": (0x38C3E585, 0xFF17, 0x412C, 0x91, 0x50, 0x4F, 0xC6, 0xF9, 0xD7, 0x2A, 0x28),
    "IID_ID3D12Heap": (0x6B3B2502, 0x6E51, 0x45B3, 0x90, 0xEE, 0x98, 0x84, 0x26, 0x5E, 0x8D, 0xF3),
    "IID_ID3D12Object": (0xC4FEC28F, 0x7966, 0x4E95, 0x9F, 0x94, 0xF4, 0x31, 0xCB, 0x56, 0xC3, 0xB8),
    "IID_ID3D12Pageable": (0x63EE58FB, 0x1268, 0x4835, 0x86, 0xDA, 0xF0, 0x08, 0xCE, 0x62, 0xF0, 0xD6),
    "IID_ID3D12PipelineLibrary": (0xC64226A8, 0x9201, 0x46AF, 0xB4, 0xCC, 0x53, 0xFB, 0x9F, 0xF7, 0x41, 0x4F),
    "IID_ID3D12PipelineLibrary1": (0x80EABF42, 0x2568, 0x4E5E, 0xBD, 0x82, 0xC3, 0x7F, 0x86, 0x96, 0x1D, 0xC3),
    "IID_ID3D12PipelineState": (0x765A30F3, 0xF624, 0x4C6F, 0xA8, 0x28, 0xAC, 0xE9, 0x48, 0x62, 0x24, 0x45),
    "IID_ID3D12Resource": (0x696442BE, 0xA72E, 0x4059, 0xBC, 0x79, 0x5B, 0x5C, 0x98, 0x04, 0x0F, 0xAD),
    "IID_ID3D12RootSignature": (0xC54A6B66, 0x72DF, 0x4EE8, 0x8B, 0xE5, 0xA9, 0x46, 0xA1, 0x42, 0x92, 0x14),
    "IID_ID3D12RootSignatureDeserializer": (0x34AB647B, 0x3CC8, 0x46AC, 0x84, 0x1B, 0xC0, 0x96, 0x56, 0x45, 0xC0, 0x46),
    "IID_ID3D12Tools": (0x7071E1F0, 0xE84B, 0x4B33, 0x97, 0x4F, 0x12, 0xFA, 0x49, 0xDE, 0x65, 0xC5),
    "IID_ID3D12VersionedRootSignatureDeserializer": (0x7F91CE67, 0x090C, 0x4BB7, 0xB7, 0x8E, 0xED, 0x8F, 0xF2, 0xE3, 0x1D, 0xA0),
    "IID_ID3DDeviceContextState": (0x5C1E0D8A, 0x7C23, 0x48F9, 0x8C, 0x59, 0xA9, 0x29, 0x58, 0xCE, 0xFF, 0x11),
    "IID_IDXGIAdapter": (0x2411E7E1, 0x12AC, 0x4CCF, 0xBD, 0x14, 0x97, 0x98, 0xE8, 0x53, 0x4D, 0xC0),
    "IID_IDXGIAdapter1": (0x29038F61, 0x3839, 0x4626, 0x91, 0xFD, 0x08, 0x68, 0x79, 0x01, 0x1A, 0x05),
    "IID_IDXGIAdapter2": (0x0AA1AE0A, 0xFA0E, 0x4B84, 0x86, 0x44, 0xE0, 0x5F, 0xF8, 0xE5, 0xAC, 0xB5),
    "IID_IDXGIDevice": (0x54EC77FA, 0x1377, 0x44E6, 0x8C, 0x32, 0x88, 0xFD, 0x5F, 0x44, 0xC8, 0x4C),
    "IID_IDXGIDevice1": (0x77DB970F, 0x6276, 0x48BA, 0xBA, 0x28, 0x07, 0x01, 0x43, 0xB4, 0x39, 0x2C),
    "IID_IDXGIDevice2": (0x05008617, 0xFBFD, 0x4051, 0xA7, 0x90, 0x14, 0x48, 0x84, 0xB4, 0xF6, 0xA9),
    "IID_IDXGIDeviceSubObject": (0x3D3E0379, 0xF9DE, 0x4D58, 0xBB, 0x6C, 0x18, 0xD6, 0x29, 0x92, 0xF1, 0xA6),
    "IID_IDXGIFactory": (0x7B7166EC, 0x21C7, 0x44AE, 0xB2, 0x1A, 0xC9, 0xAE, 0x32, 0x1A, 0xE3, 0x69),
    "IID_IDXGIFactory1": (0x770AAE78, 0xF26F, 0x4DBA, 0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
    "IID_IDXGIFactory2": (0x50C83A1C, 0xE072, 0x4C48, 0x87, 0xB0, 0x36, 0x30, 0xFA, 0x36, 0xA6, 0xD0),
    "IID_IDXGIObject": (0xAEC22FB8, 0x76F3, 0x4639, 0x9B, 0xE0, 0x28, 0xEB, 0x43, 0xA6, 0x7A, 0x2E),
    "IID_IDXGISwapChain": (0x310D36A0, 0xD2E7, 0x4C0A, 0xAA, 0x04, 0x6A, 0x9D, 0x23, 0xB8, 0x88, 0x6A),
    "IID_IDXGISwapChain1": (0x790A45F7, 0x0D42, 0x4876, 0x98, 0x3A, 0x0A, 0x55, 0xCF, 0xE6, 0xF4, 0xAA),
}

# The 10 console-only GUIDs, reported by SCARAB from the runtime's export table (doc section 4).
CONSOLE_IIDS = {
    "IID_ID3D11CounterSampleX": (0x806CEEE4, 0x280B, 0x43D9, 0x90, 0x08, 0xB4, 0xFD, 0xF7, 0xFE, 0xBE, 0xE2),
    "IID_ID3D11CounterSetX": (0xF9F6A732, 0xC5F8, 0x4A57, 0xBA, 0x36, 0xEA, 0xE7, 0xC2, 0x7B, 0xA7, 0x4A),
    "IID_ID3D11DeviceContextX": (0x48800095, 0x7134, 0x4BE7, 0x91, 0x86, 0xB8, 0x6B, 0xEC, 0xB2, 0x64, 0x77),
    "IID_ID3D11DeviceX": (0x177700F9, 0x876A, 0x4436, 0xB3, 0x68, 0x36, 0xA6, 0x04, 0xF8, 0x2C, 0xEF),
    "IID_ID3D11DmaEngineContextX": (0xA6332DDB, 0x8E02, 0x427D, 0xB0, 0xB7, 0x34, 0xA1, 0xE6, 0x1A, 0x64, 0x88),
    "IID_ID3D11PerformanceContextX": (0x9458FE06, 0xC78D, 0x47F7, 0x96, 0xA0, 0xEC, 0x7B, 0x72, 0x7B, 0xE1, 0xE9),
    "IID_ID3D11PerformanceDeviceX": (0x88671610, 0x712E, 0x4F1E, 0x84, 0xAB, 0x01, 0xB5, 0x94, 0x8B, 0xD3, 0x73),
    "IID_ID3D11UserDefinedAnnotationX": (0xB2DAAD8B, 0x03D4, 0x4DBF, 0x95, 0xEB, 0x32, 0xAB, 0x4B, 0x63, 0xD0, 0xAB),
    "IID_ID3D12ConsDmaCommandList": (0x30BC77B9, 0x8379, 0x4B25, 0x95, 0xDE, 0xB8, 0x58, 0x8C, 0x42, 0x27, 0x5D),
    "IID_IGraphicsUnknown": (0xACEEEA63, 0xE0A9, 0x4A1C, 0xBB, 0xEC, 0x71, 0xB2, 0xF4, 0x85, 0xF7, 0x58),
}

# The four console command-list interfaces whose IIDs SCARAB reports as not exported and not
# recovered (doc section 4). The local tables must not assign a value; a title trace is one
# possible future source. They must stay out of DATA_EXPORTS and CONSOLE_IIDS.
UNRECOVERED_CONSOLE_IIDS = (
    "ID3D12ConsComputeCommandList",
    "ID3D12ConsComputeCommandList6",
    "ID3D12ConsComputeCommandList8",
    "ID3D12ConsCopyCommandList6",
)

# The 11 catalogue rows labelled "method" that are not interface methods (doc section 5).
CATALOGUE_NON_METHODS = {
    "UINT_MAX",
    "UINT64_MAX",
    "DXGI_DEBUG_DX",
    "D3D12_MESSAGE_ID_INVALID_NODE_INDEX",
    "D3D12CONS_PRESENT_PLANES_MAX",
    "D3D12CONS_PRESENT_VIEWS_MAX",
    "UMDInternalDriverCallX",
    "WorldRayOriginX",
    "WorldRayDirectionX",
    "DispatchRaysDimensionsX",
    "TopLeftX",
}

SDK_NOTE = "Windows SDK 10.0.26100.0"


def guid_text(g):
    """RFC 4122 text form, for messages and the markdown table."""
    return "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x" % g


def guid_version(g):
    return (g[2] >> 12) & 0xF


def all_slots():
    """The 116 (ordinal, name, kind) rows, functions first."""
    return [(n, name, kind) for n, name, kind in FUNCTION_EXPORTS] + [
        (n, name, kind) for n, name, kind, _ev in DATA_EXPORTS
    ]


def check_slots():
    errors = []
    slots = all_slots()
    if len(slots) != 116:
        errors.append("expected 116 slots, got %d" % len(slots))
    if len(FUNCTION_EXPORTS) != 23:
        errors.append("expected 23 function exports, got %d" % len(FUNCTION_EXPORTS))
    if len(DATA_EXPORTS) != 93:
        errors.append("expected 93 data exports, got %d" % len(DATA_EXPORTS))
    ordinals = sorted(n for n, _name, _kind in slots)
    if ordinals != list(range(1, 117)):
        missing = sorted(set(range(1, 117)) - set(ordinals))
        dupes = sorted({n for n in ordinals if ordinals.count(n) > 1})
        errors.append("ordinals are not 1..116 (missing=%s duplicates=%s)" % (missing, dupes))
    names = [name for _n, name, _kind in slots]
    if len(names) != len(set(names)):
        errors.append("duplicate export names")
    # Exactly the function ordinals are 1..22 and 116; the data block is 23..115.
    func_ord = {n for n, _name, _kind in FUNCTION_EXPORTS}
    if func_ord != set(range(1, 23)) | {116}:
        errors.append("function ordinals are not 1..22 plus 116")
    if sum(1 for _n, _name, kind in FUNCTION_EXPORTS if kind == "standard") != 7:
        errors.append("expected 7 standard (PC-shared) function exports, got %d"
                      % sum(1 for _n, _name, kind in FUNCTION_EXPORTS if kind == "standard"))
    if sum(1 for _n, _name, kind in FUNCTION_EXPORTS if kind == "console") != 16:
        errors.append("expected 16 console-only function exports, got %d"
                      % sum(1 for _n, _name, kind in FUNCTION_EXPORTS if kind == "console"))
    return errors


def check_data_classes():
    errors = []
    standard = {name for _n, name, kind, _ev in DATA_EXPORTS if kind != "console"}
    console = {name for _n, name, kind, _ev in DATA_EXPORTS if kind == "console"}
    if standard != set(STANDARD_GUIDS):
        errors.append("standard data names do not equal STANDARD_GUIDS (%d vs %d)"
                      % (len(standard), len(STANDARD_GUIDS)))
    if console != set(CONSOLE_IIDS):
        errors.append("console data names do not equal CONSOLE_IIDS (%d vs %d)"
                      % (len(console), len(CONSOLE_IIDS)))
    if len(STANDARD_GUIDS) != 83:
        errors.append("expected 83 standard GUIDs, got %d" % len(STANDARD_GUIDS))
    if len(CONSOLE_IIDS) != 10:
        errors.append("expected 10 console GUIDs, got %d" % len(CONSOLE_IIDS))
    # IID ordinals are 28..115 and exactly the 88 names in the block.
    iid_ord = {n for n, name, _kind, _ev in DATA_EXPORTS if name.startswith("IID_")}
    if iid_ord != set(range(28, 116)):
        errors.append("IID ordinals are not 28..115")
    return errors


def check_guid_values():
    errors = []
    values = list(STANDARD_GUIDS.values()) + list(CONSOLE_IIDS.values())
    if len(values) != len({tuple(v) for v in values}):
        errors.append("duplicate GUID value across the standard and console tables")
    if set(STANDARD_GUIDS) & set(CONSOLE_IIDS):
        errors.append("a GUID name is in both the standard and console tables")
    for name, g in list(STANDARD_GUIDS.items()) + list(CONSOLE_IIDS.items()):
        if len(g) != 11:
            errors.append("%s: expected 11 components, got %d" % (name, len(g)))
            continue
        if g[0] > 0xFFFFFFFF or g[1] > 0xFFFF or g[2] > 0xFFFF or any(x > 0xFF for x in g[3:]):
            errors.append("%s: component out of range" % name)
        if guid_version(g) != 4:
            errors.append("%s: not an RFC 4122 version 4 UUID" % name)
    return errors


def check_unrecovered_console_iids():
    """The four unresolved command-list IIDs are absent from the local export table and CONSOLE_IIDS; the
    vendor-runtime observation is SCARAB's."""
    errors = []
    export_names = {name for _n, name, _kind, _ev in DATA_EXPORTS}
    for interface in UNRECOVERED_CONSOLE_IIDS:
        for name in (interface, "IID_" + interface):
            if name in export_names:
                errors.append("%s must not be assigned to the 93-GUID export table" % name)
            if name in CONSOLE_IIDS:
                errors.append("%s must not be assigned a local GUID value" % name)
    return errors


def check_iid_order():
    """The 88 IID names at 28..115 are alphabetical, except the DMA IID pinned at 100."""
    errors = []
    iids = [(n, name) for n, name, _kind, _ev in DATA_EXPORTS if name.startswith("IID_")]
    pinned = [x for x in iids if x[1] == "IID_ID3D12ConsDmaCommandList"]
    if len(pinned) != 1 or pinned[0][0] != 100:
        errors.append("IID_ID3D12ConsDmaCommandList is not pinned at 100")
    rest = [name for n, name in iids if name != "IID_ID3D12ConsDmaCommandList"]
    if rest != sorted(rest):
        errors.append("the IID block (excluding the DMA IID) is not alphabetical")
    return errors


def find_sdk_include():
    """Return (um_dir, shared_dir) for the newest installed Windows SDK, or (None, None)."""
    env = os.environ.get("WINDOWS_SDK_INCLUDE") or os.environ.get("SDK_INCLUDE")
    if env:
        candidates = [env]
    else:
        candidates = []
        for var in ("ProgramFiles(x86)", "ProgramFiles"):
            root = os.environ.get(var)
            if root:
                candidates += glob.glob(os.path.join(root, "Windows Kits", "10", "Include", "*"))

    best = None
    for path in candidates:
        path = os.path.normpath(path)                 # "...\um\" has an empty basename
        um = path if os.path.basename(path).lower() == "um" else os.path.join(path, "um")
        shared = os.path.join(os.path.dirname(um), "shared")
        if os.path.isdir(um):
            name = os.path.basename(os.path.dirname(um))
            try:
                key = (1, tuple(map(int, name.split("."))))
            except ValueError:
                key = (0, name)
            if best is None or key > best[0]:
                best = (key, um, shared)
    if not best:
        return None, None
    return best[1], best[2]


def parse_sdk_guids(um_dir, shared_dir):
    """name -> 11-tuple for every DEFINE_GUID in the SDK's um/ and shared/ headers."""
    found = {}
    for d in (um_dir, shared_dir):
        if not d or not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*.h")):
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except OSError:
                continue
            for m in re.finditer(r"DEFINE_GUID\(\s*((?:IID|DXGI_DEBUG)_[A-Za-z0-9_]+)\s*,\s*([^)]*)\)", text):
                nums = []
                for part in m.group(2).split(","):
                    part = part.strip().rstrip("LlUu")
                    try:
                        nums.append(int(part, 16))
                    except ValueError:
                        nums = None
                        break
                if nums and len(nums) == 11:
                    found.setdefault(m.group(1), tuple(nums))
    return found


def check_sdk():
    """Return (errors, checked_count, um_dir). checked_count is 0 when the SDK is missing."""
    um, shared = find_sdk_include()
    if not um:
        return [], 0, None
    sdk = parse_sdk_guids(um, shared)
    errors = []
    checked = 0
    for name, g in STANDARD_GUIDS.items():
        if name not in sdk:
            errors.append("%s: not found in the SDK headers" % name)
            continue
        checked += 1
        if sdk[name] != g:
            errors.append("%s: table %s != SDK %s" % (name, guid_text(g), guid_text(sdk[name])))
    for name in CONSOLE_IIDS:
        if name in sdk:
            errors.append("%s: expected a console-only IID, but the SDK defines it" % name)
    return errors, checked, um


def parse_scarab_iid_table(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for name, body in re.findall(r"GUID\s+((?:IID|DXGI_DEBUG)_[A-Za-z0-9_]+)\s*=\s*\{([^;]*)\};", text):
        nums = tuple(int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]+)", body))
        if len(nums) == 11:
            out[name] = nums
    return out


def parse_scarab_iid_console(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for m in re.finditer(r"GUID\s+(IID_[A-Za-z0-9_]+).*?ordinal\s+(\d+)", text):
        out[m.group(1)] = int(m.group(2))
    return out


def parse_scarab_catalogue(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        header = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                rows.append(tuple(parts[:3]))
    return rows


def check_scarab(root):
    """Cross-check the table against a SCARAB checkout. Returns (errors, notes)."""
    errors = []
    notes = []
    iid_table = os.path.join(root, "components", "runtime-bridge", "src", "iid_table.cpp")
    iid_console = os.path.join(root, "components", "extension-wrapper", "src", "iid_console.inc")
    catalogue = os.path.join(root, "components", "extension-wrapper", "docs",
                             "console-extension-catalogue.tsv")
    if os.path.exists(iid_table):
        theirs = parse_scarab_iid_table(iid_table)
        mine = dict(STANDARD_GUIDS)
        mine.update(CONSOLE_IIDS)
        if set(theirs) != set(mine):
            errors.append("SCARAB iid_table names differ from this table")
        for name in sorted(set(theirs) & set(mine)):
            if theirs[name] != mine[name]:
                errors.append("%s: SCARAB %s != table %s"
                              % (name, guid_text(theirs[name]), guid_text(mine[name])))
        notes.append("iid_table.cpp: %d GUID values compared" % len(theirs))
    else:
        errors.append("missing %s" % iid_table)
    if os.path.exists(iid_console):
        theirs = parse_scarab_iid_console(iid_console)
        known = {name for _n, name, _kind, _ev in DATA_EXPORTS}
        unexpected = set(theirs) - known
        if unexpected:
            errors.append("iid_console.inc has names not in the table: %s" % sorted(unexpected))
        mine = {name: n for n, name, _kind, _ev in DATA_EXPORTS
                if name in theirs}
        for name in sorted(set(theirs) & set(mine)):
            if theirs[name] != mine[name]:
                errors.append("%s: SCARAB ordinal %d != table %d"
                              % (name, theirs[name], mine[name]))
        notes.append("iid_console.inc: %d documented ordinals compared" % len(set(theirs) & set(mine)))
    else:
        errors.append("missing %s" % iid_console)
    if os.path.exists(catalogue):
        rows = parse_scarab_catalogue(catalogue)
        methods = [r for r in rows if r[0] == "method"]
        enums = [r for r in rows if r[0] == "enum"]
        notes.append("catalogue: %d method rows, %d enum rows" % (len(methods), len(enums)))
        if (len(methods), len(enums)) != (105, 106):
            notes.append("catalogue counts changed since the pin (expected 105/106)")
        present = {r[1] for r in methods}
        missing = sorted(CATALOGUE_NON_METHODS - present)
        if missing:
            errors.append("catalogue non-method rows missing: %s" % missing)
    else:
        errors.append("missing %s" % catalogue)
    return errors, notes


def markdown_table():
    """The 116-row table used in the doc, for maintenance."""
    lines = ["| # | export | kind | evidence |", "|---:|---|---|---|"]
    for n, name, kind in FUNCTION_EXPORTS:
        lines.append("| %d | `%s` | %s | documented |" % (n, name, kind))
    for n, name, kind, ev in DATA_EXPORTS:
        lines.append("| %d | `%s` | %s | %s |" % (n, name, kind, ev))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="run the invariant checks")
    ap.add_argument("--sdk", action="store_true", help="also check the 83 standard GUIDs against the SDK")
    ap.add_argument("--require-sdk", action="store_true", help="fail if the SDK is not found")
    ap.add_argument("--scarab", metavar="PATH", help="also cross-check a SCARAB checkout")
    ap.add_argument("--markdown", action="store_true", help="print the 116-row table")
    args = ap.parse_args(argv)

    if args.markdown:
        print(markdown_table())
        return 0

    errors = []
    errors += check_slots()
    errors += check_data_classes()
    errors += check_guid_values()
    errors += check_iid_order()
    errors += check_unrecovered_console_iids()

    sdk_checked = 0
    if args.sdk or args.require_sdk:
        sdk_errors, sdk_checked, um = check_sdk()
        errors += sdk_errors
        if sdk_checked:
            print("sdk: %d/%d standard GUIDs match %s" % (sdk_checked, len(STANDARD_GUIDS), um))
        elif args.require_sdk:
            errors.append("Windows SDK headers not found")
        else:
            print("sdk: SKIP (Windows SDK headers not found)")

    if args.scarab:
        scarab_errors, notes = check_scarab(args.scarab)
        errors += scarab_errors
        for note in notes:
            print("scarab: " + note)

    if errors:
        for e in errors:
            print("FAIL: " + e)
        return 1
    if args.check or args.sdk or args.scarab:
        print("PASS: 116 slots (23 functions + 93 GUID data), %d standard + %d console GUIDs, "
              "%d unresolved command-list IIDs absent from local tables"
              % (len(STANDARD_GUIDS), len(CONSOLE_IIDS), len(UNRECOVERED_CONSOLE_IIDS)))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
