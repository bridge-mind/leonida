// The subset of the public FidelityFX API that fsr_dispatch_test.cpp calls, written out from the FidelityFX SDK
// v2.4.0 headers Kits/FidelityFX/api/include/ffx_api.h, api/include/ffx_api_types.h,
// api/include/dx12/ffx_api_dx12.h and
// upscalers/include/ffx_upscale.h (https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK). Those headers carry
// the MIT license below. Only declarations are here: no AMD code, and nothing from the game's DLL.
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
// to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of
// the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
#pragma once

#include <cstdint>

namespace ffx {

using ReturnCode = uint32_t;
using StructType = uint64_t;
using Context = void*;

enum : ReturnCode {
    RETURN_OK = 0,
    RETURN_ERROR = 1,
    RETURN_ERROR_UNKNOWN_DESCTYPE = 2,
    RETURN_ERROR_RUNTIME_ERROR = 3,
    RETURN_NO_PROVIDER = 4,
    RETURN_ERROR_MEMORY = 5,
    RETURN_ERROR_PARAMETER = 6,
    RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE = 7,
};

constexpr uint32_t EFFECT_MASK = 0x00ff0000u;
constexpr uint32_t BACKEND_MASK = 0xff000000u;
constexpr uint32_t EFFECT_ID_UPSCALE = 0x00010000u;
constexpr uint32_t BACKEND_ID_DX12 = 0x00000000u;
constexpr uint32_t BACKEND_ID_XBOX = 0x02000000u;  // declared in ffx_api.h; its descriptors are not public

constexpr StructType effect_sub_id(uint32_t effect, uint32_t sub) { return (effect & EFFECT_MASK) | (sub & ~EFFECT_MASK); }
constexpr StructType backend_sub_id(uint32_t backend, uint32_t sub) {
    return (backend & BACKEND_MASK) | (sub & ~BACKEND_MASK);
}

struct Header {
    StructType type;
    Header* pNext;
};

using Message = void (*)(uint32_t type, const wchar_t* message);

constexpr StructType CONFIGURE_DESC_TYPE_GLOBALDEBUG = 7u;
constexpr uint32_t GLOBALDEBUG_LEVEL_VERBOSE = 0xfffffffu;
struct ConfigureDescGlobalDebug {
    Header header;
    uint64_t effectId;
    Message fpMessage;
    uint32_t debugLevel;
};

constexpr StructType QUERY_DESC_TYPE_GET_VERSIONS = 4u;
struct QueryDescGetVersions {
    Header header;
    uint64_t createDescType;
    void* device;
    uint64_t* outputCount;
    uint64_t* versionIds;
    const char** versionNames;
};

struct Dimensions2D {
    uint32_t width;
    uint32_t height;
};

constexpr StructType CREATE_CONTEXT_DESC_TYPE_UPSCALE = effect_sub_id(EFFECT_ID_UPSCALE, 0x00);
constexpr uint32_t UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE = 1u << 0;
constexpr uint32_t UPSCALE_ENABLE_DEPTH_INVERTED = 1u << 3;
constexpr uint32_t UPSCALE_ENABLE_AUTO_EXPOSURE = 1u << 5;
constexpr uint32_t UPSCALE_ENABLE_DEBUG_CHECKING = 1u << 7;
struct CreateContextDescUpscale {
    Header header;
    uint32_t flags;
    Dimensions2D maxRenderSize;
    Dimensions2D maxUpscaleSize;
    Message fpMessage;
};

constexpr uint32_t upscaler_version(uint32_t major, uint32_t minor, uint32_t patch) {
    return (major << 22) | (minor << 12) | patch;
}
constexpr StructType CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION = effect_sub_id(EFFECT_ID_UPSCALE, 0x0b);
struct CreateContextDescUpscaleVersion {
    Header header;
    uint32_t version;  // FFX_UPSCALER_VERSION the application was built against
};

constexpr StructType QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE = effect_sub_id(EFFECT_ID_UPSCALE, 0x02);
struct QueryDescUpscaleGetUpscaleRatioFromQualityMode {
    Header header;
    uint32_t qualityMode;
    float* pOutUpscaleRatio;
};

constexpr StructType QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE = effect_sub_id(EFFECT_ID_UPSCALE, 0x03);
struct QueryDescUpscaleGetRenderResolutionFromQualityMode {
    Header header;
    uint32_t displayWidth;
    uint32_t displayHeight;
    uint32_t qualityMode;
    uint32_t* pOutRenderWidth;
    uint32_t* pOutRenderHeight;
};

enum : uint32_t { QUALITY_MODE_NATIVEAA = 0, QUALITY_MODE_QUALITY = 1, QUALITY_MODE_PERFORMANCE = 3 };

// ffx_api_dx12.h. The DLL's own backend descriptor would be a BACKEND_ID_XBOX sub-id; its layout is not public.
constexpr StructType CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12 = backend_sub_id(BACKEND_ID_DX12, 0x02);
struct CreateBackendDX12Desc {
    Header header;
    void* device;  // ID3D12Device*
};

// ffx_api_types.h: resources and the values used below.
enum : uint32_t {
    SURFACE_FORMAT_R16G16B16A16_FLOAT = 4,
    SURFACE_FORMAT_R16G16_FLOAT = 18,
    SURFACE_FORMAT_R32_FLOAT = 28,
};
enum : uint32_t { RESOURCE_TYPE_TEXTURE2D = 2 };
enum : uint32_t { RESOURCE_USAGE_READ_ONLY = 0, RESOURCE_USAGE_UAV = 1u << 1, RESOURCE_USAGE_DEPTHTARGET = 1u << 2 };
enum : uint32_t { RESOURCE_STATE_UNORDERED_ACCESS = 1u << 1, RESOURCE_STATE_COMPUTE_READ = 1u << 2 };

struct ResourceDescription {
    uint32_t type;
    uint32_t format;
    uint32_t width;   // or size, for a buffer
    uint32_t height;  // or stride
    uint32_t depth;   // or alignment
    uint32_t mipCount;
    uint32_t flags;
    uint32_t usage;
};

struct Resource {
    void* resource;  // ID3D12Resource*
    ResourceDescription description;
    uint32_t state;
};

struct FloatCoords2D {
    float x;
    float y;
};

// ffx_upscale.h
constexpr StructType DISPATCH_DESC_TYPE_UPSCALE = effect_sub_id(EFFECT_ID_UPSCALE, 0x01);
struct DispatchDescUpscale {
    Header header;
    void* commandList;
    Resource color;
    Resource depth;
    Resource motionVectors;
    Resource exposure;
    Resource reactive;
    Resource transparencyAndComposition;
    Resource output;
    FloatCoords2D jitterOffset;
    FloatCoords2D motionVectorScale;
    Dimensions2D renderSize;
    Dimensions2D upscaleSize;
    bool enableSharpening;
    float sharpness;
    float frameTimeDelta;
    float preExposure;
    bool reset;
    float cameraNear;
    float cameraFar;
    float cameraFovAngleVertical;
    float viewSpaceToMetersFactor;
    uint32_t flags;
};

using PfnCreateContext = ReturnCode (*)(Context* context, Header* desc, const void* memCb);
using PfnDestroyContext = ReturnCode (*)(Context* context, const void* memCb);
using PfnConfigure = ReturnCode (*)(Context* context, const Header* desc);
using PfnQuery = ReturnCode (*)(Context* context, Header* desc);
using PfnDispatch = ReturnCode (*)(Context* context, const Header* desc);

inline const char* return_code_name(ReturnCode rc) {
    switch (rc) {
        case RETURN_OK: return "OK";
        case RETURN_ERROR: return "ERROR";
        case RETURN_ERROR_UNKNOWN_DESCTYPE: return "ERROR_UNKNOWN_DESCTYPE";
        case RETURN_ERROR_RUNTIME_ERROR: return "ERROR_RUNTIME_ERROR";
        case RETURN_NO_PROVIDER: return "NO_PROVIDER";
        case RETURN_ERROR_MEMORY: return "ERROR_MEMORY";
        case RETURN_ERROR_PARAMETER: return "ERROR_PARAMETER";
        case RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE: return "PROVIDER_NO_SUPPORT_NEW_DESCTYPE";
        default: return "?";
    }
}

}  // namespace ffx
