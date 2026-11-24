#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "shims.h"

namespace {

using Microsoft::WRL::ComPtr;
using SerializeFn = decltype(&::D3D12SerializeVersionedRootSignature);
using DeserializeFn = decltype(&::D3D12CreateRootSignatureDeserializer);

void require(bool condition, const char* stage) {
    if (!condition) throw std::runtime_error(stage);
}

void require_success(HRESULT hr, const char* stage) {
    if (FAILED(hr)) {
        std::fprintf(stderr, "%s returned 0x%08lX\n", stage, static_cast<unsigned long>(hr));
        throw std::runtime_error(stage);
    }
}

void require_same_failure(HRESULT direct, HRESULT shim, const char* stage) {
    std::printf("%s (direct 0x%08lX, shim 0x%08lX)\n",
                stage, static_cast<unsigned long>(direct), static_cast<unsigned long>(shim));
    require(FAILED(direct) && direct == shim, stage);
}

ComPtr<ID3DBlob> compare_serialization(SerializeFn serialize,
                                     const D3D12_VERSIONED_ROOT_SIGNATURE_DESC& desc) {
    ComPtr<ID3DBlob> direct, shim, direct_error, shim_error;
    const HRESULT direct_hr = D3D12SerializeVersionedRootSignature(&desc, &direct, &direct_error);
    const HRESULT shim_hr = serialize(&desc, &shim, &shim_error);
    require_success(direct_hr, "serialize directly");
    require_success(shim_hr, "serialize via shim");
    require(direct_hr == shim_hr && direct && shim && !direct_error && !shim_error,
            "serialization results match");

    const SIZE_T size = direct->GetBufferSize();
    require(size != 0 && size == shim->GetBufferSize() &&
            std::memcmp(direct->GetBufferPointer(), shim->GetBufferPointer(), size) == 0,
            "serialized bytes match");
    return shim;
}

void check_version_1_0(const D3D12_ROOT_SIGNATURE_DESC& desc) {
    require(desc.NumParameters == 2 && desc.NumStaticSamplers == 0 &&
            desc.Flags == D3D12_ROOT_SIGNATURE_FLAG_NONE, "1.0 root signature header");
    const auto& constants = desc.pParameters[0];
    require(constants.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS &&
            constants.Constants.ShaderRegister == 2 && constants.Constants.RegisterSpace == 3 &&
            constants.Constants.Num32BitValues == 4 &&
            constants.ShaderVisibility == D3D12_SHADER_VISIBILITY_ALL, "1.0 root constants");
    const auto& table = desc.pParameters[1];
    require(table.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
            table.DescriptorTable.NumDescriptorRanges == 1 &&
            table.ShaderVisibility == D3D12_SHADER_VISIBILITY_ALL, "1.0 descriptor table");
    const auto& range = table.DescriptorTable.pDescriptorRanges[0];
    require(range.RangeType == D3D12_DESCRIPTOR_RANGE_TYPE_SRV && range.NumDescriptors == 2 &&
            range.BaseShaderRegister == 5 && range.RegisterSpace == 6 &&
            range.OffsetInDescriptorsFromTableStart == D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            "1.0 nested descriptor range");
}

void test_version_1_0(SerializeFn serialize, DeserializeFn deserialize, ID3D12Device& device) {
    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 5, 6,
                                       D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
    std::array<D3D12_ROOT_PARAMETER, 2> parameters{};
    parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[0].Constants = {2, 3, 4};
    parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    parameters[1].DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {static_cast<UINT>(parameters.size()), parameters.data(), 0, nullptr,
                     D3D12_ROOT_SIGNATURE_FLAG_NONE};

    const auto blob = compare_serialization(serialize, desc);
    const void* data = blob->GetBufferPointer();
    const SIZE_T size = blob->GetBufferSize();

    ComPtr<ID3D12RootSignatureDeserializer> direct_layout, shim_layout;
    const HRESULT direct_hr = D3D12CreateRootSignatureDeserializer(data, size, IID_PPV_ARGS(&direct_layout));
    const HRESULT shim_hr = deserialize(data, size, IID_PPV_ARGS(&shim_layout));
    require_success(direct_hr, "deserialize 1.0 directly");
    require_success(shim_hr, "deserialize 1.0 via shim");
    require(direct_hr == shim_hr && direct_layout && shim_layout, "1.0 deserialization results match");
    check_version_1_0(*direct_layout->GetRootSignatureDesc());
    check_version_1_0(*shim_layout->GetRootSignatureDesc());

    ComPtr<ID3D12RootSignature> signature;
    require_success(device.CreateRootSignature(0, data, size, IID_PPV_ARGS(&signature)),
                    "create WARP 1.0 root signature");
    require(signature != nullptr, "WARP returned a 1.0 root signature");

    ComPtr<ID3DBlob> direct_no_error, shim_no_error;
    const HRESULT direct_optional = D3D12SerializeVersionedRootSignature(&desc, &direct_no_error, nullptr);
    const HRESULT shim_optional = serialize(&desc, &shim_no_error, nullptr);
    require_success(direct_optional, "serialize 1.0 directly with null error output");
    require_success(shim_optional, "serialize 1.0 via shim with null error output");
    require(direct_optional == shim_optional && direct_no_error && shim_no_error &&
            direct_no_error->GetBufferSize() == size && shim_no_error->GetBufferSize() == size &&
            std::memcmp(direct_no_error->GetBufferPointer(), shim_no_error->GetBufferPointer(), size) == 0,
            "serialization with null error output matches");

    const std::array<unsigned char, 4> malformed{};
    ComPtr<ID3D12RootSignatureDeserializer> direct_invalid, shim_invalid;
    const HRESULT direct_bad = D3D12CreateRootSignatureDeserializer(
        malformed.data(), malformed.size(), IID_PPV_ARGS(&direct_invalid));
    const HRESULT shim_bad = deserialize(malformed.data(), malformed.size(), IID_PPV_ARGS(&shim_invalid));
    require_same_failure(direct_bad, shim_bad, "deserialize malformed blob");
    require(!direct_invalid && !shim_invalid, "malformed blob returns no interface");

    ComPtr<ID3D12Device> direct_wrong_iid, shim_wrong_iid;
    const HRESULT direct_wrong = D3D12CreateRootSignatureDeserializer(data, size, IID_PPV_ARGS(&direct_wrong_iid));
    const HRESULT shim_wrong = deserialize(data, size, IID_PPV_ARGS(&shim_wrong_iid));
    require_same_failure(direct_wrong, shim_wrong, "deserialize with unsupported IID");
    require(!direct_wrong_iid && !shim_wrong_iid, "unsupported IID returns no interface");

    parameters[1] = parameters[0];
    ComPtr<ID3DBlob> direct_invalid_blob, shim_invalid_blob, direct_error, shim_error;
    const HRESULT direct_invalid_desc =
        D3D12SerializeVersionedRootSignature(&desc, &direct_invalid_blob, &direct_error);
    const HRESULT shim_invalid_desc = serialize(&desc, &shim_invalid_blob, &shim_error);
    require_same_failure(direct_invalid_desc, shim_invalid_desc, "serialize duplicate root constant bindings");
    require(!direct_invalid_blob && !shim_invalid_blob &&
            (direct_error.Get() != nullptr) == (shim_error.Get() != nullptr),
            "invalid root description outputs match");

    std::puts("PASS root signature 1.0 round-trip, optional error output, and negative cases");
}

void test_version_1_1(SerializeFn serialize, DeserializeFn deserialize, ID3D12Device& device) {
    const D3D12_DESCRIPTOR_RANGE1 range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 5, 6,
                                        D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
                                        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
    D3D12_ROOT_PARAMETER1 table{};
    table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    table.DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
    desc.Desc_1_1 = {1, &table, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};

    const auto blob = compare_serialization(serialize, desc);
    const void* data = blob->GetBufferPointer();
    const SIZE_T size = blob->GetBufferSize();
    ComPtr<ID3D12VersionedRootSignatureDeserializer> versioned;
    require_success(D3D12CreateVersionedRootSignatureDeserializer(data, size, IID_PPV_ARGS(&versioned)),
                    "deserialize 1.1 with Windows versioned API");
    require(versioned != nullptr, "Windows returned a versioned deserializer");
    const auto& layout = *versioned->GetUnconvertedRootSignatureDesc();
    require(layout.Version == D3D_ROOT_SIGNATURE_VERSION_1_1 &&
            layout.Desc_1_1.NumParameters == 1, "1.1 version and parameter count");
    const auto& decoded_table = layout.Desc_1_1.pParameters[0];
    require(decoded_table.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
            decoded_table.DescriptorTable.NumDescriptorRanges == 1 &&
            decoded_table.DescriptorTable.pDescriptorRanges[0].Flags == range.Flags,
            "1.1 descriptor range flag survives");

    ComPtr<ID3D12RootSignatureDeserializer> direct_legacy, shim_legacy;
    const HRESULT direct_hr = D3D12CreateRootSignatureDeserializer(data, size, IID_PPV_ARGS(&direct_legacy));
    const HRESULT shim_hr = deserialize(data, size, IID_PPV_ARGS(&shim_legacy));
    require(direct_hr == shim_hr && (direct_legacy.Get() != nullptr) == (shim_legacy.Get() != nullptr),
            "legacy deserialization of 1.1 matches Windows");
    if (SUCCEEDED(direct_hr)) {
        require(direct_legacy && shim_legacy, "legacy deserializers return interfaces on success");
        const auto& direct_desc = *direct_legacy->GetRootSignatureDesc();
        const auto& shim_desc = *shim_legacy->GetRootSignatureDesc();
        require(direct_desc.NumParameters == 1 && shim_desc.NumParameters == 1 &&
                direct_desc.Flags == shim_desc.Flags, "legacy projection header matches");
        const auto& direct_table = direct_desc.pParameters[0];
        const auto& shim_table = shim_desc.pParameters[0];
        require(direct_table.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
                shim_table.ParameterType == direct_table.ParameterType &&
                direct_table.DescriptorTable.NumDescriptorRanges == 1 &&
                shim_table.DescriptorTable.NumDescriptorRanges == 1 &&
                shim_table.DescriptorTable.pDescriptorRanges[0].BaseShaderRegister ==
                    direct_table.DescriptorTable.pDescriptorRanges[0].BaseShaderRegister,
                "legacy projection descriptor table matches");
    }

    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature{D3D_ROOT_SIGNATURE_VERSION_1_1};
    require_success(device.CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature, sizeof(feature)),
                    "query WARP root signature support");
    if (feature.HighestVersion >= D3D_ROOT_SIGNATURE_VERSION_1_1) {
        ComPtr<ID3D12RootSignature> signature;
        require_success(device.CreateRootSignature(0, data, size, IID_PPV_ARGS(&signature)),
                        "create WARP 1.1 root signature");
        require(signature != nullptr, "WARP returned a 1.1 root signature");
        std::puts("PASS WARP 1.1 root signature creation");
    } else {
        std::puts("SKIP WARP 1.1 root signature creation (feature version unavailable)");
    }
    std::puts("PASS root signature 1.1 serialization, versioned reflection, and legacy comparison");
}

void test_version_1_2(SerializeFn serialize, DeserializeFn deserialize, ID3D12Device& device) {
    // Older in-box D3D12 runtimes refuse root signature 1.2, so the whole 1.2 path is skipped there.
    // Windows 11 23H2 answers with a lower HighestVersion; Windows 10 doesn't know 1.2 and fails the
    // query with E_INVALIDARG instead.
    D3D12_FEATURE_DATA_ROOT_SIGNATURE feature{D3D_ROOT_SIGNATURE_VERSION_1_2};
    const HRESULT query = device.CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &feature, sizeof(feature));
    if (query == E_INVALIDARG) {
        std::puts("SKIP root signature 1.2 (runtime does not know version 1.2)");
        return;
    }
    require_success(query, "query WARP root signature support");
    if (feature.HighestVersion < D3D_ROOT_SIGNATURE_VERSION_1_2) {
        std::puts("SKIP root signature 1.2 (runtime HighestVersion < 1.2)");
        return;
    }

    const D3D12_DESCRIPTOR_RANGE1 range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 5, 6,
                                        D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE,
                                        D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
    D3D12_ROOT_PARAMETER1 table{};
    table.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    table.DescriptorTable = {1, &range};
    table.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_2;
    desc.Desc_1_2 = {1, &table, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};

    const auto blob = compare_serialization(serialize, desc);
    const void* data = blob->GetBufferPointer();
    const SIZE_T size = blob->GetBufferSize();
    ComPtr<ID3D12VersionedRootSignatureDeserializer> versioned;
    require_success(D3D12CreateVersionedRootSignatureDeserializer(data, size, IID_PPV_ARGS(&versioned)),
                    "deserialize 1.2 with Windows versioned API");
    require(versioned != nullptr, "Windows returned a versioned deserializer");
    const auto& layout = *versioned->GetUnconvertedRootSignatureDesc();
    require(layout.Version == D3D_ROOT_SIGNATURE_VERSION_1_2 &&
            layout.Desc_1_2.NumParameters == 1, "1.2 version and parameter count");
    const auto& decoded_table = layout.Desc_1_2.pParameters[0];
    require(decoded_table.ParameterType == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE &&
            decoded_table.DescriptorTable.NumDescriptorRanges == 1 &&
            decoded_table.DescriptorTable.pDescriptorRanges[0].Flags == range.Flags &&
            decoded_table.ShaderVisibility == table.ShaderVisibility,
            "1.2 descriptor table and shader visibility survive");
    require(layout.Desc_1_2.Flags == D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
            "1.2 root signature flags preserved");

    ComPtr<ID3D12RootSignatureDeserializer> direct_legacy, shim_legacy;
    const HRESULT direct_hr = D3D12CreateRootSignatureDeserializer(data, size, IID_PPV_ARGS(&direct_legacy));
    const HRESULT shim_hr = deserialize(data, size, IID_PPV_ARGS(&shim_legacy));
    require(direct_hr == shim_hr && (direct_legacy.Get() != nullptr) == (shim_legacy.Get() != nullptr),
            "legacy deserialization of 1.2 matches Windows");
    if (SUCCEEDED(direct_hr)) {
        require(direct_legacy && shim_legacy, "legacy deserializers return interfaces on success");
        const auto& direct_desc = *direct_legacy->GetRootSignatureDesc();
        const auto& shim_desc = *shim_legacy->GetRootSignatureDesc();
        require(direct_desc.NumParameters == 1 && shim_desc.NumParameters == 1 &&
                direct_desc.Flags == shim_desc.Flags, "legacy projection header matches");
    }

    ComPtr<ID3D12RootSignature> signature;
    require_success(device.CreateRootSignature(0, data, size, IID_PPV_ARGS(&signature)),
                    "create WARP 1.2 root signature");
    require(signature != nullptr, "WARP returned a 1.2 root signature");
    std::puts("PASS WARP 1.2 root signature creation");
    std::puts("PASS root signature 1.2 serialization, versioned reflection, and legacy comparison");
}

}  // namespace

int main() {
    try {
        const auto serialize = reinterpret_cast<SerializeFn>(
            rt::shims::find("d3d12_xs.dll", "D3D12SerializeVersionedRootSignature"));
        const auto deserialize = reinterpret_cast<DeserializeFn>(
            rt::shims::find("d3d12_xs.dll", "D3D12CreateRootSignatureDeserializer"));
        require(serialize && deserialize, "resolve both root signature imports from the shim registry");

        ComPtr<IDXGIFactory4> factory;
        ComPtr<IDXGIAdapter> adapter;
        ComPtr<ID3D12Device> device;
        require_success(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "create DXGI factory");
        if (const HRESULT warp = factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)); FAILED(warp)) {
            // Windows always has WARP; Wine doesn't. CTest reads exit code 77 as a skip (SKIP_RETURN_CODE).
            std::printf("SKIP no WARP adapter (0x%08lX)\n", static_cast<unsigned long>(warp));
            return 77;
        }
        require_success(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
                        "create WARP device");
        test_version_1_0(serialize, deserialize, *device.Get());
        test_version_1_1(serialize, deserialize, *device.Get());
        test_version_1_2(serialize, deserialize, *device.Get());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
    return 0;
}
