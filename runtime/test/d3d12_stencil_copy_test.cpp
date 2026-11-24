// Synthetic whole-resource depth/stencil copies, including the multi-mip array case that loses stencil on
// RX 5700 XT. Compare both planes with the host and with the upload pattern, including unchanged sources.
// CTest selects WARP; LEONIDA_HOST_ADAPTER can select native D3D12 for a manual second reference run.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"

namespace {
bool matches_upload(const copycmp::TexCase& tc, const copycmp::Outcome& out) {
    if (!out.error.empty()) return false;
    const UINT planes = UINT(tc.mips) * tc.depth_or_layers;
    const UINT subresources = planes * 2;
    if (out.regions.size() != size_t(subresources) * 6) return false;
    for (const UINT set : {0u, 4u}) { // CopyResource destination and original source
        for (UINT sub = 0; sub < subresources; ++sub) {
            const auto& r = out.regions[size_t(set) * subresources + sub];
            for (size_t row = 0; row < r.rows; ++row) {
                for (size_t col = 0; col < r.row_bytes; ++col) {
                    if (r.x8 && col % 4 == 3) continue; // unspecified D24 padding
                    auto expected = copycmp::fill_byte(7919u + sub, row * 131 + col, tc.floaty);
                    if (tc.floaty && sub < planes && col % 4 == 3) expected |= 0x20;
                    const auto actual = out.bytes[r.offset + row * r.pitch + col];
                    if (actual != expected) {
                        std::fprintf(stderr, "%s %s row %zu byte %zu: got %02x, expected %02x\n",
                                     tc.name, r.label.c_str(), row, col, unsigned(actual), unsigned(expected));
                        return false;
                    }
                }
            }
        }
    }
    return true;
}
} // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    copycmp::Ptr<ID3D12Device> ours, host;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
    if (copycmp::host_is_wine()) return 77;  // every case is compared with the host
    std::string adapter;
    if (!copycmp::create_host(host, adapter)) return 1;
    std::printf("depth/stencil copy reference: %s\n", adapter.c_str());
    unsigned cases = 0;
    for (const auto format : {DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT}) {
        for (const auto extent : {VkExtent2D{5, 3}, VkExtent2D{20, 12}, VkExtent2D{33, 17}}) {
            for (const UINT16 mips : {UINT16(1), UINT16(3)}) {
                for (const UINT layers : {1u, 2u}) {
                    char name[96];
                    std::snprintf(name, sizeof(name), "%s %ux%u mips=%u layers=%u",
                                  format == DXGI_FORMAT_D24_UNORM_S8_UINT ? "D24S8" : "D32S8",
                                  extent.width, extent.height, unsigned(mips), layers);
                    const copycmp::TexCase tc{name, D3D12_RESOURCE_DIMENSION_TEXTURE2D, format,
                        extent.width, extent.height, layers, mips,
                        format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT, true, false};
                    copycmp::Run a, b;
                    if (!a.init(ours.p, true) || !b.init(host.p, false)) return 1;
                    a.begin(); b.begin();
                    copycmp::Outcome actual, reference;
                    copycmp::run_texture(a, host.p, tc, actual);
                    copycmp::run_texture(b, host.p, tc, reference);
                    if (copycmp::compare(name, actual, reference) ||
                        !matches_upload(tc, reference) || !matches_upload(tc, actual)) return 1;
                    ++cases;
                }
            }
        }
    }
    std::printf("depth/stencil copies: %u cases matched host and upload patterns on both planes\n", cases);
    return vk.validation_errors() ? 1 : 0;
}
