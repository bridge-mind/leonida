// bindless::size_bindings with synthetic device limits: no GPU. The counts of set 0 come from the device's descriptor
// limits (vulkan_bindless.h): today's sizes where a device is at them, larger 2D, sampler and storage arrays where it
// has room, and less where it is below them.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_bindless.h"

#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace leonida;

namespace {

struct Device {
    VkPhysicalDeviceLimits limits{};
    VkPhysicalDeviceVulkan11Properties v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
};

// The same descriptor limit for every type, update-after-bind or not; the others are unlimited.
Device uniform(std::uint32_t plain, std::uint32_t update_after_bind) {
    Device d;
    auto& l = d.limits;
    l.maxPerStageDescriptorSampledImages = l.maxDescriptorSetSampledImages = plain;
    l.maxPerStageDescriptorSamplers = l.maxDescriptorSetSamplers = plain;
    l.maxPerStageDescriptorStorageImages = l.maxDescriptorSetStorageImages = plain;
    l.maxPerStageResources = 0xffffffffu;
    l.maxColorAttachments = 8;
    l.maxSamplerAllocationCount = 4000;  // NVIDIA's; the minimum Vulkan allows
    auto& u = d.v12;
    u.maxPerStageDescriptorUpdateAfterBindSampledImages = update_after_bind;
    u.maxDescriptorSetUpdateAfterBindSampledImages = update_after_bind;
    u.maxPerStageDescriptorUpdateAfterBindSamplers = u.maxDescriptorSetUpdateAfterBindSamplers = update_after_bind;
    u.maxPerStageDescriptorUpdateAfterBindStorageImages = update_after_bind;
    u.maxDescriptorSetUpdateAfterBindStorageImages = update_after_bind;
    u.maxPerStageUpdateAfterBindResources = 0xffffffffu;
    u.maxUpdateAfterBindDescriptorsInAllPools = 0xffffffffu;
    d.v11.maxPerSetDescriptors = 0xffffffffu;
    return d;
}

BindlessFeatures with_ms(bool update_after_bind) {
    BindlessFeatures f{};
    f.update_after_bind = update_after_bind;
    f.ms_samples.fill(VK_SAMPLE_COUNT_4_BIT);
    return f;
}

struct Totals { std::uint32_t sampled = 0, samplers = 0, storage = 0, all = 0; };
Totals totals(const BindlessFeatures& f) {
    Totals t;
    for (std::uint32_t b = 0; b < bindless::binding_count; ++b) {
        if (!bindless::binding_enabled(b, f)) continue;
        const auto type = bindless::bindings[b].type;
        if (type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) t.sampled += f.counts[b];
        if (type == VK_DESCRIPTOR_TYPE_SAMPLER) t.samplers += f.counts[b];
        if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) t.storage += f.counts[b];
        t.all += f.counts[b];
    }
    return t;
}

std::uint32_t count(const BindlessFeatures& f, ImageDimension dimension) {
    return f.counts[bindless::sampled_binding(dimension)];
}
std::uint32_t storage(const BindlessFeatures& f, ImageDimension dimension) {
    return f.counts[bindless::storage_binding(dimension, 0)];
}

// Every storage array of a dimension, the r32ui one included, holds the same indices.
bool storage_consistent(const BindlessFeatures& f) {
    for (std::uint32_t d = 0; d < 4; ++d) {
        const auto dimension = ImageDimension(d);
        const auto held = storage(f, dimension);
        if (held > count(f, dimension)) return false;
        for (std::uint32_t c = 0; c < 3; ++c)
            if (f.counts[bindless::storage_binding(dimension, c)] != held) return false;
        if (bindless::r32ui_binding(dimension) != bindless::no_binding &&
            f.counts[bindless::r32ui_binding(dimension)] != held)
            return false;
    }
    return true;
}

}  // namespace

int main() {
    using D = ImageDimension;
    constexpr auto defaults = bindless::default_counts();
    static_assert(defaults[0] == 4096 && defaults[bindless::sampler_binding] == 1024 && defaults[8] == 256);

    // An RTX 5070 (driver 621.24): 1048576 of each descriptor type, update-after-bind or not, 4000 samplers.
    for (const bool update_after_bind : {true, false}) {
        const Device d = uniform(1048576, 1048576);
        BindlessFeatures f = with_ms(update_after_bind);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12));
        EXPECT(count(f, D::dim_2d) == bindless::max_2d_images);
        EXPECT(f.counts[bindless::sampler_binding] == 4000 - bindless::reserved_samplers);
        EXPECT(f.counts[bindless::sampler_binding] > 2048);  // a full sampler heap and the fallback
        EXPECT(storage(f, D::dim_2d) == bindless::max_storage_indices);
        EXPECT(storage(f, D::dim_2d_array) == 1024 && storage(f, D::dim_3d) == 256 && storage(f, D::dim_1d) == 256);
        EXPECT(count(f, D::dim_2d_array) == 1024 && count(f, D::dim_3d) == 256 && count(f, D::dim_1d) == 256);
        EXPECT(f.ms_samples[0] == VK_SAMPLE_COUNT_4_BIT && f.ms_samples[2] == VK_SAMPLE_COUNT_4_BIT);
        EXPECT(storage_consistent(f));
        if (update_after_bind)
            std::printf("RTX 5070-like: %u 2D images, %u samplers, %u storage indices in 2D, %u descriptors\n",
                        count(f, D::dim_2d), f.counts[bindless::sampler_binding], storage(f, D::dim_2d), totals(f).all);
    }

    // At today's sizes exactly (5632 sampled images, 1024 samplers, 3584 storage images): today's set, no MS pairs.
    {
        Device d = uniform(1048576, 1048576);
        d.v12.maxPerStageDescriptorUpdateAfterBindSampledImages = 5632;
        d.v12.maxPerStageDescriptorUpdateAfterBindSamplers = 1024;
        d.v12.maxPerStageDescriptorUpdateAfterBindStorageImages = 3584;
        BindlessFeatures f = with_ms(true);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12));
        EXPECT(count(f, D::dim_2d) == 4096 && f.counts[bindless::sampler_binding] == 1024 &&
               storage(f, D::dim_2d) == 256);
        for (std::uint32_t b = 0; b < bindless::first_ms_binding; ++b)
            if (bindless::bindings[b].type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) EXPECT(f.counts[b] == defaults[b]);
        EXPECT(f.ms_samples == (std::array{VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_1_BIT}));
        EXPECT(storage_consistent(f));
    }

    // A small device below today's sizes: smaller arrays that fit, and false.
    {
        Device d = uniform(1048576, 1048576);
        d.limits.maxPerStageDescriptorSampledImages = 3000;
        d.limits.maxPerStageDescriptorSamplers = 512;
        d.limits.maxPerStageDescriptorStorageImages = 2000;
        d.limits.maxPerStageResources = 4500;
        BindlessFeatures f = with_ms(false);
        EXPECT(!bindless::size_bindings(f, d.limits, d.v11, d.v12));
        const Totals t = totals(f);
        EXPECT(t.sampled <= 3000 && t.samplers <= 512 && t.storage <= 2000 && t.sampled + t.storage + 1 + 8 <= 4500);
        EXPECT(count(f, D::dim_2d) >= 1 && count(f, D::dim_2d) < 4096 && f.counts[bindless::sampler_binding] == 512);
        EXPECT(f.ms_samples[0] == VK_SAMPLE_COUNT_1_BIT);
        EXPECT(storage_consistent(f));
    }

    // Update-after-bind and plain limits differ: each layout is sized by its own.
    {
        const Device d = uniform(16384, 1048576);
        BindlessFeatures uab = with_ms(true), plain = with_ms(false);
        EXPECT(bindless::size_bindings(uab, d.limits, d.v11, d.v12));
        EXPECT(bindless::size_bindings(plain, d.limits, d.v11, d.v12));
        EXPECT(count(uab, D::dim_2d) == bindless::max_2d_images);
        const Totals t = totals(plain);
        EXPECT(count(plain, D::dim_2d) > 4096 && count(plain, D::dim_2d) < 16384);
        EXPECT(t.sampled + bindless::growth_headroom <= 16384 && t.storage + bindless::growth_headroom <= 16384);
        EXPECT(plain.ms_samples[0] == VK_SAMPLE_COUNT_4_BIT && storage_consistent(plain));
        EXPECT(plain.counts[bindless::sampler_binding] == 4000 - bindless::reserved_samplers);
    }

    // maxPerSetDescriptors stops growth but never takes a count below today's.
    {
        Device d = uniform(1048576, 1048576);
        d.v11.maxPerSetDescriptors = 1024;
        BindlessFeatures f = with_ms(true);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12));
        EXPECT(count(f, D::dim_2d) == 4096 && f.counts[bindless::sampler_binding] == 1024 &&
               storage(f, D::dim_2d) == 256);
        d.v11.maxPerSetDescriptors = 30000;
        f = with_ms(true);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12));
        EXPECT(totals(f).all <= 30000 && count(f, D::dim_2d) > 4096);
    }

    // D3D-219: past maxPerSetDescriptors, growth follows layout_supported (vkGetDescriptorSetLayoutSupport), which
    // is asked only above that limit. Intel's Xe-LP reports 6159 and supports the full sizes.
    {
        Device d = uniform(1048576, 1048576);
        d.v11.maxPerSetDescriptors = 6159;
        d.limits.maxSamplerAllocationCount = 4000;
        unsigned asked = 0, asked_within = 0;
        const auto all_supported = [&](const BindlessFeatures& candidate) {
            ++asked;
            asked_within += totals(candidate).all <= 6159;
            return true;
        };
        BindlessFeatures f = with_ms(true);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12, all_supported));
        EXPECT(asked > 0 && asked_within == 0);
        EXPECT(count(f, D::dim_2d) == bindless::max_2d_images &&
               f.counts[bindless::sampler_binding] == 4000 - bindless::reserved_samplers);
        EXPECT(storage(f, D::dim_2d) == bindless::max_storage_indices && storage_consistent(f));

        // A driver that supports less: growth stops where it says no.
        const auto up_to_20000 = [&](const BindlessFeatures& candidate) { return totals(candidate).all <= 20000; };
        f = with_ms(true);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12, up_to_20000));
        EXPECT(totals(f).all <= 20000 && totals(f).all > 6159 && count(f, D::dim_2d) > 4096);

        // Without the callback, maxPerSetDescriptors still caps growth as before.
        f = with_ms(true);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12));
        EXPECT(totals(f).all <= 6159 || (count(f, D::dim_2d) == 4096 && f.counts[bindless::sampler_binding] == 1024));
    }

    // MoltenVK 1.4.3 on an M3 Max: update-after-bind limits of 1000000 sampled images and 500000 samplers, but
    // maxPerSetDescriptors 1212 and a vkGetDescriptorSetLayoutSupport that accepts only fewer descriptors than that
    // (MVKDevice::getDescriptorSetLayoutSupport), though the device creates and samples far larger sets. So the
    // set keeps the default counts: 1024 samplers, short of a full D3D12 sampler heap (d3d12_descriptor_test).
    {
        Device d = uniform(1048576, 1000000);
        d.v12.maxPerStageDescriptorUpdateAfterBindSamplers = d.v12.maxDescriptorSetUpdateAfterBindSamplers = 500000;
        d.limits.maxSamplerAllocationCount = 500000;
        d.v11.maxPerSetDescriptors = 1212;
        const auto below_1212 = [](const BindlessFeatures& candidate) { return totals(candidate).all < 1212; };
        BindlessFeatures f = with_ms(true);
        EXPECT(bindless::size_bindings(f, d.limits, d.v11, d.v12, below_1212));
        EXPECT(count(f, D::dim_2d) == 4096 && f.counts[bindless::sampler_binding] == 1024 &&
               storage(f, D::dim_2d) == 256);
    }

    // The default features are today's counts, which the device-less users (vk_image_fuzz) still get.
    EXPECT(BindlessFeatures{}.counts == defaults);
    std::printf("bindless sizing: synthetic limits gave the expected counts\n");
    return 0;
}
