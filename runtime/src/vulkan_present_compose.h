#pragma once

// The compute pass behind PresentX's plane composition (docs/d3d12_vulkan.md section 4.5), used by VulkanPresenter::
// present_planes(): up to two sampled planes in, each scaled from a source rectangle to a destination rectangle,
// plane 1 blended over plane 0, colour spaces converted, the result in an RGBA16F image of the swapchain's size that the
// presenter copies to the swapchain image. The shader is present_compose.comp (SPIR-V embedded, present_compose_spv.h).
// Not thread safe: the presenter calls it with its own state locked, one present at a time.

#include "vulkan_backend.h"
#include "present_compose_spv.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace leonida {

// Mirrors the shader's push constants (112 bytes).
struct PresentComposePush {
    float dest0[4], dest1[4];  // destination rectangles in output pixels: left, top, right, bottom
    float src0[4], src1[4];    // source rectangles in texels
    std::uint32_t flags0, flags1, plane_count, out_hdr;
    float sdr_white;
    std::uint32_t width, height, sdr_eotf;  // sdr_eotf: PresentPlanes::SdrEotf
    float hdr_peak;
    std::uint32_t pad[3];
};
static_assert(sizeof(PresentComposePush) == 112);

namespace compose_flag {
inline constexpr std::uint32_t point = 1, srgb_source = 2, hdr10 = 4, straight_alpha = 8;
}

class PresentComposer {
public:
    static constexpr std::uint32_t frame_slots = 2;

    PresentComposer(VulkanContext& context, const VkPhysicalDeviceMemoryProperties& memory)
        : context_(context), memory_(memory) {}

    ~PresentComposer() {
        const VkDevice device = context_.device();
        auto& vk = context_.vk;
        release_target();
        if (pipeline_) vk.vkDestroyPipeline(device, pipeline_, nullptr);
        if (pipeline_layout_) vk.vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        if (pool_) vk.vkDestroyDescriptorPool(device, pool_, nullptr);  // frees its sets
        if (set_layout_) vk.vkDestroyDescriptorSetLayout(device, set_layout_, nullptr);
    }
    PresentComposer(const PresentComposer&) = delete;
    PresentComposer& operator=(const PresentComposer&) = delete;

    // Builds the pipeline once; false when the device cannot (a function is missing, or creation fails).
    bool ready() {
        if (state_ != State::untried) return state_ == State::ready;
        state_ = State::failed;
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        if (!vk.vkCreateDescriptorSetLayout || !vk.vkCreateDescriptorPool || !vk.vkAllocateDescriptorSets ||
            !vk.vkUpdateDescriptorSets || !vk.vkCreatePipelineLayout || !vk.vkCreateShaderModule ||
            !vk.vkCreateComputePipelines || !vk.vkCmdBindPipeline || !vk.vkCmdBindDescriptorSets ||
            !vk.vkCmdPushConstants || !vk.vkCmdDispatch || !vk.vkCreateImage || !vk.vkCreateImageView ||
            !vk.vkAllocateMemory)
            return false;
        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = std::uint32_t(bindings.size());
        set_info.pBindings = bindings.data();
        if (vk.vkCreateDescriptorSetLayout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS) return false;
        const std::array<VkDescriptorPoolSize, 2> sizes{{{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * frame_slots},
                                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, frame_slots}}};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = frame_slots;
        pool_info.poolSizeCount = std::uint32_t(sizes.size());
        pool_info.pPoolSizes = sizes.data();
        if (vk.vkCreateDescriptorPool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS) return false;
        std::array<VkDescriptorSetLayout, frame_slots> layouts;
        layouts.fill(set_layout_);
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool_;
        allocate.descriptorSetCount = frame_slots;
        allocate.pSetLayouts = layouts.data();
        if (vk.vkAllocateDescriptorSets(device, &allocate, sets_.data()) != VK_SUCCESS) return false;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PresentComposePush)};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1;
        layout_info.pSetLayouts = &set_layout_;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push;
        if (vk.vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS) return false;
        VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module_info.codeSize = sizeof(present_compose_spirv);
        module_info.pCode = present_compose_spirv;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vk.vkCreateShaderModule(device, &module_info, nullptr, &module) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                               VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
        pipeline_info.layout = pipeline_layout_;
        const VkResult created =
            vk.vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
        vk.vkDestroyShaderModule(device, module, nullptr);
        if (created != VK_SUCCESS) return false;
        state_ = State::ready;
        return true;
    }

    // The RGBA16F image of `extent` the pass writes (storage + transfer source), made again when the extent changes.
    // The caller has made the queue idle first when this returns after a change (changed() says so): the old image
    // may still be in use.
    bool ensure_target(VkExtent2D extent, bool& changed) {
        changed = false;
        if (image_ && extent_.width == extent.width && extent_.height == extent.height) return true;
        changed = true;
        release_target();
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = target_format;
        info.extent = {extent.width, extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk.vkCreateImage(device, &info, nullptr, &image_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk.vkGetImageMemoryRequirements(device, image_, &requirements);
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                type = i;
                break;
            }
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        if (type == UINT32_MAX || vk.vkAllocateMemory(device, &allocate, nullptr, &memory_object_) != VK_SUCCESS ||
            vk.vkBindImageMemory(device, image_, memory_object_, 0) != VK_SUCCESS) {
            release_target();
            return false;
        }
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image_;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = target_format;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vk.vkCreateImageView(device, &view_info, nullptr, &view_) != VK_SUCCESS) {
            release_target();
            return false;
        }
        extent_ = extent;
        return true;
    }

    VkImage target() const { return image_; }
    VkExtent2D extent() const { return extent_; }
    static constexpr VkFormat target_format = VK_FORMAT_R16G16B16A16_SFLOAT;

    // The binding the image barrier helpers take for the target.
    ResourceBinding target_binding() const {
        ResourceBinding binding{};
        binding.image = image_;
        binding.format = target_format;
        binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        binding.width = extent_.width;
        binding.height = extent_.height;
        return binding;
    }

    // Records the dispatch into `commands`: the planes' views (`plane1` may repeat `plane0` for a single plane) are
    // bound with the target, the target goes UNDEFINED -> GENERAL for the writes and ends in TRANSFER_SRC_OPTIMAL for
    // the copy that follows. The planes themselves must be in SHADER_READ_ONLY_OPTIMAL (or GENERAL) already.
    void record(VkCommandBuffer commands, std::uint32_t slot, VkImageView plane0, VkImageLayout layout0,
                VkImageView plane1, VkImageLayout layout1,
                const PresentComposePush& push) {
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        const VkDescriptorSet set = sets_[slot % frame_slots];
        std::array<VkDescriptorImageInfo, 3> images{};
        images[0] = {VK_NULL_HANDLE, plane0, layout0};
        images[1] = {VK_NULL_HANDLE, plane1, layout1};
        images[2] = {VK_NULL_HANDLE, view_, VK_IMAGE_LAYOUT_GENERAL};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t i = 0; i < 3; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &images[i];
        }
        vk.vkUpdateDescriptorSets(device, std::uint32_t(writes.size()), writes.data(), 0, nullptr);

        VkImageMemoryBarrier to_write{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_write.srcAccessMask = 0;  // the previous reader (a copy) only needs the execution dependency
        to_write.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_write.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_write.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_write.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_write.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_write.image = image_;
        to_write.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_write);
        vk.vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vk.vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set, 0, nullptr);
        vk.vkCmdPushConstants(commands, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vk.vkCmdDispatch(commands, (extent_.width + 7) / 8, (extent_.height + 7) / 8, 1);
        VkImageMemoryBarrier to_read = to_write;
        to_read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        to_read.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_read.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        vk.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                nullptr, 0, nullptr, 1, &to_read);
    }


private:
    enum class State { untried, ready, failed };

    void release_target() {
        auto& vk = context_.vk;
        const VkDevice device = context_.device();
        if (view_) vk.vkDestroyImageView(device, view_, nullptr);
        if (image_) vk.vkDestroyImage(device, image_, nullptr);
        if (memory_object_) vk.vkFreeMemory(device, memory_object_, nullptr);
        view_ = VK_NULL_HANDLE;
        image_ = VK_NULL_HANDLE;
        memory_object_ = VK_NULL_HANDLE;
        extent_ = {};
    }

    VulkanContext& context_;
    VkPhysicalDeviceMemoryProperties memory_;
    State state_ = State::untried;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, frame_slots> sets_{};
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_object_ = VK_NULL_HANDLE;
    VkImageView view_ = VK_NULL_HANDLE;
    VkExtent2D extent_{};
};

}  // namespace leonida
