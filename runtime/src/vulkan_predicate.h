// GPU snapshot of D3D12's 64-bit predicate. No host readback and no per-predicate queue wait.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>
#include <memory>
#include <initializer_list>
#include "predicate_reduce_spv.h"

namespace leonida {
struct PredicateCapture {
    VkBuffer source = VK_NULL_HANDLE;
    VkDeviceSize source_offset = 0;
    VkBuffer scratch = VK_NULL_HANDLE;
    VkDeviceSize offset = 0; // two input words, then the reduced condition
    VkDescriptorSet set = VK_NULL_HANDLE;
};

class PredicateReducer {
public:
    PredicateReducer(PFN_vkGetDeviceProcAddr get, VkDevice device, VkPhysicalDeviceMemoryProperties memory,
                     VkDeviceSize max_range) : get_(get), device_(device), memory_(memory), max_range_(max_range) {}
    ~PredicateReducer() {
        if (pipeline_) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline_, nullptr);
        if (layout_) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout_, nullptr);
        if (set_layout_) fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(device_, set_layout_, nullptr);
    }
    PredicateReducer(const PredicateReducer&) = delete;
    PredicateReducer& operator=(const PredicateReducer&) = delete;

    // Owned by a command-buffer slot until its timeline ticket completes (or a failed recording is reset).
    struct Batch {
        PredicateReducer& owner;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDeviceSize bytes = 0;
        std::uint32_t count = 0, next = 0;
        explicit Batch(PredicateReducer& r) : owner(r) {}
        ~Batch() {
            if (pool) owner.fn<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(owner.device_, pool, nullptr);
            if (buffer) owner.fn<PFN_vkDestroyBuffer>("vkDestroyBuffer")(owner.device_, buffer, nullptr);
            if (memory) owner.fn<PFN_vkFreeMemory>("vkFreeMemory")(owner.device_, memory, nullptr);
        }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;
        bool prepare(VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
            if (next >= count || !source) return false;
            out = {source, offset, buffer, VkDeviceSize(next++) * 16, set};
            return true;
        }
    };

    std::unique_ptr<Batch> batch(std::uint64_t count) {
        // One descriptor starts at offset zero; a push constant selects each aligned 16-byte slot.
        if (!count || count > max_range_ / 16 || count > UINT32_MAX / 4 || !make_pipeline()) return {};
        auto b = std::make_unique<Batch>(*this);
        b->count = static_cast<std::uint32_t>(count);
        b->bytes = count * 16;
        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = b->bytes;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                       VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
        if (fn<PFN_vkCreateBuffer>("vkCreateBuffer")(device_, &buffer, nullptr, &b->buffer) != VK_SUCCESS) return {};
        VkMemoryRequirements requirements{};
        fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(device_, b->buffer, &requirements);
        std::uint32_t type = UINT32_MAX;
        for (std::uint32_t i = 0; i < memory_.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if (type == UINT32_MAX) type = i;
            if (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) { type = i; break; }
        }
        if (type == UINT32_MAX) return {};
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (fn<PFN_vkAllocateMemory>("vkAllocateMemory")(device_, &allocation, nullptr, &b->memory) != VK_SUCCESS ||
            fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(device_, b->buffer, b->memory, 0) != VK_SUCCESS) return {};
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1; pool.poolSizeCount = 1; pool.pPoolSizes = &size;
        if (fn<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(device_, &pool, nullptr, &b->pool) != VK_SUCCESS) return {};
        VkDescriptorSetAllocateInfo sets{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sets.descriptorPool = b->pool; sets.descriptorSetCount = 1; sets.pSetLayouts = &set_layout_;
        if (fn<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(device_, &sets, &b->set) != VK_SUCCESS) return {};
        const VkDescriptorBufferInfo info{b->buffer, 0, b->bytes};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = b->set; write.dstBinding = 0; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &info;
        fn<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(device_, 1, &write, 0, nullptr);
        return b;
    }

    void capture(VkCommandBuffer cb, const PredicateCapture& p) const {
        // End an earlier conditional scope before entering this helper: dispatch is itself conditional.
        // The descriptor spans the arena, so order all prior accesses before reusing it for another slot.
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
        const VkBufferCopy copy{p.source_offset, p.offset, 8};
        fn<PFN_vkCmdCopyBuffer>("vkCmdCopyBuffer")(cb, p.source, p.scratch, 1, &copy);
        // The application's PREDICATION state does not describe this internal transfer read. Preserve snapshot
        // semantics when it subsequently transitions and overwrites the source in the same command list.
        VkBufferMemoryBarrier source_done{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        source_done.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        source_done.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        source_done.srcQueueFamilyIndex = source_done.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        source_done.buffer = p.source; source_done.offset = p.source_offset; source_done.size = 8;
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 0, nullptr, 1, &source_done, 0, nullptr);
        VkMemoryBarrier compute{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        compute.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        compute.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &compute, 0, nullptr, 0, nullptr);
        fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        fn<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &p.set, 0, nullptr);
        const std::uint32_t index = static_cast<std::uint32_t>(p.offset / 4);
        fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &index);
        fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, 1, 1, 1);
        VkMemoryBarrier condition{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        condition.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        condition.dstAccessMask = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
                0, 1, &condition, 0, nullptr, 0, nullptr);
    }

private:
    template<class T> T fn(const char* name) const { return reinterpret_cast<T>(get_(device_, name)); }
    bool make_pipeline() {
        if (pipeline_) return true;
        // No entry-point assumptions for partial/mock devices: fail before allocating or recording anything.
        for (const char* name : {"vkCreateBuffer", "vkDestroyBuffer", "vkGetBufferMemoryRequirements", "vkAllocateMemory",
             "vkFreeMemory", "vkBindBufferMemory", "vkCreateDescriptorPool", "vkDestroyDescriptorPool",
             "vkAllocateDescriptorSets", "vkUpdateDescriptorSets", "vkCreateShaderModule", "vkDestroyShaderModule",
             "vkCreateDescriptorSetLayout", "vkDestroyDescriptorSetLayout", "vkCreatePipelineLayout",
             "vkDestroyPipelineLayout", "vkCreateComputePipelines", "vkDestroyPipeline", "vkCmdCopyBuffer",
             "vkCmdPipelineBarrier", "vkCmdBindPipeline", "vkCmdBindDescriptorSets", "vkCmdPushConstants", "vkCmdDispatch"})
            if (!get_(device_, name)) return false;
        VkDescriptorSetLayout set = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkShaderModule shader = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_info.bindingCount = 1; set_info.pBindings = &binding;
        bool ok = fn<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(device_, &set_info, nullptr, &set) == VK_SUCCESS;
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 1; layout_info.pSetLayouts = &set;
        layout_info.pushConstantRangeCount = 1; layout_info.pPushConstantRanges = &push;
        ok = ok && fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device_, &layout_info, nullptr, &layout) == VK_SUCCESS;
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize = sizeof(predicate_reduce_spirv); module.pCode = predicate_reduce_spirv;
        ok = ok && fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device_, &module, nullptr, &shader) == VK_SUCCESS;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; info.stage.module = shader; info.stage.pName = "main";
        info.layout = layout;
        ok = ok && fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS;
        if (shader) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, shader, nullptr);
        if (!ok) {
            if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
            if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
            if (set) fn<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(device_, set, nullptr);
            return false;
        }
        set_layout_ = set; layout_ = layout; pipeline_ = pipeline;
        return true;
    }
    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_;
    VkDeviceSize max_range_;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};
} // namespace leonida
