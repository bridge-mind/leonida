// GPU-only count gating for simple D3D12 ExecuteIndirect(DISPATCH) signatures.
// Vulkan has vkCmdDispatchIndirect but no vkCmdDispatchIndirectCount. We prepare a private indirect
// buffer with zero-group records past the GPU count, then execute all MaxCommandCount records.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>
#include "indirect_dispatch_count_spv.h"

namespace leonida {

struct CountedDispatch {
    VkBuffer indirect = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceAddress arguments = 0;
    VkDeviceAddress count = 0;
    VkDeviceAddress output = 0;
    std::uint32_t stride = 0;
    std::uint32_t max_count = 0;
};

class IndirectDispatchCounter {
public:
    IndirectDispatchCounter(PFN_vkGetDeviceProcAddr get, VkDevice device,
                            const VkPhysicalDeviceMemoryProperties& memory, const VkPhysicalDeviceLimits& limits)
        : get_(get), device_(device), memory_(memory), limits_(limits) {}
    ~IndirectDispatchCounter() {
        if (pipeline_) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline_, nullptr);
        if (layout_) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout_, nullptr);
    }
    IndirectDispatchCounter(const IndirectDispatchCounter&) = delete;
    IndirectDispatchCounter& operator=(const IndirectDispatchCounter&) = delete;

    // Retained by the command-buffer slot until its submission's timeline ticket completes.
    struct Batch {
        IndirectDispatchCounter& owner;
        struct Allocation {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
        };
        std::vector<Allocation> allocations;
        explicit Batch(IndirectDispatchCounter& value) : owner(value) {}
        ~Batch() {
            for (auto& a : allocations) {
                if (a.buffer) owner.fn<PFN_vkDestroyBuffer>("vkDestroyBuffer")(owner.device_, a.buffer, nullptr);
                if (a.memory) owner.fn<PFN_vkFreeMemory>("vkFreeMemory")(owner.device_, a.memory, nullptr);
            }
        }
        Batch(const Batch&) = delete;
        Batch& operator=(const Batch&) = delete;

        bool prepare(VkBuffer arguments, VkDeviceSize argument_offset, VkBuffer count,
                     VkDeviceSize count_offset, std::uint32_t stride, std::uint32_t max_count,
                     CountedDispatch& out) {
            out = {};
            if (!arguments || !count || !stride || stride % 4 || !max_count ||
                !owner.limits_.maxComputeWorkGroupCount[0] || !owner.make_pipeline()) return false;
            constexpr VkDeviceSize record_bytes = 3 * sizeof(std::uint32_t);
            if (VkDeviceSize(max_count) > std::numeric_limits<VkDeviceSize>::max() / record_bytes)
                return false;
            const VkDeviceSize bytes = VkDeviceSize(max_count) * record_bytes;
            VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            info.size = bytes;
            info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            Allocation scratch{};
            if (owner.fn<PFN_vkCreateBuffer>("vkCreateBuffer")(owner.device_, &info, nullptr, &scratch.buffer) != VK_SUCCESS)
                return false;
            allocations.push_back(scratch);  // all failure paths free it with the batch
            auto& a = allocations.back();
            VkMemoryRequirements requirements{};
            owner.fn<PFN_vkGetBufferMemoryRequirements>("vkGetBufferMemoryRequirements")(
                owner.device_, a.buffer, &requirements);
            std::uint32_t memory_type = UINT32_MAX;
            for (std::uint32_t i = 0; i < owner.memory_.memoryTypeCount; ++i) {
                if (!(requirements.memoryTypeBits & (1u << i))) continue;
                if (memory_type == UINT32_MAX) memory_type = i;
                if (owner.memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                    memory_type = i;
                    break;
                }
            }
            if (memory_type == UINT32_MAX) return false;
            VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
            flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = memory_type;
            if (owner.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(owner.device_, &allocation, nullptr, &a.memory) != VK_SUCCESS ||
                owner.fn<PFN_vkBindBufferMemory>("vkBindBufferMemory")(owner.device_, a.buffer, a.memory, 0) != VK_SUCCESS)
                return false;
            const auto address = owner.fn<PFN_vkGetBufferDeviceAddress>("vkGetBufferDeviceAddress");
            VkBufferDeviceAddressInfo addr{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            addr.buffer = arguments;
            const VkDeviceAddress args_va = address(owner.device_, &addr);
            addr.buffer = count;
            const VkDeviceAddress count_va = address(owner.device_, &addr);
            addr.buffer = a.buffer;
            const VkDeviceAddress output_va = address(owner.device_, &addr);
            if (!args_va || !count_va || !output_va ||
                argument_offset > UINT64_MAX - args_va || count_offset > UINT64_MAX - count_va)
                return false;
            out = {a.buffer, 0, args_va + argument_offset, count_va + count_offset,
                   output_va, stride, max_count};
            return true;
        }
    };

    std::unique_ptr<Batch> batch() { return std::make_unique<Batch>(*this); }

    void emit(VkCommandBuffer cb, const CountedDispatch& job) const {
        const auto barrier = fn<PFN_vkCmdPipelineBarrier>("vkCmdPipelineBarrier");
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &before, 0, nullptr, 0, nullptr);
        fn<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        struct Push {
            VkDeviceAddress arguments, count, output;
            std::uint32_t stride_words, first_record, records;
        };
        static_assert(sizeof(Push) == 40);
        const VkDeviceSize dispatch_limit = VkDeviceSize(limits_.maxComputeWorkGroupCount[0]) * 64;
        // Bound per-dispatch indexing to 32 bits, even when the D3D12 stride is very large.
        // The shader loads three consecutive dwords, so leave room for the final +2.
        const VkDeviceSize index_limit = (UINT32_MAX - 2u) / (job.stride / 4);
        const VkDeviceSize chunk_limit = std::min(dispatch_limit, index_limit + 1);
        for (VkDeviceSize first = 0; first < job.max_count;) {
            const std::uint32_t count = std::uint32_t(std::min<VkDeviceSize>(chunk_limit, job.max_count - first));
            const Push push{job.arguments + first * job.stride, job.count, job.output + first * 12,
                            job.stride / 4, std::uint32_t(first), count};
            fn<PFN_vkCmdPushConstants>("vkCmdPushConstants")(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                                             0, sizeof(push), &push);
            fn<PFN_vkCmdDispatch>("vkCmdDispatch")(cb, (count + 63u) / 64u, 1, 1);
            first += count;
        }
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        after.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_MEMORY_READ_BIT |
                              VK_ACCESS_MEMORY_WRITE_BIT;
        barrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 1, &after, 0, nullptr, 0, nullptr);
    }

private:
    template<typename T> T fn(const char* name) const { return reinterpret_cast<T>(get_(device_, name)); }
    bool make_pipeline() {
        if (pipeline_) return true;
        for (const char* name : {"vkCreateBuffer", "vkDestroyBuffer", "vkGetBufferMemoryRequirements",
             "vkAllocateMemory", "vkFreeMemory", "vkBindBufferMemory", "vkGetBufferDeviceAddress",
             "vkCreateShaderModule", "vkDestroyShaderModule", "vkCreatePipelineLayout",
             "vkDestroyPipelineLayout", "vkCreateComputePipelines", "vkDestroyPipeline",
             "vkCmdPipelineBarrier", "vkCmdBindPipeline", "vkCmdPushConstants", "vkCmdDispatch"})
            if (!get_(device_, name)) return false;
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader.codeSize = sizeof(indirect_dispatch_count_spirv);
        shader.pCode = indirect_dispatch_count_spirv;
        bool ok = fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device_, &shader, nullptr, &module) == VK_SUCCESS;
        const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 40};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &range;
        ok = ok && fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device_, &layout_info, nullptr, &layout) == VK_SUCCESS;
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = module;
        info.stage.pName = "main";
        info.layout = layout;
        ok = ok && fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(
            device_, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS;
        if (module) fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device_, module, nullptr);
        if (!ok) {
            if (pipeline) fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device_, pipeline, nullptr);
            if (layout) fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device_, layout, nullptr);
            return false;
        }
        layout_ = layout;
        pipeline_ = pipeline;
        return true;
    }
    PFN_vkGetDeviceProcAddr get_;
    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memory_;
    VkPhysicalDeviceLimits limits_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};
}  // namespace leonida
