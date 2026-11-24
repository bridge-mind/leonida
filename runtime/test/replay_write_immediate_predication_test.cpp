// WriteBufferImmediate is unconditional on hardware D3D12 (measured on Intel and NVIDIA D3D12; not in
// SetPredication's predicated list). On Intel's Windows Vulkan driver, vkCmdUpdateBuffer after
// PredicateReducer::capture's compute dispatch is suppressed inside an active conditional rendering scope.
// Check that emit() suspends conditional rendering around vkCmdUpdateBuffer for every write mode and restores
// it so subsequent predicated commands remain gated. No GPU or game data.
#include "../src/vulkan_replay.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace leonida;

struct Trace {
    bool cond_active = false;
    unsigned cond_begins = 0;
    unsigned cond_ends = 0;
    unsigned updates = 0;
    unsigned updates_while_cond = 0;
    unsigned clear_att = 0;
    unsigned clear_att_while_cond = 0;
    std::vector<std::string> events;
};

Trace* g_trace = nullptr;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VKAPI_ATTR void VKAPI_CALL mock_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                        std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                        const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {}

VKAPI_ATTR void VKAPI_CALL mock_begin_cond(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
    g_trace->cond_active = true;
    ++g_trace->cond_begins;
    g_trace->events.push_back("begin_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_end_cond(VkCommandBuffer) {
    g_trace->cond_active = false;
    ++g_trace->cond_ends;
    g_trace->events.push_back("end_cond");
}

VKAPI_ATTR void VKAPI_CALL mock_update_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, const void*) {
    ++g_trace->updates;
    if (g_trace->cond_active) ++g_trace->updates_while_cond;
    g_trace->events.push_back("update");
}

VKAPI_ATTR void VKAPI_CALL mock_begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL mock_end_rendering(VkCommandBuffer) {}

VKAPI_ATTR void VKAPI_CALL mock_clear_attachments(VkCommandBuffer, std::uint32_t, const VkClearAttachment*,
                                                  std::uint32_t, const VkClearRect*) {
    ++g_trace->clear_att;
    if (g_trace->cond_active) ++g_trace->clear_att_while_cond;
    g_trace->events.push_back("clear_att");
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    constexpr std::uint64_t va_buf = 0x10000;
    ResourceBinding buf{};
    buf.buffer = handle<VkBuffer>(1);
    buf.size = 64;
    buf.va = va_buf;

    ResourceBinding pred{};
    pred.buffer = handle<VkBuffer>(2);
    pred.size = 16;

    ResourceBinding rt{};
    rt.image = handle<VkImage>(3);
    rt.format = VK_FORMAT_R8G8B8A8_UNORM;
    rt.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rt.width = rt.height = 4;
    rt.depth = rt.mip_levels = rt.array_layers = rt.samples = 1;
    rt.color_attachment = true;

    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        if (id == 1) return &buf;
        if (id == 2) return &pred;
        if (id == 3) return &rt;
        return nullptr;
    };

    ReplayFunctions vk{};
    vk.pipeline_barrier = mock_barrier;
    vk.update_buffer = mock_update_buffer;
    vk.begin_rendering = mock_begin_rendering;
    vk.end_rendering = mock_end_rendering;
    vk.clear_attachments = mock_clear_attachments;
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) { return handle<VkImageView>(6); };
    vk.begin_conditional_rendering = mock_begin_cond;
    vk.end_conditional_rendering = mock_end_cond;
    vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
        out = {source, offset, handle<VkBuffer>(4), 0, handle<VkDescriptorSet>(5)};
        return true;
    };
    vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) { g_trace->events.push_back("capture"); };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t length, VkBuffer& out_buf, VkDeviceSize& out_off) {
        if (va < va_buf || va - va_buf >= buf.size || length > buf.size - (va - va_buf)) return false;
        out_buf = buf.buffer;
        out_off = va - va_buf;
        return true;
    };

    namespace m = write_immediate_mode;

    // 1. All three WriteBufferImmediate modes under active predication suspend conditional rendering around
    //    vkCmdUpdateBuffer and restore it before a subsequent predicated clear.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.set_predication({2, 0, d3d12_predication::equal_zero}), "record predication");
        expect(list.write_immediate({va_buf + 0, 0x11111111, m::normal}), "record normal write");
        expect(list.write_immediate({va_buf + 4, 0x22222222, m::marker_in}), "record marker_in write");
        expect(list.write_immediate({va_buf + 8, 0x33333333, m::marker_out}), "record marker_out write");
        expect(list.clear_render_target({3, 0, 0, 1}, {0, 0, 1, 1}), "record predicated clear after writes");
        expect(list.close(), "close predicated list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay predicated list");
        expect(trace.updates == 3 && trace.updates_while_cond == 0,
               "predicated writes: all three modes run outside conditional rendering");
        expect(trace.clear_att == 1 && trace.clear_att_while_cond == 1,
               "predicated clear after writes: runs with conditional rendering restored");
        expect(trace.cond_begins == 4 && trace.cond_ends == 4 && !trace.cond_active,
               "predicated writes: conditional rendering balanced");
        const std::vector<std::string> want{
            "capture",    "begin_cond", "end_cond", "update",    "begin_cond", "end_cond", "update",
            "begin_cond", "end_cond",   "update",   "begin_cond", "clear_att", "end_cond",
        };
        expect(trace.events == want, "predicated writes: exact suspend/resume event order");
    }

    // 2. Unpredicated WriteBufferImmediate never touches conditional rendering.
    {
        Trace trace;
        g_trace = &trace;
        CommandList list;
        expect(list.write_immediate({va_buf + 0, 0x11111111, m::normal}), "record unpredicated normal");
        expect(list.write_immediate({va_buf + 4, 0x22222222, m::marker_in}), "record unpredicated marker_in");
        expect(list.write_immediate({va_buf + 8, 0x33333333, m::marker_out}), "record unpredicated marker_out");
        expect(list.close(), "close unpredicated list");
        expect(replay(list, VK_NULL_HANDLE, resolve, vk) == ReplayResult::success, "replay unpredicated list");
        expect(trace.updates == 3 && trace.cond_begins == 0 && trace.cond_ends == 0,
               "unpredicated writes: conditional rendering untouched");
    }

    if (failures) return 1;
    std::puts("replay_write_immediate_predication_test: WriteBufferImmediate suspends and restores conditional rendering");
    return 0;
}
