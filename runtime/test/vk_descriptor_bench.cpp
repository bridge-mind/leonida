// GPU micro-benchmark for the descriptor hash table (set 0, binding 2 contract, D3D-56).
// Measures T# and S# hash-table lookups at 1k, 10k, and 100k entries across wave-uniform and lane-divergent
// access patterns, hits and misses, and buffer memory placements (DEVICE_LOCAL vs HOST_VISIBLE vs ReBAR).
//
// The compute SPIR-V modules are built in memory using the exact FNV-1a + 8-slot linear probe loop emitted by
// tools/rdna2spirv.py (desc_probe), so this tool has no glslangValidator build dependency and checks every
// dispatch against CPU DescriptorTable::lookup() before timing.
//
// Usage:
//   vk_descriptor_bench [--sizes 1000,10000,100000] [--groups 4096] [--iters 64] [--reps 25] [--warmup 5] [--warmup-ms 300]
//                       [--memory all|device|host|rebar] [--device N] [--list-devices] [--quick] [--csv]
//                       [--dump-spv PREFIX] [--self-test]
//   LEONIDA_VK_VALIDATION=1 enables VK_LAYER_KHRONOS_validation; any validation error fails the run.
#define VK_NO_PROTOTYPES
#include "vulkan_loader.h"
#include "vk_device_pick.h"
#include "../src/descriptor_table.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#define CHECK_VK(expr)                                                                       \
    do {                                                                                     \
        const VkResult vk_res_ = (expr);                                                     \
        if (vk_res_ != VK_SUCCESS) {                                                         \
            std::fprintf(stderr, "FAILED %s -> %d (line %d)\n", #expr, int(vk_res_), __LINE__); \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

namespace {

using namespace leonida;

constexpr int kUnsupportedExit = 77;
constexpr std::uint32_t kWorkgroupSize = 64;
unsigned g_validation_errors = 0;

VKAPI_ATTR VkBool32 VKAPI_CALL on_debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                VkDebugUtilsMessageTypeFlagsEXT types,
                                                const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
        (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)) {
        ++g_validation_errors;
    }
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

// ---- SPIR-V 1.3 binary builder matching tools/rdna2spirv.py desc_probe ----

namespace spv_op {
constexpr std::uint32_t kMagic = 0x07230203u;
constexpr std::uint32_t kVersion13 = 0x00010300u;

constexpr std::uint32_t kCapShader = 1u;
constexpr std::uint32_t kAddrLogical = 0u;
constexpr std::uint32_t kMemGlsl450 = 1u;
constexpr std::uint32_t kExecGLCompute = 5u;
constexpr std::uint32_t kExecModeLocalSize = 17u;

constexpr std::uint32_t kScInput = 1u;
constexpr std::uint32_t kScFunction = 7u;
constexpr std::uint32_t kScPushConstant = 9u;
constexpr std::uint32_t kScStorageBuffer = 12u;

constexpr std::uint32_t kDecBlock = 2u;
constexpr std::uint32_t kDecArrayStride = 6u;
constexpr std::uint32_t kDecBuiltIn = 11u;
constexpr std::uint32_t kDecNonWritable = 24u;
constexpr std::uint32_t kDecNonReadable = 25u;
constexpr std::uint32_t kDecBinding = 33u;
constexpr std::uint32_t kDecDescriptorSet = 34u;
constexpr std::uint32_t kDecOffset = 35u;
constexpr std::uint32_t kBuiltInGlobalInvocationId = 28u;

constexpr std::uint32_t kOpMemoryModel = 14u;
constexpr std::uint32_t kOpEntryPoint = 15u;
constexpr std::uint32_t kOpExecutionMode = 16u;
constexpr std::uint32_t kOpCapability = 17u;
constexpr std::uint32_t kOpTypeVoid = 19u;
constexpr std::uint32_t kOpTypeBool = 20u;
constexpr std::uint32_t kOpTypeInt = 21u;
constexpr std::uint32_t kOpTypeVector = 23u;
constexpr std::uint32_t kOpTypeRuntimeArray = 29u;
constexpr std::uint32_t kOpTypeStruct = 30u;
constexpr std::uint32_t kOpTypePointer = 32u;
constexpr std::uint32_t kOpTypeFunction = 33u;
constexpr std::uint32_t kOpConstantTrue = 41u;
constexpr std::uint32_t kOpConstant = 43u;
constexpr std::uint32_t kOpFunction = 54u;
constexpr std::uint32_t kOpFunctionParameter = 55u;
constexpr std::uint32_t kOpFunctionEnd = 56u;
constexpr std::uint32_t kOpFunctionCall = 57u;
constexpr std::uint32_t kOpVariable = 59u;
constexpr std::uint32_t kOpLoad = 61u;
constexpr std::uint32_t kOpStore = 62u;
constexpr std::uint32_t kOpAccessChain = 65u;
constexpr std::uint32_t kOpDecorate = 71u;
constexpr std::uint32_t kOpMemberDecorate = 72u;
constexpr std::uint32_t kOpIAdd = 128u;
constexpr std::uint32_t kOpIMul = 132u;
constexpr std::uint32_t kOpLogicalAnd = 167u;
constexpr std::uint32_t kOpIEqual = 170u;
constexpr std::uint32_t kOpULessThan = 176u;
constexpr std::uint32_t kOpShiftRightLogical = 194u;
constexpr std::uint32_t kOpShiftLeftLogical = 196u;
constexpr std::uint32_t kOpBitwiseXor = 198u;
constexpr std::uint32_t kOpBitwiseAnd = 199u;
constexpr std::uint32_t kOpLoopMerge = 246u;
constexpr std::uint32_t kOpSelectionMerge = 247u;
constexpr std::uint32_t kOpLabel = 248u;
constexpr std::uint32_t kOpBranch = 249u;
constexpr std::uint32_t kOpBranchConditional = 250u;
constexpr std::uint32_t kOpReturn = 253u;
constexpr std::uint32_t kOpReturnValue = 254u;
} // namespace spv_op

enum class ShaderKind {
    Baseline,
    SingleLookup,
    PairTSLookup,
};

struct alignas(4) PushConstants {
    std::uint32_t key_mask;     // pc[0]: mask for T#/Miss ring (power of two minus 1)
    std::uint32_t iters;        // pc[1]: loop iterations per invocation
    std::uint32_t group_stride; // pc[2]: start index delta per 64-invocation group
    std::uint32_t lane_stride;  // pc[3]: start index delta per lane (0 => wave-uniform, 1 => divergent)
    std::uint32_t iter_stride;  // pc[4]: index delta per loop iteration
    std::uint32_t base_index;   // pc[5]: starting key offset within bank
    std::uint32_t dep_mask;     // pc[6]: data-dependency mask from previous result into next key index
    std::uint32_t t_base;       // pc[7]: base key slot inside keys_buf for primary key
    std::uint32_t s_base;       // pc[8]: base key slot inside keys_buf for S# key (PairTSLookup)
    std::uint32_t s_mask;       // pc[9]: mask for S# ring (PairTSLookup)
};
static_assert(sizeof(PushConstants) == 40);

class SpirvBuilder {
public:
    std::vector<std::uint32_t> build(ShaderKind kind) {
        const bool has_desc_map = kind != ShaderKind::Baseline;

        const std::uint32_t ty_void = id();
        const std::uint32_t ty_bool = id();
        const std::uint32_t ty_uint = id();
        const std::uint32_t ty_v3u = id();
        const std::uint32_t ty_rt_u = id();
        const std::uint32_t ty_buf = id();
        const std::uint32_t ty_pc = id();
        const std::uint32_t ptr_in_v3u = id();
        const std::uint32_t ptr_in_u = id();
        const std::uint32_t ptr_sb_buf = id();
        const std::uint32_t ptr_sb_u = id();
        const std::uint32_t ptr_pc = id();
        const std::uint32_t ptr_pc_u = id();
        const std::uint32_t ptr_fn_u = id();
        const std::uint32_t ty_fn_main = id();
        const std::uint32_t ty_fn_lookup = has_desc_map ? id() : 0u;
        const std::uint32_t true_id = has_desc_map ? id() : 0u;

        const std::uint32_t gid_var = id();
        const std::uint32_t keys_var = id();
        const std::uint32_t map_var = has_desc_map ? id() : 0u;
        const std::uint32_t out_var = id();
        const std::uint32_t pc_var = id();
        const std::uint32_t fn_lookup = has_desc_map ? id() : 0u;
        const std::uint32_t fn_main = id();

        // Prologue: Capability, MemoryModel, EntryPoint, ExecutionMode
        emit(preamble_, spv_op::kOpCapability, {spv_op::kCapShader});
        emit(preamble_, spv_op::kOpMemoryModel, {spv_op::kAddrLogical, spv_op::kMemGlsl450});
        emit(preamble_, spv_op::kOpEntryPoint,
             {spv_op::kExecGLCompute, fn_main, 0x6E69616Du, 0x00000000u, gid_var});
        emit(preamble_, spv_op::kOpExecutionMode,
             {fn_main, spv_op::kExecModeLocalSize, kWorkgroupSize, 1u, 1u});

        // Annotations
        emit(decs_, spv_op::kOpDecorate, {gid_var, spv_op::kDecBuiltIn, spv_op::kBuiltInGlobalInvocationId});
        emit(decs_, spv_op::kOpDecorate, {ty_rt_u, spv_op::kDecArrayStride, 4u});
        emit(decs_, spv_op::kOpMemberDecorate, {ty_buf, 0u, spv_op::kDecOffset, 0u});
        emit(decs_, spv_op::kOpDecorate, {ty_buf, spv_op::kDecBlock});

        emit(decs_, spv_op::kOpDecorate, {keys_var, spv_op::kDecDescriptorSet, 0u});
        emit(decs_, spv_op::kOpDecorate, {keys_var, spv_op::kDecBinding, 0u});
        emit(decs_, spv_op::kOpDecorate, {keys_var, spv_op::kDecNonWritable});

        if (has_desc_map) {
            emit(decs_, spv_op::kOpDecorate, {map_var, spv_op::kDecDescriptorSet, 0u});
            emit(decs_, spv_op::kOpDecorate, {map_var, spv_op::kDecBinding, 2u});
            emit(decs_, spv_op::kOpDecorate, {map_var, spv_op::kDecNonWritable});
        }

        emit(decs_, spv_op::kOpDecorate, {out_var, spv_op::kDecDescriptorSet, 0u});
        emit(decs_, spv_op::kOpDecorate, {out_var, spv_op::kDecBinding, 3u});
        emit(decs_, spv_op::kOpDecorate, {out_var, spv_op::kDecNonReadable});

        for (std::uint32_t m = 0; m < 10u; ++m) {
            emit(decs_, spv_op::kOpMemberDecorate, {ty_pc, m, spv_op::kDecOffset, m * 4u});
        }
        emit(decs_, spv_op::kOpDecorate, {ty_pc, spv_op::kDecBlock});

        // Types
        emit(types_, spv_op::kOpTypeVoid, {ty_void});
        emit(types_, spv_op::kOpTypeBool, {ty_bool});
        emit(types_, spv_op::kOpTypeInt, {ty_uint, 32u, 0u});
        emit(types_, spv_op::kOpTypeVector, {ty_v3u, ty_uint, 3u});
        emit(types_, spv_op::kOpTypeRuntimeArray, {ty_rt_u, ty_uint});
        emit(types_, spv_op::kOpTypeStruct, {ty_buf, ty_rt_u});
        emit(types_, spv_op::kOpTypeStruct,
             {ty_pc, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint});
        emit(types_, spv_op::kOpTypePointer, {ptr_in_v3u, spv_op::kScInput, ty_v3u});
        emit(types_, spv_op::kOpTypePointer, {ptr_in_u, spv_op::kScInput, ty_uint});
        emit(types_, spv_op::kOpTypePointer, {ptr_sb_buf, spv_op::kScStorageBuffer, ty_buf});
        emit(types_, spv_op::kOpTypePointer, {ptr_sb_u, spv_op::kScStorageBuffer, ty_uint});
        emit(types_, spv_op::kOpTypePointer, {ptr_pc, spv_op::kScPushConstant, ty_pc});
        emit(types_, spv_op::kOpTypePointer, {ptr_pc_u, spv_op::kScPushConstant, ty_uint});
        emit(types_, spv_op::kOpTypePointer, {ptr_fn_u, spv_op::kScFunction, ty_uint});
        emit(types_, spv_op::kOpTypeFunction, {ty_fn_main, ty_void});
        if (has_desc_map) {
            emit(types_, spv_op::kOpTypeFunction,
                 {ty_fn_lookup, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint, ty_uint});
            emit(consts_, spv_op::kOpConstantTrue, {ty_bool, true_id});
        }

        // Global variables (placed after constants & pointer types)
        emit(gvars_, spv_op::kOpVariable, {ptr_in_v3u, gid_var, spv_op::kScInput});
        emit(gvars_, spv_op::kOpVariable, {ptr_sb_buf, keys_var, spv_op::kScStorageBuffer});
        if (has_desc_map) {
            emit(gvars_, spv_op::kOpVariable, {ptr_sb_buf, map_var, spv_op::kScStorageBuffer});
        }
        emit(gvars_, spv_op::kOpVariable, {ptr_sb_buf, out_var, spv_op::kScStorageBuffer});
        emit(gvars_, spv_op::kOpVariable, {ptr_pc, pc_var, spv_op::kScPushConstant});

        ty_bool_ = ty_bool;
        ty_uint_ = ty_uint;
        ptr_sb_u_ = ptr_sb_u;
        ptr_fn_u_ = ptr_fn_u;

        // %desc_lookup_fn(w0..w7) -> uint: identical probe loop to tools/rdna2spirv.py desc_probe()
        if (has_desc_map) {
            emit_desc_lookup_fn(fn_lookup, ty_fn_lookup, map_var, true_id);
        }

        // %main()
        emit_main_fn(kind, fn_main, ty_void, ty_fn_main, ptr_in_u, ptr_pc_u, gid_var, keys_var, out_var, pc_var,
                     fn_lookup);

        std::vector<std::uint32_t> spv;
        spv.reserve(5 + preamble_.size() + decs_.size() + types_.size() + consts_.size() + gvars_.size() + funcs_.size());
        spv.push_back(spv_op::kMagic);
        spv.push_back(spv_op::kVersion13);
        spv.push_back(0u);
        spv.push_back(next_id_);
        spv.push_back(0u);
        append(spv, preamble_);
        append(spv, decs_);
        append(spv, types_);
        append(spv, consts_);
        append(spv, gvars_);
        append(spv, funcs_);
        return spv;
    }

private:
    std::uint32_t id() {
        return next_id_++;
    }

    std::uint32_t uconst(std::uint32_t value) {
        const auto [it, inserted] = uconsts_.emplace(value, 0u);
        if (inserted) {
            it->second = id();
            emit(consts_, spv_op::kOpConstant, {ty_uint_, it->second, value});
        }
        return it->second;
    }

    static void emit(std::vector<std::uint32_t>& dst, std::uint32_t opcode, std::initializer_list<std::uint32_t> ops) {
        const std::uint32_t words = static_cast<std::uint32_t>(1 + ops.size());
        dst.push_back((words << 16) | opcode);
        for (const std::uint32_t w : ops) dst.push_back(w);
    }

    static void emit_vec(std::vector<std::uint32_t>& dst, std::uint32_t opcode, const std::vector<std::uint32_t>& ops) {
        const std::uint32_t words = static_cast<std::uint32_t>(1 + ops.size());
        dst.push_back((words << 16) | opcode);
        for (const std::uint32_t w : ops) dst.push_back(w);
    }

    static void append(std::vector<std::uint32_t>& dst, const std::vector<std::uint32_t>& src) {
        dst.insert(dst.end(), src.begin(), src.end());
    }

    std::uint32_t binop(std::uint32_t opcode, std::uint32_t res_type, std::uint32_t a, std::uint32_t b) {
        const std::uint32_t r = id();
        emit(funcs_, opcode, {res_type, r, a, b});
        return r;
    }

    std::uint32_t load(std::uint32_t res_type, std::uint32_t ptr) {
        const std::uint32_t r = id();
        emit(funcs_, spv_op::kOpLoad, {res_type, r, ptr});
        return r;
    }

    std::uint32_t access2(std::uint32_t ptr_type, std::uint32_t base, std::uint32_t idx0, std::uint32_t idx1) {
        const std::uint32_t r = id();
        emit(funcs_, spv_op::kOpAccessChain, {ptr_type, r, base, idx0, idx1});
        return r;
    }

    void emit_desc_lookup_fn(std::uint32_t fn_id, std::uint32_t fn_type, std::uint32_t map_var,
                             std::uint32_t true_id) {
        const std::uint32_t c0 = uconst(0u);
        const std::uint32_t c1 = uconst(1u);
        const std::uint32_t c8 = uconst(kDescriptorProbes);
        const std::uint32_t c9 = uconst(9u);
        const std::uint32_t c10 = uconst(10u);
        const std::uint32_t c_offset = uconst(kFnvOffsetBasis);
        const std::uint32_t c_prime = uconst(kFnvPrime);

        emit(funcs_, spv_op::kOpFunction, {ty_uint_, fn_id, 0u, fn_type});
        std::array<std::uint32_t, 8> params{};
        for (int i = 0; i < 8; ++i) {
            params[i] = id();
            emit(funcs_, spv_op::kOpFunctionParameter, {ty_uint_, params[i]});
        }

        const std::uint32_t lbl_entry = id();
        const std::uint32_t lbl_head = id();
        const std::uint32_t lbl_body = id();
        const std::uint32_t lbl_hit = id();
        const std::uint32_t lbl_next = id();
        const std::uint32_t lbl_cont = id();
        const std::uint32_t lbl_end = id();

        emit(funcs_, spv_op::kOpLabel, {lbl_entry});
        const std::uint32_t kv = id();
        const std::uint32_t iv = id();
        emit(funcs_, spv_op::kOpVariable, {ptr_fn_u_, kv, spv_op::kScFunction, c0});
        emit(funcs_, spv_op::kOpVariable, {ptr_fn_u_, iv, spv_op::kScFunction, c0});

        std::uint32_t h = c_offset;
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t x = binop(spv_op::kOpBitwiseXor, ty_uint_, h, params[i]);
            h = binop(spv_op::kOpIMul, ty_uint_, x, c_prime);
        }
        const std::uint32_t mask_ptr = access2(ptr_sb_u_, map_var, c0, c0);
        const std::uint32_t mask = load(ty_uint_, mask_ptr);
        emit(funcs_, spv_op::kOpBranch, {lbl_head});

        emit(funcs_, spv_op::kOpLabel, {lbl_head});
        emit(funcs_, spv_op::kOpLoopMerge, {lbl_end, lbl_cont, 0u});
        emit(funcs_, spv_op::kOpBranch, {lbl_body});

        emit(funcs_, spv_op::kOpLabel, {lbl_body});
        const std::uint32_t k = load(ty_uint_, kv);
        const std::uint32_t s = binop(spv_op::kOpIAdd, ty_uint_, h, k);
        const std::uint32_t slot = binop(spv_op::kOpBitwiseAnd, ty_uint_, s, mask);
        const std::uint32_t base = binop(spv_op::kOpIMul, ty_uint_, slot, c10);
        std::uint32_t match = true_id;
        for (std::uint32_t j = 0; j < 8u; ++j) {
            const std::uint32_t off = binop(spv_op::kOpIAdd, ty_uint_, base, uconst(1u + j));
            const std::uint32_t p = access2(ptr_sb_u_, map_var, c0, off);
            const std::uint32_t v = load(ty_uint_, p);
            const std::uint32_t eq = binop(spv_op::kOpIEqual, ty_bool_, v, params[j]);
            match = binop(spv_op::kOpLogicalAnd, ty_bool_, match, eq);
        }
        emit(funcs_, spv_op::kOpSelectionMerge, {lbl_next, 0u});
        emit(funcs_, spv_op::kOpBranchConditional, {match, lbl_hit, lbl_next});

        emit(funcs_, spv_op::kOpLabel, {lbl_hit});
        const std::uint32_t idx_off = binop(spv_op::kOpIAdd, ty_uint_, base, c9);
        const std::uint32_t idx_ptr = access2(ptr_sb_u_, map_var, c0, idx_off);
        const std::uint32_t idx_val = load(ty_uint_, idx_ptr);
        emit(funcs_, spv_op::kOpStore, {iv, idx_val});
        emit(funcs_, spv_op::kOpStore, {kv, c8});
        emit(funcs_, spv_op::kOpBranch, {lbl_next});

        emit(funcs_, spv_op::kOpLabel, {lbl_next});
        emit(funcs_, spv_op::kOpBranch, {lbl_cont});

        emit(funcs_, spv_op::kOpLabel, {lbl_cont});
        const std::uint32_t k2 = load(ty_uint_, kv);
        const std::uint32_t k3 = binop(spv_op::kOpIAdd, ty_uint_, k2, c1);
        emit(funcs_, spv_op::kOpStore, {kv, k3});
        const std::uint32_t more = binop(spv_op::kOpULessThan, ty_bool_, k3, c8);
        emit(funcs_, spv_op::kOpBranchConditional, {more, lbl_head, lbl_end});

        emit(funcs_, spv_op::kOpLabel, {lbl_end});
        const std::uint32_t ret = load(ty_uint_, iv);
        emit(funcs_, spv_op::kOpReturnValue, {ret});
        emit(funcs_, spv_op::kOpFunctionEnd, {});
    }

    void emit_main_fn(ShaderKind kind, std::uint32_t fn_id, std::uint32_t ty_void, std::uint32_t ty_fn_main,
                      std::uint32_t ptr_in_u, std::uint32_t ptr_pc_u, std::uint32_t gid_var, std::uint32_t keys_var,
                      std::uint32_t out_var, std::uint32_t pc_var, std::uint32_t fn_lookup) {
        const std::uint32_t c0 = uconst(0u);
        const std::uint32_t c1 = uconst(1u);
        const std::uint32_t c3 = uconst(3u);
        const std::uint32_t c6 = uconst(6u);
        const std::uint32_t c63 = uconst(63u);
        const std::uint32_t c_offset = uconst(kFnvOffsetBasis);
        const std::uint32_t c_prime = uconst(kFnvPrime);
        const std::uint32_t c_ff = uconst(0xFFFFFFFFu);

        emit(funcs_, spv_op::kOpFunction, {ty_void, fn_id, 0u, ty_fn_main});
        const std::uint32_t lbl_entry = id();
        const std::uint32_t lbl_head = id();
        const std::uint32_t lbl_body = id();
        const std::uint32_t lbl_cont = id();
        const std::uint32_t lbl_end = id();

        emit(funcs_, spv_op::kOpLabel, {lbl_entry});
        const std::uint32_t i_var = id();
        const std::uint32_t acc_var = id();
        const std::uint32_t prev_var = id();
        emit(funcs_, spv_op::kOpVariable, {ptr_fn_u_, i_var, spv_op::kScFunction, c0});
        emit(funcs_, spv_op::kOpVariable, {ptr_fn_u_, acc_var, spv_op::kScFunction, c_offset});
        emit(funcs_, spv_op::kOpVariable, {ptr_fn_u_, prev_var, spv_op::kScFunction, c0});

        const std::uint32_t gid_ptr = id();
        emit(funcs_, spv_op::kOpAccessChain, {ptr_in_u, gid_ptr, gid_var, c0});
        const std::uint32_t gid_x = load(ty_uint_, gid_ptr);

        std::array<std::uint32_t, 10> pc{};
        for (std::uint32_t m = 0; m < 10u; ++m) {
            const std::uint32_t p = id();
            emit(funcs_, spv_op::kOpAccessChain, {ptr_pc_u, p, pc_var, uconst(m)});
            pc[m] = load(ty_uint_, p);
        }

        const std::uint32_t wg = binop(spv_op::kOpShiftRightLogical, ty_uint_, gid_x, c6);
        const std::uint32_t lid = binop(spv_op::kOpBitwiseAnd, ty_uint_, gid_x, c63);
        const std::uint32_t wg_term = binop(spv_op::kOpIMul, ty_uint_, wg, pc[2]);
        const std::uint32_t lid_term = binop(spv_op::kOpIMul, ty_uint_, lid, pc[3]);
        const std::uint32_t s0 = binop(spv_op::kOpIAdd, ty_uint_, pc[5], wg_term);
        const std::uint32_t start = binop(spv_op::kOpIAdd, ty_uint_, s0, lid_term);
        emit(funcs_, spv_op::kOpBranch, {lbl_head});

        emit(funcs_, spv_op::kOpLabel, {lbl_head});
        const std::uint32_t iv = load(ty_uint_, i_var);
        const std::uint32_t cond = binop(spv_op::kOpULessThan, ty_bool_, iv, pc[1]);
        emit(funcs_, spv_op::kOpLoopMerge, {lbl_end, lbl_cont, 0u});
        emit(funcs_, spv_op::kOpBranchConditional, {cond, lbl_body, lbl_end});

        emit(funcs_, spv_op::kOpLabel, {lbl_body});
        const std::uint32_t prev = load(ty_uint_, prev_var);
        const std::uint32_t dep = binop(spv_op::kOpBitwiseAnd, ty_uint_, prev, pc[6]);
        const std::uint32_t iter_delta = binop(spv_op::kOpIMul, ty_uint_, iv, pc[4]);
        const std::uint32_t idx_base = binop(spv_op::kOpIAdd, ty_uint_, start, iter_delta);
        const std::uint32_t idx1 = binop(spv_op::kOpIAdd, ty_uint_, idx_base, dep);

        const std::uint32_t masked_t = binop(spv_op::kOpBitwiseAnd, ty_uint_, idx1, pc[0]);
        const std::uint32_t kidx = binop(spv_op::kOpIAdd, ty_uint_, pc[7], masked_t);
        const std::uint32_t koff = binop(spv_op::kOpShiftLeftLogical, ty_uint_, kidx, c3);

        std::array<std::uint32_t, 8> w{};
        for (std::uint32_t j = 0; j < 8u; ++j) {
            const std::uint32_t ko = binop(spv_op::kOpIAdd, ty_uint_, koff, uconst(j));
            const std::uint32_t kp = access2(ptr_sb_u_, keys_var, c0, ko);
            w[j] = load(ty_uint_, kp);
        }

        if (kind == ShaderKind::Baseline) {
            std::uint32_t mix = w[0];
            for (int j = 1; j < 8; ++j) {
                mix = binop(spv_op::kOpBitwiseXor, ty_uint_, mix, w[j]);
            }
            const std::uint32_t acc0 = load(ty_uint_, acc_var);
            const std::uint32_t acc1 = binop(spv_op::kOpIMul, ty_uint_, acc0, c_prime);
            const std::uint32_t acc2 = binop(spv_op::kOpBitwiseXor, ty_uint_, acc1, mix);
            emit(funcs_, spv_op::kOpStore, {acc_var, acc2});
            emit(funcs_, spv_op::kOpStore, {prev_var, mix});
        } else if (kind == ShaderKind::SingleLookup) {
            const std::uint32_t res = id();
            emit_vec(funcs_, spv_op::kOpFunctionCall,
                     {ty_uint_, res, fn_lookup, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]});
            const std::uint32_t acc0 = load(ty_uint_, acc_var);
            const std::uint32_t acc1 = binop(spv_op::kOpIMul, ty_uint_, acc0, c_prime);
            const std::uint32_t acc2 = binop(spv_op::kOpBitwiseXor, ty_uint_, acc1, res);
            emit(funcs_, spv_op::kOpStore, {acc_var, acc2});
            emit(funcs_, spv_op::kOpStore, {prev_var, res});
        } else {
            const std::uint32_t res_t = id();
            emit_vec(funcs_, spv_op::kOpFunctionCall,
                     {ty_uint_, res_t, fn_lookup, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]});
            const std::uint32_t masked_s = binop(spv_op::kOpBitwiseAnd, ty_uint_, idx1, pc[9]);
            const std::uint32_t skidx = binop(spv_op::kOpIAdd, ty_uint_, pc[8], masked_s);
            const std::uint32_t soff = binop(spv_op::kOpShiftLeftLogical, ty_uint_, skidx, c3);
            std::array<std::uint32_t, 4> s{};
            for (std::uint32_t j = 0; j < 4u; ++j) {
                const std::uint32_t so = binop(spv_op::kOpIAdd, ty_uint_, soff, uconst(j));
                const std::uint32_t sp = access2(ptr_sb_u_, keys_var, c0, so);
                s[j] = load(ty_uint_, sp);
            }
            const std::uint32_t res_s = id();
            emit_vec(funcs_, spv_op::kOpFunctionCall,
                     {ty_uint_, res_s, fn_lookup, s[0], s[1], s[2], s[3], c_ff, c_ff, c_ff, c_ff});
            const std::uint32_t acc0 = load(ty_uint_, acc_var);
            const std::uint32_t acc1 = binop(spv_op::kOpIMul, ty_uint_, acc0, c_prime);
            const std::uint32_t acc2 = binop(spv_op::kOpBitwiseXor, ty_uint_, acc1, res_t);
            const std::uint32_t acc3 = binop(spv_op::kOpIMul, ty_uint_, acc2, c_prime);
            const std::uint32_t acc4 = binop(spv_op::kOpBitwiseXor, ty_uint_, acc3, res_s);
            emit(funcs_, spv_op::kOpStore, {acc_var, acc4});
            emit(funcs_, spv_op::kOpStore, {prev_var, res_t});
        }
        emit(funcs_, spv_op::kOpBranch, {lbl_cont});

        emit(funcs_, spv_op::kOpLabel, {lbl_cont});
        const std::uint32_t i_cur = load(ty_uint_, i_var);
        const std::uint32_t i_nxt = binop(spv_op::kOpIAdd, ty_uint_, i_cur, c1);
        emit(funcs_, spv_op::kOpStore, {i_var, i_nxt});
        emit(funcs_, spv_op::kOpBranch, {lbl_head});

        emit(funcs_, spv_op::kOpLabel, {lbl_end});
        const std::uint32_t final_acc = load(ty_uint_, acc_var);
        const std::uint32_t out_ptr = access2(ptr_sb_u_, out_var, c0, gid_x);
        emit(funcs_, spv_op::kOpStore, {out_ptr, final_acc});
        emit(funcs_, spv_op::kOpReturn, {});
        emit(funcs_, spv_op::kOpFunctionEnd, {});
    }

    std::uint32_t next_id_ = 1u;
    std::uint32_t ty_bool_ = 0u;
    std::uint32_t ty_uint_ = 0u;
    std::uint32_t ptr_sb_u_ = 0u;
    std::uint32_t ptr_fn_u_ = 0u;
    std::map<std::uint32_t, std::uint32_t> uconsts_;
    std::vector<std::uint32_t> preamble_;
    std::vector<std::uint32_t> decs_;
    std::vector<std::uint32_t> types_;
    std::vector<std::uint32_t> consts_;
    std::vector<std::uint32_t> gvars_;
    std::vector<std::uint32_t> funcs_;
};

std::vector<std::uint32_t> build_bench_spv(ShaderKind kind) {
    SpirvBuilder b;
    return b.build(kind);
}

bool check_spv_structure(const std::vector<std::uint32_t>& words, const char* label) {
    if (words.size() < 5u || words[0] != spv_op::kMagic || words[1] != spv_op::kVersion13 || words[3] < 2u) {
        std::fprintf(stderr, "self-test: invalid SPIR-V header for %s\n", label);
        return false;
    }
    const std::uint32_t bound = words[3];
    std::size_t idx = 5u;
    int open_fns = 0;
    bool seen_entry = false;
    while (idx < words.size()) {
        const std::uint32_t head = words[idx];
        const std::uint32_t wc = head >> 16;
        const std::uint32_t op = head & 0xFFFFu;
        if (wc == 0u || idx + wc > words.size()) {
            std::fprintf(stderr, "self-test: truncated instruction in %s at word %zu\n", label, idx);
            return false;
        }
        if (op == spv_op::kOpEntryPoint) seen_entry = true;
        if (op == spv_op::kOpFunction) {
            if (wc < 3u || words[idx + 2u] >= bound) return false;
            ++open_fns;
        } else if (op == spv_op::kOpFunctionEnd) {
            if (open_fns <= 0) return false;
            --open_fns;
        }
        idx += wc;
    }
    if (!seen_entry || open_fns != 0) {
        std::fprintf(stderr, "self-test: malformed function or entry point in %s\n", label);
        return false;
    }
    return true;
}

bool write_spv_file(const std::string& path, const std::vector<std::uint32_t>& words) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "could not open %s for writing\n", path.c_str());
        return false;
    }
    f.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * sizeof(std::uint32_t)));
    if (f.fail()) {
        std::fprintf(stderr, "failed writing %s\n", path.c_str());
        return false;
    }
    return true;
}

// ---- Workload generation & exact CPU oracle ----

std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

std::uint32_t next_pow2_u32(std::uint32_t n) {
    std::uint32_t p = 1u;
    while (p < n) p <<= 1;
    return p;
}

DescriptorKey make_bench_tsharp(std::uint32_t id) {
    const std::uint64_t va = 0x0000010000000000ULL + static_cast<std::uint64_t>(id) * 256ULL;
    const std::uint32_t fmt = 10u + (id % 24u);
    const std::uint32_t w = 63u + ((id * 37u) % 2048u);
    const std::uint32_t h = 63u + ((id * 53u) % 2048u);
    return {
        static_cast<std::uint32_t>(va >> 8),
        (static_cast<std::uint32_t>(va >> 40) & 0xFFu) | 0x2000u | (fmt << 20),
        (w & 0x3FFFu) | ((h & 0x3FFFu) << 14),
        0x000AA688u ^ (id & 0xFFu),
        id ^ 0x9E3779B9u,
        (id * 0x85EBCA6Bu) & 0x00FFFFFFu,
        0x00000010u | ((id % 7u) << 12),
        0x80000000u ^ (id << 16),
    };
}

DescriptorKey make_bench_ssharp(std::uint32_t id) {
    const std::uint32_t raw[4] = {
        0x00000092u | ((id & 0x7u) << 20),
        0x00C00000u ^ (id << 8),
        0x14500000u | (id & 0xFFFu),
        id ^ 0xA5A50001u,
    };
    return make_ssharp_key(raw);
}

DescriptorKey make_bench_miss(std::uint32_t id) {
    return {
        0xDEAD0000u ^ (id * 0x9E3779B9u),
        0xBEEF0000u ^ id,
        0xCAFE0001u + id,
        0xBAAD0002u ^ (id << 3),
        0x13572468u,
        0x24681357u,
        0xFACEB00Cu,
        0x40000000u | ((id & 0xFFFFFFu) << 1),
    };
}

// Mirrors the shader's exact probe loop against the exported storage buffer, returning {index, 1-based probe count}.
std::pair<std::uint32_t, std::uint32_t> probe_gpu_table(const std::vector<std::uint32_t>& table_buf,
                                                        const DescriptorKey& key) {
    const std::uint32_t h = hash_descriptor_key(key);
    const std::uint32_t mask = table_buf[0];
    for (std::uint32_t k = 0; k < kDescriptorProbes; ++k) {
        const std::uint32_t base = ((h + k) & mask) * 10u;
        bool match = true;
        for (std::uint32_t j = 0; j < 8u; ++j) {
            match = match && (table_buf[1u + base + j] == key[j]);
        }
        if (match) return {table_buf[base + 9u], k + 1u};
    }
    return {0u, kDescriptorProbes};
}

struct TableDataset {
    std::uint32_t entry_count = 0;
    std::uint32_t t_count = 0;
    std::uint32_t s_count = 0;
    std::uint32_t capacity = 0;
    double load_pct = 0.0;
    double mean_probes = 0.0;
    double probe1_pct = 0.0;
    std::uint32_t max_probes = 0;
    std::uint32_t t_pow2 = 0;
    std::uint32_t s_pow2 = 0;
    std::uint32_t t_base = 0;
    std::uint32_t s_base = 0;
    std::uint32_t miss_base = 0;
    std::vector<std::uint32_t> gpu_table;
    std::vector<std::uint32_t> keys_buf;          // [T# (t_pow2) | S# (s_pow2) | Miss (t_pow2)] x 8 u32
    std::vector<std::uint32_t> expected_baseline; // per key slot: XOR of 8 key words
    std::vector<std::uint32_t> expected_lookup;   // per key slot: expected bindless index
    std::vector<std::uint32_t> slot_entry_id;     // per key slot: unique entry index within its bank
};

bool build_dataset(std::uint32_t entry_count, TableDataset& out) {
    out = TableDataset{};
    out.entry_count = entry_count;
    out.s_count = std::min(entry_count / 2u, std::max<std::uint32_t>(16u, entry_count / 10u));
    out.t_count = entry_count - out.s_count;

    DescriptorTable table(std::max<std::uint32_t>(32u, next_pow2_u32(entry_count * 2u)));
    std::vector<DescriptorKey> t_keys(out.t_count);
    std::vector<DescriptorKey> s_keys(out.s_count);

    for (std::uint32_t i = 0; i < out.t_count; ++i) {
        t_keys[i] = make_bench_tsharp(i + 1u);
        table.insert(t_keys[i], 1u + i);
    }
    for (std::uint32_t i = 0; i < out.s_count; ++i) {
        s_keys[i] = make_bench_ssharp(i + 1u);
        table.insert(s_keys[i], 1u + out.t_count + i);
    }

    out.capacity = static_cast<std::uint32_t>(table.capacity());
    out.load_pct = 100.0 * static_cast<double>(entry_count) / static_cast<double>(out.capacity);
    out.gpu_table = table.build_gpu_buffer();

    std::uint64_t probe_sum = 0;
    std::uint32_t probe1 = 0;
    std::uint32_t max_p = 0;
    auto check_live = [&](const DescriptorKey& k, std::uint32_t expected_idx) -> bool {
        const auto [got, probes] = probe_gpu_table(out.gpu_table, k);
        if (got != expected_idx || table.lookup(k) != expected_idx) {
            std::fprintf(stderr, "table oracle mismatch: got=%u cpu=%u expected=%u\n", got, table.lookup(k),
                         expected_idx);
            return false;
        }
        probe_sum += probes;
        if (probes == 1u) ++probe1;
        if (probes > max_p) max_p = probes;
        return true;
    };
    for (std::uint32_t i = 0; i < out.t_count; ++i) {
        if (!check_live(t_keys[i], 1u + i)) return false;
    }
    for (std::uint32_t i = 0; i < out.s_count; ++i) {
        if (!check_live(s_keys[i], 1u + out.t_count + i)) return false;
    }
    out.mean_probes = static_cast<double>(probe_sum) / static_cast<double>(entry_count);
    out.probe1_pct = 100.0 * static_cast<double>(probe1) / static_cast<double>(entry_count);
    out.max_probes = max_p;

    // Shuffle key rings deterministically so unit-stride walks visit scattered table slots across L2/DRAM.
    std::uint64_t rng = 0x4C454F4E49444156ULL ^ static_cast<std::uint64_t>(entry_count);
    std::vector<std::uint32_t> t_perm(out.t_count);
    std::iota(t_perm.begin(), t_perm.end(), 0u);
    for (std::uint32_t i = out.t_count; i > 1u; --i) {
        const std::uint32_t j = static_cast<std::uint32_t>(splitmix64(rng) % i);
        std::swap(t_perm[i - 1u], t_perm[j]);
    }
    // Pin t_perm[0] to a depth-1 entry so uniform_hot_tsharp measures identical probe depth across sizes.
    for (std::uint32_t k = 0; k < out.t_count; ++k) {
        if (probe_gpu_table(out.gpu_table, t_keys[t_perm[k]]).second == 1u) {
            std::swap(t_perm[0], t_perm[k]);
            break;
        }
    }
    std::vector<std::uint32_t> s_perm(out.s_count);
    std::iota(s_perm.begin(), s_perm.end(), 0u);
    for (std::uint32_t i = out.s_count; i > 1u; --i) {
        const std::uint32_t j = static_cast<std::uint32_t>(splitmix64(rng) % i);
        std::swap(s_perm[i - 1u], s_perm[j]);
    }

    out.t_pow2 = next_pow2_u32(out.t_count);
    out.s_pow2 = next_pow2_u32(out.s_count);
    out.t_base = 0u;
    out.s_base = out.t_pow2;
    out.miss_base = out.t_pow2 + out.s_pow2;

    const std::uint32_t total_slots = out.t_pow2 + out.s_pow2 + out.t_pow2;
    out.keys_buf.resize(static_cast<std::size_t>(total_slots) * 8u);
    out.expected_baseline.resize(total_slots);
    out.expected_lookup.resize(total_slots);
    out.slot_entry_id.resize(total_slots);

    const auto store_slot = [&](std::uint32_t slot, const DescriptorKey& key, std::uint32_t expected_idx,
                                std::uint32_t entry_id) {
        std::uint32_t xor_mix = 0u;
        for (std::uint32_t j = 0; j < 8u; ++j) {
            out.keys_buf[static_cast<std::size_t>(slot) * 8u + j] = key[j];
            xor_mix ^= key[j];
        }
        out.expected_baseline[slot] = xor_mix;
        out.expected_lookup[slot] = expected_idx;
        out.slot_entry_id[slot] = entry_id;
    };

    for (std::uint32_t i = 0; i < out.t_pow2; ++i) {
        const std::uint32_t idx = t_perm[i % out.t_count];
        store_slot(out.t_base + i, t_keys[idx], 1u + idx, idx);
    }
    for (std::uint32_t i = 0; i < out.s_pow2; ++i) {
        const std::uint32_t idx = s_perm[i % out.s_count];
        store_slot(out.s_base + i, s_keys[idx], 1u + out.t_count + idx, idx);
    }
    std::uint32_t miss_seq = 0u;
    for (std::uint32_t i = 0; i < out.t_pow2; ++i) {
        DescriptorKey mk{};
        for (;;) {
            mk = make_bench_miss(miss_seq++);
            if (!table.contains(mk) && probe_gpu_table(out.gpu_table, mk).first == 0u) break;
        }
        store_slot(out.miss_base + i, mk, 0u, i);
    }
    return true;
}

struct ScenarioSpec {
    const char* name;
    ShaderKind shader;
    bool divergent;
    bool group_stride_is_iters;
    std::uint32_t group_stride;
    std::uint32_t lane_stride;
    std::uint32_t iter_stride;
    bool dep_chain;
    enum class Bank { TSharp, SSharp, Miss } bank;
    double lookups_per_iter;
};

PushConstants make_push_constants(const ScenarioSpec& sc, const TableDataset& ds, std::uint32_t iters) {
    PushConstants pc{};
    pc.iters = iters;
    pc.group_stride = sc.group_stride_is_iters ? iters : sc.group_stride;
    pc.lane_stride = sc.lane_stride;
    pc.iter_stride = sc.iter_stride;
    pc.base_index = 0u;
    pc.s_base = ds.s_base;
    pc.s_mask = ds.s_pow2 - 1u;
    if (sc.bank == ScenarioSpec::Bank::TSharp) {
        pc.t_base = ds.t_base;
        pc.key_mask = ds.t_pow2 - 1u;
    } else if (sc.bank == ScenarioSpec::Bank::SSharp) {
        pc.t_base = ds.s_base;
        pc.key_mask = ds.s_pow2 - 1u;
    } else {
        pc.t_base = ds.miss_base;
        pc.key_mask = ds.t_pow2 - 1u;
    }
    pc.dep_mask = sc.dep_chain ? pc.key_mask : 0u;
    return pc;
}

std::uint32_t count_unique_keys(const ScenarioSpec& sc, const TableDataset& ds, const PushConstants& pc,
                                std::uint32_t eff_groups) {
    const std::uint32_t primary_distinct =
        (sc.bank == ScenarioSpec::Bank::TSharp) ? ds.t_count
        : (sc.bank == ScenarioSpec::Bank::SSharp) ? ds.s_count
                                                  : ds.t_pow2;
    const std::uint32_t eff_invocations = eff_groups * kWorkgroupSize;
    if (sc.dep_chain) {
        if (eff_invocations >= primary_distinct) return primary_distinct;
        std::vector<std::uint8_t> seen(primary_distinct, 0u);
        std::uint32_t count = 0u;
        for (std::uint32_t tid = 0; tid < eff_invocations; ++tid) {
            const std::uint32_t wg = tid >> 6;
            const std::uint32_t lid = tid & 63u;
            const std::uint32_t start = pc.base_index + wg * pc.group_stride + lid * pc.lane_stride;
            std::uint32_t prev = 0u;
            for (std::uint32_t i = 0; i < pc.iters; ++i) {
                const std::uint32_t idx1 = start + i * pc.iter_stride + (prev & pc.dep_mask);
                const std::uint32_t kidx = pc.t_base + (idx1 & pc.key_mask);
                const std::uint32_t eid = ds.slot_entry_id[kidx];
                if (!seen[eid]) {
                    seen[eid] = 1u;
                    ++count;
                }
                prev = ds.expected_lookup[kidx];
            }
        }
        return count;
    }

    std::uint64_t span = 1u;
    if (pc.group_stride == 0u && pc.lane_stride == 0u && pc.iter_stride == 0u) {
        span = 1u;
    } else if (pc.lane_stride == 0u && pc.iter_stride == 1u && pc.group_stride == pc.iters) {
        span = static_cast<std::uint64_t>(eff_groups) * static_cast<std::uint64_t>(pc.iters);
    } else if (pc.group_stride == kWorkgroupSize && pc.lane_stride == 1u && pc.iter_stride == 1u) {
        span = static_cast<std::uint64_t>(eff_groups) * kWorkgroupSize + pc.iters - 1u;
    } else {
        span = static_cast<std::uint64_t>(eff_invocations) * static_cast<std::uint64_t>(pc.iters);
    }

    const std::uint32_t primary_seen =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(primary_distinct, span));
    if (sc.shader == ShaderKind::PairTSLookup) {
        const std::uint32_t s_seen = static_cast<std::uint32_t>(std::min<std::uint64_t>(ds.s_count, span));
        return primary_seen + s_seen;
    }
    return primary_seen;
}

std::uint32_t eval_invocation_oracle(const ScenarioSpec& sc, const TableDataset& ds, const PushConstants& pc,
                                     std::uint32_t tid) {
    const std::uint32_t wg = tid >> 6;
    const std::uint32_t lid = tid & 63u;
    const std::uint32_t start = pc.base_index + wg * pc.group_stride + lid * pc.lane_stride;
    std::uint32_t acc = kFnvOffsetBasis;
    std::uint32_t prev = 0u;

    for (std::uint32_t i = 0; i < pc.iters; ++i) {
        const std::uint32_t dep = prev & pc.dep_mask;
        const std::uint32_t idx1 = start + i * pc.iter_stride + dep;
        const std::uint32_t kidx = pc.t_base + (idx1 & pc.key_mask);
        if (sc.shader == ShaderKind::Baseline) {
            const std::uint32_t mix = ds.expected_baseline[kidx];
            acc = (acc * kFnvPrime) ^ mix;
            prev = mix;
        } else if (sc.shader == ShaderKind::SingleLookup) {
            const std::uint32_t res = ds.expected_lookup[kidx];
            acc = (acc * kFnvPrime) ^ res;
            prev = res;
        } else {
            const std::uint32_t res_t = ds.expected_lookup[kidx];
            const std::uint32_t skidx = pc.s_base + (idx1 & pc.s_mask);
            const std::uint32_t res_s = ds.expected_lookup[skidx];
            acc = (acc * kFnvPrime) ^ res_t;
            acc = (acc * kFnvPrime) ^ res_s;
            prev = res_t;
        }
    }
    return acc;
}

bool verify_gpu_output(const ScenarioSpec& sc, const TableDataset& ds, const PushConstants& pc,
                       const std::uint32_t* gpu_out, std::uint32_t total_invocations) {
    const std::uint64_t total_work = static_cast<std::uint64_t>(total_invocations) * pc.iters;
    std::uint32_t mismatches = 0u;
    auto check_tid = [&](std::uint32_t tid) {
        const std::uint32_t want = eval_invocation_oracle(sc, ds, pc, tid);
        const std::uint32_t got = gpu_out[tid];
        if (got != want) {
            if (mismatches < 5u) {
                std::fprintf(stderr, "mismatch %s tid=%u: got=0x%08x want=0x%08x\n", sc.name, tid, got, want);
            }
            ++mismatches;
        }
    };

    if (total_work <= 4000000ULL) {
        for (std::uint32_t tid = 0; tid < total_invocations; ++tid) check_tid(tid);
    } else {
        const std::uint32_t edge = std::min<std::uint32_t>(512u, total_invocations);
        for (std::uint32_t tid = 0; tid < edge; ++tid) check_tid(tid);
        const std::uint32_t tail = total_invocations - edge;
        for (std::uint32_t tid = std::max(edge, tail); tid < total_invocations; ++tid) check_tid(tid);
        const std::uint32_t step = std::max<std::uint32_t>(65u, total_invocations / 4096u);
        for (std::uint32_t tid = edge; tid < tail; tid += step) check_tid(tid);
    }
    return mismatches == 0u;
}

// ---- Vulkan helpers & CLI options ----

enum class MemoryFilter { All, Device, Host, Rebar };

struct Options {
    std::vector<std::uint32_t> sizes = {1000u, 10000u, 100000u};
    std::uint32_t groups = 4096u;
    std::uint32_t iters = 64u;
    std::uint32_t reps = 25u;
    std::uint32_t warmup = 5u;
    double max_dispatch_ms = 120.0;
    double warmup_ms = 300.0;   // keep running the timed dispatch until this much time has passed (0 = off)
    std::optional<std::uint32_t> device_index;
    MemoryFilter mem_filter = MemoryFilter::All;
    bool list_devices = false;
    bool csv = false;
    bool self_test = false;
    std::string dump_spv_prefix;
};

void print_usage() {
    std::printf(
        "usage: vk_descriptor_bench [options]\n"
        "  --sizes LIST         comma-separated table entry counts (default: 1000,10000,100000)\n"
        "  --groups N           workgroups of 64 invocations per dispatch (default: 4096)\n"
        "  --iters N            lookups per invocation per dispatch (default: 64)\n"
        "  --reps N             timed repetitions per scenario (default: 25)\n"
        "  --warmup N           warmup dispatches before timing (default: 5)\n"
        "  --warmup-ms F        then keep dispatching until F ms have passed, so the GPU clocks are up before the\n"
        "                       first timed run, 0=off (default: 300; --quick turns it off)\n"
        "  --max-dispatch-ms F  auto-scale slow dispatches to avoid OS TDR, 0=off (default: 120)\n"
        "  --memory MODE        all|device|host|rebar (default: all)\n"
        "  --device N           select physical device index from --list-devices\n"
        "  --list-devices       list Vulkan physical devices and exit\n"
        "  --quick              fast check profile (sizes 1000,10000; groups 16; iters 4; reps 3)\n"
        "  --csv                emit CSV with timing_status and empty fields for unavailable measurements\n"
        "  --dump-spv PREFIX    write the 3 SPIR-V compute modules to PREFIX_*.spv and exit\n"
        "  --self-test          CPU-only checks for CLI, SPIR-V encoding and descriptor lookup\n"
        "  --help               print this message\n");
}

bool parse_u32(const char* s, std::uint32_t& out) {
    if (!s || *s < '0' || *s > '9') return false;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (*end != '\0' || v > std::numeric_limits<std::uint32_t>::max()) return false;
    out = static_cast<std::uint32_t>(v);
    return true;
}

bool parse_args(int argc, const char* const* argv, Options& opt, bool& show_help) {
    show_help = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto need_val = [&](const char*& val) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing argument for %.*s\n", int(arg.size()), arg.data());
                return false;
            }
            val = argv[++i];
            return true;
        };
        if (arg == "--help" || arg == "-h") {
            show_help = true;
            return true;
        } else if (arg == "--self-test") {
            opt.self_test = true;
        } else if (arg == "--list-devices") {
            opt.list_devices = true;
        } else if (arg == "--csv") {
            opt.csv = true;
        } else if (arg == "--quick") {
            opt.sizes = {1000u, 10000u};
            opt.groups = 16u;
            opt.iters = 4u;
            opt.warmup = 1u;
            opt.warmup_ms = 0.0;
            opt.reps = 3u;
        } else if (arg == "--groups") {
            const char* v = nullptr;
            if (!need_val(v) || !parse_u32(v, opt.groups) || opt.groups == 0u) return false;
        } else if (arg == "--iters") {
            const char* v = nullptr;
            if (!need_val(v) || !parse_u32(v, opt.iters) || opt.iters == 0u) return false;
        } else if (arg == "--reps") {
            const char* v = nullptr;
            if (!need_val(v) || !parse_u32(v, opt.reps) || opt.reps == 0u) return false;
        } else if (arg == "--warmup") {
            const char* v = nullptr;
            if (!need_val(v) || !parse_u32(v, opt.warmup)) return false;
        } else if (arg == "--max-dispatch-ms") {
            const char* v = nullptr;
            if (!need_val(v)) return false;
            char* end = nullptr;
            const double val = std::strtod(v, &end);
            if (*v == '\0' || *end != '\0' || !std::isfinite(val) || val < 0.0) return false;
            opt.max_dispatch_ms = val;
        } else if (arg == "--warmup-ms") {
            const char* v = nullptr;
            if (!need_val(v)) return false;
            char* end = nullptr;
            const double val = std::strtod(v, &end);
            if (*v == '\0' || *end != '\0' || !std::isfinite(val) || val < 0.0) return false;
            opt.warmup_ms = val;
        } else if (arg == "--device") {
            const char* v = nullptr;
            std::uint32_t idx = 0u;
            if (!need_val(v) || !parse_u32(v, idx)) return false;
            opt.device_index = idx;
        } else if (arg == "--dump-spv") {
            const char* v = nullptr;
            if (!need_val(v)) return false;
            opt.dump_spv_prefix = v;
        } else if (arg == "--memory") {
            const char* v = nullptr;
            if (!need_val(v)) return false;
            const std::string_view mv(v);
            if (mv == "all") opt.mem_filter = MemoryFilter::All;
            else if (mv == "device") opt.mem_filter = MemoryFilter::Device;
            else if (mv == "host") opt.mem_filter = MemoryFilter::Host;
            else if (mv == "rebar") opt.mem_filter = MemoryFilter::Rebar;
            else {
                std::fprintf(stderr, "invalid --memory value: %s\n", v);
                return false;
            }
        } else if (arg == "--sizes") {
            const char* v = nullptr;
            if (!need_val(v)) return false;
            opt.sizes.clear();
            std::string_view sv(v);
            while (!sv.empty()) {
                const std::size_t comma = sv.find(',');
                const std::string part(sv.substr(0, comma));
                std::uint32_t sz = 0u;
                if (!parse_u32(part.c_str(), sz) || sz < 16u) {
                    std::fprintf(stderr, "invalid size in --sizes (must be >= 16): %s\n", part.c_str());
                    return false;
                }
                opt.sizes.push_back(sz);
                if (comma == std::string_view::npos) break;
                sv.remove_prefix(comma + 1u);
            }
            if (opt.sizes.empty()) return false;
        } else {
            std::fprintf(stderr, "unknown option: %.*s\n", int(arg.size()), arg.data());
            return false;
        }
    }
    return true;
}

const char* device_type_name(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

std::string format_mem_flags(VkMemoryPropertyFlags flags) {
    std::string s;
    auto add = [&](VkMemoryPropertyFlags bit, const char* name) {
        if ((flags & bit) == bit) {
            if (!s.empty()) s += '|';
            s += name;
        }
    };
    add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL");
    add(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE");
    add(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT");
    add(VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED");
    return s.empty() ? "NONE" : s;
}

std::uint32_t pick_memory_type_exact_or_fallback(const VkPhysicalDeviceMemoryProperties& mp, std::uint32_t type_bits,
                                                 VkMemoryPropertyFlags req, VkMemoryPropertyFlags avoid) {
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & req) == req && (f & avoid) == 0u) return i;
    }
    for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(type_bits & (1u << i))) continue;
        if ((mp.memoryTypes[i].propertyFlags & req) == req) return i;
    }
    return 0xFFFFFFFFu;
}

struct GpuBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    std::uint32_t memory_type_index = 0xFFFFFFFFu;
    void* mapped = nullptr;
};

struct MemoryPlacement {
    const char* label;
    std::uint32_t memory_type_index;
    std::uint32_t heap_index;
    VkDeviceSize heap_size;
    VkMemoryPropertyFlags flags;
    bool use_staging;
};

int run_self_test() {
    const char* const device_args[] = {"vk_descriptor_bench", "--device", "4294967295"};
    Options device_options;
    bool show_help = false;
    if (device_options.device_index.has_value() || !parse_args(3, device_args, device_options, show_help) ||
        device_options.device_index != std::numeric_limits<std::uint32_t>::max()) {
        std::fprintf(stderr, "self-test: explicit device index became automatic selection\n");
        return 1;
    }

    {
        const char* const ok_args[] = {"vk_descriptor_bench", "--warmup-ms", "0"};
        const char* const bad_args[] = {"vk_descriptor_bench", "--warmup-ms", "-1"};
        const char* const quick_args[] = {"vk_descriptor_bench", "--quick"};
        Options o_ok, o_bad, o_quick, o_default;
        bool h = false;
        if (o_default.warmup_ms != 300.0 || !parse_args(3, ok_args, o_ok, h) || o_ok.warmup_ms != 0.0 ||
            parse_args(3, bad_args, o_bad, h) || !parse_args(2, quick_args, o_quick, h) || o_quick.warmup_ms != 0.0) {
            std::fprintf(stderr, "self-test: --warmup-ms parsing or its default is wrong\n");
            return 1;
        }
    }

    const struct {
        const char* name;
        ShaderKind kind;
    } modules[3] = {
        {"baseline", ShaderKind::Baseline},
        {"single", ShaderKind::SingleLookup},
        {"pair_ts", ShaderKind::PairTSLookup},
    };
    for (const auto& m : modules) {
        const auto spv = build_bench_spv(m.kind);
        if (!check_spv_structure(spv, m.name)) return 1;
    }

    TableDataset ds;
    if (!build_dataset(1000u, ds)) return 1;
    if (ds.entry_count != 1000u || ds.capacity < 2048u || ds.max_probes > kDescriptorProbes) {
        std::fprintf(stderr, "self-test: unexpected 1k table metrics\n");
        return 1;
    }
    const ScenarioSpec single_sc{"self_test", ShaderKind::SingleLookup, false, true, 0u, 0u, 1u, false,
                                 ScenarioSpec::Bank::TSharp, 1.0};
    const PushConstants pc = make_push_constants(single_sc, ds, 8u);
    if (eval_invocation_oracle(single_sc, ds, pc, 0u) == kFnvOffsetBasis) {
        std::fprintf(stderr, "self-test: degenerate checksum on lane 0\n");
        return 1;
    }
    const ScenarioSpec hot_sc{"hot", ShaderKind::SingleLookup, false, false, 0u, 0u, 0u, false,
                              ScenarioSpec::Bank::TSharp, 1.0};
    const ScenarioSpec pair_sc{"pair", ShaderKind::PairTSLookup, false, true, 0u, 0u, 1u, false,
                               ScenarioSpec::Bank::TSharp, 2.0};
    const std::uint32_t hot_u = count_unique_keys(hot_sc, ds, make_push_constants(hot_sc, ds, 64u), 4096u);
    const std::uint32_t q_u = count_unique_keys(single_sc, ds, make_push_constants(single_sc, ds, 4u), 16u);
    const std::uint32_t full_u = count_unique_keys(single_sc, ds, make_push_constants(single_sc, ds, 64u), 4096u);
    const std::uint32_t pair_u = count_unique_keys(pair_sc, ds, make_push_constants(pair_sc, ds, 64u), 4096u);
    if (hot_u != 1u || q_u != 64u || full_u != ds.t_count || pair_u != ds.entry_count) {
        std::fprintf(stderr, "self-test: unexpected unique_keys (%u, %u, %u, %u)\n", hot_u, q_u, full_u, pair_u);
        return 1;
    }
    std::printf("vk_descriptor_bench self-test passed (1k table cap=%u, load=%.1f%%, mean_probes=%.3f, max=%u)\n",
                ds.capacity, ds.load_pct, ds.mean_probes, ds.max_probes);
    return 0;
}

int run_dump_spv(const std::string& prefix) {
    const struct {
        const char* suffix;
        ShaderKind kind;
    } items[3] = {
        {"_baseline.spv", ShaderKind::Baseline},
        {"_single.spv", ShaderKind::SingleLookup},
        {"_pair_ts.spv", ShaderKind::PairTSLookup},
    };
    for (const auto& item : items) {
        const std::string path = prefix + item.suffix;
        const auto spv = build_bench_spv(item.kind);
        if (!check_spv_structure(spv, path.c_str()) || !write_spv_file(path, spv)) return 1;
        std::printf("wrote %s (%zu bytes)\n", path.c_str(), spv.size() * sizeof(std::uint32_t));
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options opt;
    bool show_help = false;
    if (!parse_args(argc, argv, opt, show_help)) {
        print_usage();
        return 2;
    }
    if (show_help) {
        print_usage();
        return 0;
    }
    if (opt.self_test) return run_self_test();
    if (!opt.dump_spv_prefix.empty()) return run_dump_spv(opt.dump_spv_prefix);

    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) return 1;

#define LI(n) const auto n = reinterpret_cast<PFN_##n>(gipa(inst, #n))
    VkInstance inst = VK_NULL_HANDLE;
    LI(vkCreateInstance);
    LI(vkEnumerateInstanceLayerProperties);
    if (!vkCreateInstance) return 1;

    const char* val_env = std::getenv("LEONIDA_VK_VALIDATION");
    const bool validation = val_env && std::strcmp(val_env, "0") != 0 && *val_env != '\0';
    const char* layer = "VK_LAYER_KHRONOS_validation";
    const char* debug_utils = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;

    VkDebugUtilsMessengerCreateInfoEXT messenger_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    messenger_info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    messenger_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    messenger_info.pfnUserCallback = on_debug_message;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vk_descriptor_bench";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (validation) {
        std::uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> layers(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
        bool found = false;
        for (const auto& lp : layers) {
            if (std::strcmp(lp.layerName, layer) == 0) found = true;
        }
        if (!found) {
            std::fprintf(stderr, "LEONIDA_VK_VALIDATION is set but %s is not installed\n", layer);
            return 1;
        }
        ici.pNext = &messenger_info;
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &debug_utils;
    }
    CHECK_VK(vkCreateInstance(&ici, nullptr, &inst));

    LI(vkDestroyInstance);
    LI(vkEnumeratePhysicalDevices);
    LI(vkGetPhysicalDeviceProperties);
    LI(vkGetPhysicalDeviceProperties2);
    LI(vkGetPhysicalDeviceQueueFamilyProperties);
    LI(vkGetPhysicalDeviceMemoryProperties);
    LI(vkCreateDevice);
    LI(vkGetDeviceProcAddr);

    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT = nullptr;
    if (validation) {
        LI(vkCreateDebugUtilsMessengerEXT);
        vkDestroyDebugUtilsMessengerEXT =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(gipa(inst, "vkDestroyDebugUtilsMessengerEXT"));
        CHECK_VK(vkCreateDebugUtilsMessengerEXT(inst, &messenger_info, nullptr, &messenger));
    }

    auto destroy_instance = [&]() {
        if (validation && messenger && vkDestroyDebugUtilsMessengerEXT) {
            vkDestroyDebugUtilsMessengerEXT(inst, messenger, nullptr);
        }
        if (inst && vkDestroyInstance) vkDestroyInstance(inst, nullptr);
    };

    std::uint32_t pd_count = 0;
    vkEnumeratePhysicalDevices(inst, &pd_count, nullptr);
    std::vector<VkPhysicalDevice> pds(pd_count);
    if (pd_count > 0) vkEnumeratePhysicalDevices(inst, &pd_count, pds.data());

    if (opt.list_devices) {
        for (std::uint32_t i = 0; i < pd_count; ++i) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(pds[i], &p);
            std::printf("[%u] %s (%s, vendor=0x%04x, device=0x%04x, api=%u.%u.%u)\n", i, p.deviceName,
                        device_type_name(p.deviceType), p.vendorID, p.deviceID, VK_API_VERSION_MAJOR(p.apiVersion),
                        VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        }
        destroy_instance();
        return 0;
    }

    auto find_compute_queue = [&](VkPhysicalDevice cand, std::uint32_t& out_qi,
                                  std::uint32_t& out_ts_bits) -> bool {
        std::uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(cand, &qn, qf.data());
        for (std::uint32_t i = 0; i < qn; ++i) {
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                out_qi = i;
                out_ts_bits = qf[i].timestampValidBits;
                return true;
            }
        }
        return false;
    };

    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties pd_props{};
    std::uint32_t qi = 0;
    std::uint32_t ts_valid_bits = 0;
    if (opt.device_index.has_value()) {
        const std::uint32_t device_index = *opt.device_index;
        if (device_index >= pd_count) {
            std::fprintf(stderr, "invalid --device index %u (found %u devices)\n", device_index, pd_count);
            destroy_instance();
            return 2;
        }
        VkPhysicalDevice cand = pds[device_index];
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(cand, &p);
        if (p.apiVersion < VK_API_VERSION_1_1 || !find_compute_queue(cand, qi, ts_valid_bits)) {
            std::printf("unsupported: selected device lacks Vulkan 1.1 or compute queue\n");
            destroy_instance();
            return kUnsupportedExit;
        }
        pd = cand;
        pd_props = p;
    } else {
        auto score = [](VkPhysicalDeviceType t, std::uint32_t vendor) -> int {
            return leonida::vk_test::device_rank(t, vendor);
        };
        int best_score = -1;
        for (VkPhysicalDevice cand : pds) {
            VkPhysicalDeviceProperties p{};
            vkGetPhysicalDeviceProperties(cand, &p);
            std::uint32_t c_qi = 0, c_ts = 0;
            if (p.apiVersion < VK_API_VERSION_1_1 || !find_compute_queue(cand, c_qi, c_ts)) continue;
            const int s = score(p.deviceType, p.vendorID);
            if (s > best_score) {
                best_score = s;
                pd = cand;
                pd_props = p;
                qi = c_qi;
                ts_valid_bits = c_ts;
            }
        }
    }

    if (!pd) {
        std::printf("unsupported: no Vulkan 1.1+ compute physical device\n");
        destroy_instance();
        return kUnsupportedExit;
    }

    if (opt.groups > pd_props.limits.maxComputeWorkGroupCount[0]) {
        std::fprintf(stderr, "--groups %u exceeds maxComputeWorkGroupCount[0] (%u)\n", opt.groups,
                     pd_props.limits.maxComputeWorkGroupCount[0]);
        destroy_instance();
        return 2;
    }

    std::uint32_t subgroup_size = 0;
    std::string driver_desc = "unknown";
    if (vkGetPhysicalDeviceProperties2) {
        VkPhysicalDeviceSubgroupProperties sg_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceVulkan12Properties vk12_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        props2.pNext = &sg_props;
        if (pd_props.apiVersion >= VK_API_VERSION_1_2) {
            sg_props.pNext = &vk12_props;
        }
        vkGetPhysicalDeviceProperties2(pd, &props2);
        subgroup_size = sg_props.subgroupSize;
        if (pd_props.apiVersion >= VK_API_VERSION_1_2 && vk12_props.driverName[0] != '\0') {
            driver_desc = vk12_props.driverName;
            if (vk12_props.driverInfo[0] != '\0') {
                driver_desc += " (";
                driver_desc += vk12_props.driverInfo;
                driver_desc += ')';
            }
        }
    }

    const float q_prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qi;
    qci.queueCount = 1;
    qci.pQueuePriorities = &q_prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;

    VkDevice dev = VK_NULL_HANDLE;
    CHECK_VK(vkCreateDevice(pd, &dci, nullptr, &dev));

#define LD(n) const auto n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(dev, #n))
    LD(vkGetDeviceQueue);
    LD(vkDeviceWaitIdle);
    LD(vkDestroyDevice);
    LD(vkCreateBuffer);
    LD(vkDestroyBuffer);
    LD(vkGetBufferMemoryRequirements);
    LD(vkAllocateMemory);
    LD(vkFreeMemory);
    LD(vkBindBufferMemory);
    LD(vkMapMemory);
    LD(vkCreateShaderModule);
    LD(vkDestroyShaderModule);
    LD(vkCreateDescriptorSetLayout);
    LD(vkDestroyDescriptorSetLayout);
    LD(vkCreatePipelineLayout);
    LD(vkDestroyPipelineLayout);
    LD(vkCreateComputePipelines);
    LD(vkDestroyPipeline);
    LD(vkCreateDescriptorPool);
    LD(vkDestroyDescriptorPool);
    LD(vkAllocateDescriptorSets);
    LD(vkUpdateDescriptorSets);
    LD(vkCreateCommandPool);
    LD(vkDestroyCommandPool);
    LD(vkAllocateCommandBuffers);
    LD(vkBeginCommandBuffer);
    LD(vkEndCommandBuffer);
    LD(vkResetCommandBuffer);
    LD(vkCmdBindPipeline);
    LD(vkCmdBindDescriptorSets);
    LD(vkCmdPushConstants);
    LD(vkCmdDispatch);
    LD(vkCmdCopyBuffer);
    LD(vkCmdPipelineBarrier);
    LD(vkCreateFence);
    LD(vkDestroyFence);
    LD(vkWaitForFences);
    LD(vkResetFences);
    LD(vkQueueSubmit);
    LD(vkCreateQueryPool);
    LD(vkDestroyQueryPool);
    LD(vkGetQueryPoolResults);
    LD(vkCmdResetQueryPool);
    LD(vkCmdWriteTimestamp);
#undef LD

    VkQueue queue = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev, qi, 0, &queue);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    const bool use_gpu_timestamps = (ts_valid_bits > 0u) && (pd_props.limits.timestampPeriod > 0.0f) &&
                                    vkCreateQueryPool && vkCmdResetQueryPool && vkCmdWriteTimestamp &&
                                    vkGetQueryPoolResults;

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer util_cmd = VK_NULL_HANDLE;
    VkCommandBuffer bench_cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool qpool = VK_NULL_HANDLE;
    std::array<VkShaderModule, 3> sms{};
    std::array<VkPipeline, 3> pipes{};

    auto destroy_buf = [&](GpuBuffer& b) {
        if (b.buffer) vkDestroyBuffer(dev, b.buffer, nullptr);
        if (b.memory) vkFreeMemory(dev, b.memory, nullptr);
        b = GpuBuffer{};
    };

    auto cleanup_device = [&]() {
        if (dev && vkDeviceWaitIdle) vkDeviceWaitIdle(dev);
        for (VkPipeline p : pipes) {
            if (p) vkDestroyPipeline(dev, p, nullptr);
        }
        for (VkShaderModule m : sms) {
            if (m) vkDestroyShaderModule(dev, m, nullptr);
        }
        if (qpool) vkDestroyQueryPool(dev, qpool, nullptr);
        if (fence) vkDestroyFence(dev, fence, nullptr);
        if (cpool) vkDestroyCommandPool(dev, cpool, nullptr);
        if (dpool) vkDestroyDescriptorPool(dev, dpool, nullptr);
        if (pl) vkDestroyPipelineLayout(dev, pl, nullptr);
        if (dsl) vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        if (dev) vkDestroyDevice(dev, nullptr);
        destroy_instance();
    };

    // Set 0 bindings: 0 = keys_buf, 2 = desc_map, 3 = out_buf
    const std::array<VkDescriptorSetLayoutBinding, 3> dsl_bindings{{
        {0u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3u, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1u, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    }};
    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = static_cast<std::uint32_t>(dsl_bindings.size());
    dsli.pBindings = dsl_bindings.data();
    if (vkCreateDescriptorSetLayout(dev, &dsli, nullptr, &dsl) != VK_SUCCESS) {
        cleanup_device();
        return 1;
    }

    const VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0u, static_cast<std::uint32_t>(sizeof(PushConstants))};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1u;
    pli.pSetLayouts = &dsl;
    pli.pushConstantRangeCount = 1u;
    pli.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(dev, &pli, nullptr, &pl) != VK_SUCCESS) {
        cleanup_device();
        return 1;
    }

    const std::array<ShaderKind, 3> kinds{ShaderKind::Baseline, ShaderKind::SingleLookup, ShaderKind::PairTSLookup};
    for (std::size_t i = 0; i < kinds.size(); ++i) {
        const auto spv = build_bench_spv(kinds[i]);
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = spv.size() * sizeof(std::uint32_t);
        smi.pCode = spv.data();
        if (vkCreateShaderModule(dev, &smi, nullptr, &sms[i]) != VK_SUCCESS) {
            cleanup_device();
            return 1;
        }
        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = sms[i];
        cpci.stage.pName = "main";
        cpci.layout = pl;
        if (vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1u, &cpci, nullptr, &pipes[i]) != VK_SUCCESS) {
            cleanup_device();
            return 1;
        }
    }

    const VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3u};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1u;
    dpi.poolSizeCount = 1u;
    dpi.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool) != VK_SUCCESS) {
        cleanup_device();
        return 1;
    }
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1u;
    dsai.pSetLayouts = &dsl;
    if (vkAllocateDescriptorSets(dev, &dsai, &dset) != VK_SUCCESS) {
        cleanup_device();
        return 1;
    }

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = qi;
    if (vkCreateCommandPool(dev, &cpci, nullptr, &cpool) != VK_SUCCESS) {
        cleanup_device();
        return 1;
    }
    VkCommandBuffer cmds[2] = {};
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2u;
    if (vkAllocateCommandBuffers(dev, &cbai, cmds) != VK_SUCCESS) {
        cleanup_device();
        return 1;
    }
    util_cmd = cmds[0];
    bench_cmd = cmds[1];

    const VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, 0u};
    if (vkCreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS) {
        cleanup_device();
        return 1;
    }
    if (use_gpu_timestamps) {
        VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpi.queryCount = 2u;
        if (vkCreateQueryPool(dev, &qpi, nullptr, &qpool) != VK_SUCCESS) {
            cleanup_device();
            return 1;
        }
    }

    auto alloc_buffer_mt = [&](VkDeviceSize bytes, VkBufferUsageFlags usage, std::uint32_t mt_idx, bool map_mem,
                               GpuBuffer& out_buf) -> bool {
        out_buf = GpuBuffer{};
        out_buf.size = bytes;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = usage;
        // Every failure says which call failed and with what result, so a run that stops is never silent.
        if (const VkResult r = vkCreateBuffer(dev, &bci, nullptr, &out_buf.buffer); r != VK_SUCCESS) {
            std::fprintf(stderr, "vkCreateBuffer (%llu bytes) failed: VkResult %d\n",
                         static_cast<unsigned long long>(bytes), static_cast<int>(r));
            return false;
        }
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(dev, out_buf.buffer, &mr);
        if (mt_idx >= mp.memoryTypeCount || !(mr.memoryTypeBits & (1u << mt_idx))) {
            std::fprintf(stderr, "memory type %u is not allowed for this buffer (memoryTypeBits 0x%x)\n", mt_idx,
                         mr.memoryTypeBits);
            destroy_buf(out_buf);
            return false;
        }
        out_buf.memory_type_index = mt_idx;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = mt_idx;
        if (const VkResult r = vkAllocateMemory(dev, &mai, nullptr, &out_buf.memory); r != VK_SUCCESS) {
            std::fprintf(stderr, "vkAllocateMemory (%llu bytes, memory type %u) failed: VkResult %d\n",
                         static_cast<unsigned long long>(mr.size), mt_idx, static_cast<int>(r));
            out_buf.memory = VK_NULL_HANDLE;
            destroy_buf(out_buf);
            return false;
        }
        if (const VkResult r = vkBindBufferMemory(dev, out_buf.buffer, out_buf.memory, 0u); r != VK_SUCCESS) {
            std::fprintf(stderr, "vkBindBufferMemory failed: VkResult %d\n", static_cast<int>(r));
            destroy_buf(out_buf);
            return false;
        }
        if (map_mem) {
            if (const VkResult r = vkMapMemory(dev, out_buf.memory, 0u, bytes, 0u, &out_buf.mapped); r != VK_SUCCESS) {
                std::fprintf(stderr, "vkMapMemory (%llu bytes) failed: VkResult %d\n",
                             static_cast<unsigned long long>(bytes), static_cast<int>(r));
                destroy_buf(out_buf);
                return false;
            }
        }
        return true;
    };

    auto alloc_typed_buffer = [&](VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags req,
                                  VkMemoryPropertyFlags avoid, bool map_mem, GpuBuffer& out_buf) -> bool {
        VkBuffer probe = VK_NULL_HANDLE;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = usage;
        if (vkCreateBuffer(dev, &bci, nullptr, &probe) != VK_SUCCESS) return false;
        VkMemoryRequirements mr{};
        vkGetBufferMemoryRequirements(dev, probe, &mr);
        vkDestroyBuffer(dev, probe, nullptr);
        const std::uint32_t mt_idx = pick_memory_type_exact_or_fallback(mp, mr.memoryTypeBits, req, avoid);
        return alloc_buffer_mt(bytes, usage, mt_idx, map_mem, out_buf);
    };

    auto submit_and_wait = [&](VkCommandBuffer cmd) -> bool {
        VkResult r = vkResetFences(dev, 1u, &fence);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "FAILED vkResetFences -> %d\n", int(r));
            return false;
        }
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1u;
        si.pCommandBuffers = &cmd;
        r = vkQueueSubmit(queue, 1u, &si, fence);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "FAILED vkQueueSubmit -> %d\n", int(r));
            return false;
        }
        r = vkWaitForFences(dev, 1u, &fence, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "FAILED vkWaitForFences -> %d\n", int(r));
            return false;
        }
        return true;
    };

    auto upload_via_staging = [&](const void* src_data, VkDeviceSize bytes, GpuBuffer& dst_buf) -> bool {
        GpuBuffer stage{};
        if (!alloc_typed_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0u, true,
                                stage)) {
            return false;
        }
        std::memcpy(stage.mapped, src_data, static_cast<std::size_t>(bytes));

        if (vkResetCommandBuffer(util_cmd, 0u) != VK_SUCCESS) {
            destroy_buf(stage);
            return false;
        }
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(util_cmd, &bi) != VK_SUCCESS) {
            destroy_buf(stage);
            return false;
        }
        const VkBufferCopy copy{0u, 0u, bytes};
        vkCmdCopyBuffer(util_cmd, stage.buffer, dst_buf.buffer, 1u, &copy);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(util_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 1u, &mb,
                             0u, nullptr, 0u, nullptr);
        if (vkEndCommandBuffer(util_cmd) != VK_SUCCESS || !submit_and_wait(util_cmd)) {
            destroy_buf(stage);
            return false;
        }
        destroy_buf(stage);
        return true;
    };

    const std::uint32_t total_invocations = opt.groups * kWorkgroupSize;
    const VkDeviceSize out_bytes = static_cast<VkDeviceSize>(total_invocations) * sizeof(std::uint32_t);

    // Output buffer (device local when available) + host readback staging buffer
    GpuBuffer out_buf{};
    GpuBuffer readback_buf{};
    if (!alloc_typed_buffer(out_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, false,
                            out_buf) ||
        !alloc_typed_buffer(out_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0u, true,
                            readback_buf)) {
        destroy_buf(out_buf);
        destroy_buf(readback_buf);
        cleanup_device();
        return 1;
    }

    auto cleanup_all = [&](GpuBuffer& a, GpuBuffer& b) {
        destroy_buf(a);
        destroy_buf(b);
        destroy_buf(out_buf);
        destroy_buf(readback_buf);
        cleanup_device();
    };

    const double total_single_lookups = static_cast<double>(total_invocations) * static_cast<double>(opt.iters);
    const std::uint64_t ts_mask =
        (ts_valid_bits == 64u) ? ~0ULL : (ts_valid_bits == 0u ? 0ULL : ((1ULL << ts_valid_bits) - 1ULL));

    if (!opt.csv) {
        std::printf("device: %s (%s, vendor=0x%04x, device=0x%04x, subgroup=%u, driver=%s)\n", pd_props.deviceName,
                    device_type_name(pd_props.deviceType), pd_props.vendorID, pd_props.deviceID, subgroup_size,
                    driver_desc.c_str());
        if (use_gpu_timestamps) {
            std::printf("timing: Vulkan GPU timestamps (valid_bits=%u, period=%.3f ns); validation=%s\n", ts_valid_bits,
                        pd_props.limits.timestampPeriod, validation ? "on" : "off");
        } else {
            std::printf("timing: host wall-clock fallback; validation=%s\n", validation ? "on" : "off");
        }
        std::printf("dispatch: %u workgroups x %u threads = %u invocations x %u iters = %.2fM single / %.2fM pair lookups "
                    "(warmup=%u, reps=%u)\n",
                    opt.groups, kWorkgroupSize, total_invocations, opt.iters, total_single_lookups / 1e6,
                    (total_single_lookups * 2.0) / 1e6, opt.warmup, opt.reps);
    } else {
        std::printf("# device,%s,%s\n", pd_props.deviceName, driver_desc.c_str());
        std::printf("entries,capacity,load_pct,size_bytes,mean_probes,max_probes,placement,mem_type,flags,scenario,"
                    "groups,iters,reps,unique_keys,med_ms,min_ms,max_ms,us_per_1m_lk,net_us_per_1m_lk,glk_per_s,timing_status\n");
    }

    const std::array<ScenarioSpec, 11> scenarios{{
        {"baseline_uniform", ShaderKind::Baseline, false, true, 0u, 0u, 1u, false, ScenarioSpec::Bank::TSharp, 0.0},
        {"baseline_divergent", ShaderKind::Baseline, true, false, 64u, 1u, 1u, false, ScenarioSpec::Bank::TSharp, 0.0},
        {"uniform_hot_tsharp", ShaderKind::SingleLookup, false, false, 0u, 0u, 0u, false, ScenarioSpec::Bank::TSharp,
         1.0},
        {"uniform_tsharp_hit", ShaderKind::SingleLookup, false, true, 0u, 0u, 1u, false, ScenarioSpec::Bank::TSharp,
         1.0},
        {"uniform_ssharp_hit", ShaderKind::SingleLookup, false, true, 0u, 0u, 1u, false, ScenarioSpec::Bank::SSharp,
         1.0},
        {"uniform_pair_ts", ShaderKind::PairTSLookup, false, true, 0u, 0u, 1u, false, ScenarioSpec::Bank::TSharp, 2.0},
        {"uniform_miss", ShaderKind::SingleLookup, false, true, 0u, 0u, 1u, false, ScenarioSpec::Bank::Miss, 1.0},
        {"divergent_tsharp_hit", ShaderKind::SingleLookup, true, false, 64u, 1u, 1u, false, ScenarioSpec::Bank::TSharp,
         1.0},
        {"divergent_pair_ts", ShaderKind::PairTSLookup, true, false, 64u, 1u, 1u, false, ScenarioSpec::Bank::TSharp,
         2.0},
        {"divergent_miss", ShaderKind::SingleLookup, true, false, 64u, 1u, 1u, false, ScenarioSpec::Bank::Miss, 1.0},
        {"latency_divergent", ShaderKind::SingleLookup, true, false, 64u, 1u, 1u, true, ScenarioSpec::Bank::TSharp,
         1.0},
    }};

    bool seen_rebar = false;
    for (const std::uint32_t entry_count : opt.sizes) {
        TableDataset ds;
        if (!build_dataset(entry_count, ds)) {
            GpuBuffer unused_a{}, unused_b{};
            cleanup_all(unused_a, unused_b);
            return 1;
        }
        const VkDeviceSize keys_bytes = static_cast<VkDeviceSize>(ds.keys_buf.size()) * sizeof(std::uint32_t);
        const VkDeviceSize map_bytes = static_cast<VkDeviceSize>(ds.gpu_table.size()) * sizeof(std::uint32_t);

        if (map_bytes > pd_props.limits.maxStorageBufferRange || keys_bytes > pd_props.limits.maxStorageBufferRange) {
            std::printf("unsupported: table (%llu bytes) exceeds maxStorageBufferRange (%u)\n",
                        static_cast<unsigned long long>(map_bytes), pd_props.limits.maxStorageBufferRange);
            GpuBuffer unused_a{}, unused_b{};
            cleanup_all(unused_a, unused_b);
            return kUnsupportedExit;
        }

        // Upload keys_buf once per size into DEVICE_LOCAL (fallback HOST_VISIBLE).
        GpuBuffer keys_buf{};
        GpuBuffer map_buf{};
        if (!alloc_typed_buffer(keys_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, false,
                                keys_buf) ||
            !upload_via_staging(ds.keys_buf.data(), keys_bytes, keys_buf)) {
            cleanup_all(keys_buf, map_buf);
            return 1;
        }

        // Query desc_map memoryRequirements to discover valid placements.
        std::uint32_t map_type_bits = 0;
        {
            VkBuffer probe = VK_NULL_HANDLE;
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = map_bytes;
            bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vkCreateBuffer(dev, &bci, nullptr, &probe) != VK_SUCCESS) {
                cleanup_all(keys_buf, map_buf);
                return 1;
            }
            VkMemoryRequirements mr{};
            vkGetBufferMemoryRequirements(dev, probe, &mr);
            vkDestroyBuffer(dev, probe, nullptr);
            map_type_bits = mr.memoryTypeBits;
        }

        std::vector<MemoryPlacement> placements;
        auto add_placement = [&](const char* label, std::uint32_t mt_idx, bool use_staging) {
            if (mt_idx == 0xFFFFFFFFu) return;
            for (const auto& existing : placements) {
                if (existing.memory_type_index == mt_idx) {
                    if (!opt.csv) {
                        std::printf("  note: skipping placement %s (same memoryTypeIndex %u as %s)\n", label, mt_idx,
                                    existing.label);
                    }
                    return;
                }
            }
            const auto& mt = mp.memoryTypes[mt_idx];
            placements.push_back(
                {label, mt_idx, mt.heapIndex, mp.memoryHeaps[mt.heapIndex].size, mt.propertyFlags, use_staging});
        };

        if (opt.mem_filter == MemoryFilter::All || opt.mem_filter == MemoryFilter::Device) {
            const std::uint32_t dev_mt = pick_memory_type_exact_or_fallback(
                mp, map_type_bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            add_placement("device_local", dev_mt, true);
        }
        if (opt.mem_filter == MemoryFilter::All || opt.mem_filter == MemoryFilter::Host) {
            // Exact BindlessSet::create() search order: first HOST_VISIBLE | HOST_COHERENT memory type.
            const std::uint32_t bindless_mt = pick_memory_type_exact_or_fallback(
                mp, map_type_bits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0u);
            add_placement("host_visible", bindless_mt, false);
        }
        if (opt.mem_filter == MemoryFilter::All || opt.mem_filter == MemoryFilter::Rebar) {
            std::uint32_t rebar_mt = 0xFFFFFFFFu;
            const VkMemoryPropertyFlags want =
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            for (std::uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
                if (!(map_type_bits & (1u << i))) continue;
                if ((mp.memoryTypes[i].propertyFlags & want) != want) continue;
                if (mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size >= map_bytes * 2u) {
                    rebar_mt = i;
                    break;
                }
            }
            if (rebar_mt != 0xFFFFFFFFu) {
                seen_rebar = true;
                add_placement("host_device_local", rebar_mt, false);
            }
        }

        if (!opt.csv) {
            std::printf(
                "\n=== entries=%u (T#=%u, S#=%u) | cap=%u | load=%.1f%% | size=%.2f MiB (%.1f KiB) | "
                "probes mean=%.3f max=%u p1=%.1f%% ===\n",
                ds.entry_count, ds.t_count, ds.s_count, ds.capacity, ds.load_pct,
                static_cast<double>(map_bytes) / (1024.0 * 1024.0), static_cast<double>(map_bytes) / 1024.0,
                ds.mean_probes, ds.max_probes, ds.probe1_pct);
        }

        for (const auto& pl_cfg : placements) {
            destroy_buf(map_buf);
            const VkBufferUsageFlags map_usage =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (pl_cfg.use_staging ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0u);
            if (!alloc_buffer_mt(map_bytes, map_usage, pl_cfg.memory_type_index, !pl_cfg.use_staging, map_buf)) {
                // The ReBAR placement is optional: a driver can list a DEVICE_LOCAL | HOST_VISIBLE type yet refuse
                // the allocation (RTX 4080 SUPER, Windows 617.14: VK_ERROR_OUT_OF_DEVICE_MEMORY for 160 KiB).
                // Skip it, as when no such type exists, so the other placements and table sizes still run.
                if (std::strcmp(pl_cfg.label, "host_device_local") == 0) {
                    if (!opt.csv)
                        std::printf("  note: skipping placement host_device_local (memory type %u can't be "
                                    "allocated, see stderr)\n", pl_cfg.memory_type_index);
                    continue;
                }
                std::fprintf(stderr, "placement %s: allocating the %llu-byte table failed\n", pl_cfg.label,
                             static_cast<unsigned long long>(map_bytes));
                cleanup_all(keys_buf, map_buf);
                return 1;
            }
            if (pl_cfg.use_staging) {
                if (!upload_via_staging(ds.gpu_table.data(), map_bytes, map_buf)) {
                    cleanup_all(keys_buf, map_buf);
                    return 1;
                }
            } else {
                std::memcpy(map_buf.mapped, ds.gpu_table.data(), static_cast<std::size_t>(map_bytes));
            }

            const std::array<VkDescriptorBufferInfo, 3> buf_infos{{
                {keys_buf.buffer, 0u, keys_bytes},
                {map_buf.buffer, 0u, map_bytes},
                {out_buf.buffer, 0u, out_bytes},
            }};
            const std::array<VkWriteDescriptorSet, 3> writes{{
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 0u, 0u, 1u,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf_infos[0], nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 2u, 0u, 1u,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf_infos[1], nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 3u, 0u, 1u,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buf_infos[2], nullptr},
            }};
            vkUpdateDescriptorSets(dev, static_cast<std::uint32_t>(writes.size()), writes.data(), 0u, nullptr);

            if (!opt.csv) {
                std::printf("-- placement: %s (type=%u, heap=%u %.0f MiB, flags=%s) --\n", pl_cfg.label,
                            pl_cfg.memory_type_index, pl_cfg.heap_index,
                            static_cast<double>(pl_cfg.heap_size) / (1024.0 * 1024.0),
                            format_mem_flags(pl_cfg.flags).c_str());
                std::printf("  %-22s %7s %9s %9s %9s %11s %12s %9s\n", "scenario", "keys", "med_ms", "min_ms",
                            "max_ms", "us/1M_lk", "net_us/1M", "Glk/s");
            }

            double base_uniform_us_per_1m = 0.0;
            double base_divergent_us_per_1m = 0.0;

            for (const auto& sc : scenarios) {
                const VkPipeline pipe = pipes[static_cast<std::size_t>(sc.shader)];

                std::uint32_t eff_groups = opt.groups;
                std::uint32_t eff_iters = opt.iters;
                std::uint32_t eff_reps = opt.reps;
                std::uint32_t eff_warmup = opt.warmup;

                // Probe slow scenarios on a small grid so heavy PCIe/sysram lookups stay well below Windows TDR.
                if (opt.max_dispatch_ms > 0.0 && opt.groups > 64u) {
                    const std::uint32_t probe_groups = 64u;
                    const std::uint32_t probe_iters = std::min(opt.iters, 4u);
                    const PushConstants probe_pc = make_push_constants(sc, ds, probe_iters);
                    if (vkResetCommandBuffer(util_cmd, 0u) != VK_SUCCESS) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    if (vkBeginCommandBuffer(util_cmd, &bi) != VK_SUCCESS) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    if (use_gpu_timestamps) vkCmdResetQueryPool(util_cmd, qpool, 0u, 2u);
                    vkCmdBindPipeline(util_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                    vkCmdBindDescriptorSets(util_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0u, 1u, &dset, 0u, nullptr);
                    vkCmdPushConstants(util_cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                                       static_cast<std::uint32_t>(sizeof(probe_pc)), &probe_pc);
                    if (use_gpu_timestamps) vkCmdWriteTimestamp(util_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0u);
                    vkCmdDispatch(util_cmd, probe_groups, 1u, 1u);
                    if (use_gpu_timestamps) {
                        vkCmdWriteTimestamp(util_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1u);
                    }
                    if (vkEndCommandBuffer(util_cmd) != VK_SUCCESS) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    const auto pt0 = std::chrono::steady_clock::now();
                    if (!submit_and_wait(util_cmd)) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    const auto pt1 = std::chrono::steady_clock::now();
                    double probe_ms = 0.0;
                    if (use_gpu_timestamps) {
                        std::uint64_t ts[2] = {};
                        if (vkGetQueryPoolResults(dev, qpool, 0u, 2u, sizeof(ts), ts, sizeof(std::uint64_t),
                                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
                            cleanup_all(keys_buf, map_buf);
                            return 1;
                        }
                        const std::uint64_t ticks = (ts[1] - ts[0]) & ts_mask;
                        probe_ms =
                            static_cast<double>(ticks) * static_cast<double>(pd_props.limits.timestampPeriod) * 1e-6;
                    } else {
                        probe_ms = std::chrono::duration<double, std::milli>(pt1 - pt0).count();
                    }

                    const double est_full_ms =
                        std::max(0.0, probe_ms - 0.005) *
                        (static_cast<double>(opt.groups) / static_cast<double>(probe_groups)) *
                        (static_cast<double>(opt.iters) / static_cast<double>(probe_iters));
                    if (est_full_ms > opt.max_dispatch_ms) {
                        const double target_ms = opt.max_dispatch_ms * 0.5;
                        const double work_scale = target_ms / est_full_ms;
                        const std::uint32_t min_iters = std::min(opt.iters, 4u);
                        const double max_iter_reduction =
                            static_cast<double>(opt.iters) / static_cast<double>(min_iters);
                        if (work_scale * max_iter_reduction >= 1.0) {
                            eff_iters = std::max(
                                min_iters, static_cast<std::uint32_t>(std::floor(static_cast<double>(opt.iters) * work_scale)));
                        } else {
                            eff_iters = min_iters;
                            const double rem_scale = work_scale * max_iter_reduction;
                            eff_groups = std::max<std::uint32_t>(
                                64u, static_cast<std::uint32_t>(std::floor(static_cast<double>(opt.groups) * rem_scale)));
                        }
                        eff_reps = std::min(opt.reps, 9u);
                        eff_warmup = std::min(opt.warmup, 2u);
                    }
                }

                const PushConstants pc = make_push_constants(sc, ds, eff_iters);
                const std::uint32_t eff_invocations = eff_groups * kWorkgroupSize;
                const VkDeviceSize eff_out_bytes = static_cast<VkDeviceSize>(eff_invocations) * sizeof(std::uint32_t);
                const double eff_single_lookups = static_cast<double>(eff_invocations) * static_cast<double>(eff_iters);
                const bool was_scaled = (eff_groups != opt.groups || eff_iters != opt.iters);
                const std::uint32_t unique_keys = count_unique_keys(sc, ds, pc, eff_groups);

                // Warmup + correctness verification with readback
                for (std::uint32_t w = 0; w <= eff_warmup; ++w) {
                    const bool readback = (w == 0u);
                    if (vkResetCommandBuffer(util_cmd, 0u) != VK_SUCCESS) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    if (vkBeginCommandBuffer(util_cmd, &bi) != VK_SUCCESS) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    vkCmdBindPipeline(util_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                    vkCmdBindDescriptorSets(util_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0u, 1u, &dset, 0u, nullptr);
                    vkCmdPushConstants(util_cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                                       static_cast<std::uint32_t>(sizeof(pc)), &pc);
                    vkCmdDispatch(util_cmd, eff_groups, 1u, 1u);
                    if (readback) {
                        VkMemoryBarrier b1{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        b1.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        b1.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        vkCmdPipelineBarrier(util_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0u, 1u, &b1, 0u, nullptr, 0u, nullptr);
                        const VkBufferCopy copy{0u, 0u, eff_out_bytes};
                        vkCmdCopyBuffer(util_cmd, out_buf.buffer, readback_buf.buffer, 1u, &copy);
                        VkMemoryBarrier b2{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        b2.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                        vkCmdPipelineBarrier(util_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0u,
                                             1u, &b2, 0u, nullptr, 0u, nullptr);
                    }
                    if (vkEndCommandBuffer(util_cmd) != VK_SUCCESS || !submit_and_wait(util_cmd)) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    if (readback) {
                        const auto* gpu_out = static_cast<const std::uint32_t*>(readback_buf.mapped);
                        if (!verify_gpu_output(sc, ds, pc, gpu_out, eff_invocations)) {
                            cleanup_all(keys_buf, map_buf);
                            return 1;
                        }
                    }
                }

                // Record reusable timed command buffer (dispatch + timestamps only)
                if (vkResetCommandBuffer(bench_cmd, 0u) != VK_SUCCESS) {
                    cleanup_all(keys_buf, map_buf);
                    return 1;
                }
                {
                    const VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, 0u, nullptr};
                    if (vkBeginCommandBuffer(bench_cmd, &bi) != VK_SUCCESS) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    if (use_gpu_timestamps) vkCmdResetQueryPool(bench_cmd, qpool, 0u, 2u);
                    vkCmdBindPipeline(bench_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                    vkCmdBindDescriptorSets(bench_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0u, 1u, &dset, 0u, nullptr);
                    vkCmdPushConstants(bench_cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                                       static_cast<std::uint32_t>(sizeof(pc)), &pc);
                    if (use_gpu_timestamps) vkCmdWriteTimestamp(bench_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0u);
                    vkCmdDispatch(bench_cmd, eff_groups, 1u, 1u);
                    if (use_gpu_timestamps) {
                        vkCmdWriteTimestamp(bench_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1u);
                    }
                    if (vkEndCommandBuffer(bench_cmd) != VK_SUCCESS) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                }

                // A handful of dispatches is not enough to bring the clocks up on every GPU: with --warmup 5 an idle
                // RX 7600 measured the first placement ~1.8x slower than with --warmup 200, and the RTX 4050 ~1.3x
                // (TEST-277). Keep running the command that is about to be timed for --warmup-ms.
                if (opt.warmup_ms > 0.0) {
                    const auto w0 = std::chrono::steady_clock::now();
                    for (;;) {
                        if (!submit_and_wait(bench_cmd)) {
                            cleanup_all(keys_buf, map_buf);
                            return 1;
                        }
                        const double spent =
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
                        if (spent >= opt.warmup_ms) break;
                    }
                }

                std::vector<double> samples_ms;
                samples_ms.reserve(eff_reps);
                for (std::uint32_t r = 0; r < eff_reps; ++r) {
                    const auto t0 = std::chrono::steady_clock::now();
                    if (!submit_and_wait(bench_cmd)) {
                        cleanup_all(keys_buf, map_buf);
                        return 1;
                    }
                    const auto t1 = std::chrono::steady_clock::now();
                    if (use_gpu_timestamps) {
                        std::uint64_t ts[2] = {};
                        if (vkGetQueryPoolResults(dev, qpool, 0u, 2u, sizeof(ts), ts, sizeof(std::uint64_t),
                                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
                            cleanup_all(keys_buf, map_buf);
                            return 1;
                        }
                        const std::uint64_t ticks = (ts[1] - ts[0]) & ts_mask;
                        const double ms =
                            static_cast<double>(ticks) * static_cast<double>(pd_props.limits.timestampPeriod) * 1e-6;
                        samples_ms.push_back(ms);
                    } else {
                        samples_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                    }
                }

                std::sort(samples_ms.begin(), samples_ms.end());
                const double med_ms = samples_ms[samples_ms.size() / 2u];
                const double min_ms = samples_ms.front();
                const double max_ms = samples_ms.back();

                const bool is_lookup = sc.lookups_per_iter > 0.0;
                double& baseline_us = sc.divergent ? base_divergent_us_per_1m : base_uniform_us_per_1m;
                if (!is_lookup) baseline_us = med_ms * 1e9 / eff_single_lookups;

                const bool have_times = max_ms > 0.0;
                const bool have_rate = is_lookup && med_ms > 0.0;
                const double total_lookups = eff_single_lookups * sc.lookups_per_iter;
                const double us_per_1m = have_rate ? med_ms * 1e9 / total_lookups : 0.0;
                const double net_us_per_1m = have_rate ? std::max(0.0, us_per_1m - baseline_us / sc.lookups_per_iter) : 0.0;
                const double glk_s = med_ms > 1e-9 ? total_lookups / (med_ms * 1e6) : 0.0;

                const char* status = nullptr;
                if (!have_times) status = use_gpu_timestamps ? "GPU timestamps did not advance" : "host clock did not advance";
                else if (med_ms == 0.0) status = "median below timer resolution";
                else if (is_lookup && baseline_us == 0.0) status = "baseline timing unavailable";

                if (opt.csv) {
                    std::printf("%u,%u,%.2f,%llu,%.4f,%u,%s,%u,\"%s\",%s,%u,%u,%u,%u", ds.entry_count, ds.capacity,
                                ds.load_pct, static_cast<unsigned long long>(map_bytes), ds.mean_probes, ds.max_probes,
                                pl_cfg.label, pl_cfg.memory_type_index, format_mem_flags(pl_cfg.flags).c_str(), sc.name,
                                eff_groups, eff_iters, eff_reps, unique_keys);
                } else {
                    std::printf("  %-22s %7u", sc.name, unique_keys);
                }
                const auto metric = [&](double value, bool available, int width, int precision, int csv_precision) {
                    if (opt.csv) {
                        std::putchar(',');
                        if (available) std::printf("%.*f", csv_precision, value);
                    } else if (available) {
                        std::printf(" %*.*f", width, precision, value);
                    } else {
                        std::printf(" %*s", width, "-");
                    }
                };
                metric(med_ms, have_times, 9, 4, 6);
                metric(min_ms, have_times, 9, 4, 6);
                metric(max_ms, have_times, 9, 4, 6);
                metric(us_per_1m, have_rate, 11, 3, 4);
                metric(net_us_per_1m, have_rate && baseline_us > 0.0, 12, 3, 4);
                metric(glk_s, have_rate, 9, 2, 4);
                if (opt.csv) {
                    std::printf(",%s", status ? status : "ok");
                } else {
                    if (was_scaled) std::printf(" [scaled %ux%u, n=%u]", eff_groups, eff_iters, eff_reps);
                    if (status) std::printf(" [%s]", status);
                }
                std::putchar('\n');
                std::fflush(stdout);
            }
        }
        destroy_buf(keys_buf);
        destroy_buf(map_buf);
    }

    if (opt.mem_filter == MemoryFilter::Rebar && !seen_rebar) {
        if (!opt.csv) {
            std::printf("unsupported: no HOST_VISIBLE | DEVICE_LOCAL (ReBAR) memory type large enough for table\n");
        }
        destroy_buf(out_buf);
        destroy_buf(readback_buf);
        cleanup_device();
        return kUnsupportedExit;
    }

    destroy_buf(out_buf);
    destroy_buf(readback_buf);
    cleanup_device();

    if (validation && g_validation_errors > 0u) {
        std::fprintf(stderr, "%u Vulkan validation errors\n", g_validation_errors);
        return 1;
    }
    return 0;
}
