#pragma once

// Per-pipeline `--f16-interp-fallback` half placement and SPIR-V specialization (SHADER-378).
//
// Under `--f16-interp-fallback` (MoltenVK / devices without PerVertexKHR), a pixel shader reads decoded f16 halves
// as ordinary smooth vec4 inputs recorded in `index.json` under `f16_halves: {param: [lo_loc, hi_loc]}`. When a
// producer (VS) exports many params (up to 16) or a PS reads high attributes alongside f16 attributes, no fixed
// static location map fits all pairs within MoltenVK's 30 locations (0..29, 124 output components counting
// gl_Position). At graphics PSO creation, once both VS and PS are known, `link_f16_halves`:
//   1. Fast-paths unchanged when every PS `f16_halves` entry is already exported at the exact same locations by VS.
//   2. Chooses mutually non-colliding `[lo_loc, hi_loc]` slots in `0..top_location` (keeping PS locations unchanged
//      whenever they do not collide with any raw VS output; otherwise relocating PS `OpDecorate Location` in place).
//   3. Specializes the VS SPIR-V binary for the pair: reuses matching static `f16_halves` outputs (updating their
//      `Location` decorations), demotes unneeded static `f16_halves` outputs to `Private` (freeing their locations),
//      and synthesizes `UnpackHalf2x16` half exports right before every `OpReturn` in the Vertex entry function for
//      any param the VS did not emit statically.
//
// Pure C++20 stdlib, no Vulkan or Windows headers required.

#include "shader_library.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace leonida {

namespace spirv_detail {

constexpr std::uint32_t kSpirvMagic = 0x07230203u;
constexpr std::uint32_t kSpirvVersion14 = 0x00010400u;

constexpr std::uint16_t kOpSourceContinued = 2;
constexpr std::uint16_t kOpSource = 3;
constexpr std::uint16_t kOpSourceExtension = 4;
constexpr std::uint16_t kOpName = 5;
constexpr std::uint16_t kOpMemberName = 6;
constexpr std::uint16_t kOpString = 7;
constexpr std::uint16_t kOpExtension = 10;
constexpr std::uint16_t kOpExtInstImport = 11;
constexpr std::uint16_t kOpExtInst = 12;
constexpr std::uint16_t kOpMemoryModel = 14;
constexpr std::uint16_t kOpEntryPoint = 15;
constexpr std::uint16_t kOpExecutionMode = 16;
constexpr std::uint16_t kOpCapability = 17;
constexpr std::uint16_t kOpTypeInt = 21;
constexpr std::uint16_t kOpTypeFloat = 22;
constexpr std::uint16_t kOpTypeVector = 23;
constexpr std::uint16_t kOpTypePointer = 32;
constexpr std::uint16_t kOpConstantNull = 46;
constexpr std::uint16_t kOpFunction = 54;
constexpr std::uint16_t kOpFunctionEnd = 56;
constexpr std::uint16_t kOpVariable = 59;
constexpr std::uint16_t kOpLoad = 61;
constexpr std::uint16_t kOpStore = 62;
constexpr std::uint16_t kOpAccessChain = 65;
constexpr std::uint16_t kOpInBoundsAccessChain = 66;
constexpr std::uint16_t kOpDecorate = 71;
constexpr std::uint16_t kOpMemberDecorate = 72;
constexpr std::uint16_t kOpDecorationGroup = 73;
constexpr std::uint16_t kOpGroupDecorate = 74;
constexpr std::uint16_t kOpGroupMemberDecorate = 75;
constexpr std::uint16_t kOpCompositeConstruct = 80;
constexpr std::uint16_t kOpCompositeExtract = 81;
constexpr std::uint16_t kOpBitcast = 124;
constexpr std::uint16_t kOpReturn = 253;
constexpr std::uint16_t kOpModuleProcessed = 330;
constexpr std::uint16_t kOpExecutionModeId = 331;
constexpr std::uint16_t kOpDecorateId = 332;
constexpr std::uint16_t kOpDecorateString = 5632;
constexpr std::uint16_t kOpMemberDecorateString = 5633;

constexpr std::uint32_t kExecModelVertex = 0;
constexpr std::uint32_t kExecModelFragment = 4;

constexpr std::uint32_t kStorageInput = 1;
constexpr std::uint32_t kStorageOutput = 3;
constexpr std::uint32_t kStoragePrivate = 6;

constexpr std::uint32_t kDecorationLocation = 30;
constexpr std::uint32_t kGlslUnpackHalf2x16 = 62;

struct Inst {
    std::uint16_t opcode = 0;
    std::vector<std::uint32_t> words;
};

inline Inst make_inst(std::uint16_t opcode, std::initializer_list<std::uint32_t> operands) {
    Inst inst;
    inst.opcode = opcode;
    inst.words.reserve(1 + operands.size());
    inst.words.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
    for (std::uint32_t w : operands) inst.words.push_back(w);
    return inst;
}

inline Inst make_inst_vec(std::uint16_t opcode, const std::vector<std::uint32_t>& operands) {
    Inst inst;
    inst.opcode = opcode;
    inst.words.reserve(1 + operands.size());
    inst.words.push_back((std::uint32_t(1 + operands.size()) << 16) | opcode);
    inst.words.insert(inst.words.end(), operands.begin(), operands.end());
    return inst;
}

struct Module {
    std::array<std::uint32_t, 5> header{};
    std::vector<Inst> insts;

    bool parse(std::span<const std::uint32_t> words) {
        if (words.size() < 5 || words[0] != kSpirvMagic || words[3] == 0) return false;
        for (std::size_t i = 0; i < 5; ++i) header[i] = words[i];
        insts.clear();
        std::size_t pos = 5;
        while (pos < words.size()) {
            const std::uint32_t head = words[pos];
            const std::uint32_t wc = head >> 16;
            const std::uint16_t op = std::uint16_t(head & 0xffffu);
            if (wc == 0 || pos + wc > words.size()) return false;
            Inst inst;
            inst.opcode = op;
            inst.words.assign(words.begin() + std::ptrdiff_t(pos), words.begin() + std::ptrdiff_t(pos + wc));
            insts.push_back(std::move(inst));
            pos += wc;
        }
        return true;
    }

    std::vector<std::uint32_t> serialize() const {
        std::size_t total = 5;
        for (const Inst& inst : insts) total += inst.words.size();
        std::vector<std::uint32_t> out;
        out.reserve(total);
        out.insert(out.end(), header.begin(), header.end());
        for (const Inst& inst : insts) out.insert(out.end(), inst.words.begin(), inst.words.end());
        return out;
    }

    std::uint32_t alloc_id() { return header[3]++; }
};

inline bool is_pre_type_opcode(std::uint16_t op) {
    switch (op) {
    case kOpCapability:
    case kOpExtension:
    case kOpExtInstImport:
    case kOpMemoryModel:
    case kOpEntryPoint:
    case kOpExecutionMode:
    case kOpExecutionModeId:
    case kOpString:
    case kOpSourceContinued:
    case kOpSource:
    case kOpSourceExtension:
    case kOpName:
    case kOpMemberName:
    case kOpModuleProcessed:
    case kOpDecorate:
    case kOpMemberDecorate:
    case kOpDecorationGroup:
    case kOpGroupDecorate:
    case kOpGroupMemberDecorate:
    case kOpDecorateId:
    case kOpDecorateString:
    case kOpMemberDecorateString:
        return true;
    default:
        return false;
    }
}

inline std::string decode_literal_string(std::span<const std::uint32_t> words) {
    std::string s;
    for (std::uint32_t w : words) {
        for (int b = 0; b < 4; ++b) {
            const char c = char((w >> (8 * b)) & 0xffu);
            if (c == '\0') return s;
            s.push_back(c);
        }
    }
    return s;
}

inline std::size_t entry_point_interface_start(const Inst& ep) {
    // OpEntryPoint <ExecModel> <EntryPointId> <Name...> <InterfaceIds...>
    for (std::size_t i = 3; i < ep.words.size(); ++i) {
        const std::uint32_t w = ep.words[i];
        if ((w & 0x000000ffu) == 0 || (w & 0x0000ff00u) == 0 || (w & 0x00ff0000u) == 0 || (w & 0xff000000u) == 0)
            return i + 1;
    }
    return ep.words.size();
}

// Collects global OpVariables of `storage_class` that carry `OpDecorate <id> Location <loc>`.
// Returns map: location -> variable_id (first variable found at that location).
inline std::map<std::uint32_t, std::uint32_t> collect_located_vars(const Module& mod, std::uint32_t storage_class) {
    std::map<std::uint32_t, std::uint32_t> var_storage;
    for (const Inst& inst : mod.insts) {
        if (inst.opcode == kOpFunction) break;
        if (inst.opcode == kOpVariable && inst.words.size() >= 4) var_storage[inst.words[2]] = inst.words[3];
    }
    std::map<std::uint32_t, std::uint32_t> by_loc;
    for (const Inst& inst : mod.insts) {
        if (inst.opcode == kOpDecorate && inst.words.size() >= 4 && inst.words[2] == kDecorationLocation) {
            const std::uint32_t target = inst.words[1];
            const auto it = var_storage.find(target);
            if (it != var_storage.end() && it->second == storage_class) by_loc.emplace(inst.words[3], target);
        }
    }
    return by_loc;
}

}  // namespace spirv_detail

// Specializes/patches `vs_words` and (if needed) `ps_words` so that every entry in `ps_entry.f16_halves` is
// exported by the VS at the exact smooth location the PS reads, within `0..top_location` (default 29, MoltenVK's
// 124-component vertex output limit counting gl_Position).
//
// Returns true on success (including the no-op fast path when no patching is needed). On failure returns false and
// writes a concise reason to `*error` if non-null; the caller's word vectors are only replaced on success.
inline bool link_f16_halves(std::vector<std::uint32_t>& vs_words, const ShaderEntry& vs_entry,
                            std::vector<std::uint32_t>& ps_words, const ShaderEntry& ps_entry,
                            std::string* error = nullptr, std::uint32_t top_location = 29) {
    const auto fail = [&](const std::string& msg) {
        if (error) *error = msg;
        return false;
    };
    if (ps_entry.f16_halves.empty()) return true;

    // Fast path: every PS half pair is already exported at the exact same locations by VS.
    bool already_linked = true;
    for (const auto& [param, ps_pair] : ps_entry.f16_halves) {
        const auto it = vs_entry.f16_halves.find(param);
        if (it == vs_entry.f16_halves.end() || it->second != ps_pair) {
            already_linked = false;
            break;
        }
    }
    if (already_linked) return true;

    using namespace spirv_detail;
    Module vs_mod, ps_mod;
    if (!vs_mod.parse(vs_words)) return fail("f16_halves: malformed VS SPIR-V");
    if (!ps_mod.parse(ps_words)) return fail("f16_halves: malformed PS SPIR-V");

    const auto vs_outputs = collect_located_vars(vs_mod, kStorageOutput);
    const auto ps_inputs = collect_located_vars(ps_mod, kStorageInput);

    std::set<std::uint32_t> vs_static_half_locs;
    for (const auto& [param, pair] : vs_entry.f16_halves) {
        vs_static_half_locs.insert(pair[0]);
        vs_static_half_locs.insert(pair[1]);
    }
    std::set<std::uint32_t> ps_static_half_locs;
    for (const auto& [param, pair] : ps_entry.f16_halves) {
        ps_static_half_locs.insert(pair[0]);
        ps_static_half_locs.insert(pair[1]);
    }

    // Raw outputs of VS (all located Output variables excluding VS's own static f16_halves).
    std::map<std::uint32_t, std::uint32_t> vs_raw_outputs;
    std::set<std::uint32_t> vs_raw_locs;
    for (const auto& [loc, var_id] : vs_outputs) {
        if (!vs_static_half_locs.count(loc)) {
            vs_raw_outputs[loc] = var_id;
            vs_raw_locs.insert(loc);
        }
    }
    std::set<std::uint32_t> ps_raw_locs;
    for (const auto& [loc, var_id] : ps_inputs) {
        if (!ps_static_half_locs.count(loc)) ps_raw_locs.insert(loc);
    }

    // Choose paired [lo, hi] locations for every param in ps_entry.f16_halves.
    // Keep PS's existing locations unchanged whenever none collides with VS raw outputs or exceeds top_location.
    bool keep_ps_locs = true;
    std::set<std::uint32_t> seen_ps_halves;
    for (const auto& [param, pair] : ps_entry.f16_halves) {
        for (std::uint32_t loc : pair) {
            if (loc > top_location || vs_raw_locs.count(loc) || !seen_ps_halves.insert(loc).second) {
                keep_ps_locs = false;
                break;
            }
        }
        if (!keep_ps_locs) break;
    }

    std::map<std::uint32_t, std::array<std::uint32_t, 2>> pair_halves;
    if (keep_ps_locs) {
        pair_halves = ps_entry.f16_halves;
    } else {
        std::set<std::uint32_t> reserved = vs_raw_locs;
        reserved.insert(ps_raw_locs.begin(), ps_raw_locs.end());
        std::vector<std::uint32_t> free_locs;
        for (std::int64_t loc = top_location; loc >= 0; --loc)
            if (!reserved.count(std::uint32_t(loc))) free_locs.push_back(std::uint32_t(loc));
        const std::size_t need = 2 * ps_entry.f16_halves.size();
        if (free_locs.size() < need)
            return fail("f16_halves: need " + std::to_string(need) + " free locations in 0.." +
                        std::to_string(top_location) + ", only " + std::to_string(free_locs.size()) + " available");
        std::size_t k = 0;
        for (const auto& [param, old_pair] : ps_entry.f16_halves) {
            const std::uint32_t hi = free_locs[2 * k];
            const std::uint32_t lo = free_locs[2 * k + 1];
            pair_halves[param] = {lo, hi};
            ++k;
        }
    }

    // Patch PS OpDecorate Location in place if we relocated any PS half input.
    bool ps_changed = false;
    if (!keep_ps_locs) {
        std::map<std::uint32_t, std::uint32_t> ps_var_new_loc;
        for (const auto& [param, old_pair] : ps_entry.f16_halves) {
            const auto& new_pair = pair_halves[param];
            for (std::size_t h = 0; h < 2; ++h) {
                const auto it = ps_inputs.find(old_pair[h]);
                if (it != ps_inputs.end()) ps_var_new_loc[it->second] = new_pair[h];
            }
        }
        for (Inst& inst : ps_mod.insts) {
            if (inst.opcode == kOpDecorate && inst.words.size() >= 4 && inst.words[2] == kDecorationLocation) {
                const auto it = ps_var_new_loc.find(inst.words[1]);
                if (it != ps_var_new_loc.end() && inst.words[3] != it->second) {
                    inst.words[3] = it->second;
                    ps_changed = true;
                }
            }
        }
    }

    // Inspect VS types, globals, entry point, and uses of existing static f16_halves variables.
    std::uint32_t float_ty = 0, uint_ty = 0, v2float_ty = 0, v4float_ty = 0;
    std::uint32_t ptr_out_v4_ty = 0, ptr_priv_v4_ty = 0, null_v4_id = 0, glsl_id = 0;
    std::map<std::uint32_t, std::uint32_t> ptr_pointee;     // ptr_type_id -> pointee_type_id
    std::map<std::uint32_t, std::uint32_t> var_ptr_type;    // var_id -> ptr_type_id
    std::uint32_t vs_entry_fn = 0;
    std::size_t vs_ep_idx = vs_mod.insts.size();

    for (std::size_t idx = 0; idx < vs_mod.insts.size(); ++idx) {
        const Inst& inst = vs_mod.insts[idx];
        if (inst.opcode == kOpExtInstImport && inst.words.size() >= 3) {
            if (decode_literal_string(std::span<const std::uint32_t>(inst.words).subspan(2)) == "GLSL.std.450")
                glsl_id = inst.words[1];
        } else if (inst.opcode == kOpEntryPoint && inst.words.size() >= 4 && inst.words[1] == kExecModelVertex) {
            if (vs_ep_idx == vs_mod.insts.size()) {
                vs_ep_idx = idx;
                vs_entry_fn = inst.words[2];
            }
        } else if (inst.opcode == kOpTypeFloat && inst.words.size() == 3 && inst.words[2] == 32) {
            float_ty = inst.words[1];
        } else if (inst.opcode == kOpTypeInt && inst.words.size() == 4 && inst.words[2] == 32 && inst.words[3] == 0) {
            uint_ty = inst.words[1];
        } else if (inst.opcode == kOpTypeVector && inst.words.size() == 4) {
            if (float_ty && inst.words[2] == float_ty && inst.words[3] == 2) v2float_ty = inst.words[1];
            if (float_ty && inst.words[2] == float_ty && inst.words[3] == 4) v4float_ty = inst.words[1];
        } else if (inst.opcode == kOpTypePointer && inst.words.size() == 4) {
            ptr_pointee[inst.words[1]] = inst.words[3];
            if (v4float_ty && inst.words[3] == v4float_ty) {
                if (inst.words[2] == kStorageOutput) ptr_out_v4_ty = inst.words[1];
                if (inst.words[2] == kStoragePrivate) ptr_priv_v4_ty = inst.words[1];
            }
        } else if (inst.opcode == kOpConstantNull && inst.words.size() == 3 && v4float_ty && inst.words[1] == v4float_ty) {
            null_v4_id = inst.words[2];
        } else if (inst.opcode == kOpVariable && inst.words.size() >= 4) {
            var_ptr_type[inst.words[2]] = inst.words[1];
        } else if (inst.opcode == kOpFunction) {
            break;
        }
    }
    if (vs_ep_idx == vs_mod.insts.size() || !vs_entry_fn) return fail("f16_halves: VS has no Vertex entry point");

    // Check which existing VS variables are addressed by OpAccessChain (those cannot be trivially demoted to Private).
    std::set<std::uint32_t> access_chained_vars;
    for (const Inst& inst : vs_mod.insts) {
        if ((inst.opcode == kOpAccessChain || inst.opcode == kOpInBoundsAccessChain) && inst.words.size() >= 4)
            access_chained_vars.insert(inst.words[3]);
    }

    // Decide per VS static f16_halves entry: reuse (update Location) or demote (Output -> Private, drop Location).
    std::map<std::uint32_t, std::uint32_t> vs_var_new_loc;
    std::set<std::uint32_t> demote_vars;
    std::set<std::uint32_t> satisfied_params;
    for (const auto& [param, old_pair] : vs_entry.f16_halves) {
        const auto lo_it = vs_outputs.find(old_pair[0]);
        const auto hi_it = vs_outputs.find(old_pair[1]);
        const auto want_it = pair_halves.find(param);
        if (want_it != pair_halves.end() && lo_it != vs_outputs.end() && hi_it != vs_outputs.end()) {
            vs_var_new_loc[lo_it->second] = want_it->second[0];
            vs_var_new_loc[hi_it->second] = want_it->second[1];
            satisfied_params.insert(param);
        } else {
            for (const auto& it : {lo_it, hi_it}) {
                if (it == vs_outputs.end()) continue;
                const std::uint32_t vid = it->second;
                const auto pty = var_ptr_type.find(vid);
                const bool is_v4 = pty != var_ptr_type.end() && v4float_ty && ptr_pointee[pty->second] == v4float_ty;
                if (!is_v4 || access_chained_vars.count(vid))
                    return fail("f16_halves: cannot demote non-vec4 or access-chained VS static half output");
                demote_vars.insert(vid);
            }
        }
    }

    // Params that must be synthesized in the VS epilogue.
    struct SynthParam {
        std::uint32_t param = 0;
        std::uint32_t src_var = 0;  // 0 if VS does not export this param (emit null vec4)
        std::uint32_t lo_loc = 0;
        std::uint32_t hi_loc = 0;
        std::uint32_t lo_var = 0;
        std::uint32_t hi_var = 0;
    };
    std::vector<SynthParam> synth;
    bool need_null_v4 = false;
    for (const auto& [param, target_pair] : pair_halves) {
        if (satisfied_params.count(param)) continue;
        SynthParam sp;
        sp.param = param;
        sp.lo_loc = target_pair[0];
        sp.hi_loc = target_pair[1];
        const auto raw_it = vs_raw_outputs.find(param);
        if (raw_it != vs_raw_outputs.end()) {
            const auto pty = var_ptr_type.find(raw_it->second);
            if (pty == var_ptr_type.end() || !v4float_ty || ptr_pointee[pty->second] != v4float_ty)
                return fail("f16_halves: VS param " + std::to_string(param) + " output is not vec4 float");
            sp.src_var = raw_it->second;
        } else {
            need_null_v4 = true;
        }
        synth.push_back(sp);
    }

    // Create any missing types / GLSL.std.450 import needed for demotion or synthesis.
    std::vector<Inst> new_types;
    const bool need_glsl = std::any_of(synth.begin(), synth.end(), [](const SynthParam& s) { return s.src_var != 0; });
    bool insert_glsl = false;
    if (need_glsl && !glsl_id) {
        glsl_id = vs_mod.alloc_id();
        insert_glsl = true;
    }
    if ((!synth.empty() || !demote_vars.empty()) && !float_ty) {
        float_ty = vs_mod.alloc_id();
        new_types.push_back(make_inst(kOpTypeFloat, {float_ty, 32}));
    }
    if (need_glsl && !uint_ty) {
        uint_ty = vs_mod.alloc_id();
        new_types.push_back(make_inst(kOpTypeInt, {uint_ty, 32, 0}));
    }
    if (need_glsl && !v2float_ty) {
        v2float_ty = vs_mod.alloc_id();
        new_types.push_back(make_inst(kOpTypeVector, {v2float_ty, float_ty, 2}));
    }
    if ((!synth.empty() || !demote_vars.empty()) && !v4float_ty) {
        v4float_ty = vs_mod.alloc_id();
        new_types.push_back(make_inst(kOpTypeVector, {v4float_ty, float_ty, 4}));
    }
    if (!synth.empty() && !ptr_out_v4_ty) {
        ptr_out_v4_ty = vs_mod.alloc_id();
        new_types.push_back(make_inst(kOpTypePointer, {ptr_out_v4_ty, kStorageOutput, v4float_ty}));
    }
    if (!demote_vars.empty() && !ptr_priv_v4_ty) {
        ptr_priv_v4_ty = vs_mod.alloc_id();
        new_types.push_back(make_inst(kOpTypePointer, {ptr_priv_v4_ty, kStoragePrivate, v4float_ty}));
    }
    if (need_null_v4 && !null_v4_id) {
        null_v4_id = vs_mod.alloc_id();
        new_types.push_back(make_inst(kOpConstantNull, {v4float_ty, null_v4_id}));
    }

    // Allocate Output OpVariables for synthesized params.
    std::vector<Inst> new_decorates;
    std::vector<Inst> new_vars;
    for (SynthParam& sp : synth) {
        sp.lo_var = vs_mod.alloc_id();
        sp.hi_var = vs_mod.alloc_id();
        new_decorates.push_back(make_inst(kOpDecorate, {sp.lo_var, kDecorationLocation, sp.lo_loc}));
        new_decorates.push_back(make_inst(kOpDecorate, {sp.hi_var, kDecorationLocation, sp.hi_loc}));
        new_vars.push_back(make_inst(kOpVariable, {ptr_out_v4_ty, sp.lo_var, kStorageOutput}));
        new_vars.push_back(make_inst(kOpVariable, {ptr_out_v4_ty, sp.hi_var, kStorageOutput}));
    }

    // Rebuild VS instruction stream in exact SPIR-V logical layout order.
    // Demoted OpVariables are emitted right after new_types (before the first OpFunction) so ptr_priv_v4_ty is
    // always defined before every OpVariable that uses it.
    const bool drop_private_from_ep = vs_mod.header[1] < kSpirvVersion14;
    std::vector<Inst> demoted_var_insts;
    std::vector<Inst> out_insts;
    out_insts.reserve(vs_mod.insts.size() + new_types.size() + new_decorates.size() + new_vars.size() +
                      synth.size() * 32);

    bool glsl_inserted = !insert_glsl;
    bool decorates_inserted = new_decorates.empty();
    bool globals_inserted = new_types.empty() && new_vars.empty() && demote_vars.empty();
    bool in_entry_fn = false;
    bool found_entry_return = false;

    for (std::size_t idx = 0; idx < vs_mod.insts.size(); ++idx) {
        Inst inst = vs_mod.insts[idx];

        // Insert OpExtInstImport "GLSL.std.450" right before OpMemoryModel if the module didn't already have it.
        if (!glsl_inserted && inst.opcode == kOpMemoryModel) {
            // "GLSL.std.450\0" -> 4 little-endian words: 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000
            out_insts.push_back(
                make_inst(kOpExtInstImport, {glsl_id, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0x00000000u}));
            glsl_inserted = true;
        }

        // Insert new OpDecorate instructions at the end of the Annotations section (before the first type/global).
        if (!decorates_inserted && !is_pre_type_opcode(inst.opcode)) {
            out_insts.insert(out_insts.end(), new_decorates.begin(), new_decorates.end());
            decorates_inserted = true;
        }

        // Insert new Types / Constants, demoted Private OpVariables, and new Output OpVariables before OpFunction.
        if (!globals_inserted && inst.opcode == kOpFunction) {
            out_insts.insert(out_insts.end(), new_types.begin(), new_types.end());
            out_insts.insert(out_insts.end(), demoted_var_insts.begin(), demoted_var_insts.end());
            out_insts.insert(out_insts.end(), new_vars.begin(), new_vars.end());
            globals_inserted = true;
        }

        // Update Vertex OpEntryPoint interface list.
        if (idx == vs_ep_idx) {
            const std::size_t if_start = entry_point_interface_start(inst);
            std::vector<std::uint32_t> operands(inst.words.begin() + 1, inst.words.begin() + std::ptrdiff_t(if_start));
            for (std::size_t i = if_start; i < inst.words.size(); ++i) {
                const std::uint32_t id = inst.words[i];
                if (drop_private_from_ep && demote_vars.count(id)) continue;
                operands.push_back(id);
            }
            for (const SynthParam& sp : synth) {
                operands.push_back(sp.lo_var);
                operands.push_back(sp.hi_var);
            }
            out_insts.push_back(make_inst_vec(kOpEntryPoint, operands));
            continue;
        }

        // Update or remove OpDecorate Location on static VS f16_halves outputs.
        if (inst.opcode == kOpDecorate && inst.words.size() >= 4 && inst.words[2] == kDecorationLocation) {
            const std::uint32_t target = inst.words[1];
            if (demote_vars.count(target)) continue;  // drop Location decoration from demoted Private variable
            const auto it = vs_var_new_loc.find(target);
            if (it != vs_var_new_loc.end()) inst.words[3] = it->second;
            out_insts.push_back(std::move(inst));
            continue;
        }

        // Demote unneeded static VS f16_halves OpVariable from Output to Private (emitted after new_types).
        if (!globals_inserted && inst.opcode == kOpVariable && inst.words.size() >= 4 &&
            demote_vars.count(inst.words[2])) {
            inst.words[1] = ptr_priv_v4_ty;
            inst.words[3] = kStoragePrivate;
            demoted_var_insts.push_back(std::move(inst));
            continue;
        }

        // Track Vertex entry function and inject UnpackHalf2x16 epilogue before every OpReturn inside it.
        if (inst.opcode == kOpFunction && inst.words.size() >= 3 && inst.words[2] == vs_entry_fn) {
            in_entry_fn = true;
        } else if (inst.opcode == kOpFunctionEnd && in_entry_fn) {
            in_entry_fn = false;
        } else if (in_entry_fn && inst.opcode == kOpReturn && !synth.empty()) {
            found_entry_return = true;
            for (const SynthParam& sp : synth) {
                if (sp.src_var == 0) {
                    out_insts.push_back(make_inst(kOpStore, {sp.lo_var, null_v4_id}));
                    out_insts.push_back(make_inst(kOpStore, {sp.hi_var, null_v4_id}));
                    continue;
                }
                const std::uint32_t raw_id = vs_mod.alloc_id();
                out_insts.push_back(make_inst(kOpLoad, {v4float_ty, raw_id, sp.src_var}));
                std::array<std::uint32_t, 4> lo_c{}, hi_c{};
                for (std::uint32_t c = 0; c < 4; ++c) {
                    const std::uint32_t e_id = vs_mod.alloc_id();
                    const std::uint32_t u_id = vs_mod.alloc_id();
                    const std::uint32_t h2_id = vs_mod.alloc_id();
                    const std::uint32_t a_id = vs_mod.alloc_id();
                    const std::uint32_t b_id = vs_mod.alloc_id();
                    out_insts.push_back(make_inst(kOpCompositeExtract, {float_ty, e_id, raw_id, c}));
                    out_insts.push_back(make_inst(kOpBitcast, {uint_ty, u_id, e_id}));
                    out_insts.push_back(make_inst(kOpExtInst, {v2float_ty, h2_id, glsl_id, kGlslUnpackHalf2x16, u_id}));
                    out_insts.push_back(make_inst(kOpCompositeExtract, {float_ty, a_id, h2_id, 0}));
                    out_insts.push_back(make_inst(kOpCompositeExtract, {float_ty, b_id, h2_id, 1}));
                    lo_c[c] = a_id;
                    hi_c[c] = b_id;
                }
                const std::uint32_t lo_vec = vs_mod.alloc_id();
                const std::uint32_t hi_vec = vs_mod.alloc_id();
                out_insts.push_back(make_inst(kOpCompositeConstruct, {v4float_ty, lo_vec, lo_c[0], lo_c[1], lo_c[2], lo_c[3]}));
                out_insts.push_back(make_inst(kOpCompositeConstruct, {v4float_ty, hi_vec, hi_c[0], hi_c[1], hi_c[2], hi_c[3]}));
                out_insts.push_back(make_inst(kOpStore, {sp.lo_var, lo_vec}));
                out_insts.push_back(make_inst(kOpStore, {sp.hi_var, hi_vec}));
            }
        }

        out_insts.push_back(std::move(inst));
    }

    if (!synth.empty() && !found_entry_return)
        return fail("f16_halves: Vertex entry point has no OpReturn to emit decoded halves");
    if (!glsl_inserted || !decorates_inserted || !globals_inserted)
        return fail("f16_halves: unexpected VS SPIR-V layout");

    vs_mod.insts = std::move(out_insts);
    vs_words = vs_mod.serialize();
    if (ps_changed) ps_words = ps_mod.serialize();
    return true;
}

}  // namespace leonida
