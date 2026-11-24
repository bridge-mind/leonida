#pragma once

// D24_UNORM_S8_UINT on a GPU without D24 (the layer stores D32_SFLOAT_S8_UINT, depth24.h) in draws (D3D-309).
//
// Two D3D rules break when the depth attachment is float (found by @marquezinii, measured by @Vonsek on #291):
//   * the integer RasterizerState.DepthBias is in fixed D24 units, r = 1 / (2^24 - 1) ("the minimum representable
//     value > 0 in the depth-buffer format"), while Vulkan scales a float attachment's constant bias by the
//     primitive's exponent;
//   * clears are quantised to the D24 code (depth24_clear_value), fragment depth is not, so an EQUAL test at a
//     cleared value fails where D24 hardware passes.
// The fix moves both into the pixel stage: the fragment writes FragDepth = quantise(z + bias), with z the rasterised
// depth, bias = DepthBias * r + SlopeScaledDepthBias * max(|dz/dx|, |dz/dy|) clamped by DepthBiasClamp (D3D's
// formula), and quantise = depth24_to_float(depth24_quantize(.)) from depth24_convert.comp, so a fragment holds
// exactly the float a D24 code reads back as. The rasterizer's own bias is then turned off for the pipeline.
// A shader that writes its own depth (SV_Depth) keeps its value, quantised; D3D applies no bias to it.
//
// depth24_patch_fragment edits a translated pixel shader's SPIR-V; depth24_depth_only_fragment builds the stage
// for a pipeline without a pixel shader (depth-only passes such as shadow maps). Both are plain word editing with
// no dependency beyond the standard library; d3d12_depth24_draw_test runs them (with LEONIDA_VK_VALIDATION=1 the
// validation layer checks the modules).
//
// Not covered (logged by the caller): a pixel shader with EarlyFragmentTests (its FragDepth write would be ignored),
// multisampled targets (FragDepth is per pixel, which would flatten per-sample depth), and a viewport depth range
// narrower than the biased depth (D3D clamps to MinDepth..MaxDepth before quantising; the quantisation here clamps
// to 0..1 and Vulkan's clamp to the viewport range follows it).

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace leonida {

struct Depth24Bias {
    int constant = 0;     // D3D12_RASTERIZER_DESC::DepthBias
    float slope = 0.0f;   // SlopeScaledDepthBias
    float clamp = 0.0f;   // DepthBiasClamp: > 0 caps the bias, < 0 floors it, 0 leaves it
    bool any() const { return constant != 0 || slope != 0.0f; }
};

// D3D's constant term in float: (float)DepthBias * r, with r the float nearest 1 / (2^24 - 1).
inline float depth24_bias_constant(int bias) {
    return float(bias) * float(1.0 / 16777215.0);
}

namespace spirv24 {

enum : std::uint32_t {
    op_name = 5, op_line = 8, op_ext_inst_import = 11, op_ext_inst = 12, op_memory_model = 14, op_entry_point = 15,
    op_execution_mode = 16, op_capability = 17, op_type_void = 19, op_type_bool = 20, op_type_int = 21,
    op_type_float = 22, op_type_vector = 23, op_type_pointer = 32, op_type_function = 33, op_constant = 43,
    op_function = 54, op_function_end = 56, op_variable = 59, op_load = 61, op_store = 62, op_decorate = 71,
    op_composite_extract = 81, op_convert_f_to_u = 109, op_convert_u_to_f = 112, op_bitcast = 124, op_i_add = 128,
    op_f_add = 129, op_i_sub = 130, op_f_mul = 133, op_select = 169, op_i_equal = 170, op_u_greater_than = 172,
    op_u_greater_than_equal = 174, op_shift_left_logical = 196, op_dpdx = 207, op_dpdy = 208, op_label = 248,
    op_return = 253, op_no_line = 317, op_execution_mode_id = 331,
};
enum : std::uint32_t { storage_input = 1, storage_output = 3 };
enum : std::uint32_t { decoration_builtin = 11, builtin_frag_coord = 15, builtin_frag_depth = 22 };
enum : std::uint32_t { model_fragment = 4 };
enum : std::uint32_t {
    mode_origin_upper_left = 7, mode_early_fragment_tests = 9, mode_depth_replacing = 12, mode_depth_greater = 14,
    mode_depth_less = 15, mode_depth_unchanged = 16,
};
enum : std::uint32_t { glsl_round_even = 2, glsl_fabs = 4, glsl_fmin = 37, glsl_fmax = 40, glsl_find_umsb = 75 };

using Inst = std::vector<std::uint32_t>;  // a whole instruction, word 0 included

inline std::uint32_t opcode(const Inst& i) { return i[0] & 0xffffu; }
inline Inst make(std::uint32_t op, std::initializer_list<std::uint32_t> operands) {
    Inst i;
    i.reserve(operands.size() + 1);
    i.push_back(std::uint32_t(operands.size() + 1) << 16 | op);
    i.insert(i.end(), operands.begin(), operands.end());
    return i;
}
inline void append_string(Inst& i, const char* s) {
    const std::size_t n = std::strlen(s) + 1, words = (n + 3) / 4;
    const std::size_t at = i.size();
    i.resize(at + words, 0);
    std::memcpy(&i[at], s, n);
}
inline void fix_length(Inst& i) { i[0] = std::uint32_t(i.size()) << 16 | opcode(i); }
inline std::uint32_t bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
// Words a literal string takes in an instruction, starting at `at`.
inline std::size_t string_words(const Inst& i, std::size_t at) {
    std::size_t w = at;
    while (w < i.size()) {
        const std::uint32_t word = i[w++];
        if ((word & 0xff000000u) == 0 || (word & 0x00ff0000u) == 0 || (word & 0x0000ff00u) == 0 ||
            (word & 0x000000ffu) == 0)
            break;
    }
    return w - at;
}

inline bool is_global_decl(std::uint32_t op) {
    return op == 1 /* OpUndef */ || (op >= op_type_void && op <= 39) || (op >= 41 && op <= 52) || op == op_variable;
}

class Module {
public:
    bool parse(const std::vector<std::uint32_t>& words, std::string& why) {
        if (words.size() < 5 || words[0] != 0x07230203u) {
            why = "not a SPIR-V module";
            return false;
        }
        header_.assign(words.begin(), words.begin() + 5);
        for (std::size_t w = 5; w < words.size();) {
            const std::uint32_t count = words[w] >> 16;
            if (count == 0 || w + count > words.size()) {
                why = "truncated instruction";
                return false;
            }
            insts_.emplace_back(words.begin() + std::ptrdiff_t(w), words.begin() + std::ptrdiff_t(w + count));
            w += count;
        }
        return true;
    }
    std::vector<std::uint32_t> words() const {
        std::vector<std::uint32_t> out = header_;
        for (const Inst& i : insts_) out.insert(out.end(), i.begin(), i.end());
        return out;
    }
    std::uint32_t fresh() { return header_[3]++; }
    std::uint32_t version() const { return header_[1]; }
    std::vector<Inst>& insts() { return insts_; }
    std::vector<std::uint32_t>& header() { return header_; }

private:
    std::vector<std::uint32_t> header_;
    std::vector<Inst> insts_;
};

// The edit, on a parsed module. Globals are collected first and spliced in at their section's end.
class Patch {
public:
    Patch(Module& m, std::uint32_t entry) : m_(m), entry_(entry) {
        // Copies: apply() moves the module's instructions while the code at returns still looks types up.
        for (const Inst& i : m.insts()) {
            if (opcode(i) == op_function) break;
            existing_.push_back(i);
        }
    }

    // A type, reused when the module already declares it (SPIR-V forbids duplicate non-aggregate types).
    std::uint32_t type(std::uint32_t op, std::initializer_list<std::uint32_t> operands) {
        auto same = [&](const Inst& i) {
            if (opcode(i) != op || i.size() != operands.size() + 2) return false;
            std::size_t k = 2;
            for (std::uint32_t o : operands)
                if (i[k++] != o) return false;
            return true;
        };
        for (const Inst& ex : existing_)
            if (same(ex)) return ex[1];
        for (const Inst& i : globals_)
            if (same(i)) return i[1];
        const std::uint32_t id = m_.fresh();
        Inst i = make(op, {id});
        i.insert(i.end(), operands.begin(), operands.end());
        fix_length(i);
        globals_.push_back(std::move(i));
        return id;
    }
    std::uint32_t t_float() { return type(op_type_float, {32}); }
    std::uint32_t t_uint() { return type(op_type_int, {32, 0}); }
    std::uint32_t t_bool() { return type(op_type_bool, {}); }
    // Constants are cached by type and value: all of them are declared before apply() writes the globals out.
    std::uint32_t constant(std::uint32_t t, std::uint32_t value) {
        for (const auto& c : constants_)
            if (c[0] == t && c[1] == value) return c[2];
        const std::uint32_t id = m_.fresh();
        globals_.push_back(make(op_constant, {t, id, value}));
        constants_.push_back({t, value, id});
        return id;
    }
    // Declares what quantise() uses, so it can run inside apply() (at returns) without adding globals.
    void prime_quantise() {
        t_bool();
        glsl();
        for (float f : {16777215.0f, 0.0f}) f32(f);
        for (std::uint32_t u : {0x3f800000u, 0xffffffu, 0x7f800000u, 0u, 23u, 1u, 80u}) u32(u);
    }
    std::uint32_t f32(float f) { return constant(t_float(), bits(f)); }
    std::uint32_t u32(std::uint32_t u) { return constant(t_uint(), u); }

    std::uint32_t glsl() {
        if (glsl_) return glsl_;
        for (const Inst& ex : existing_) {
            if (opcode(ex) != op_ext_inst_import || ex.size() < 3) continue;
            std::string name((ex.size() - 2) * 4, '\0');
            std::memcpy(name.data(), &ex[2], name.size());
            if (std::strcmp(name.c_str(), "GLSL.std.450") == 0) return glsl_ = ex[1];
        }
        glsl_ = m_.fresh();
        Inst i = make(op_ext_inst_import, {glsl_});
        append_string(i, "GLSL.std.450");
        fix_length(i);
        import_ = std::move(i);
        return glsl_;
    }

    // A built-in variable already decorated in the module, or 0.
    std::uint32_t builtin_variable(std::uint32_t builtin) const {
        for (const Inst& ex : existing_)
            if (opcode(ex) == op_decorate && ex.size() == 4 && ex[2] == decoration_builtin && ex[3] == builtin)
                return ex[1];
        return 0;
    }
    std::uint32_t variable(std::uint32_t pointer_type, std::uint32_t storage, std::uint32_t builtin) {
        const std::uint32_t id = m_.fresh();
        globals_.push_back(make(op_variable, {pointer_type, id, storage}));
        decorations_.push_back(make(op_decorate, {id, decoration_builtin, builtin}));
        interface_.push_back(id);
        return id;
    }
    void ensure_interface(std::uint32_t id) { interface_.push_back(id); }
    void mode(std::uint32_t m) { modes_.push_back(make(op_execution_mode, {entry_, m})); }

    // Instructions in a function body: emit returns the result id.
    std::uint32_t emit(std::vector<Inst>& code, std::uint32_t op, std::uint32_t type,
                       std::initializer_list<std::uint32_t> operands) {
        const std::uint32_t id = m_.fresh();
        Inst i = make(op, {type, id});
        i.insert(i.end(), operands.begin(), operands.end());
        fix_length(i);
        code.push_back(std::move(i));
        return id;
    }
    std::uint32_t ext(std::vector<Inst>& code, std::uint32_t type, std::uint32_t inst,
                      std::initializer_list<std::uint32_t> operands) {
        const std::uint32_t id = m_.fresh();
        Inst i = make(op_ext_inst, {type, id, glsl(), inst});
        i.insert(i.end(), operands.begin(), operands.end());
        fix_length(i);
        code.push_back(std::move(i));
        return id;
    }

    // depth24_to_float(depth24_from_bits(z)) of depth24_convert.comp, branch-free.
    std::uint32_t quantise(std::vector<Inst>& code, std::uint32_t z) {
        const std::uint32_t tf = t_float(), tu = t_uint(), tb = t_bool();
        const std::uint32_t b = emit(code, op_bitcast, tu, {z});
        const std::uint32_t product = emit(code, op_f_mul, tf, {z, f32(16777215.0f)});
        const std::uint32_t rounded = ext(code, tf, glsl_round_even, {product});
        // Clamped before the conversion so an out-of-range float never reaches OpConvertFToU (undefined there).
        const std::uint32_t safe = ext(code, tf, glsl_fmin, {ext(code, tf, glsl_fmax, {rounded, f32(0.0f)}),
                                                            f32(16777215.0f)});
        const std::uint32_t code_mid = emit(code, op_convert_f_to_u, tu, {safe});
        const std::uint32_t at_least_one = emit(code, op_u_greater_than_equal, tb, {b, u32(0x3f800000u)});
        const std::uint32_t code_hi = emit(code, op_select, tu, {at_least_one, u32(0xffffffu), code_mid});
        const std::uint32_t negative_or_nan = emit(code, op_u_greater_than, tb, {b, u32(0x7f800000u)});
        const std::uint32_t d = emit(code, op_select, tu, {negative_or_nan, u32(0), code_hi});
        // float(m + 1) * 2^(e - 47) with e = findMSB(d) and m = d << (23 - e); d == 0 gives 0.
        const std::uint32_t e = ext(code, tu, glsl_find_umsb, {d});
        const std::uint32_t m = emit(code, op_shift_left_logical, tu, {d, emit(code, op_i_sub, tu, {u32(23), e})});
        const std::uint32_t mf = emit(code, op_convert_u_to_f, tf, {emit(code, op_i_add, tu, {m, u32(1)})});
        const std::uint32_t exponent = emit(code, op_shift_left_logical, tu, {emit(code, op_i_add, tu, {e, u32(80)}),
                                                                            u32(23)});
        const std::uint32_t scale = emit(code, op_bitcast, tf, {exponent});
        const std::uint32_t value = emit(code, op_f_mul, tf, {mf, scale});
        const std::uint32_t zero = emit(code, op_i_equal, tb, {d, u32(0)});
        return emit(code, op_select, tf, {zero, f32(0.0f), value});
    }

    // Splices everything into the module.
    void apply(std::vector<Inst> entry_code, std::vector<Inst> (*at_return)(Patch&, void*), void* ctx,
               bool drop_conservative_modes) {
        auto& in = m_.insts();
        std::size_t last_entry_point = 0;
        for (std::size_t k = 0; k < in.size(); ++k)
            if (opcode(in[k]) == op_entry_point) last_entry_point = k;
        std::vector<Inst> out;
        out.reserve(in.size() + globals_.size() + entry_code.size() + 16);
        bool imported = import_.empty(), decorated = false, declared = false;
        bool in_main = false, entry_done = false;
        for (std::size_t k = 0; k < in.size(); ++k) {
            Inst& i = in[k];
            const std::uint32_t op = opcode(i);
            if (!imported && op == op_memory_model) {
                out.push_back(std::move(import_));
                imported = true;
            }
            if (op == op_execution_mode && i.size() >= 3 && i[1] == entry_ && drop_conservative_modes &&
                (i[2] == mode_depth_greater || i[2] == mode_depth_less || i[2] == mode_depth_unchanged ||
                 i[2] == mode_depth_replacing))
                continue;  // replaced by the DepthReplacing in modes_
            if (op == op_entry_point && i.size() > 2 && i[2] == entry_) {
                const std::size_t names = 3 + string_words(i, 3);
                for (std::uint32_t id : interface_) {
                    bool listed = false;
                    for (std::size_t w = names; w < i.size(); ++w) listed |= i[w] == id;
                    if (!listed) i.push_back(id);
                }
                fix_length(i);
            }
            if (!decorated && (is_global_decl(op) || op == op_function)) {
                for (Inst& d : decorations_) out.push_back(std::move(d));
                decorated = true;
            }
            if (!declared && op == op_function) {
                for (const Inst& g : globals_) out.push_back(g);  // copies: type() may still read them
                declared = true;
            }
            if (op == op_function && i.size() > 2 && i[2] == entry_) in_main = true;
            if (in_main && !entry_done && op != op_function && op != op_label && op != op_variable &&
                op != op_line && op != op_no_line && op != 55 /* OpFunctionParameter */) {
                for (Inst& c : entry_code) out.push_back(std::move(c));
                entry_done = true;
            }
            if (in_main && op == op_return && at_return) {
                for (Inst& c : at_return(*this, ctx)) out.push_back(std::move(c));
            }
            if (op == op_function_end) in_main = false;
            out.push_back(std::move(i));
            if (k == last_entry_point)  // execution modes follow the entry points
                for (Inst& mo : modes_) out.push_back(std::move(mo));
        }
        in = std::move(out);
    }

private:
    Module& m_;
    std::uint32_t entry_;
    std::vector<Inst> existing_;
    std::vector<Inst> globals_, decorations_, modes_;
    Inst import_;
    std::vector<std::uint32_t> interface_;
    std::vector<std::array<std::uint32_t, 3>> constants_;  // type, value, id
    std::uint32_t glsl_ = 0;
};

}  // namespace spirv24

// Edits a pixel shader so it writes the D24-quantised (and, unless it writes depth itself, D3D-biased) depth.
// Returns the new words, or an empty vector with `why` set when the shader can't take the edit.
inline std::vector<std::uint32_t> depth24_patch_fragment(const std::vector<std::uint32_t>& words,
                                                         const Depth24Bias& bias, std::string& why) {
    using namespace spirv24;
    Module m;
    if (!m.parse(words, why)) return {};
    std::uint32_t entry = 0;
    for (const Inst& i : m.insts()) {
        if (opcode(i) == op_entry_point && i.size() > 2 && i[1] == model_fragment) {
            if (entry) {
                why = "more than one fragment entry point";
                return {};
            }
            entry = i[2];
        }
        if (opcode(i) == op_function) break;
    }
    if (!entry) {
        why = "no fragment entry point";
        return {};
    }
    for (const Inst& i : m.insts()) {
        if (opcode(i) == op_execution_mode && i.size() >= 3 && i[1] == entry && i[2] == mode_early_fragment_tests) {
            why = "the shader forces early fragment tests, which ignore a FragDepth write";
            return {};
        }
        if (opcode(i) == op_function) break;
    }

    Patch p(m, entry);
    const std::uint32_t tf = p.t_float();
    p.prime_quantise();
    std::uint32_t frag_depth = p.builtin_variable(builtin_frag_depth);
    const bool shader_writes_depth = frag_depth != 0;
    if (!shader_writes_depth)
        frag_depth = p.variable(p.type(op_type_pointer, {storage_output, tf}), storage_output, builtin_frag_depth);
    else
        p.ensure_interface(frag_depth);
    p.mode(mode_depth_replacing);

    std::vector<Inst> entry_code;
    if (!shader_writes_depth) {
        std::uint32_t frag_coord = p.builtin_variable(builtin_frag_coord);
        const std::uint32_t tv4 = p.type(op_type_vector, {tf, 4});
        if (!frag_coord)
            frag_coord = p.variable(p.type(op_type_pointer, {storage_input, tv4}), storage_input, builtin_frag_coord);
        else
            p.ensure_interface(frag_coord);
        const std::uint32_t coord = p.emit(entry_code, op_load, tv4, {frag_coord});
        std::uint32_t z = p.emit(entry_code, op_composite_extract, tf, {coord, 2});
        if (bias.any()) {
            // In the entry block: uniform control flow, so the derivatives are defined.
            const std::uint32_t dx = p.ext(entry_code, tf, glsl_fabs, {p.emit(entry_code, op_dpdx, tf, {z})});
            const std::uint32_t dy = p.ext(entry_code, tf, glsl_fabs, {p.emit(entry_code, op_dpdy, tf, {z})});
            const std::uint32_t max_slope = p.ext(entry_code, tf, glsl_fmax, {dx, dy});
            std::uint32_t amount = p.emit(entry_code, op_f_mul, tf, {p.f32(bias.slope), max_slope});
            amount = p.emit(entry_code, op_f_add, tf, {amount, p.f32(depth24_bias_constant(bias.constant))});
            if (bias.clamp > 0.0f) amount = p.ext(entry_code, tf, glsl_fmin, {amount, p.f32(bias.clamp)});
            if (bias.clamp < 0.0f) amount = p.ext(entry_code, tf, glsl_fmax, {amount, p.f32(bias.clamp)});
            z = p.emit(entry_code, op_f_add, tf, {z, amount});
        }
        const std::uint32_t q = p.quantise(entry_code, z);
        entry_code.push_back(make(op_store, {frag_depth, q}));
        p.apply(std::move(entry_code), nullptr, nullptr, true);
    } else {
        struct Ctx { std::uint32_t var, tf; } ctx{frag_depth, tf};
        p.apply({}, [](Patch& patch, void* c) {
            const auto& x = *static_cast<Ctx*>(c);
            std::vector<Inst> code;
            const std::uint32_t written = patch.emit(code, op_load, x.tf, {x.var});
            code.push_back(make(op_store, {x.var, patch.quantise(code, written)}));
            return code;
        }, &ctx, true);
    }
    return m.words();
}

// A pixel stage for a pipeline without one: no colour output, only the biased, quantised depth.
inline std::vector<std::uint32_t> depth24_depth_only_fragment(const Depth24Bias& bias) {
    using namespace spirv24;
    // OpCapability Shader; OpMemoryModel Logical GLSL450; OpEntryPoint Fragment %main "main";
    // OpExecutionMode %main OriginUpperLeft; %void; %fn = void(); %main = function { label; return }
    enum : std::uint32_t { id_main = 1, id_void = 2, id_fn = 3, id_label = 4, bound = 5 };
    std::vector<std::uint32_t> w = {0x07230203u, 0x00010000u, 0, bound, 0};
    auto put = [&](Inst i) { w.insert(w.end(), i.begin(), i.end()); };
    put(make(op_capability, {1}));
    put(make(op_memory_model, {0, 1}));
    Inst ep = make(op_entry_point, {model_fragment, id_main});
    append_string(ep, "main");
    fix_length(ep);
    put(ep);
    put(make(op_execution_mode, {id_main, mode_origin_upper_left}));
    put(make(op_type_void, {id_void}));
    put(make(op_type_function, {id_fn, id_void}));
    put(make(op_function, {id_void, id_main, 0, id_fn}));
    put(make(op_label, {id_label}));
    put(make(op_return, {}));
    put(make(op_function_end, {}));
    std::string why;
    return depth24_patch_fragment(w, bias, why);
}

}  // namespace leonida
