// leonida::spirv_declares_writable_storage (src/spirv_stores.h) over hand-built SPIR-V declarations: which modules
// need fragmentStoresAndAtomics / vertexPipelineStoresAndAtomics outside a compute shader. No Vulkan, no GPU, no
// shader files: only the instructions the rule looks at (types, pointers, variables, decorations) after a header.
#include "../src/spirv_stores.h"

#include <cstdio>
#include <initializer_list>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "spirv_stores_test: FAILED: %s\n", what);
        ++failures;
    }
}

using Words = std::vector<std::uint32_t>;

Words header() { return {0x07230203, 0x00010600, 0, 100, 0}; }

void inst(Words& words, std::uint16_t op, std::initializer_list<std::uint32_t> operands) {
    words.push_back((static_cast<std::uint32_t>(operands.size() + 1) << 16) | op);
    words.insert(words.end(), operands);
}

// Opcodes and enumerants, from the SPIR-V specification.
constexpr std::uint16_t OpTypeInt = 21, OpTypeImage = 25, OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32,
                        OpTypeArray = 28, OpVariable = 59, OpDecorate = 71, OpMemberDecorate = 72, OpDecorationGroup = 73,
                        OpGroupDecorate = 74, OpGroupMemberDecorate = 75;
constexpr std::uint32_t Dim2D = 1, DimSubpassData = 6;
constexpr std::uint32_t Block = 2, BufferBlock = 3, NonWritable = 24;
constexpr std::uint32_t UniformConstant = 0, Uniform = 2, StorageBuffer = 12;
// ids
constexpr std::uint32_t uint_t = 1, image_t = 2, array_t = 3, pointer_t = 4, var = 5, struct_t = 6, length_c = 7;

// %image = OpTypeImage %uint <dim> 0 0 0 <sampled> <format>; `sampled` 2 is a storage image (or, with SubpassData, an
// input attachment), 1 a sampled one.
Words image_module(std::uint32_t sampled, bool arrayed, bool runtime, bool non_writable, std::uint32_t dim = Dim2D) {
    Words w = header();
    if (non_writable) inst(w, OpDecorate, {var, NonWritable});
    inst(w, OpTypeInt, {uint_t, 32, 0});
    inst(w, OpTypeImage, {image_t, uint_t, dim, 0, 0, 0, sampled, dim == DimSubpassData ? 0u : 33u});
    std::uint32_t pointee = image_t;
    if (arrayed) {
        if (runtime) {
            inst(w, OpTypeRuntimeArray, {array_t, image_t});
        } else {
            inst(w, OpTypeArray, {array_t, image_t, length_c});
        }
        pointee = array_t;
    }
    inst(w, OpTypePointer, {pointer_t, UniformConstant, pointee});
    inst(w, OpVariable, {pointer_t, var, UniformConstant});
    return w;
}

// A two-member block in `storage_class`, decorated Block or BufferBlock, with `read_only_members` of its members
// (and optionally the variable itself) NonWritable.
Words buffer_module(std::uint32_t storage_class, std::uint32_t block_decoration, unsigned read_only_members,
                    bool variable_non_writable) {
    Words w = header();
    inst(w, OpDecorate, {struct_t, block_decoration});
    for (std::uint32_t member = 0; member < read_only_members; ++member)
        inst(w, OpMemberDecorate, {struct_t, member, NonWritable});
    if (variable_non_writable) inst(w, OpDecorate, {var, NonWritable});
    inst(w, OpTypeInt, {uint_t, 32, 0});
    inst(w, OpTypeStruct, {struct_t, uint_t, uint_t});
    inst(w, OpTypePointer, {pointer_t, storage_class, struct_t});
    inst(w, OpVariable, {pointer_t, var, storage_class});
    return w;
}

}  // namespace

int main() {
    using leonida::spirv_declares_writable_storage;

    // Storage images: the translator's form (no decoration), read-only, and inside arrays, as set 0 declares them.
    check(spirv_declares_writable_storage(image_module(2, false, false, false)), "a storage image needs the feature");
    check(!spirv_declares_writable_storage(image_module(2, false, false, true)), "a NonWritable storage image does not");
    check(spirv_declares_writable_storage(image_module(2, true, true, false)),
          "a runtime array of storage images needs it");
    check(spirv_declares_writable_storage(image_module(2, true, false, false)), "a sized array of storage images too");
    check(!spirv_declares_writable_storage(image_module(2, true, true, true)),
          "a NonWritable array of storage images does not");
    check(!spirv_declares_writable_storage(image_module(1, false, false, false)), "a sampled image does not");
    check(!spirv_declares_writable_storage(image_module(1, true, true, false)), "an array of sampled images does not");
    // An input attachment (subpassInput) is Sampled 2 with Dim SubpassData, and read-only: no feature.
    check(!spirv_declares_writable_storage(image_module(2, false, false, false, DimSubpassData)),
          "an input attachment does not");
    check(!spirv_declares_writable_storage(image_module(2, true, false, false, DimSubpassData)),
          "an array of input attachments does not");

    // NonWritable through a decoration group (OpDecorationGroup + OpGroupDecorate) counts like a direct one.
    {
        constexpr std::uint32_t group = 20;
        const auto grouped = [&](std::uint32_t target) {
            Words w = header();
            inst(w, OpDecorate, {group, NonWritable});
            inst(w, OpDecorationGroup, {group});
            inst(w, OpGroupDecorate, {group, target});
            inst(w, OpTypeInt, {uint_t, 32, 0});
            inst(w, OpTypeImage, {image_t, uint_t, Dim2D, 0, 0, 0, 2, 33});
            inst(w, OpTypePointer, {pointer_t, UniformConstant, image_t});
            inst(w, OpVariable, {pointer_t, var, UniformConstant});
            return w;
        };
        check(!spirv_declares_writable_storage(grouped(var)),
              "a storage image made NonWritable through a decoration group does not");
        check(spirv_declares_writable_storage(grouped(var + 100)),
              "a group that targets another id does not cover the variable");
    }
    // And OpGroupMemberDecorate for every member of a storage buffer block.
    {
        constexpr std::uint32_t group = 20;
        Words w = header();
        inst(w, OpDecorate, {group, NonWritable});
        inst(w, OpDecorationGroup, {group});
        inst(w, OpDecorate, {struct_t, Block});
        inst(w, OpGroupMemberDecorate, {group, struct_t, 0, struct_t, 1});
        inst(w, OpTypeInt, {uint_t, 32, 0});
        inst(w, OpTypeStruct, {struct_t, uint_t, uint_t});
        inst(w, OpTypePointer, {pointer_t, StorageBuffer, struct_t});
        inst(w, OpVariable, {pointer_t, var, StorageBuffer});
        check(!spirv_declares_writable_storage(w), "a block whose members are NonWritable through a group does not");
    }

    // Storage buffers: read-only when the variable or every member is.
    check(spirv_declares_writable_storage(buffer_module(StorageBuffer, Block, 0, false)),
          "a storage buffer block needs the feature");
    check(spirv_declares_writable_storage(buffer_module(StorageBuffer, Block, 1, false)),
          "a block with one writable member still needs it");
    // Out-of-range member decorations must not satisfy the count for an undecorated writable member.
    {
        Words w = header();
        inst(w, OpDecorate, {struct_t, Block});
        inst(w, OpMemberDecorate, {struct_t, 0, NonWritable});
        inst(w, OpMemberDecorate, {struct_t, 5, NonWritable});
        inst(w, OpTypeInt, {uint_t, 32, 0});
        inst(w, OpTypeStruct, {struct_t, uint_t, uint_t});
        inst(w, OpTypePointer, {pointer_t, StorageBuffer, struct_t});
        inst(w, OpVariable, {pointer_t, var, StorageBuffer});
        check(spirv_declares_writable_storage(w),
              "an out-of-range OpMemberDecorate cannot hide a writable member");
    }
    {
        constexpr std::uint32_t group = 20;
        Words w = header();
        inst(w, OpDecorate, {group, NonWritable});
        inst(w, OpDecorationGroup, {group});
        inst(w, OpDecorate, {struct_t, Block});
        inst(w, OpGroupMemberDecorate, {group, struct_t, 0, struct_t, 2});
        inst(w, OpTypeInt, {uint_t, 32, 0});
        inst(w, OpTypeStruct, {struct_t, uint_t, uint_t});
        inst(w, OpTypePointer, {pointer_t, StorageBuffer, struct_t});
        inst(w, OpVariable, {pointer_t, var, StorageBuffer});
        check(spirv_declares_writable_storage(w),
              "an out-of-range OpGroupMemberDecorate cannot hide a writable member");
    }
    check(!spirv_declares_writable_storage(buffer_module(StorageBuffer, Block, 2, false)),
          "a block whose members are all NonWritable does not (GLSL `readonly buffer`)");
    check(!spirv_declares_writable_storage(buffer_module(StorageBuffer, Block, 0, true)),
          "a NonWritable variable does not (the translator's descriptor map)");
    check(spirv_declares_writable_storage(buffer_module(Uniform, BufferBlock, 0, false)),
          "the SPIR-V 1.0 form of a storage buffer (Uniform + BufferBlock) needs it");
    check(!spirv_declares_writable_storage(buffer_module(Uniform, BufferBlock, 2, false)),
          "and does not when it is read-only");
    check(!spirv_declares_writable_storage(buffer_module(Uniform, Block, 0, false)), "a uniform buffer does not");

    // Not a module: nothing to need.
    check(!spirv_declares_writable_storage({}), "no words");
    Words bad_magic = image_module(2, false, false, false);
    bad_magic[0] = 0;
    check(!spirv_declares_writable_storage(bad_magic), "a wrong magic");
    Words truncated = image_module(2, false, false, false);
    truncated.pop_back();  // the OpVariable now runs past the end
    check(!spirv_declares_writable_storage(truncated), "an instruction that runs past the end");
    Words zero_length = header();
    zero_length.push_back(OpVariable);  // word count 0
    check(!spirv_declares_writable_storage(zero_length), "an instruction of length 0");
    check(!spirv_declares_writable_storage(header()), "a header with no instructions");
    // An array type that names itself as its element: the walk through array types stops instead of spinning.
    Words self_array = header();
    inst(self_array, OpTypeRuntimeArray, {array_t, array_t});
    inst(self_array, OpTypePointer, {pointer_t, UniformConstant, array_t});
    inst(self_array, OpVariable, {pointer_t, var, UniformConstant});
    check(!spirv_declares_writable_storage(self_array), "a self-referential array type ends, and needs nothing");
    // The same module with a real storage image declared after it is still seen.
    Words mixed = self_array;
    inst(mixed, OpTypeInt, {uint_t, 32, 0});
    inst(mixed, OpTypeImage, {image_t, uint_t, 1, 0, 0, 0, 2, 33});
    inst(mixed, OpTypePointer, {struct_t, UniformConstant, image_t});
    inst(mixed, OpVariable, {struct_t, length_c, UniformConstant});
    check(spirv_declares_writable_storage(mixed), "a storage image after a malformed array is still found");

    if (failures) return 1;
    std::printf("spirv_stores_test: OK (storage images, arrays of them, storage buffers in both forms, framing, a "
                "self-referential array, input attachments, decoration groups)\n");
    return 0;
}
