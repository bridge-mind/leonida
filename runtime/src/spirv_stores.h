#pragma once

// Does a SPIR-V module declare a storage image, storage texel buffer or storage buffer it may write?
//
// Vulkan lets a shader stage other than compute write such a variable only when the device enabled a feature for
// it: fragmentStoresAndAtomics for the fragment stage (VUID-RuntimeSpirv-NonWritable-06340), and
// vertexPipelineStoresAndAtomics for the vertex, tessellation and geometry stages (-06341). Without the feature,
// every such variable of the stage must carry the NonWritable decoration. The translator declares the storage
// images of a shader that stores or does an image atomic without it, so the device has to know which modules need
// the feature before it builds a pipeline from them.
//
// The scan is conservative the way the rule is: it looks at declarations, not at whether an instruction writes.
// No Vulkan headers; the caller has already checked the binary's framing (spirv_words.h).

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace leonida {

inline bool spirv_declares_writable_storage(std::span<const std::uint32_t> words) {
    constexpr std::uint32_t magic = 0x07230203;
    constexpr std::uint16_t op_type_image = 25, op_type_array = 28, op_type_runtime_array = 29, op_type_struct = 30,
                            op_type_pointer = 32, op_variable = 59, op_decorate = 71, op_member_decorate = 72,
                            op_group_decorate = 74, op_group_member_decorate = 75;
    constexpr std::uint32_t decoration_buffer_block = 3, decoration_non_writable = 24;
    constexpr std::uint32_t dim_subpass_data = 6;  // an input attachment: Sampled 2 as well, but read-only
    constexpr std::uint32_t class_uniform_constant = 0, class_uniform = 2, class_storage_buffer = 12;
    if (words.size() < 5 || words[0] != magic) return false;

    struct Pointer {
        std::uint32_t storage_class, pointee;
    };
    struct Variable {
        std::uint32_t pointer_type, id, storage_class;
    };
    std::unordered_set<std::uint32_t> storage_images;                  // OpTypeImage with Sampled == 2
    std::unordered_map<std::uint32_t, std::uint32_t> element_of;       // array type -> element type
    std::unordered_map<std::uint32_t, std::uint32_t> member_count;     // struct type -> members
    std::unordered_map<std::uint32_t, Pointer> pointers;
    std::unordered_set<std::uint32_t> non_writable, buffer_blocks;     // decorated ids
    std::unordered_map<std::uint32_t, std::unordered_set<std::uint32_t>> non_writable_members;
    std::vector<Variable> variables;
    // OpGroupDecorate / OpGroupMemberDecorate: a decoration group's decorations reach these targets. Resolved after
    // the scan, once every OpDecorate of the group has been seen.
    std::vector<std::pair<std::uint32_t, std::uint32_t>> group_targets;                      // group, target
    std::vector<std::pair<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>> group_members;  // group, (type, member)

    for (std::size_t at = 5; at < words.size();) {
        const std::uint32_t count = words[at] >> 16;
        const std::uint16_t op = static_cast<std::uint16_t>(words[at] & 0xffff);
        if (count == 0 || at + count > words.size()) return false;  // not a module the driver would take either
        const std::uint32_t* w = &words[at];
        switch (op) {
        case op_type_image:  // result, sampled type, dim, depth, arrayed, ms, sampled, format
            if (count >= 9 && w[7] == 2 && w[3] != dim_subpass_data) storage_images.insert(w[1]);
            break;
        case op_type_array:
        case op_type_runtime_array:
            if (count >= 3) element_of[w[1]] = w[2];
            break;
        case op_type_struct:
            if (count >= 2) member_count[w[1]] = count - 2;
            break;
        case op_type_pointer:
            if (count >= 4) pointers[w[1]] = {w[2], w[3]};
            break;
        case op_variable:
            if (count >= 4) variables.push_back({w[1], w[2], w[3]});
            break;
        case op_decorate:
            if (count >= 3 && w[2] == decoration_non_writable) non_writable.insert(w[1]);
            if (count >= 3 && w[2] == decoration_buffer_block) buffer_blocks.insert(w[1]);
            break;
        case op_member_decorate:
            if (count >= 4 && w[3] == decoration_non_writable) non_writable_members[w[1]].insert(w[2]);
            break;
        case op_group_decorate:  // group, targets...
            for (std::uint32_t i = 2; i < count; ++i) group_targets.push_back({w[1], w[i]});
            break;
        case op_group_member_decorate:  // group, (target, member) pairs
            for (std::uint32_t i = 2; i + 1 < count; i += 2) group_members.push_back({w[1], {w[i], w[i + 1]}});
            break;
        default:
            break;
        }
        at += count;
    }

    for (const auto& [group, target] : group_targets) {
        if (non_writable.count(group)) non_writable.insert(target);
        if (buffer_blocks.count(group)) buffer_blocks.insert(target);
    }
    for (const auto& [group, member] : group_members)
        if (non_writable.count(group)) non_writable_members[member.first].insert(member.second);

    for (const Variable& variable : variables) {
        const auto pointer = pointers.find(variable.pointer_type);
        if (pointer == pointers.end()) continue;
        // An array of images or of buffer blocks: the element decides. Arrays nest a few levels at most; the bound
        // is for a malformed module whose array type names itself, which would otherwise never end.
        constexpr int max_array_depth = 16;
        std::uint32_t type = pointer->second.pointee;
        int depth = 0;
        for (auto element = element_of.find(type); element != element_of.end() && depth < max_array_depth;
             element = element_of.find(type), ++depth)
            type = element->second;
        if (depth == max_array_depth) continue;  // not a type a driver would accept either
        if (non_writable.count(variable.id)) continue;
        if (variable.storage_class == class_uniform_constant) {
            if (storage_images.count(type)) return true;  // a storage image or a storage texel buffer
            continue;
        }
        const bool storage_buffer = variable.storage_class == class_storage_buffer ||
                                    (variable.storage_class == class_uniform && buffer_blocks.count(type));
        if (!storage_buffer) continue;
        // A block is read-only only when every declared member [0, total) is NonWritable; counting the set's size
        // alone would let an out-of-range member decoration mask an undecorated writable member.
        const auto members = member_count.find(type);
        const auto read_only = non_writable_members.find(type);
        const std::uint32_t total = members == member_count.end() ? 0 : members->second;
        if (total == 0 || read_only == non_writable_members.end()) return true;
        for (std::uint32_t member = 0; member < total; ++member)
            if (!read_only->second.count(member)) return true;
    }
    return false;
}

}  // namespace leonida
