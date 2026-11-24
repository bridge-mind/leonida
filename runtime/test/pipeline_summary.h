// vk_pipeline_check's isolate summary stage set, without a Vulkan device.
#pragma once
#include <set>
#include <string>

namespace leonida {

// Every stage with a result in any of the maps. big_skipped counts: a BIG SKIP is a result too, and a stage that
// only ever skipped large modules must still appear in the parent's summary.
template <class OkMap, class BadMap, class StatsMap, class BigMap>
std::set<std::string> summary_stages(const OkMap& modules_ok, const BadMap& modules_bad, const StatsMap& stats,
                                     const BigMap& big_skipped) {
    std::set<std::string> out;
    for (const auto& entry : modules_ok) out.insert(entry.first);
    for (const auto& entry : modules_bad) out.insert(entry.first);
    for (const auto& entry : stats) out.insert(entry.first);
    for (const auto& entry : big_skipped) out.insert(entry.first);
    return out;
}

}  // namespace leonida
