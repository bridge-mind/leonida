// vk_pipeline_check's summary stage set (pipeline_summary.h), without a Vulkan device.
#include "pipeline_summary.h"

#include <cstdio>
#include <map>
#include <set>
#include <string>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

int main() {
    using Counts = std::map<std::string, unsigned>;
    const Counts ok{{"cs", 3}};
    const Counts bad{{"ps", 1}};
    const std::map<std::string, int> stats{{"vs", 1}};
    const Counts big{{"ms", 2}};
    const std::set<std::string> all{"cs", "ps", "vs", "ms"};
    CHECK(leonida::summary_stages(ok, bad, stats, big) == all);

    // A stage that exists only in big_skipped is the regression this pins: its BIG SKIP lines must put it in the
    // parent's summary even though no module, pipeline or statistic was recorded for it.
    const std::set<std::string> only_big{"gs"};
    CHECK(leonida::summary_stages(Counts{}, Counts{}, std::map<std::string, int>{}, Counts{{"gs", 1}}) == only_big);

    std::puts("pipeline_summary: ok");
    return 0;
}
