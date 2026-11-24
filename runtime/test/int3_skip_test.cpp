// LEONIDA_SKIP_INT3 (shims/startup_trace.h): with the variable set and a trace open, an int3 is logged once per
// address and stepped over, so execution continues. A separate executable because the variable is read once.
#include <windows.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what);
    if (!ok) ++g_failed;
}

int count(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
    return n;
}

// noinline: the optimizer must not unroll the loop into two separate int3 sites.
__declspec(noinline) void site_a(int& reached) {
    __debugbreak();
    ++reached;
}
__declspec(noinline) void site_b(int& reached) {
    __debugbreak();
    ++reached;
}

}  // namespace

int main() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", L"1");  // before the first exception: the flag is read once
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    const std::wstring path = std::wstring(temp) + L"int3_skip_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
    DeleteFileW(path.c_str());
    check(rt::startup_trace::open(path.c_str()), "open the startup trace file");

    int reached = 0;
    for (int i = 0; i < 2; ++i) site_a(reached);  // the same site twice: one event
    site_b(reached);                              // another site
    check(reached == 3, "execution continues past every int3");
    rt::startup_trace::close();

    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    const std::string log = text.str();
    in.close();
    DeleteFileW(path.c_str());
    check(count(log, "\"kind\":\"int3_skipped\"") == 2, "exactly two int3_skipped events (one per address)");
    check(count(log, "\"kind\":\"exception\"") == 0, "a skipped int3 is not also logged as an exception");
    check(log.find("\"at\":\"host:") != std::string::npos, "the event names the address");
    std::printf("%s\n", g_failed ? "FAILED" : "all checks passed");
    return g_failed ? 1 : 0;
}
