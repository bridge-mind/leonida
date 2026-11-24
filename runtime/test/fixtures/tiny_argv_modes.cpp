// Original fixture linked with the real MSVC default, setargv/wsetargv, and noarg startup options.
// The optional sequence checks later configuration and no-op pointer identity in the same process.
#include <windows.h>
#include <corecrt_startup.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace {
template<class C> void dump(C** values) {
    if (!values) { std::printf("null"); return; }
    for (size_t i = 0; values[i]; ++i) {
        std::printf("[");
        for (const C* p = values[i]; *p; ++p) {
            using U = std::make_unsigned_t<C>;
            std::printf(sizeof(C) == 1 ? "%02X" : "%04X", static_cast<unsigned>(static_cast<U>(*p)));
        }
        std::printf("]");
    }
}
void snapshot(const char* label) {
    std::printf("STATE %s argc=%d A=", label, *__p___argc());
    dump(*__p___argv());
    std::printf(" W=");
    dump(*__p___wargv());
    std::puts("");
}
void no_op(bool wide) {
    const int argc = *__p___argc();
    char** a = *__p___argv();
    wchar_t** w = *__p___wargv();
    const int status = wide ? _configure_wide_argv(_crt_argv_no_arguments)
                            : _configure_narrow_argv(_crt_argv_no_arguments);
    std::printf("NOOP %c status=%d unchanged=%u\n", wide ? 'W' : 'A', status,
                static_cast<unsigned>(argc == *__p___argc() && a == *__p___argv() && w == *__p___wargv()));
}
int body() {
    snapshot("entry");
    if (GetEnvironmentVariableW(L"LEONIDA_ARGV_SEQUENCE", nullptr, 0)) {
        no_op(false); no_op(true); snapshot("initial-noops");
        std::printf("CONFIG A2 status=%d\n", _configure_narrow_argv(_crt_argv_expanded_arguments));
        snapshot("A2"); no_op(false); snapshot("A2-noop");
        std::printf("CONFIG W1 status=%d\n", _configure_wide_argv(_crt_argv_unexpanded_arguments));
        snapshot("W1"); no_op(true); snapshot("W1-noop");
        std::printf("CONFIG A1 status=%d\n", _configure_narrow_argv(_crt_argv_unexpanded_arguments));
        snapshot("A1");
        std::printf("CONFIG W2 status=%d\n", _configure_wide_argv(_crt_argv_expanded_arguments));
        snapshot("W2"); no_op(false); no_op(true); snapshot("final-noops");
    }
    return 0;
}
}
#ifdef APP_WIDE
int wmain(int argc, wchar_t** argv) {
    if (argc != *__p___argc() || argv != *__p___wargv()) return 3;
    return body();
}
#else
int main(int argc, char** argv) {
    if (argc != *__p___argc() || argv != *__p___argv()) return 3;
    return body();
}
#endif
