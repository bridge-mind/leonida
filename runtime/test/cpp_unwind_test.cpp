#include <cstdio>

namespace {

struct UnwindMarker {
    bool& destroyed;
    ~UnwindMarker() { destroyed = true; }
};

struct TestException {};

// Keep the object in a separate frame, outside the catch handler's function.
__declspec(noinline) void throw_with_marker(bool& destroyed) {
    UnwindMarker marker{destroyed};
    throw TestException{};
}

}  // namespace

int main() {
    bool destroyed = false;
    try {
        throw_with_marker(destroyed);
    } catch (const TestException&) {
        if (destroyed) return 0;
        std::fputs("C++ exception unwound without running a destructor\n", stderr);
        return 1;
    }
    std::fputs("Expected a C++ exception\n", stderr);
    return 1;
}
