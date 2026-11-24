// A stand-in program for `gta6rt --run`: atexit handlers, static destructors and the ways a program ends.
// Test code, not game content. check_atexit.cmake runs it natively and through the loader with each mode below
// and compares the lines it prints.
//   return        main returns: handlers and static destructors run, in reverse order of registration
//   exit          exit(7): the same
//   _exit         _exit(8): no handlers, no destructors
//   quick_exit    quick_exit(9): no atexit handlers, no destructors
//   ExitProcess   ExitProcess(10): no handlers, no destructors
//   cexit         _cexit() runs every handler and returns; main then returns 11
//   c_exit        _c_exit() runs none and returns; main then returns 12, and exit runs them all
//   concurrent_exit  a second thread calls _exit(7) while exit(7) runs a handler: it waits for them all
// One handler registers another while the handlers run: it must still run, next.
#include <windows.h>
#include <process.h>  // _cexit, _c_exit

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void say(const char* what) {
    std::printf("tiny_atexit: %s\n", what);
    std::fflush(stdout);  // every line reaches the pipe before any way out, natively and under the loader
}

struct Tracer {
    const char* name;
    explicit Tracer(const char* n) : name(n) {
        std::printf("tiny_atexit: ctor %s\n", name);
        std::fflush(stdout);
    }
    ~Tracer() {
        std::printf("tiny_atexit: dtor %s\n", name);
        std::fflush(stdout);
    }
};
Tracer g_first("first");
Tracer g_second("second");

void handler_one() { say("atexit one"); }
void handler_two() { say("atexit two"); }
void handler_late() { say("atexit late"); }
void handler_registers() {
    say("atexit registers late");
    std::atexit(handler_late);
}

// concurrent_exit: while exit() runs this handler on the main thread, a second thread calls _exit(). The CRT lets
// it go once the handlers are done, so either call may set the exit code: both use 7.
HANDLE g_slow_started = nullptr;
void handler_slow() {
    say("slow handler started");
    SetEvent(g_slow_started);
    Sleep(300);
    say("slow handler done");
}
DWORD WINAPI quick_exit_thread(void*) {
    WaitForSingleObject(g_slow_started, INFINITE);
    _exit(7);
}

}  // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "return";
    std::atexit(handler_one);
    std::atexit(handler_registers);
    std::atexit(handler_two);
    say("main");
    if (!std::strcmp(mode, "concurrent_exit")) {
        g_slow_started = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        std::atexit(handler_slow);
        CloseHandle(CreateThread(nullptr, 0, quick_exit_thread, nullptr, 0, nullptr));
        std::exit(7);
    }
    if (!std::strcmp(mode, "exit")) std::exit(7);
    if (!std::strcmp(mode, "_exit")) _exit(8);
    if (!std::strcmp(mode, "quick_exit")) std::quick_exit(9);
    if (!std::strcmp(mode, "ExitProcess")) ExitProcess(10);
    if (!std::strcmp(mode, "cexit")) {
        _cexit();
        say("after _cexit");
        return 11;
    }
    if (!std::strcmp(mode, "c_exit")) {
        _c_exit();
        say("after _c_exit");
        return 12;
    }
    return 3;
}
