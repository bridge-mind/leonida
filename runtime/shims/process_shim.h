// The game's command line and exit, for `gta6rt --run` (shims/process.cpp).
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace rt::process {

// Set GetCommandLine to `program` then `args`, quoted where needed, before process-attach callbacks. The wide text
// holds the arguments exactly; the narrow text uses the ANSI code page (RT-84). Reset CRT argv to argc=0/null:
// its startup configuration selects no arguments, the original arguments, or wildcard expansion for each width.
void set_command_line(const std::wstring& program, const std::vector<std::wstring>& args);

// Runs a mapped EXE's entry point on a new thread with `stack_bytes` of reserved stack and returns the game's
// exit code: the entry point's return value, or what it passed to exit/ExitProcess (3 for abort). Returns when
// the first of those happens; other game threads are left running (the host process ends them when it exits).
int run_entry(void* entry, std::size_t stack_bytes = 8u << 20);

}  // namespace rt::process
