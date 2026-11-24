// Check the exact process exit code returned by gta6rt after the synthetic guest exits.
#include <windows.h>

#include <cstdio>
#include <string>
#include <string_view>

namespace {

constexpr DWORD kProcessWaitTimeoutMs = 30000;

enum class ExitQueryStatus {
    exited,
    timed_out,
    wait_failed,
    unexpected_wait_result,
    exit_code_failed,
};

struct ExitQueryResult {
    ExitQueryStatus status;
    DWORD wait_result;
    DWORD win32_error;
    DWORD exit_code;
};

struct ProcessApi {
    DWORD(WINAPI* wait)(HANDLE, DWORD);
    BOOL(WINAPI* get_exit_code)(HANDLE, LPDWORD);
    DWORD(WINAPI* last_error)();
};

struct FakeProcessApiState {
    DWORD wait_result = WAIT_OBJECT_0;
    BOOL get_exit_code_succeeds = TRUE;
    DWORD error = ERROR_SUCCESS;
    DWORD exit_code = 42;
    unsigned wait_calls = 0;
    unsigned get_exit_code_calls = 0;
    unsigned last_error_calls = 0;
    unsigned last_operation = 0;
    unsigned last_error_observed_after = 0;
};

FakeProcessApiState g_fake_api_state;

DWORD WINAPI fake_wait(HANDLE, DWORD) {
    ++g_fake_api_state.wait_calls;
    g_fake_api_state.last_operation = 1;
    if (g_fake_api_state.wait_result == WAIT_FAILED) {
        g_fake_api_state.error = ERROR_INVALID_HANDLE;
    }
    return g_fake_api_state.wait_result;
}

BOOL WINAPI fake_get_exit_code(HANDLE, LPDWORD exit_code) {
    ++g_fake_api_state.get_exit_code_calls;
    g_fake_api_state.last_operation = 2;
    if (!g_fake_api_state.get_exit_code_succeeds) {
        g_fake_api_state.error = ERROR_ACCESS_DENIED;
        return FALSE;
    }
    *exit_code = g_fake_api_state.exit_code;
    return TRUE;
}

DWORD WINAPI fake_last_error() {
    ++g_fake_api_state.last_error_calls;
    g_fake_api_state.last_error_observed_after = g_fake_api_state.last_operation;
    return g_fake_api_state.error;
}

ExitQueryResult query_exit_code(HANDLE process, const ProcessApi& api) {
    const DWORD wait_result = api.wait(process, kProcessWaitTimeoutMs);
    if (wait_result == WAIT_TIMEOUT) {
        return {ExitQueryStatus::timed_out, wait_result, ERROR_SUCCESS, 0};
    }
    if (wait_result == WAIT_FAILED) {
        return {ExitQueryStatus::wait_failed, wait_result, api.last_error(), 0};
    }
    if (wait_result != WAIT_OBJECT_0) {
        return {ExitQueryStatus::unexpected_wait_result, wait_result, ERROR_SUCCESS, 0};
    }

    DWORD exit_code = 0;
    if (!api.get_exit_code(process, &exit_code)) {
        return {ExitQueryStatus::exit_code_failed, wait_result, api.last_error(), 0};
    }
    return {ExitQueryStatus::exited, wait_result, ERROR_SUCCESS, exit_code};
}

std::string format_exit_query_failure(const ExitQueryResult& result) {
    char message[160]{};
    switch (result.status) {
    case ExitQueryStatus::timed_out:
        std::snprintf(message, sizeof(message), "timed out waiting for gta6rt after %lu ms (wait result %#lx)\n",
                      kProcessWaitTimeoutMs, result.wait_result);
        break;
    case ExitQueryStatus::wait_failed:
        std::snprintf(message, sizeof(message), "WaitForSingleObject failed with Win32 error %lu (wait result %#lx)\n",
                      result.win32_error, result.wait_result);
        break;
    case ExitQueryStatus::exit_code_failed:
        std::snprintf(message, sizeof(message), "GetExitCodeProcess failed with Win32 error %lu after wait result %#lx\n",
                      result.win32_error, result.wait_result);
        break;
    case ExitQueryStatus::unexpected_wait_result:
        std::snprintf(message, sizeof(message), "WaitForSingleObject returned unexpected wait result %#lx\n",
                      result.wait_result);
        break;
    case ExitQueryStatus::exited:
        break;
    }
    return message;
}

bool expect(bool condition, std::string_view description) {
    if (condition) {
        return true;
    }
    std::fprintf(stderr, "exit-code diagnostic self-test failed: %.*s\n", static_cast<int>(description.size()),
                 description.data());
    return false;
}

void reset_fake_api() {
    g_fake_api_state = {};
}

bool run_diagnostic_self_tests() {
    bool passed = true;
    const ProcessApi fake_api{fake_wait, fake_get_exit_code, fake_last_error};

    reset_fake_api();
    g_fake_api_state.wait_result = WAIT_FAILED;
    ExitQueryResult result = query_exit_code(nullptr, fake_api);
    passed &= expect(result.status == ExitQueryStatus::wait_failed, "WAIT_FAILED is classified as a wait failure");
    passed &= expect(result.win32_error == ERROR_INVALID_HANDLE, "wait failure preserves GetLastError");
    passed &= expect(g_fake_api_state.get_exit_code_calls == 0, "exit code is not queried after a wait failure");
    passed &= expect(g_fake_api_state.last_error_calls == 1 && g_fake_api_state.last_error_observed_after == 1,
                     "wait error is captured immediately after WaitForSingleObject");
    passed &= expect(format_exit_query_failure(result) ==
                         "WaitForSingleObject failed with Win32 error 6 (wait result 0xffffffff)\n",
                     "wait failure diagnostic names the failing API and error");

    reset_fake_api();
    g_fake_api_state.get_exit_code_succeeds = FALSE;
    result = query_exit_code(nullptr, fake_api);
    passed &= expect(result.status == ExitQueryStatus::exit_code_failed,
                     "GetExitCodeProcess failure is classified separately");
    passed &= expect(result.win32_error == ERROR_ACCESS_DENIED, "exit code failure preserves GetLastError");
    passed &= expect(g_fake_api_state.last_error_calls == 1 && g_fake_api_state.last_error_observed_after == 2,
                     "exit code error is captured immediately after GetExitCodeProcess");
    passed &= expect(format_exit_query_failure(result) ==
                         "GetExitCodeProcess failed with Win32 error 5 after wait result 0\n",
                     "exit code failure diagnostic names the failing API and error");

    reset_fake_api();
    g_fake_api_state.wait_result = WAIT_TIMEOUT;
    result = query_exit_code(nullptr, fake_api);
    passed &= expect(result.status == ExitQueryStatus::timed_out, "timeout remains distinct from API failure");
    passed &= expect(g_fake_api_state.get_exit_code_calls == 0 && g_fake_api_state.last_error_calls == 0,
                     "timeout neither queries the exit code nor reports a stale Win32 error");
    passed &= expect(format_exit_query_failure(result) ==
                         "timed out waiting for gta6rt after 30000 ms (wait result 0x102)\n",
                     "timeout diagnostic retains its duration and wait result");

    reset_fake_api();
    g_fake_api_state.wait_result = WAIT_ABANDONED;
    result = query_exit_code(nullptr, fake_api);
    passed &= expect(result.status == ExitQueryStatus::unexpected_wait_result,
                     "unexpected wait results are not reported as API failures");
    passed &= expect(g_fake_api_state.get_exit_code_calls == 0 && g_fake_api_state.last_error_calls == 0,
                     "unexpected wait results do not query the exit code or report a stale Win32 error");
    passed &= expect(format_exit_query_failure(result) == "WaitForSingleObject returned unexpected wait result 0x80\n",
                     "unexpected wait diagnostic includes the returned value");

    reset_fake_api();
    result = query_exit_code(nullptr, fake_api);
    passed &= expect(result.status == ExitQueryStatus::exited && result.exit_code == 42,
                     "successful wait returns the process exit code");
    passed &= expect(g_fake_api_state.last_error_calls == 0, "successful query does not read a stale Win32 error");
    return passed;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
    if (!run_diagnostic_self_tests()) {
        return 1;
    }
    if (argc == 2 && std::wstring_view(argv[1]) == L"--self-test") {
        return 0;
    }
    if (argc != 3) {
        std::fprintf(stderr, "usage: tiny_game_exit_code_test <gta6rt.exe> <tiny_game.exe>\n");
        return 1;
    }

    std::wstring command = L"\"" + std::wstring(argv[1]) + L"\" --run \"" + argv[2] + L"\" -- hello 42 exit-code";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    if (!CreateProcessW(argv[1], command.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &child)) {
        std::fprintf(stderr, "could not start gta6rt (Win32 error %lu)\n", GetLastError());
        return 1;
    }
    CloseHandle(child.hThread);

    const ProcessApi process_api{WaitForSingleObject, GetExitCodeProcess, GetLastError};
    const ExitQueryResult result = query_exit_code(child.hProcess, process_api);
    if (result.status == ExitQueryStatus::timed_out) {
        TerminateProcess(child.hProcess, 1);
    }
    CloseHandle(child.hProcess);
    if (result.status != ExitQueryStatus::exited) {
        const std::string message = format_exit_query_failure(result);
        std::fputs(message.c_str(), stderr);
        return 1;
    }
    if (result.exit_code != 42) {
        std::fprintf(stderr, "gta6rt exited with code %lu, expected guest exit code 42\n", result.exit_code);
        return 1;
    }
    std::printf("gta6rt returned the expected guest exit code 42\n");
    return 0;
}
