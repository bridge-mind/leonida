#include "diagnostics.h"

#include <cstdio>
#include <string>

namespace {

// One line shape for every failure so a log grep for "FAIL " finds all of them.
void PrintDiagnostic(const char* file, int line, const char* call, const std::string& detail,
                     const char* what_to_check) {
    std::fprintf(stderr, "FAIL %s:%d %s -> %s. Check: %s\n", file, line, call, detail.c_str(),
                 what_to_check);
    std::fflush(stderr);
}

// Hex is how HRESULTs are looked up in the SDK headers and in Event Viewer.
std::string FormatHex32(unsigned long value, const char* prefix) {
    char text[32];
    std::snprintf(text, sizeof text, "%s0x%08lX", prefix, value);
    return std::string(text);
}

}  // namespace

bool HrSucceeded(HRESULT result, const char* file, int line, const char* call,
                 const char* what_to_check) {
    if (SUCCEEDED(result)) {
        return true;
    }
    PrintDiagnostic(file, line, call, FormatHex32(static_cast<unsigned long>(result), "HRESULT "),
                    what_to_check);
    return false;
}

bool Win32Succeeded(BOOL result, const char* file, int line, const char* call,
                    const char* what_to_check) {
    if (result != FALSE) {
        return true;
    }
    const unsigned long last_error = static_cast<unsigned long>(GetLastError());
    char detail[64];
    std::snprintf(detail, sizeof detail, "Win32 error %lu (0x%08lX)", last_error, last_error);
    PrintDiagnostic(file, line, call, detail, what_to_check);
    return false;
}

void ReportFailure(const char* file, int line, const char* call, const std::string& message,
                   const char* what_to_check) {
    PrintDiagnostic(file, line, call, message, what_to_check);
}
