#pragma once

#include <windows.h>

#include <string>

// Failure reporting for this tool. Every fallible call goes through one of these so a
// failed run always names the file, the call, the numeric error and the next thing to
// check. Each function returns true only when the call succeeded, so call sites branch
// on a single boolean and never need a nested if.

// Reports a failed HRESULT. Use for anything that returns HRESULT, including
// GetDeviceRemovedReason, where the value is the reason rather than a call failure.
bool HrSucceeded(HRESULT result, const char* file, int line, const char* call,
                 const char* what_to_check);

// Reports a failed Win32 call using GetLastError. Use for CreateEventW and friends.
bool Win32Succeeded(BOOL result, const char* file, int line, const char* call,
                    const char* what_to_check);

// Reports a failure that carries no error code (missing file, bad argument, a check the
// tool itself does not implement). message states what went wrong and may name a path.
void ReportFailure(const char* file, int line, const char* call, const std::string& message,
                   const char* what_to_check);
