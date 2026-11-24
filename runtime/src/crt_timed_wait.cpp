// The timed condition-variable wait of every runtime binary built against the dynamic CRT (/MD), routed through
// msvcp140.dll so it uses the same _Mtx_t layout as the mutex it unlocks.
//
// In the 14.4x STL, condition_variable::wait_for/wait_until call _Cnd_timedwait_for_unchecked. <xthreads.h> declares
// it without _CRTIMP2_PURE, so it is not imported: it is linked statically from msvcprt.lib (its sharedmutex.obj),
// and it unlocks the mutex by reaching into Windows' _Mtx_t layout for its SRW lock. _Mtx_lock/_Mtx_unlock are
// imported from msvcp140.dll. Wine's builtin msvcp140 laid _Mtx_t out differently until wine-10.1 ("msvcp140: Fix
// field order in _Mtx_t", 757bd7909fda), so on Wine 9.0 (Ubuntu 24.04's wine) a timed wait released nothing and
// kept the mutex for its whole timeout: the thread that would set the predicate could not take the lock, and every
// wait_for timed out instead of being woken (issue #3796, measured 2000 ms vs 0.3 ms for a 2 s wait).
//
// Defining the public symbols of that msvcprt.lib member here keeps it out of the link. The timed wait then goes
// through msvcp140's exported _Cnd_timedwait, which locks and waits on its own layout, whatever DLL provides it
// (Windows, any Wine). The other definitions are the ones the member would bring in, with the same behaviour:
// std::shared_mutex on an SRW lock, and this_thread::sleep_for. Only /MD targets get this file (CMakeLists.txt);
// a /MT target links the whole STL statically, so its mutex and wait already agree. No game files.
#include <windows.h>

#include <time.h>
#include <xthreads.h>

extern "C" {

// Exported by every msvcp140.dll, Windows' and Wine's; no longer declared by the 14.4x headers. The deadline is
// absolute, in TIME_UTC seconds and nanoseconds, and the result uses _Thrd_result's values (0, or 2 on timeout).
__declspec(dllimport) int __cdecl _Cnd_timedwait(_Cnd_t, _Mtx_t, const _timespec64*) noexcept;

_Thrd_result __stdcall _Cnd_timedwait_for_unchecked(_Cnd_t cond, _Mtx_t mtx, unsigned int ms) noexcept {
    _timespec64 deadline{};
    if (_timespec64_get(&deadline, TIME_UTC) != TIME_UTC) deadline = {};
    deadline.tv_sec += ms / 1000;
    deadline.tv_nsec += static_cast<long>(ms % 1000) * 1000000L;
    if (deadline.tv_nsec >= 1000000000L) {
        deadline.tv_sec += 1;
        deadline.tv_nsec -= 1000000000L;
    }
    // A spurious or early return is allowed ("unchecked"): the STL's callers check the clock themselves.
    return static_cast<_Thrd_result>(_Cnd_timedwait(cond, mtx, &deadline));
}

// Not declared by the 14.4x headers, but defined by the same member: keep a caller of the older name off it too.
_Thrd_result __stdcall _Cnd_timedwait_for(_Cnd_t cond, _Mtx_t mtx, unsigned int ms) noexcept {
    return _Cnd_timedwait_for_unchecked(cond, mtx, ms);
}

void __cdecl _Smtx_lock_exclusive(_Smtx_t* lock) noexcept {
    AcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock));
}

void __cdecl _Smtx_lock_shared(_Smtx_t* lock) noexcept {
    AcquireSRWLockShared(reinterpret_cast<PSRWLOCK>(lock));
}

int __cdecl _Smtx_try_lock_exclusive(_Smtx_t* lock) noexcept {
    return TryAcquireSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock));
}

int __cdecl _Smtx_try_lock_shared(_Smtx_t* lock) noexcept {
    return TryAcquireSRWLockShared(reinterpret_cast<PSRWLOCK>(lock));
}

void __cdecl _Smtx_unlock_exclusive(_Smtx_t* lock) noexcept {
    ReleaseSRWLockExclusive(reinterpret_cast<PSRWLOCK>(lock));
}

void __cdecl _Smtx_unlock_shared(_Smtx_t* lock) noexcept {
    ReleaseSRWLockShared(reinterpret_cast<PSRWLOCK>(lock));
}

void __stdcall _Thrd_sleep_for(unsigned long ms) noexcept {
    Sleep(ms);
}

}  // extern "C"
