# XTaskQueue / XAsync (from microsoft/libHttpClient)

The GDK's task queue and async-block implementation, taken unmodified from
[microsoft/libHttpClient](https://github.com/microsoft/libHttpClient) at commit
`c5007fd9cbdc785518ea6dccc5751b125d451cc1`. **MIT licence** (see `LICENSE`), which is compatible with Leonida's
GPL-3.0-or-later.

Only what the task queue needs is copied: `Include/XAsync.h`, `Include/XAsyncProvider.h`, `Include/XTaskQueue.h`,
the Win32 sources and headers of `Source/Task/`, and `Source/Common/ResultMacros.h`. The HTTP and WebSocket stack is
not included and nothing here goes online.

libHttpClient's own `pch.h` pulls in that whole stack, so `runtime/shims/xtaskqueue/pch.h` stands in for it with
the few macros and error codes `Source/Task` uses. To update: copy the same files from a newer commit, change the
commit above and rebuild `xgameruntime_test`.
