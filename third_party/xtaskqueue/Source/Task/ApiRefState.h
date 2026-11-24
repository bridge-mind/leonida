// Copyright (c) Microsoft Corporation
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

#pragma once

#include <atomic>
#include <cstdint>
#include <exception>

// Atomic reference count with an irreversible rundown marker.
// The final Release atomically sets the dying bit.
class ApiRefState final
{
public:
    struct ReleaseResult
    {
        uint32_t remaining;
        bool startRundown;
    };

    // AddRef is also used internally during rundown.
    uint32_t AddRef() noexcept
    {
        uint32_t current = m_state.load();

        for (;;)
        {
            uint32_t refs = current & kRefMask;

            if (refs == kRefMask)
            {
                std::terminate();
            }

            if (m_state.compare_exchange_weak(current, current + 1))
            {
                return refs;
            }
        }
    }

    // Never acquire a new external reference after rundown starts.
    bool TryAddRef() noexcept
    {
        uint32_t current = m_state.load();

        for (;;)
        {
            uint32_t refs = current & kRefMask;

            if ((current & kDyingBit) != 0 ||
                refs == 0 ||
                refs == kRefMask)
            {
                return false;
            }

            if (m_state.compare_exchange_weak(current, current + 1))
            {
                return true;
            }
        }
    }

    ReleaseResult Release() noexcept
    {
        uint32_t current = m_state.load();

        for (;;)
        {
            uint32_t refs = current & kRefMask;

            if (refs == 0)
            {
                std::terminate();
            }

            const bool startRundown =
                refs == 1 && (current & kDyingBit) == 0;

            uint32_t desired =
                (current & kDyingBit) | (refs - 1);

            if (startRundown)
            {
                desired |= kDyingBit;
            }

            if (m_state.compare_exchange_weak(current, desired))
            {
                return { refs - 1, startRundown };
            }
        }
    }

    bool IsDying() const noexcept
    {
        return (m_state.load() & kDyingBit) != 0;
    }

    uint32_t RefCount() const noexcept
    {
        return m_state.load() & kRefMask;
    }

private:
    static constexpr uint32_t kDyingBit = 0x80000000u;
    static constexpr uint32_t kRefMask = 0x7FFFFFFFu;

    std::atomic<uint32_t> m_state{ 0 };
};
