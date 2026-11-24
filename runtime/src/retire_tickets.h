#pragma once

#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

namespace leonida {

// Global submission tickets may finish out of order on independent Vulkan queues. Only the
// contiguous completed prefix is safe for deferred resource destruction. The caller serializes
// record(), completed() and queue_idle() with the device submission mutex. record() must
// receive strictly increasing global ticket numbers; otherwise a duplicate or backward ticket
// can make the completed prefix stall or move backward.
template<class Semaphore>
class RetireTickets {
public:
    void record(std::uint64_t ticket, Semaphore semaphore, std::uint64_t value) {
        pending_.push_back({ticket, semaphore, value, false});
    }

    template<class Query>
    std::uint64_t completed(Query&& query) {
        while (!pending_.empty()) {
            const Entry& front = pending_.front();
            if (!front.idle && !query(front.semaphore, front.value)) break;
            completed_ = front.ticket;
            pending_.pop_front();
        }
        return completed_;
    }

    // The (semaphore, value) of every submission up to `ticket` that has not been seen complete: what a caller
    // that must have them done (and may not hold the lock that guards this object while it waits) waits for.
    std::vector<std::pair<Semaphore, std::uint64_t>> pending_through(std::uint64_t ticket) const {
        std::vector<std::pair<Semaphore, std::uint64_t>> out;
        for (const Entry& entry : pending_)
            if (entry.ticket <= ticket && !entry.idle) out.emplace_back(entry.semaphore, entry.value);
        return out;
    }

    // Call only after the owning queue's wait-idle succeeded. Merely requesting idle does not
    // prove its submissions completed and must not advance the retirement prefix.
    void queue_idle(Semaphore semaphore) {
        for (Entry& entry : pending_)
            if (entry.semaphore == semaphore) entry.idle = true;
    }

private:
    struct Entry {
        std::uint64_t ticket;
        Semaphore semaphore;
        std::uint64_t value;
        bool idle;
    };
    std::deque<Entry> pending_;
    std::uint64_t completed_ = 0;
};

}  // namespace leonida
