#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace leonida {

// Cross-queue Wait-before-Signal (docs section 4.4, "Wait before Signal across queues").
//
// A queue replays its ExecuteCommandLists against one device-wide layout tracker, in the order the calls reach the
// layer. When queue B calls Wait(fence, v) and the Signal(fence, v) on queue A is only submitted afterwards, B's
// lists would be replayed against the layouts from before A's work. So a queue holds a batch (and everything it
// is asked to do after it) while any fence it waits on has no submitted Signal of at least that value, and
// releases it, in order, once one has been submitted: the layouts then follow the dependency order, while Vulkan's
// own timeline semaphores still order the GPU work.
//
// This class is the device-wide half: for every fence, the highest value a Signal has been submitted for (or the
// CPU has set it to), which is not the completed value; the held queues, which it asks to release whatever became
// ready; and "after held work" callbacks (the presenter), which run once every operation held when they were
// registered has been released. It owns no Vulkan objects. Semaphores it was never told about count as satisfied,
// so a queue used with a plain VkSemaphore behaves as before.
//
// Threads. No GPU wait happens here, but the thread that makes a wait satisfiable (a queue's Signal, a CPU Signal)
// does the release itself, after it has dropped its own queue's state lock: it records and submits the held
// batches (a held ExecuteCommandLists costs what it would have cost when called, which includes whatever
// execute_now does: the bindless upload's wait for the device is one) and runs the callbacks. The COM wrapper
// still holds the mutex of the queue whose call that thread is in, so that queue's other calls wait meanwhile.
// One thread releases at a time; a thread that finds another one releasing leaves a note and that one takes
// another pass. detach() and cancel_after() (and remove_fence() from outside a release) wait for a running
// release.
template<class Semaphore>
class QueueHolds {
public:
    // A queue that can hold batches. The calls take the queue's own lock; none is made with it held.
    struct Holder {
        // Releases the held operations at the front of the queue whose waits are satisfied. True if any was.
        virtual bool release_ready() = 0;
        // Drops the held operations that wait on `semaphore` (its fence is gone), reporting each.
        virtual void discard_waiting_on(Semaphore semaphore) = 0;
        // Sequence number (next_seq()) of the oldest held operation, 0 when nothing is held.
        virtual std::uint64_t oldest_held_seq() = 0;

    protected:
        ~Holder() = default;
    };

    void add_fence(Semaphore semaphore, std::uint64_t initial) {
        std::lock_guard lock(mutex_);
        fences_[semaphore] = initial;
    }

    // The fence is gone. Operations that still reference it are dropped and reported (the COM layer keeps a fence
    // alive while a held operation names it, so this is a safety net for direct users of VulkanFence). Not done
    // from inside a release: the releasing thread already holds the lock, and what it is releasing cannot
    // reference a fence whose last reference it is dropping, since that reference is the one it holds.
    void remove_fence(Semaphore semaphore) {
        {
            std::lock_guard lock(mutex_);
            fences_.erase(semaphore);
        }
        if (held_.load(std::memory_order_acquire) == 0 || in_pump()) return;
        {
            std::lock_guard pump(pump_mutex_);
            in_pump() = true;  // what is dropped may release other fences: they must not wait for this lock
            for (Holder* holder : snapshot()) holder->discard_waiting_on(semaphore);
            in_pump() = false;
        }
        pump_ready();
    }

    // A Signal of `value` was submitted to the GPU (or the CPU set the fence). Call after the submission, with no
    // queue lock held; then pump_ready() to release what it satisfied.
    void signaled(Semaphore semaphore, std::uint64_t value) {
        std::lock_guard lock(mutex_);
        const auto it = fences_.find(semaphore);
        if (it != fences_.end() && it->second < value) it->second = value;
    }

    // A queue is about to submit a Signal of `value`; call inside the device submit lock, and record the value with
    // signaled() only once the vkQueueSubmit succeeded (a failed submit signals nothing: waiters must stay held and a
    // retry must go through). False when a Signal of at least `value` was already submitted: the timeline cannot go
    // down or stay (VUID-VkSubmitInfo-pSignalSemaphores-03242 and the timeline rules), and a wait for `value` is
    // already satisfied, so the Vulkan signal is left out. This happens when a held Signal(F, 2) is released after an
    // unheld Signal(F, 3). Unknown semaphores are always signaled.
    bool would_signal(Semaphore semaphore, std::uint64_t value) const {
        std::lock_guard lock(mutex_);
        const auto it = fences_.find(semaphore);
        return it == fences_.end() || it->second < value;
    }

    // true when no submitted Signal reaches `value` yet. Unknown semaphores are satisfied.
    bool pending(Semaphore semaphore, std::uint64_t value) const {
        std::lock_guard lock(mutex_);
        const auto it = fences_.find(semaphore);
        return it != fences_.end() && it->second < value;
    }

    void attach(Holder* holder) {
        std::lock_guard lock(mutex_);
        holders_.push_back(holder);
    }

    // Returns once no thread is using `holder`.
    void detach(Holder* holder) {
        {
            std::lock_guard pump(pump_mutex_);
            std::lock_guard lock(mutex_);
            for (auto it = holders_.begin(); it != holders_.end(); ++it)
                if (*it == holder) {
                    holders_.erase(it);
                    break;
                }
        }
        pump_ready();
    }

    // Sequence numbers of held operations, in the order they were held (across queues).
    std::uint64_t next_seq() { return seq_.fetch_add(1, std::memory_order_acq_rel) + 1; }
    std::uint64_t current_seq() const { return seq_.load(std::memory_order_acquire); }

    // Runs `fn` once every operation held up to now has been released (or dropped), on the thread that releases the
    // last of them, or right here if none is held. Callbacks of one owner run in registration order. Used by the
    // presenter: a Present queued behind held work shows what that work draws.
    void after_held(const void* owner, std::function<void()> fn) {
        {
            std::lock_guard lock(mutex_);
            barriers_.push_back({owner, current_seq(), std::move(fn)});
            barrier_count_.fetch_add(1, std::memory_order_acq_rel);
        }
        pump_ready();
    }

    // Drops the callbacks of `owner` that have not run, and waits for one that is running.
    void cancel_after(const void* owner) {
        {
            std::lock_guard pump(pump_mutex_);
            std::lock_guard lock(mutex_);
            for (auto it = barriers_.begin(); it != barriers_.end();)
                if (it->owner == owner) {
                    it = barriers_.erase(it);
                    barrier_count_.fetch_sub(1, std::memory_order_acq_rel);
                } else {
                    ++it;
                }
        }
        pump_ready();
    }

    // Held operations over all queues, for the cheap "is there anything to do" test.
    void held_changed(std::int64_t delta) { held_.fetch_add(delta, std::memory_order_acq_rel); }
    bool any_held() const { return held_.load(std::memory_order_acquire) > 0; }

    // Releases every held operation that has become ready, in dependency order: a release can submit a Signal
    // that makes another queue's front ready, so passes repeat until one finds nothing new. Then the callbacks
    // whose held work is all gone.
    void pump_ready() {
        if (!any_held() && barrier_count_.load(std::memory_order_acquire) == 0) return;
        dirty_.store(true, std::memory_order_release);
        if (in_pump()) return;  // a release on this thread submitted a Signal: the running pass will go again
        for (;;) {
            std::unique_lock pump(pump_mutex_, std::try_to_lock);
            if (!pump) return;  // another thread is releasing and will see dirty_
            in_pump() = true;
            while (dirty_.exchange(false, std::memory_order_acq_rel)) {
                bool progress = true;
                while (progress) {
                    progress = false;
                    for (Holder* holder : snapshot()) progress |= holder->release_ready();
                }
                run_barriers();
            }
            in_pump() = false;
            pump.unlock();
            if (!dirty_.load(std::memory_order_acquire)) return;
        }
    }

private:
    struct Barrier {
        const void* owner;
        std::uint64_t seq;
        std::function<void()> fn;
    };

    static bool& in_pump() {
        static thread_local bool pumping = false;
        return pumping;
    }

    std::vector<Holder*> snapshot() const {
        std::lock_guard lock(mutex_);
        return holders_;
    }

    // Called with pump_mutex_ held. A callback runs when no held operation is older than the moment it was asked.
    void run_barriers() {
        for (;;) {
            Barrier front;
            {
                std::lock_guard lock(mutex_);
                if (barriers_.empty()) return;
                front = barriers_.front();
            }
            for (Holder* holder : snapshot()) {
                const std::uint64_t oldest = holder->oldest_held_seq();
                if (oldest && oldest <= front.seq) return;
            }
            {
                std::lock_guard lock(mutex_);
                if (barriers_.empty() || barriers_.front().seq != front.seq || barriers_.front().owner != front.owner)
                    continue;  // cancelled meanwhile
                barriers_.pop_front();
                barrier_count_.fetch_sub(1, std::memory_order_acq_rel);
            }
            front.fn();
        }
    }

    mutable std::mutex mutex_;  // leaf: fences_, holders_ and barriers_
    std::mutex pump_mutex_;     // one releasing thread; held across release_ready(), so detach() waits for it
    std::unordered_map<Semaphore, std::uint64_t> fences_;
    std::vector<Holder*> holders_;
    std::deque<Barrier> barriers_;
    std::atomic<std::int64_t> held_{0};
    std::atomic<std::int64_t> barrier_count_{0};
    std::atomic<std::uint64_t> seq_{0};
    std::atomic<bool> dirty_{false};
};

}  // namespace leonida
