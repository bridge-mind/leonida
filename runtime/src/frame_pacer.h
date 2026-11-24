#pragma once

// Frame pacing for the Xbox presentation path (docs/d3d12_vulkan.md section 4.5): what SetFrameIntervalX,
// ScheduleFrameEventX, WaitFrameEventX and GetFrameStatisticsX do, and the token check and flip bookkeeping that
// PresentX needs. No Vulkan and no Windows headers: time comes from an injectable FrameClock, so the tests drive it
// with a fake clock and the real runtime uses a steady high-resolution timer (a host vblank source can replace it).
//
// What is known and what is ours. From the public GDKX samples (section 4.5): the call order, the argument counts, that
// the interval call takes an allowed frame latency (buffer count - 1 in one sample, 1 in another), that the origin
// event is scheduled once and waited on every frame, that the wait yields a token that PresentX carries, and that
// frame events must be registered again after ResumeX. Everything below the call shapes is our own model, chosen to
// be deterministic and to behave like the samples expect; none of it is measured on the console:
//   * An interval is a period in microseconds (the real enum values are unmeasured).
//   * Vblanks happen at epoch + n * period (n = 1, 2, ...), the epoch being the time of SetFrameIntervalX. The
//     first origin after ScheduleFrameEventX is the next vblank. The origin event of vblank n is signalled
//     offset microseconds after it (ScheduleFrameEventX's offset).
//   * A frame is "in flight" from PresentX until the vblank that shows it. Frames show one per vblank, in order.
//     WaitFrameEventX does not return while `latency` frames are in flight. A frame presented while the origin event
//     is not armed is not paced (PresentX does not hold it for a vblank): it shows at once and is never in flight.
//   * A wait that comes late (after the next origin already passed) returns at once with the newest passed origin;
//     the missed origins are skipped, not queued.
//   * Tokens are 1, 2, 3, ... (0 is the null token), one per successful wait, never reused, across suspend/resume.
//   * A frame is due at the vblank after the origin that handed out its token; a PresentX after that vblank is late by
//     the time since it (present_timing), and ImmediateThresholdPercent decides whether that lateness is shown at once
//     (decide_present_sync).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>

namespace leonida {

// HRESULT values as plain numbers, so this header stands alone.
namespace pacer_hr {
inline constexpr std::int32_t ok = 0;
inline constexpr std::int32_t false_ = 1;                                  // S_FALSE
inline constexpr std::int32_t invalid_arg = std::int32_t(0x80070057);      // E_INVALIDARG
inline constexpr std::int32_t invalid_state = std::int32_t(0x8007139F);    // HRESULT_FROM_WIN32(ERROR_INVALID_STATE)
inline constexpr std::int32_t timeout = std::int32_t(0x80070102);          // HRESULT_FROM_WIN32(WAIT_TIMEOUT)
}  // namespace pacer_hr

inline constexpr std::uint64_t no_deadline = std::numeric_limits<std::uint64_t>::max();
inline constexpr std::uint32_t infinite_timeout_ms = 0xFFFFFFFFu;  // INFINITE

class FrameClock {
public:
    virtual ~FrameClock() = default;
    virtual std::uint64_t now_us() = 0;
    // Called with `lock` held on the pacer's mutex. Returns when `cv` is notified, `deadline_us` (in now_us() time;
    // no_deadline for none) has passed, or spuriously; the caller re-checks its condition.
    virtual void wait(std::unique_lock<std::mutex>& lock, std::condition_variable& cv, std::uint64_t deadline_us) = 0;
    // The pacer gives a function that takes its mutex and notifies; a fake clock calls it when time moves.
    virtual void set_wake(std::function<void()> wake) { (void)wake; }
};

// Real time: std::chrono::steady_clock (QueryPerformanceCounter on Windows).
class SteadyFrameClock final : public FrameClock {
public:
    std::uint64_t now_us() override {
        return std::uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::steady_clock::now() - start_).count());
    }
    void wait(std::unique_lock<std::mutex>& lock, std::condition_variable& cv, std::uint64_t deadline_us) override {
        if (deadline_us == no_deadline) cv.wait(lock);
        else cv.wait_until(lock, start_ + std::chrono::microseconds(deadline_us));
    }

private:
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
};

// Time that only moves when the test says so. A pacer waiting on it blocks until advance_us() is called.
class ManualFrameClock final : public FrameClock {
public:
    std::uint64_t now_us() override { return now_.load(); }
    void wait(std::unique_lock<std::mutex>& lock, std::condition_variable& cv, std::uint64_t) override { cv.wait(lock); }
    void set_wake(std::function<void()> wake) override {
        std::lock_guard guard(wake_mutex_);
        wake_ = std::move(wake);
    }
    void advance_us(std::uint64_t microseconds) {
        now_ += microseconds;
        // Called under the lock, so a pacer that is being destroyed (set_wake({})) is never woken after it is gone.
        std::lock_guard guard(wake_mutex_);
        if (wake_) wake_();
    }

private:
    std::atomic<std::uint64_t> now_{0};
    std::mutex wake_mutex_;
    std::function<void()> wake_;
};

// Provisional definitions of ours: the real constants and layouts of the console API are unmeasured.
inline constexpr std::uint32_t frame_event_origin = 0;
// Statistics types (bit mask). Interval and render entries are never produced here.
inline constexpr std::uint32_t stat_frame_interval = 0x01;
inline constexpr std::uint32_t stat_frame_event = 0x02;
inline constexpr std::uint32_t stat_render = 0x04;
inline constexpr std::uint32_t stat_present = 0x08;
inline constexpr std::uint32_t stat_display = 0x10;

// How a frame that is being presented is put on the display.
enum class PresentSync {
    vblank,     // at the next vblank (FIFO)
    immediate,  // now, mid-scan (a tear)
};

// D3D12XBOX_PRESENT_PARAMETERS::ImmediateThresholdPercent (docs/d3d12_vulkan.md section 4.5): a frame that missed its
// vblank is flipped at once, mid-scan, when the scan has not got further than `threshold_percent` of the screen,
// instead of waiting a whole interval. `late_by_us` is how long after the vblank the frame was ready (0: on time),
// `interval_us` the refresh period; the vblank grid is regular, so the scan is `late_by_us % interval_us` of the way
// down. 0 (or a negative or NaN value) never tears, 100 or more tears wherever the scan is. A frame on time, or an
// interval of 0 (unpaced), is never torn. Pure, so the tests can sweep it.
inline PresentSync decide_present_sync(std::uint64_t late_by_us, std::uint32_t interval_us, float threshold_percent) {
    if (!(threshold_percent > 0.0f) || late_by_us == 0 || interval_us == 0) return PresentSync::vblank;
    const double scanned_percent = 100.0 * double(late_by_us % interval_us) / double(interval_us);
    return scanned_percent <= double(std::min(threshold_percent, 100.0f)) ? PresentSync::immediate
                                                                          : PresentSync::vblank;
}

// Where the scan of a frame that was shown mid-scan stood, in percent (0 for a frame shown at a vblank).
inline float scan_percent(std::uint64_t late_by_us, std::uint32_t interval_us) {
    return interval_us ? float(100.0 * double(late_by_us % interval_us) / double(interval_us)) : 0.0f;
}

struct FrameStatistic {
    std::uint32_t type = 0;          // one stat_* bit
    std::uint32_t plane_index = 0;
    std::uint64_t token = 0;
    std::uint64_t time_us = 0;       // pacer clock
    float percent_scanned = 0.0f;    // display entries: where the scan was when the frame came up (0 at a vblank)
};

class FramePacer {
public:
    explicit FramePacer(FrameClock& clock) : clock_(clock) {
        clock_.set_wake([this] {
            { std::lock_guard guard(mutex_); }
            cv_.notify_all();
        });
    }
    ~FramePacer() {
        clock_.set_wake({});
        {
            std::lock_guard guard(mutex_);
            shutdown_ = true;
        }
        cv_.notify_all();
        std::unique_lock lock(mutex_);  // waiters leave before the members go
        cv_.wait(lock, [this] { return waiters_ == 0; });
    }
    FramePacer(const FramePacer&) = delete;
    FramePacer& operator=(const FramePacer&) = delete;

    // SetFrameIntervalX(output, interval, latency, flags). `period_us` is the interval as a period; `latency` is
    // the number of frames allowed in flight (0 counts as 1). The vblank grid is re-based: its epoch is now. A
    // scheduled origin event stays armed (with its offset, kept below the new period) and its next origin is one
    // new period from now; waiters blocked on the old grid pick the new one up.
    std::int32_t set_frame_interval(std::uint32_t period_us, std::uint32_t latency, std::uint32_t flags) {
        (void)flags;
        if (period_us == 0) return pacer_hr::invalid_arg;
        std::lock_guard guard(mutex_);
        if (suspended_) return pacer_hr::invalid_state;
        period_us_ = period_us;
        latency_ = std::max(latency, 1u);
        epoch_us_ = clock_.now_us();
        interval_set_ = true;
        offset_us_ = std::min(offset_us_, period_us - 1);
        last_origin_ = 0;  // origin 0 is the epoch itself, which has passed
        cv_.notify_all();
        return pacer_hr::ok;
    }

    // ScheduleFrameEventX(event, offset, ..., flags). Only the origin event exists here.
    std::int32_t schedule_frame_event(std::uint32_t event, std::uint32_t offset_us, std::uint32_t flags) {
        (void)flags;
        if (event != frame_event_origin) return pacer_hr::invalid_arg;
        std::lock_guard guard(mutex_);
        if (suspended_ || !interval_set_) return pacer_hr::invalid_state;
        if (offset_us >= period_us_) return pacer_hr::invalid_arg;
        offset_us_ = offset_us;
        armed_ = true;
        // The first origin is the next one that has not come yet: origin n is due at epoch + n * period + offset.
        const std::uint64_t now = clock_.now_us();
        last_origin_ = now >= epoch_us_ + offset_us_ ? std::int64_t((now - epoch_us_ - offset_us_) / period_us_) : 0;
        cv_.notify_all();
        return pacer_hr::ok;
    }

    // WaitFrameEventX(event, timeout, ..., flags, &token). Blocks until the next origin, and while `latency` frames
    // are in flight. pacer_hr::timeout when `timeout_ms` passes first (infinite_timeout_ms: never); the token is left
    // alone then. invalid_state when nothing is registered, the pacer is suspended, or one of those happens while waiting.
    std::int32_t wait_frame_event(std::uint32_t event, std::uint32_t timeout_ms, std::uint32_t flags,
                                  std::uint64_t* token) {
        (void)flags;
        if (event != frame_event_origin || !token) return pacer_hr::invalid_arg;
        std::unique_lock lock(mutex_);
        ++waiters_;
        const std::int32_t result = wait_locked(lock, timeout_ms, token);
        if (--waiters_ == 0) cv_.notify_all();
        return result;
    }

    // GetFrameStatisticsX(token, type_mask, &count, array). `count` is the room in `out` on entry and the number of
    // entries written on return. S_OK when every producible requested type was written (event, present, display),
    // S_FALSE with count 0 or fewer when they are not (there yet, or the token is unknown or has aged out of the
    // last 64). With a null `out`, count only receives how many entries are available.
    std::int32_t get_frame_statistics(std::uint64_t token, std::uint32_t type_mask, std::uint32_t* count,
                                      FrameStatistic* out) {
        if (!count) return pacer_hr::invalid_arg;
        std::lock_guard guard(mutex_);
        retire_locked(clock_.now_us());
        const Record* record = nullptr;
        for (const Record& r : records_)
            if (r.token == token) record = &r;
        FrameStatistic found[3];
        std::uint32_t n = 0, expected = 0;
        auto add = [&](std::uint32_t type, bool present, std::uint64_t time) {
            if (!(type_mask & type)) return;
            ++expected;
            if (!record || !present) return;
            found[n++] = FrameStatistic{type, 0, token, time, type == stat_display ? record->percent_scanned : 0.0f};
        };
        if (record) {
            add(stat_frame_event, true, record->origin_us);
            add(stat_present, record->presented, record->present_us);
            add(stat_display, record->displayed, record->display_us);
        } else {
            expected = ((type_mask & stat_frame_event) ? 1u : 0u) + ((type_mask & stat_present) ? 1u : 0u) +
                       ((type_mask & stat_display) ? 1u : 0u);
        }
        if (!out) {
            const std::uint32_t available = n;
            *count = available;
            return available && available == expected ? pacer_hr::ok : pacer_hr::false_;
        }
        const std::uint32_t written = std::min(n, *count);
        for (std::uint32_t i = 0; i < written; ++i) out[i] = found[i];
        *count = written;
        return written && written == expected ? pacer_hr::ok : pacer_hr::false_;
    }

    // Queue side. check_present_token before the frame is presented, commit_present_token after it was queued.
    // What real hardware does with a null, repeated or out-of-order token is unmeasured, and a title that presents
    // with Token = 0 (a loading screen), re-presents with one token, or presents from a render thread out of order
    // must not freeze. So the only refusal is a token that was never handed out (invalid_arg, which is a bug in the
    // caller); null, stale (older than the last presented) and repeated tokens return ok and say which kind they
    // were in `issue`, so the caller can log it. invalid_state while suspended.
    enum class TokenIssue { none, null_token, stale, repeated };
    std::int32_t check_present_token(std::uint64_t token, TokenIssue* issue = nullptr) {
        std::lock_guard guard(mutex_);
        if (issue) *issue = TokenIssue::none;
        if (suspended_) return pacer_hr::invalid_state;
        if (token > issued_) return pacer_hr::invalid_arg;
        if (issue) {
            if (token == 0) *issue = TokenIssue::null_token;
            else if (token == last_presented_) *issue = TokenIssue::repeated;
            else if (token < last_presented_) *issue = TokenIssue::stale;
        }
        return pacer_hr::ok;
    }
    // How late the frame of `token` is if it is presented now (see the model above). late_us stays 0 when the frame
    // could not be shown at once anyway: unpaced, null or unknown token, or earlier frames still waiting for their
    // vblank (a flip cannot overtake them), or the token was already presented or is older than the last presented
    // (a repeated or stale token never tears).
    struct PresentTiming {
        std::uint64_t late_us = 0;
        std::uint32_t period_us = 0;   // 0: unpaced
    };
    PresentTiming present_timing(std::uint64_t token) {
        std::lock_guard guard(mutex_);
        PresentTiming timing;
        if (!interval_set_ || !token) return timing;
        const std::uint64_t now = clock_.now_us();
        retire_locked(now);
        timing.period_us = period_us_;
        if (!pending_.empty()) return timing;
        for (const Record& r : records_)
            if (r.token == token && !r.presented && token > last_presented_ && now > r.due_us)
                timing.late_us = now - r.due_us;
        return timing;
    }

    // `sync` immediate: the frame is shown at once (display time = present time) instead of at the next vblank, and
    // percent_scanned is the lateness measured at this commit, so it agrees with the display time. The caller got
    // `sync` from decide_present_sync and the presenter confirmed it.
    void commit_present_token(std::uint64_t token, std::uint32_t plane_count, PresentSync sync = PresentSync::vblank) {
        (void)plane_count;
        std::lock_guard guard(mutex_);
        const std::uint64_t now = clock_.now_us();
        retire_locked(now);
        last_presented_ = std::max(last_presented_, token);
        if (!interval_set_ || !armed_) {  // not paced (PresentX holds no vblank until the event is armed): shows at once
            for (Record& r : records_)
                if (r.token == token) { r.presented = r.displayed = true; r.present_us = r.display_us = now; }
            return;
        }
        // The vblank after now, and not before the one after the previous queued frame.
        const std::uint64_t since = now > epoch_us_ ? now - epoch_us_ : 0;
        std::uint64_t flip = epoch_us_ + (since / period_us_ + 1) * period_us_;
        if (!pending_.empty()) flip = std::max(flip, pending_.back().flip_us + period_us_);
        float percent = 0.0f;
        if (sync == PresentSync::immediate && pending_.empty()) {
            flip = now;
            std::uint64_t late_us = 0;
            for (const Record& r : records_)
                if (r.token == token && now > r.due_us) late_us = now - r.due_us;
            percent = scan_percent(late_us, period_us_);
            ++immediate_presents_;
        }
        pending_.push_back(Pending{token, flip, percent});
        for (Record& r : records_)
            if (r.token == token) { r.presented = true; r.present_us = now; }
        retire_locked(now);
        cv_.notify_all();
    }
    // A blank present (no planes) carries no token: it is only counted as a frame to show. With no interval set or
    // the origin event not armed it is not paced: it shows at once and is not counted.
    void note_untokened_present() {
        std::lock_guard guard(mutex_);
        if (!interval_set_ || !armed_) return;
        const std::uint64_t now = clock_.now_us();
        retire_locked(now);
        const std::uint64_t since = now > epoch_us_ ? now - epoch_us_ : 0;
        std::uint64_t flip = epoch_us_ + (since / period_us_ + 1) * period_us_;
        if (!pending_.empty()) flip = std::max(flip, pending_.back().flip_us + period_us_);
        pending_.push_back(Pending{0, flip, 0.0f});
        cv_.notify_all();
    }

    // SuspendX: presentation stops, the registration and the frames in flight are dropped, waiters fail.
    void suspend() {
        std::lock_guard guard(mutex_);
        suspended_ = true;
        interval_set_ = armed_ = false;
        pending_.clear();
        cv_.notify_all();
    }
    // ResumeX: presents work again, but the interval and the origin event must be set up again.
    void resume() {
        std::lock_guard guard(mutex_);
        suspended_ = false;
        interval_set_ = armed_ = false;
        cv_.notify_all();
    }

    bool suspended() const { std::lock_guard guard(mutex_); return suspended_; }
    std::uint64_t tokens_issued() const { std::lock_guard guard(mutex_); return issued_; }
    std::uint32_t frames_in_flight() {
        std::lock_guard guard(mutex_);
        retire_locked(clock_.now_us());
        return std::uint32_t(pending_.size());
    }
    // Frames that were flipped mid-scan because of ImmediateThresholdPercent.
    std::uint64_t immediate_presents() const { std::lock_guard guard(mutex_); return immediate_presents_; }
    std::uint32_t period_us() const { std::lock_guard guard(mutex_); return period_us_; }
    bool registered() const { std::lock_guard guard(mutex_); return interval_set_ && armed_; }

private:
    struct Pending {
        std::uint64_t token;
        std::uint64_t flip_us;
        float percent_scanned;  // where the scan was for a frame shown mid-scan, else 0
    };
    struct Record {
        std::uint64_t token = 0, origin_us = 0, present_us = 0, display_us = 0;
        std::uint64_t due_us = 0;  // vblank after this token's origin, fixed when issued
        float percent_scanned = 0.0f;
        bool presented = false, displayed = false;
    };

    // Frames whose vblank has come are shown.
    void retire_locked(std::uint64_t now) {
        while (!pending_.empty() && pending_.front().flip_us <= now) {
            for (Record& r : records_)
                if (r.token == pending_.front().token && pending_.front().token) {
                    r.displayed = true;
                    r.display_us = pending_.front().flip_us;
                    r.percent_scanned = pending_.front().percent_scanned;
                }
            pending_.pop_front();
        }
    }

    std::int32_t wait_locked(std::unique_lock<std::mutex>& lock, std::uint32_t timeout_ms, std::uint64_t* token) {
        const std::uint64_t start = clock_.now_us();
        const std::uint64_t give_up = timeout_ms == infinite_timeout_ms ? no_deadline
                                                                         : start + std::uint64_t(timeout_ms) * 1000;
        for (;;) {
            if (shutdown_ || suspended_ || !interval_set_ || !armed_) return pacer_hr::invalid_state;
            const std::uint64_t now = clock_.now_us();
            retire_locked(now);
            const std::uint64_t first = epoch_us_ + offset_us_;  // origin 0
            const std::int64_t next = last_origin_ + 1;
            const std::uint64_t next_us = first + std::uint64_t(next) * period_us_;
            std::uint64_t wake = next_us;
            if (now >= next_us) {
                if (pending_.size() < latency_) {
                    const std::int64_t newest = std::int64_t((now - first) / period_us_);
                    const std::int64_t origin = std::max(next, newest);
                    last_origin_ = origin;
                    *token = ++issued_;
                    Record record;
                    record.token = *token;
                    record.origin_us = first + std::uint64_t(origin) * period_us_;
                    record.due_us = record.origin_us - offset_us_ + period_us_;
                    records_.push_back(record);
                    if (records_.size() > 64) records_.pop_front();
                    return pacer_hr::ok;
                }
                wake = pending_.front().flip_us;  // blocked by the latency: until the oldest frame is shown
            }
            if (give_up != no_deadline && now >= give_up) return pacer_hr::timeout;
            clock_.wait(lock, cv_, give_up == no_deadline ? wake : std::min(wake, give_up));
        }
    }

    FrameClock& clock_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool shutdown_ = false;
    std::uint32_t waiters_ = 0;

    bool suspended_ = false;
    bool interval_set_ = false;
    bool armed_ = false;
    std::uint32_t period_us_ = 0;
    std::uint32_t latency_ = 1;
    std::uint32_t offset_us_ = 0;
    std::uint64_t epoch_us_ = 0;
    std::int64_t last_origin_ = 0;  // origin 0 is the epoch itself, which has passed
    std::uint64_t issued_ = 0;
    std::uint64_t last_presented_ = 0;
    std::uint64_t immediate_presents_ = 0;
    std::deque<Pending> pending_;
    std::deque<Record> records_;
};

}  // namespace leonida
