#include "xbox_present.h"

#include "../src/vulkan_present.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rt::d3d12 {

namespace {

using leonida::FramePacer;
namespace hr = leonida::pacer_hr;

HRESULT from_pacer(std::int32_t value) { return HRESULT(value); }

// The fixed text of a log line is printed once per object.
void log_once(bool& done, const char* text) {
    if (done) return;
    done = true;
    std::fprintf(stderr, "[rt] present: %s\n", text);
}

}  // namespace

XboxPresentation::XboxPresentation(std::unique_ptr<WindowPresenter> presenter, leonida::FrameClock* clock)
    : presenter_(std::move(presenter)),
      own_clock_(clock ? nullptr : std::make_unique<leonida::SteadyFrameClock>()),
      pacer_(clock ? *clock : *own_clock_) {
    // A present waiting for a swapchain image gives up once SuspendX has been called.
    presenter_->set_abort([this] { return pacer_.suspended(); });
    // LEONIDA_SDR_EOTF=gamma22 (or 2.2, any case) decodes SDR with a pure power of 2.2 instead of piecewise sRGB; "srgb" or
    // unset keeps the default; anything else (also a value too long for the buffer, ERANGE) is logged once and ignored.
    char eotf[32] = {};
    std::size_t eotf_length = 0;
    const errno_t env = getenv_s(&eotf_length, eotf, "LEONIDA_SDR_EOTF");
    if (env == 0 && eotf_length > 1) {
        if (!_stricmp(eotf, "gamma22") || !_stricmp(eotf, "2.2")) sdr_eotf_ = leonida::PresentPlanes::SdrEotf::gamma22;
        else if (_stricmp(eotf, "srgb") != 0)
            std::fprintf(stderr, "[rt] present: LEONIDA_SDR_EOTF=%s is not srgb, gamma22 or 2.2: piecewise sRGB used\n", eotf);
    } else if (env == ERANGE) {
        std::fprintf(stderr, "[rt] present: LEONIDA_SDR_EOTF is too long to be srgb, gamma22 or 2.2: piecewise sRGB used\n");
    }
    display_white_thread_ = std::thread([this] { display_white_worker(); });
}

XboxPresentation::~XboxPresentation() {
    {
        std::lock_guard lock(worker_mutex_);
        worker_stop_ = true;
    }
    worker_wake_.notify_all();
    display_white_thread_.join();
    presenter_->set_abort(nullptr);
}

// Windows' SDR white level costs a QueryDisplayConfig and two DisplayConfigGetDeviceInfo calls (measured in
// xbox_present_test, docs section 4.5): too much for the present path, which would hitch once a second and feed the pacer.
// So it is read here, about once a second while an HDR10 swapchain is in use, and PresentX reads the cached value.
void XboxPresentation::display_white_worker() {
    std::unique_lock lock(worker_mutex_);
    while (!worker_stop_) {
        if (white_wanted_.load()) {
            lock.unlock();
            const float nits = presenter_->display_sdr_white_nits();
            if (nits > 0.0f) display_white_ = nits;  // a failed query keeps the last good value
            lock.lock();
        }
        worker_wake_.wait_for(lock, std::chrono::seconds(1), [this] { return worker_stop_; });
    }
}

float XboxPresentation::effective_sdr_white_nits() {
    if (sdr_white_override_ > 0.0f) return sdr_white_override_;
    if (!presenter_->hdr_active()) return 203.0f;  // the Windows setting belongs to HDR mode
    if (!white_wanted_.exchange(true)) {  // first HDR10 present: have it read now
        std::lock_guard lock(worker_mutex_);
        worker_wake_.notify_all();
    }
    const float cached = display_white_;
    return cached > 0.0f ? cached : 203.0f;
}

HRESULT XboxPresentation::SetFrameIntervalX(IUnknown*, UINT interval_period_us, UINT latency, UINT flags) {
    return from_pacer(pacer_.set_frame_interval(interval_period_us, latency, flags));
}

HRESULT XboxPresentation::ScheduleFrameEventX(UINT event, UINT offset_us, UINT flags) {
    return from_pacer(pacer_.schedule_frame_event(event, offset_us, flags));
}

HRESULT XboxPresentation::WaitFrameEventX(UINT event, DWORD timeout_ms, UINT flags, std::uint64_t* token) {
    return from_pacer(pacer_.wait_frame_event(event, timeout_ms, flags, token));
}

HRESULT XboxPresentation::GetFrameStatisticsX(std::uint64_t token, UINT type_mask, UINT* count,
                                              leonida::FrameStatistic* out) {
    return from_pacer(pacer_.get_frame_statistics(token, type_mask, count, out));
}

HRESULT XboxPresentation::PresentX(UINT plane_count, const XboxPresentPlaneParameters* planes,
                                   const XboxPresentParameters* params) {
    std::lock_guard serialise(present_mutex_);
    if (plane_count > 2 || (plane_count && !planes)) return E_INVALIDARG;
    if (pacer_.suspended()) return from_pacer(hr::invalid_state);
    // A zero, negative or NaN threshold never tears.
    const float threshold = params ? params->ImmediateThresholdPercent : 0.0f;
    const bool wants_tear = threshold > 0.0f;
    if (wants_tear && !presenter_->tear_supported())
        log_once(logged_threshold_,
                 "ImmediateThresholdPercent has no effect on this swapchain (tearing disabled by LEONIDA_PRESENT_NO_TEAR, "
                 "no VK_KHR/EXT_swapchain_maintenance1, or no IMMEDIATE/FIFO_RELAXED mode the surface can switch to): "
                 "a late frame waits for the next vblank");

    // Unpaced (no interval registered): do not hold the queue for vblanks.
    const UINT sync_interval = pacer_.registered() ? 1 : 0;

    if (plane_count == 0) {
        const HRESULT result = presenter_->present_blank(sync_interval);
        if (SUCCEEDED(result)) {
            pacer_.note_untokened_present();
            ++presents_;
        }
        return result;
    }

    // All planes carry the token of one frame.
    const std::uint64_t token = planes[0].Token;
    for (UINT i = 0; i < plane_count; ++i) {
        const XboxPresentPlaneParameters& plane = planes[i];
        if (plane.Token != token || plane.ResourceCount == 0 || !plane.ppResources || !plane.ppResources[0])
            return E_INVALIDARG;
        // Both paths read ppResources[0] only: a plane with several resources would silently lose the others and could show
        // the wrong frame, so it is refused (before anything is presented) until it is modelled. What the console does
        // with ResourceCount above 1 (views? stereo?) is unmeasured.
        if (plane.ResourceCount != 1) {
            log_once(logged_resource_count_, "PresentX plane with ResourceCount other than 1: not implemented, refused");
            return DXGI_ERROR_UNSUPPORTED;
        }
    }
    // Only a token that was never handed out is refused. Null, stale and repeated tokens are presented anyway (what the
    // console does with them is unmeasured, and a title that does it must not freeze); each kind is logged once.
    FramePacer::TokenIssue issue{};
    if (const std::int32_t checked = pacer_.check_present_token(token, &issue); checked != hr::ok)
        return from_pacer(checked);
    switch (issue) {
    case FramePacer::TokenIssue::null_token:
        log_once(logged_null_token_, "PresentX with the null token: presented anyway (real behaviour unmeasured)");
        break;
    case FramePacer::TokenIssue::stale:
        log_once(logged_stale_token_, "PresentX with a token older than the last presented: presented anyway");
        break;
    case FramePacer::TokenIssue::repeated:
        log_once(logged_repeated_token_, "PresentX with the token of the last present again: presented anyway");
        break;
    case FramePacer::TokenIssue::none:
        break;
    }

    // What each plane asks for. Colour spaces other than HDR10 are composed as the SDR default (logged once).
    const auto is_hdr10 = [&](const XboxPresentPlaneParameters& p) {
        if (p.ColorSpace != xbox_color_space_g22_p709 && p.ColorSpace != xbox_color_space_g2084_p2020)
            log_once(logged_color_space_, "a present plane's DXGI colour space is neither G22/P709 nor G2084/P2020: composed as SDR G22/P709");
        return p.ColorSpace == xbox_color_space_g2084_p2020;
    };
    // The extra fields are unmeasured, so they are read defensively: a placement that is not a sane sub-rectangle of the output
    // (also NaN) counts as none, an unknown filter as the default, unknown flag bits are ignored, extended descriptors are
    // not read; each is logged once and the frame is shown anyway (a title must not freeze on a field we misread).
    const auto placement = [&](const XboxPresentPlaneParameters& p) -> const XboxViewRect* {
        const XboxViewRect* r = p.pDestPlacementBase;
        if (!r) return nullptr;
        if (!(r->left >= 0.0f && r->left < r->right && r->right <= 1.0f && r->top >= 0.0f && r->top < r->bottom &&
              r->bottom <= 1.0f)) {
            log_once(logged_extras_, "a present plane's pDestPlacementBase is not a sub-rectangle of 0..1: ignored (whole output)");
            return nullptr;
        }
        return (r->left == 0.0f && r->top == 0.0f && r->right == 1.0f && r->bottom == 1.0f) ? nullptr : r;
    };
    const auto filter_of = [&](const XboxPresentPlaneParameters& p) {
        if (p.ScaleFilter == xbox_scale_filter_point || p.ScaleFilter == xbox_scale_filter_linear) return p.ScaleFilter;
        if (p.ScaleFilter != xbox_scale_filter_default)
            log_once(logged_extras_, "a present plane's ScaleFilter is not a known value: default filter");
        return int(xbox_scale_filter_default);
    };
    for (UINT i = 0; i < plane_count; ++i)
        if (planes[i].ExtendedDescCount)
            log_once(logged_extras_, "a present plane has extended descriptors: ignored");
    const bool hdr0 = is_hdr10(planes[0]);
    // Plane 1 is composed over plane 0 only when flagged an overlay (hypothesis). Otherwise it is not shown, as before: in
    // the public HDR sample it is the GameDVR copy of the frame, and blending a plane we cannot tell apart from it risks
    // drawing the frame twice with an alpha that was never meant to be blended.
    const bool overlay = plane_count == 2 && (planes[1].Flags & xbox_plane_flag_overlay) != 0;
    if (plane_count == 2 && !overlay)
        log_once(logged_planes_, "PresentX with 2 planes and no overlay flag on plane 1: only plane 0 is shown (plane 1 is taken "
                                 "as the GameDVR copy of the frame)");
    const bool compose = hdr0 || placement(planes[0]) || filter_of(planes[0]) != xbox_scale_filter_default || overlay;

    if (compose) {
        // Every shown plane: a 2D texture, a source rectangle inside it, a placement inside the output.
        leonida::PresentPlanes composed;
        composed.count = overlay ? 2 : 1;
        composed.sdr_white_nits = effective_sdr_white_nits();
        composed.sdr_eotf = sdr_eotf_;
        composed.hdr_peak_nits = hdr_peak_nits_;
        for (UINT i = 0; i < composed.count; ++i) {
            const XboxPresentPlaneParameters& p = planes[i];
            ID3D12Resource* source = p.ppResources[0];
            const D3D12_RESOURCE_DESC source_desc = source->GetDesc();
            if (source_desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return E_INVALIDARG;
            const D3D12_RECT* source_rect = p.pSrcViewRects;
            if (source_rect && (source_rect->left < 0 || source_rect->top < 0 || source_rect->left >= source_rect->right ||
                                source_rect->top >= source_rect->bottom || UINT64(source_rect->right) > source_desc.Width ||
                                UINT(source_rect->bottom) > source_desc.Height))
                return E_INVALIDARG;
            const XboxViewRect* place = placement(p);
            leonida::PresentPlane& out = composed.plane[i];
            out.source = resource_id(source);
            if (source_rect) {
                out.source_rect = {source_rect->left, source_rect->top, source_rect->right, source_rect->bottom};
                out.has_source_rect = true;
            }
            if (place) {
                out.dest[0] = place->left;
                out.dest[1] = place->top;
                out.dest[2] = place->right;
                out.dest[3] = place->bottom;
            }
            out.point_filter = filter_of(p) == xbox_scale_filter_point;
            out.hdr10 = is_hdr10(p);
            out.straight_alpha = (p.Flags & xbox_plane_flag_straight_alpha) != 0;
        }
        bool tear = false;
        if (wants_tear && sync_interval > 0 && presenter_->tear_supported()) {
            const FramePacer::PresentTiming timing = pacer_.present_timing(token);
            tear = leonida::decide_present_sync(timing.late_us, timing.period_us, threshold) == leonida::PresentSync::immediate;
        }
        const HRESULT result = presenter_->present_planes(composed, sync_interval, tear);
        if (FAILED(result)) return pacer_.suspended() ? from_pacer(hr::invalid_state) : result;
        pacer_.commit_present_token(token, plane_count,
                                    tear && presenter_->last_present_tore() ? leonida::PresentSync::immediate
                                                                            : leonida::PresentSync::vblank);
        ++presents_;
        return S_OK;
    }

    const XboxPresentPlaneParameters& plane = planes[0];
    ID3D12Resource* resource = plane.ppResources[0];
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) return E_INVALIDARG;
    D3D12_HEAP_PROPERTIES heap{};
    D3D12_HEAP_FLAGS heap_flags = D3D12_HEAP_FLAG_NONE;
    if (SUCCEEDED(resource->GetHeapProperties(&heap, &heap_flags)) && !(heap_flags & D3D12_HEAP_FLAG_ALLOW_DISPLAY))
        log_once(logged_display_, "a present plane's resource was created without D3D12_HEAP_FLAG_ALLOW_DISPLAY; presenting it anyway");

    const D3D12_RECT* rect = plane.pSrcViewRects;
    if (rect && (rect->left < 0 || rect->top < 0 || rect->left >= rect->right || rect->top >= rect->bottom ||
                 UINT64(rect->right) > desc.Width || UINT(rect->bottom) > desc.Height))
        return E_INVALIDARG;

    // How late the frame is, now that everything that can refuse it has: past the vblank it was due at, by what part of
    // the interval the scan has moved on. The presenter says whether it really queued it as a tear.
    FramePacer::PresentTiming timing;
    bool tear = false;
    if (wants_tear && sync_interval > 0 && presenter_->tear_supported()) {
        timing = pacer_.present_timing(token);
        tear = leonida::decide_present_sync(timing.late_us, timing.period_us, threshold) == leonida::PresentSync::immediate;
    }
    const HRESULT result = presenter_->present(resource, sync_interval, rect, tear);
    if (FAILED(result)) return pacer_.suspended() ? from_pacer(hr::invalid_state) : result;
    pacer_.commit_present_token(token, plane_count,
                                tear && presenter_->last_present_tore() ? leonida::PresentSync::immediate
                                                                        : leonida::PresentSync::vblank);
    ++presents_;
    return S_OK;
}

// SuspendX and ResumeX do not take present_mutex_: a PresentX stuck waiting for a swapchain image must not hold them
// (it polls the suspended state and gives up). A present that was already past that point finishes normally.
HRESULT XboxPresentation::SuspendX(UINT) {
    pacer_.suspend();
    return S_OK;
}

HRESULT XboxPresentation::ResumeX() {
    pacer_.resume();
    return S_OK;
}

}  // namespace rt::d3d12
