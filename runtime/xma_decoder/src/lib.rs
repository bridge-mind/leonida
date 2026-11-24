#![deny(unsafe_op_in_unsafe_fn)]

use std::alloc::Layout;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::Mutex;

use xma::{Xma2Stream, XMA_FRAME_SAMPLES, XMA_PACKET_BYTES};

pub const LEONIDA_XMA2_OK: i32 = 0;
pub const LEONIDA_XMA2_INVALID_ARGUMENT: i32 = 1;
pub const LEONIDA_XMA2_INVALID_FORMAT: i32 = 2;
pub const LEONIDA_XMA2_BAD_PACKET_SIZE: i32 = 3;
pub const LEONIDA_XMA2_OUT_OF_MEMORY: i32 = 4;
pub const LEONIDA_XMA2_DECODER_FAILURE: i32 = 5;
pub const LEONIDA_XMA2_PANIC: i32 = 6;

const MAX_FRAMES_PER_PACKET: usize = 64;

#[repr(C)]
pub struct LeonidaXma2Decoder {
    stream: Mutex<Xma2Stream>,
    channels: usize,
}

#[repr(C)]
pub struct LeonidaXma2PcmBlock {
    samples: Vec<i16>,
}

#[cfg(not(test))]
use std::alloc::alloc as allocate_handle;
#[cfg(test)]
use tests::allocate_handle;

fn try_new_handle<T>(value: T) -> Result<Box<T>, i32> {
    const { assert!(std::mem::size_of::<T>() > 0) };
    // Both the allocation and Box's eventual deallocation use the global allocator
    // with T's exact layout. Initialize the storage before handing it to Box.
    let storage = unsafe { allocate_handle(Layout::new::<T>()) }.cast::<T>();
    if storage.is_null() {
        return Err(LEONIDA_XMA2_OUT_OF_MEMORY);
    }
    unsafe {
        storage.write(value);
        Ok(Box::from_raw(storage))
    }
}

/// Creates a decoder for one mono or stereo XMA2 stream.
///
/// # Safety
/// `out_decoder` must be writable for one pointer. The returned decoder must be destroyed once,
/// after all calls using it have finished.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_decoder_create(
    sample_rate: u32,
    channels: u32,
    out_decoder: *mut *mut LeonidaXma2Decoder,
) -> i32 {
    if out_decoder.is_null() {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    unsafe { *out_decoder = ptr::null_mut() };
    if sample_rate == 0 || !(channels == 1 || channels == 2) {
        return LEONIDA_XMA2_INVALID_FORMAT;
    }

    match catch_unwind(AssertUnwindSafe(|| {
        let stream = Xma2Stream::new(sample_rate, channels as usize)
            .map_err(|_| LEONIDA_XMA2_INVALID_FORMAT)?;
        try_new_handle(LeonidaXma2Decoder {
            stream: Mutex::new(stream),
            channels: channels as usize,
        })
    })) {
        Ok(Ok(decoder)) => {
            unsafe { *out_decoder = Box::into_raw(decoder) };
            LEONIDA_XMA2_OK
        }
        Ok(Err(status)) => status,
        Err(_) => LEONIDA_XMA2_PANIC,
    }
}

/// Decodes a buffer containing one or more complete 2048-byte XMA2 packets.
///
/// # Safety
/// `decoder` must be a live handle returned by `leonida_xma2_decoder_create`.
/// `input` must reference `input_len` readable bytes and `out_block` must be writable.
/// Do not destroy a decoder concurrently with this call. The returned PCM block remains valid
/// until `leonida_xma2_pcm_destroy` is called.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_decode_buffer(
    decoder: *mut LeonidaXma2Decoder,
    input: *const u8,
    input_len: usize,
    out_block: *mut *mut LeonidaXma2PcmBlock,
) -> i32 {
    if out_block.is_null() {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    unsafe { *out_block = ptr::null_mut() };

    if decoder.is_null() || input.is_null() || input_len > isize::MAX as usize {
        return LEONIDA_XMA2_INVALID_ARGUMENT;
    }
    if input_len == 0 || !input_len.is_multiple_of(XMA_PACKET_BYTES) {
        return LEONIDA_XMA2_BAD_PACKET_SIZE;
    }

    let decoded = catch_unwind(AssertUnwindSafe(|| {
        let handle = unsafe { &*decoder };
        let bytes = unsafe { slice::from_raw_parts(input, input_len) };
        let mut stream = handle
            .stream
            .lock()
            .map_err(|_| LEONIDA_XMA2_DECODER_FAILURE)?;
        let max_packet_samples = MAX_FRAMES_PER_PACKET
            .checked_mul(XMA_FRAME_SAMPLES)
            .and_then(|samples| samples.checked_mul(handle.channels))
            .ok_or(LEONIDA_XMA2_INVALID_ARGUMENT)?;
        let mut block = try_new_handle(LeonidaXma2PcmBlock {
            samples: Vec::new(),
        })?;
        let samples = &mut block.samples;

        let (packets, _) = bytes.as_chunks::<XMA_PACKET_BYTES>();
        for packet in packets {
            let requested = samples
                .len()
                .checked_add(max_packet_samples)
                .ok_or(LEONIDA_XMA2_INVALID_ARGUMENT)?;
            if requested > isize::MAX as usize / std::mem::size_of::<i16>() {
                return Err(LEONIDA_XMA2_INVALID_ARGUMENT);
            }
            samples
                .try_reserve(max_packet_samples)
                .map_err(|_| LEONIDA_XMA2_OUT_OF_MEMORY)?;
            stream.decode_packet(packet, samples);
        }

        if !samples.len().is_multiple_of(handle.channels) {
            return Err(LEONIDA_XMA2_DECODER_FAILURE);
        }
        Ok(block)
    }));

    match decoded {
        Ok(Ok(block)) => {
            unsafe { *out_block = Box::into_raw(block) };
            LEONIDA_XMA2_OK
        }
        Ok(Err(status)) => status,
        Err(_) => LEONIDA_XMA2_PANIC,
    }
}

/// Destroys a decoder. Passing null is a no-op.
///
/// # Safety
/// A non-null pointer must be a live handle returned by `leonida_xma2_decoder_create`,
/// and no calls may be using it concurrently.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_decoder_destroy(decoder: *mut LeonidaXma2Decoder) {
    if !decoder.is_null() {
        unsafe { drop(Box::from_raw(decoder)) };
    }
}

/// Returns the immutable interleaved signed-16 PCM pointer for a block.
///
/// # Safety
/// A non-null pointer must be a live block returned by `leonida_xma2_decode_buffer`.
/// The returned pointer is valid only until the block is destroyed.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_pcm_data(block: *const LeonidaXma2PcmBlock) -> *const i16 {
    unsafe { block.as_ref() }.map_or(ptr::null(), |pcm| pcm.samples.as_ptr())
}

/// Returns the number of signed-16 samples in a block.
///
/// # Safety
/// A non-null pointer must be a live block returned by `leonida_xma2_decode_buffer`.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_pcm_sample_count(block: *const LeonidaXma2PcmBlock) -> usize {
    unsafe { block.as_ref() }.map_or(0, |pcm| pcm.samples.len())
}

/// Destroys a PCM block. Passing null is a no-op.
///
/// # Safety
/// A non-null pointer must be a live block returned by `leonida_xma2_decode_buffer`.
#[no_mangle]
pub unsafe extern "C" fn leonida_xma2_pcm_destroy(block: *mut LeonidaXma2PcmBlock) {
    if !block.is_null() {
        unsafe { drop(Box::from_raw(block)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        static FAIL_NEXT_HANDLE_ALLOCATION: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) unsafe fn allocate_handle(layout: Layout) -> *mut u8 {
        if FAIL_NEXT_HANDLE_ALLOCATION.replace(false) {
            ptr::null_mut()
        } else {
            unsafe { std::alloc::alloc(layout) }
        }
    }

    #[test]
    fn decoder_handle_allocation_failure_clears_output() {
        let mut decoder = ptr::dangling_mut();
        FAIL_NEXT_HANDLE_ALLOCATION.set(true);
        let status = unsafe { leonida_xma2_decoder_create(48_000, 2, &mut decoder) };
        assert_eq!(status, LEONIDA_XMA2_OUT_OF_MEMORY);
        assert!(decoder.is_null());

        assert_eq!(
            unsafe { leonida_xma2_decoder_create(48_000, 2, &mut decoder) },
            LEONIDA_XMA2_OK
        );
        assert!(!decoder.is_null());
        unsafe { leonida_xma2_decoder_destroy(decoder) };
    }

    #[test]
    fn pcm_handle_allocation_failure_clears_output_and_allows_retry() {
        let mut decoder = ptr::null_mut();
        assert_eq!(
            unsafe { leonida_xma2_decoder_create(48_000, 2, &mut decoder) },
            LEONIDA_XMA2_OK
        );
        let packet = [0u8; XMA_PACKET_BYTES];
        let mut block = ptr::dangling_mut();
        FAIL_NEXT_HANDLE_ALLOCATION.set(true);
        let status = unsafe {
            leonida_xma2_decode_buffer(decoder, packet.as_ptr(), packet.len(), &mut block)
        };
        assert_eq!(status, LEONIDA_XMA2_OUT_OF_MEMORY);
        assert!(block.is_null());

        assert_eq!(
            unsafe {
                leonida_xma2_decode_buffer(decoder, packet.as_ptr(), packet.len(), &mut block)
            },
            LEONIDA_XMA2_OK
        );
        assert!(!block.is_null());
        unsafe {
            leonida_xma2_pcm_destroy(block);
            leonida_xma2_decoder_destroy(decoder);
        }
    }
}
