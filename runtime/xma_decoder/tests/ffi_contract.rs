use leonida_xma2_decoder::*;
use std::ptr;
use xma::bits::BitWriter;
use xma::Xma2Stream;

const PACKET_BYTES: usize = 2048;

fn synthetic_silent_packet(channels: usize) -> [u8; PACKET_BYTES] {
    assert!(channels == 1 || channels == 2);

    let mut frame = BitWriter::default();
    frame.put(1, 1); // Fixed-size tile header.
    frame.put(1, 0); // One full-length subframe.
    if channels == 2 {
        frame.put(1, 0); // No optional stereo decorrelation matrix.
    }
    frame.put(8, 0); // No dynamic range adjustment.
    frame.put(1, 0); // No trim metadata.
    frame.put(1, 0); // No fill data.
    frame.put(1, 0); // No corrupt-subframe marker.
    if channels == 2 {
        frame.put(1, 0); // No stereo transform error.
        frame.put(1, 1); // Two-channel group without decorrelation.
        frame.put(1, 0);
    }
    for _ in 0..channels {
        frame.put(1, 0); // No transmitted coefficients: silence.
    }
    frame.put(1, 0); // Align to the next frame boundary.
    frame.put(1, 0); // This packet contains one frame.

    let frame_bits = 15 + frame.bits;
    let mut packet_writer = BitWriter::default();
    packet_writer.put(6, 1);
    packet_writer.put(15, frame_bits as u32);
    packet_writer.put(3, 1);
    packet_writer.put(8, 0);
    packet_writer.put(15, frame_bits as u32);
    packet_writer.copy_from(&frame.buf, 0, frame.bits);

    let mut packet = [0u8; PACKET_BYTES];
    packet[..packet_writer.buf.len()].copy_from_slice(&packet_writer.buf);
    packet
}

fn make_decoder(rate: u32, channels: u32) -> *mut LeonidaXma2Decoder {
    let mut decoder = ptr::null_mut();
    let result = unsafe { leonida_xma2_decoder_create(rate, channels, &mut decoder) };
    assert_eq!(result, LEONIDA_XMA2_OK);
    assert!(!decoder.is_null());
    decoder
}

#[test]
fn creation_rejects_invalid_outputs_and_formats() {
    assert_eq!(
        unsafe { leonida_xma2_decoder_create(48_000, 2, ptr::null_mut()) },
        LEONIDA_XMA2_INVALID_ARGUMENT
    );

    let mut decoder = ptr::dangling_mut::<LeonidaXma2Decoder>();
    assert_eq!(
        unsafe { leonida_xma2_decoder_create(48_000, 3, &mut decoder) },
        LEONIDA_XMA2_INVALID_FORMAT
    );
    assert!(decoder.is_null());

    assert_eq!(
        unsafe { leonida_xma2_decoder_create(0, 1, &mut decoder) },
        LEONIDA_XMA2_INVALID_FORMAT
    );
    assert!(decoder.is_null());
}

#[test]
fn synthetic_silent_packet_returns_owned_pcm_block_after_decoder_destruction() {
    let decoder = make_decoder(48_000, 2);
    let packet = synthetic_silent_packet(2);
    let mut input = [0u8; PACKET_BYTES * 3];
    input[PACKET_BYTES..PACKET_BYTES * 2].copy_from_slice(&packet);
    input[PACKET_BYTES * 2..].copy_from_slice(&packet);
    let mut block = ptr::null_mut();

    assert_eq!(
        unsafe { leonida_xma2_decode_buffer(decoder, input.as_ptr(), input.len(), &mut block,) },
        LEONIDA_XMA2_OK
    );
    assert!(!block.is_null());

    let samples = unsafe { leonida_xma2_pcm_sample_count(block) };
    assert_eq!(samples % 2, 0);
    assert!(samples > 0);
    let pcm = unsafe { leonida_xma2_pcm_data(block) };
    assert!(!pcm.is_null());
    let decoded = unsafe { std::slice::from_raw_parts(pcm, samples) };
    assert!(decoded.iter().all(|sample| *sample == 0));

    unsafe { leonida_xma2_decoder_destroy(decoder) };
    assert_eq!(unsafe { leonida_xma2_pcm_sample_count(block) }, samples);
    assert_eq!(
        unsafe { std::slice::from_raw_parts(leonida_xma2_pcm_data(block), samples) },
        decoded
    );
    unsafe { leonida_xma2_pcm_destroy(block) };
}

#[test]
fn synthetic_silent_mono_stream_decodes_to_silent_pcm() {
    let decoder = make_decoder(44_100, 1);
    let packet = synthetic_silent_packet(1);
    let mut input = [0u8; PACKET_BYTES * 3];
    input[PACKET_BYTES..PACKET_BYTES * 2].copy_from_slice(&packet);
    input[PACKET_BYTES * 2..].copy_from_slice(&packet);
    let mut block = ptr::null_mut();

    assert_eq!(
        unsafe { leonida_xma2_decode_buffer(decoder, input.as_ptr(), input.len(), &mut block) },
        LEONIDA_XMA2_OK
    );
    assert!(!block.is_null());
    let samples = unsafe { leonida_xma2_pcm_sample_count(block) };
    assert!(samples > 0);
    let pcm = unsafe { leonida_xma2_pcm_data(block) };
    assert!(!pcm.is_null());
    assert!(unsafe { std::slice::from_raw_parts(pcm, samples) }
        .iter()
        .all(|sample| *sample == 0));

    unsafe {
        leonida_xma2_pcm_destroy(block);
        leonida_xma2_decoder_destroy(decoder);
    }
}

#[test]
fn decode_rejects_nulls_and_non_packet_aligned_buffers() {
    let decoder = make_decoder(44_100, 1);
    let packet = [0u8; PACKET_BYTES];
    let mut block = ptr::dangling_mut::<LeonidaXma2PcmBlock>();

    assert_eq!(
        unsafe { leonida_xma2_decode_buffer(decoder, packet.as_ptr(), 0, &mut block) },
        LEONIDA_XMA2_BAD_PACKET_SIZE
    );
    assert!(block.is_null());

    assert_eq!(
        unsafe {
            leonida_xma2_decode_buffer(decoder, packet.as_ptr(), PACKET_BYTES - 1, &mut block)
        },
        LEONIDA_XMA2_BAD_PACKET_SIZE
    );
    assert!(block.is_null());

    assert_eq!(
        unsafe { leonida_xma2_decode_buffer(decoder, ptr::null(), PACKET_BYTES, &mut block) },
        LEONIDA_XMA2_INVALID_ARGUMENT
    );
    assert!(block.is_null());

    unsafe { leonida_xma2_decoder_destroy(decoder) };
}

#[test]
fn deterministic_synthetic_packets_do_not_unwind_across_ffi() {
    let decoder = make_decoder(48_000, 2);
    for seed in 1..17u32 {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        let mut packet = [0u8; PACKET_BYTES];
        for byte in &mut packet {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = state as u8;
        }

        let mut block = ptr::null_mut();
        let result = unsafe {
            leonida_xma2_decode_buffer(decoder, packet.as_ptr(), packet.len(), &mut block)
        };
        assert!(
            result == LEONIDA_XMA2_OK
                || result == LEONIDA_XMA2_DECODER_FAILURE
                || result == LEONIDA_XMA2_PANIC,
            "unexpected status {result}"
        );
        if !block.is_null() {
            unsafe { leonida_xma2_pcm_destroy(block) };
        }
    }

    unsafe { leonida_xma2_decoder_destroy(decoder) };
}

#[test]
fn multiple_packet_buffer_matches_serial_decoder_calls() {
    let decoder = make_decoder(48_000, 2);
    let packet = synthetic_silent_packet(2);
    let mut input = [0u8; PACKET_BYTES * 3];
    input[PACKET_BYTES..PACKET_BYTES * 2].copy_from_slice(&packet);
    input[PACKET_BYTES * 2..].copy_from_slice(&packet);

    let mut reference = Xma2Stream::new(48_000, 2).unwrap();
    let mut expected = Vec::new();
    for packet in input.as_chunks::<PACKET_BYTES>().0 {
        reference.decode_packet(packet, &mut expected);
    }

    let mut block = ptr::null_mut();
    assert_eq!(
        unsafe { leonida_xma2_decode_buffer(decoder, input.as_ptr(), input.len(), &mut block) },
        LEONIDA_XMA2_OK
    );
    assert!(!block.is_null());
    let samples = unsafe { leonida_xma2_pcm_sample_count(block) };
    assert!(samples > 0);
    assert_eq!(samples, expected.len());
    assert_eq!(samples % 2, 0);
    let pcm = unsafe { leonida_xma2_pcm_data(block) };
    let actual = unsafe { std::slice::from_raw_parts(pcm, samples) };
    assert_eq!(actual, expected.as_slice());

    unsafe {
        leonida_xma2_pcm_destroy(block);
        leonida_xma2_decoder_destroy(decoder);
    }
}
