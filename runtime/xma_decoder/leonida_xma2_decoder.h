#ifndef LEONIDA_XMA2_DECODER_H
#define LEONIDA_XMA2_DECODER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LEONIDA_XMA2_PACKET_BYTES 2048u

typedef struct LeonidaXma2Decoder LeonidaXma2Decoder;
typedef struct LeonidaXma2PcmBlock LeonidaXma2PcmBlock;

enum LeonidaXma2Status {
  LEONIDA_XMA2_OK = 0,
  LEONIDA_XMA2_INVALID_ARGUMENT = 1,
  LEONIDA_XMA2_INVALID_FORMAT = 2,
  LEONIDA_XMA2_BAD_PACKET_SIZE = 3,
  LEONIDA_XMA2_OUT_OF_MEMORY = 4,
  LEONIDA_XMA2_DECODER_FAILURE = 5,
  LEONIDA_XMA2_PANIC = 6,
};

int32_t leonida_xma2_decoder_create(uint32_t sample_rate, uint32_t channels,
                                    LeonidaXma2Decoder **out_decoder);
int32_t leonida_xma2_decode_buffer(LeonidaXma2Decoder *decoder,
                                   const uint8_t *input, size_t input_len,
                                   LeonidaXma2PcmBlock **out_block);
void leonida_xma2_decoder_destroy(LeonidaXma2Decoder *decoder);
const int16_t *leonida_xma2_pcm_data(const LeonidaXma2PcmBlock *block);
size_t leonida_xma2_pcm_sample_count(const LeonidaXma2PcmBlock *block);
void leonida_xma2_pcm_destroy(LeonidaXma2PcmBlock *block);

#ifdef __cplusplus
}
#endif

#endif
