#ifndef H3_LUP_H
#define H3_LUP_H

#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_weights.h"
#include "h3_text_encoder.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_lup h3_lup;

typedef struct {
    float scale;
    uint32_t steps;
    h3_layout *layout;
    h3_text_embedding *text;
} h3_lup_ctx;

typedef void (*h3_lup_progress)(int completed_blocks, int total_blocks, void *opaque);

int h3_lup_scale_latent(int latent, float scale);

h3_lup *h3_lup_load(h3_gpu *gpu, const char *weight_directory,
                    float scale, uint32_t steps,
                    h3_layout *layout_upscaled, h3_text_embedding *text,
                    h3_lup_progress progress, void *progress_opaque,
                    char *error, size_t error_size);

void h3_lup_free(h3_lup *lup);

int h3_lup_forward(h3_lup *lup, float **video_latent,
                    int *latent_t, int *latent_h, int *latent_w,
                    h3_lup_progress progress, void *progress_opaque,
                    char *error, size_t error_size);

h3_lup_ctx *h3_lup_get_ctx(h3_lup *lup);

#endif /* H3_LUP_H */
