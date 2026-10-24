#include "h3_lup.h"
#include "h3_gpu.h"
#include "h3_host.h"
#include "h3_text_encoder.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NAME_BUFFER_LENGTH 160

/* Latent mean values for normalization (24 channels) */
static const float LATENTS_MEAN[24] = {
    0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
    -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
    -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
    -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
    -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
    -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f
};

/* Latent standard deviation values for normalization (24 channels) */
static const float LATENTS_STD[24] = {
    1.2223774194717407f, 1.2767263650894165f, 1.6831774711608887f, 1.7549455165863037f,
    1.5636216402053833f, 2.194143533706665f, 0.9653137922286987f, 1.0569885969161987f,
    0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
    0.7996809482574463f, 0.44988900423049927f, 0.7197399735450745f, 0.6936293244361877f,
    2.961095094680786f, 2.7694199085235596f, 3.0496184825897217f, 2.1088054180145264f,
    3.276226282119751f, 3.1627357006073f, 2.2816812992095947f, 2.6127843856811523f
};

enum {
    IN_BLOCKS = 6,
    OUT_BLOCKS = 6,
    IN_CHANNELS = 24,
    CHANNELS = 512,
    EMBED_KERNEL = 3,
    TEMPORAL_KERNEL = 5,
    SCALE_EMBED = 64,
    TEMPORAL_CHUNK = 32,
    NORM_GROUPS = 32,
    //TIME_INPUT = 256,
    //TIME_HIDDEN = 5376,
    //BLOCK_OUTPUT = H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN,
    //FINAL_OUTPUT = 2 * H3_DIT_HIDDEN
};

// ResBlockEmb3D
typedef struct {
    // in_layers: norm (0) + conv3d (2)
    h3_gpu_tensor *in_layers_0_w;
    h3_gpu_tensor *in_layers_0_b;
    h3_gpu_tensor *in_layers_2_w;
    h3_gpu_tensor *in_layers_2_b;
    // emb_layers: linear projection for scale/shift (1)
    h3_gpu_tensor *emb_layers_1_w;
    h3_gpu_tensor *emb_layers_1_b;
    // out_norm
    h3_gpu_tensor *out_norm_w;
    h3_gpu_tensor *out_norm_b;
    // out_layers: conv3d (2)
    h3_gpu_tensor *out_layers_2_w;
    h3_gpu_tensor *out_layers_2_b;
} h3_lup_res_emb_block;

// TemporalConv: Pre-Norm -> SiLU -> 1D Temporal DWConv -> 1x1x1 PWConv -> Residual Add
typedef struct {
    // norm
    h3_gpu_tensor *norm_w;
    h3_gpu_tensor *norm_b;
    // depthwise temporal conv: kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=channels
    h3_gpu_tensor *dwconv_w;
    h3_gpu_tensor *dwconv_b;
    // pointwise 1x1x1 conv
    h3_gpu_tensor *pwconv_w;
    h3_gpu_tensor *pwconv_b;
} h3_lup_temporal_block;

typedef struct {
    h3_lup_res_emb_block res_emb_in;
    h3_lup_temporal_block temporal;
    h3_lup_res_emb_block res_emb_out;
} h3_lup_block;

struct h3_lup {
    h3_gpu *gpu;
    h3_weight_store *weights;

    h3_lup_ctx ctx;

    h3_gpu_tensor *conv_in_w;
    h3_gpu_tensor *conv_in_b;
    h3_gpu_tensor *embed_0_w;
    h3_gpu_tensor *embed_0_b;
    h3_gpu_tensor *embed_2_w;
    h3_gpu_tensor *embed_2_b;
    h3_lup_block in_blocks[IN_BLOCKS];
    h3_lup_block out_blocks[OUT_BLOCKS];
    h3_gpu_tensor *norm_out_w;
    h3_gpu_tensor *norm_out_b;
    h3_gpu_tensor *conv_out_w;
    h3_gpu_tensor *conv_out_b;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static void report(h3_lup_progress progress, void *opaque, int completed, int total) {
    if (progress) progress(completed, total, opaque);
}

static uint16_t bfloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    bits += 0x7FFF + ((bits >> 16) & 1);
    return (uint16_t)(bits >> 16);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static inline h3_gpu_tensor *bf16_1(h3_gpu *gpu, h3_weight_store *weights, h3_gpu_tensor **tensor, const char *name,
                                    uint64_t width, char *error, size_t error_size) {
    uint64_t shape[] = {width};
    *tensor = h3_weight_load_bf16(weights, gpu, name, 1, shape, error, error_size);
    return *tensor;
}

static inline h3_gpu_tensor *bf16_2(h3_gpu *gpu, h3_weight_store *weights, h3_gpu_tensor **tensor, const char *name,
                                    uint64_t rows, uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    *tensor = h3_weight_load_bf16(weights, gpu, name, 2, shape, error, error_size);
    return *tensor;
}

static inline h3_gpu_tensor *bf16_5(h3_gpu *gpu, h3_weight_store *weights, h3_gpu_tensor **tensor, const char *name,
                                    uint64_t b, uint64_t c, uint64_t t, uint64_t h, uint64_t w, char *error, size_t error_size) {
    uint64_t shape[] = {b, c, t, h, w};
    *tensor = h3_weight_load_bf16(weights, gpu, name, 5, shape, error, error_size);
    return *tensor;
}

static inline h3_gpu_tensor *bf16_1_block(h3_gpu *gpu, h3_weight_store *weights, h3_gpu_tensor **tensor, char *name, const char *prefix, const char *suffix,
                                        uint64_t width, char *error, size_t error_size) {
    snprintf(name, NAME_BUFFER_LENGTH, "%s%s", prefix, suffix);
    uint64_t shape[] = {width};
    *tensor = h3_weight_load_bf16(weights, gpu, name, 1, shape, error, error_size);
    return *tensor;
}

static inline h3_gpu_tensor *bf16_2_block(h3_gpu *gpu, h3_weight_store *weights, h3_gpu_tensor **tensor, char *name, const char *prefix, const char *suffix,
                                        uint64_t rows, uint64_t cols, char *error, size_t error_size) {
    snprintf(name, NAME_BUFFER_LENGTH, "%s%s", prefix, suffix);
    uint64_t shape[] = {rows, cols};
    *tensor = h3_weight_load_bf16(weights, gpu, name, 2, shape, error, error_size);
    return *tensor;
}

static inline h3_gpu_tensor *bf16_5_block(h3_gpu *gpu, h3_weight_store *weights, h3_gpu_tensor **tensor, char *name, const char *prefix, const char *suffix,
                                        uint64_t b, uint64_t c, uint64_t t, uint64_t h, uint64_t w, char *error, size_t error_size) {
    snprintf(name, NAME_BUFFER_LENGTH, "%s%s", prefix, suffix);
    uint64_t shape[] = {b, c, t, h, w};
    *tensor = h3_weight_load_bf16(weights, gpu, name, 5, shape, error, error_size);
    return *tensor;
}

static inline int min(int x, int y) {
    return (x < y) ? x : y;
}

static inline int max(int x, int y) {
    return (x > y) ? x : y;
}

static inline uint32_t align_up(uint32_t val, uint32_t alignment) {
    if (alignment <= 1) return val;
    return ((val + alignment - 1) / alignment) * alignment;
}

int h3_lup_scale_latent(int latent, float scale) {
    uint32_t raw = (uint32_t)roundf((float)latent * scale);
    return align_up(raw, 2);
}

static int load_res_emb_block(h3_gpu *gpu, h3_weight_store *weights, h3_lup_res_emb_block *block, const char *prefix,
                      char *error, size_t error_size) {
    char name[NAME_BUFFER_LENGTH];
#define LOAD1_BF16(field, suffix, width) do {                                                               \
    if (!bf16_1_block(gpu, weights, &block->field, name, prefix, suffix, width, error, error_size)) return 0;        \
} while (0)
#define LOAD2_BF16(field, suffix, rows, cols) do {                                                          \
    if (!bf16_2_block(gpu, weights, &block->field, name, prefix, suffix, rows, cols, error, error_size)) return 0;   \
} while (0)
#define LOAD5_BF16(field, suffix, b, c, t, h, w) do {                                                          \
    if (!bf16_5_block(gpu, weights, &block->field, name, prefix, suffix, b, c, t, h, w, error, error_size)) return 0;   \
} while (0)
    LOAD1_BF16(in_layers_0_w, "in_layers.0.weight", CHANNELS);
    LOAD1_BF16(in_layers_0_b, "in_layers.0.bias", CHANNELS);
    LOAD5_BF16(in_layers_2_w, "in_layers.2.weight", CHANNELS, CHANNELS, EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL);
    LOAD1_BF16(in_layers_2_b, "in_layers.2.bias", CHANNELS);
    LOAD2_BF16(emb_layers_1_w, "emb_layers.1.weight", CHANNELS * 2, SCALE_EMBED);
    LOAD1_BF16(emb_layers_1_b, "emb_layers.1.bias", CHANNELS * 2);
    LOAD1_BF16(out_norm_w, "out_norm.weight", CHANNELS);
    LOAD1_BF16(out_norm_b, "out_norm.bias", CHANNELS);
    LOAD5_BF16(out_layers_2_w, "out_layers.2.weight", CHANNELS, CHANNELS, EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL);
    LOAD1_BF16(out_layers_2_b, "out_layers.2.bias", CHANNELS);
#undef LOAD1_BF16
#undef LOAD2_BF16
#undef LOAD5_BF16
    return 1;
}

static void free_res_emb_block(h3_lup_res_emb_block *block) {
    free_tensor(&block->in_layers_0_w);
    free_tensor(&block->in_layers_0_b);
    free_tensor(&block->in_layers_2_w);
    free_tensor(&block->in_layers_2_b);
    free_tensor(&block->emb_layers_1_w);
    free_tensor(&block->emb_layers_1_b);
    free_tensor(&block->out_norm_w);
    free_tensor(&block->out_norm_b);
    free_tensor(&block->out_layers_2_w);
    free_tensor(&block->out_layers_2_b);
}

static int load_temporal_block(h3_gpu *gpu, h3_weight_store *weights, h3_lup_temporal_block *block, const char *prefix,
                      char *error, size_t error_size) {
    char name[NAME_BUFFER_LENGTH];
#define LOAD1_BF16(field, suffix, width) do {                                                               \
    if (!bf16_1_block(gpu, weights, &block->field, name, prefix, suffix, width, error, error_size)) return 0;        \
} while (0)
#define LOAD5_BF16(field, suffix, b, c, t, h, w) do {                                                          \
    if (!bf16_5_block(gpu, weights, &block->field, name, prefix, suffix, b, c, t, h, w, error, error_size)) return 0;   \
} while (0)
    LOAD1_BF16(norm_w, "norm.weight", CHANNELS);
    LOAD1_BF16(norm_b, "norm.bias", CHANNELS);
    LOAD5_BF16(dwconv_w, "dwconv.weight", CHANNELS, 1, TEMPORAL_KERNEL, 1, 1);
    LOAD1_BF16(dwconv_b, "dwconv.bias", CHANNELS);
    LOAD5_BF16(pwconv_w, "pwconv.weight", CHANNELS, CHANNELS, 1, 1, 1);
    LOAD1_BF16(pwconv_b, "pwconv.bias", CHANNELS);
#undef LOAD1_BF16
#undef LOAD5_BF16
    return 1;
}

static void free_temporal_block(h3_lup_temporal_block *block) {
    free_tensor(&block->norm_w);
    free_tensor(&block->norm_b);
    free_tensor(&block->dwconv_w);
    free_tensor(&block->dwconv_b);
    free_tensor(&block->pwconv_w);
    free_tensor(&block->pwconv_b);
}

static int load_core(h3_lup *lup, h3_lup_progress progress, void *opaque,
                     char *error, size_t error_size) {
    unsigned total_blocks = (IN_BLOCKS + OUT_BLOCKS) * 3;
    unsigned blocks_index = 0;
    char prefix[64];
    report(progress, opaque, 0, total_blocks);
    for (unsigned i = 0; i < IN_BLOCKS; i++) {
        for (unsigned j = 0; j < 3; j++) {
            snprintf(prefix, sizeof(prefix), "in_blocks.%u.", i * 3 + j);
            if (j == 0) {
                if (!load_res_emb_block(lup->gpu, lup->weights, &lup->in_blocks[i].res_emb_in, prefix, error, error_size))
                    return 0;
            } else if (j == 1) {
                if (!load_temporal_block(lup->gpu, lup->weights, &lup->in_blocks[i].temporal, prefix, error, error_size))
                    return 0;
            } else {
                if (!load_res_emb_block(lup->gpu, lup->weights, &lup->in_blocks[i].res_emb_out, prefix, error, error_size))
                    return 0;
            }
            report(progress, opaque, ++blocks_index, total_blocks);
        }
    }
    for (unsigned i = 0; i < OUT_BLOCKS; i++) {
        for (unsigned j = 0; j < 3; j++) {
            snprintf(prefix, sizeof(prefix), "out_blocks.%u.", i * 3 + j);
            if (j == 0) {
                if (!load_res_emb_block(lup->gpu, lup->weights, &lup->out_blocks[i].res_emb_in, prefix, error, error_size))
                    return 0;
            } else if (j == 1) {
                if (!load_temporal_block(lup->gpu, lup->weights, &lup->out_blocks[i].temporal, prefix, error, error_size))
                    return 0;
            } else {
                if (!load_res_emb_block(lup->gpu, lup->weights, &lup->out_blocks[i].res_emb_out, prefix, error, error_size))
                    return 0;
            }
            report(progress, opaque, ++blocks_index, total_blocks);
        }
    }
#define LOAD1_BF16(field, name, width) do {                                                               \
    if (!bf16_1(lup->gpu, lup->weights, &lup->field, name, width, error, error_size)) return 0;        \
} while (0)
#define LOAD2_BF16(field, name, rows, cols) do {                                                          \
    if (!bf16_2(lup->gpu, lup->weights, &lup->field, name, rows, cols, error, error_size)) return 0;   \
} while (0)
#define LOAD5_BF16(field, name, b, c, t, h, w) do {                                                          \
    if (!bf16_5(lup->gpu, lup->weights, &lup->field, name, b, c, t, h, w, error, error_size)) return 0;   \
} while (0)
    LOAD5_BF16(conv_in_w, "conv_in.weight", CHANNELS, IN_CHANNELS, EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL);
    LOAD1_BF16(conv_in_b, "conv_in.bias", CHANNELS);
    LOAD2_BF16(embed_0_w, "embed.0.weight", SCALE_EMBED, 1);
    LOAD1_BF16(embed_0_b, "embed.0.bias", SCALE_EMBED);
    LOAD2_BF16(embed_2_w, "embed.2.weight", SCALE_EMBED, SCALE_EMBED);
    LOAD1_BF16(embed_2_b, "embed.2.bias", SCALE_EMBED);
    LOAD1_BF16(norm_out_w, "norm_out.weight", CHANNELS);
    LOAD1_BF16(norm_out_b, "norm_out.bias", CHANNELS);
    LOAD5_BF16(conv_out_w, "conv_out.weight", IN_CHANNELS, CHANNELS, EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL);
    LOAD1_BF16(conv_out_b, "conv_out.bias", IN_CHANNELS);
#undef LOAD1_BF16
#undef LOAD2_BF16
#undef LOAD5_BF16
    return 1;
}

h3_lup_ctx *h3_lup_get_ctx(h3_lup *lup) {
    return &lup->ctx;
}

h3_lup *h3_lup_load(h3_gpu *gpu, const char *weight_directory,
                    float scale, uint32_t steps,
                    h3_layout *layout, h3_text_embedding *text,
                    h3_lup_progress progress, void *progress_opaque,
                    char *error, size_t error_size) {
    h3_lup *lup = calloc(1, sizeof(*lup));
    if (!lup) {
        fail(error, error_size, "out of memory creating latent upscaler");
        return NULL;
    }
    lup->gpu = gpu;
    lup->weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!lup->weights) goto failed;

    lup->ctx.scale = scale;
    lup->ctx.steps = steps;
    lup->ctx.layout = layout;
    lup->ctx.text = text;

    if (!load_core(lup, progress, progress_opaque, error, error_size)) goto failed;

    return lup;
failed:
    h3_lup_free(lup);
    return NULL;
}

void h3_lup_free(h3_lup *lup) {
    if (!lup) return;
    for (unsigned i = 0; i < IN_BLOCKS; i++) {
        free_res_emb_block(&lup->in_blocks[i].res_emb_in);
        free_temporal_block(&lup->in_blocks[i].temporal);
        free_res_emb_block(&lup->in_blocks[i].res_emb_out);
    }
    for (unsigned i = 0; i < OUT_BLOCKS; i++) {
        free_res_emb_block(&lup->out_blocks[i].res_emb_in);
        free_temporal_block(&lup->out_blocks[i].temporal);
        free_res_emb_block(&lup->out_blocks[i].res_emb_out);
    }
    free_tensor(&lup->conv_in_w);
    free_tensor(&lup->conv_in_b);
    free_tensor(&lup->embed_0_w);
    free_tensor(&lup->embed_0_b);
    free_tensor(&lup->embed_2_w);
    free_tensor(&lup->embed_2_b);
    free_tensor(&lup->norm_out_w);
    free_tensor(&lup->norm_out_b);
    free_tensor(&lup->conv_out_w);
    free_tensor(&lup->conv_out_b);
    h3_weight_store_free(lup->weights);
    h3_text_embedding_free(lup->ctx.text);
    h3_layout_free(lup->ctx.layout);
    free(lup);
}

static int h3_lup_forward_res_emb(h3_gpu* gpu, h3_lup_res_emb_block* block,
                                    h3_gpu_tensor *output, h3_gpu_tensor *input, h3_gpu_tensor *residual,
                                    h3_gpu_tensor *emb_out, const h3_gpu_tensor *emb_in,
                                    uint32_t b, uint32_t c, uint32_t t, uint32_t h, uint32_t w,
                                    char *error, size_t error_size) {
    int ok = 1;
#define OP(call, label) do {                                                    \
    if (!gpu_op(gpu, (call), error, error_size, label)) { ok = 0; goto end; }   \
} while (0)
    // Residual
    size_t elements = (size_t)b * c * t * h * w;
    OP(h3_gpu_copy_bf16(gpu, residual, 0, input, 0, elements),
        "lup res_emb residual");
    // In
    OP(h3_gpu_group_norm_bf16(gpu, output, input, block->in_layers_0_w, block->in_layers_0_b, b, c, t, h, w, NORM_GROUPS),
        "lup res_emb in norm");
    OP(h3_gpu_silu_bf16(gpu, input, output, (uint32_t)elements),
        "lup res_emb in silu");
    OP(h3_gpu_conv3d_bf16(gpu, output, input,
                        block->in_layers_2_w, block->in_layers_2_b,
                        b, c, t, h, w, c, 1,
                        EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL,
                        1, 1, 1),
        "lup res_emb in conv3d");
    // Scale Embed
    //OP(h3_gpu_silu_bf16(gpu, emb, emb, SCALE_EMBED), // Moved upstream to save redundant SiLUs
    //    "lup res_emb emb silu");
    OP(h3_gpu_linear_bf16(gpu, emb_out, emb_in, block->emb_layers_1_w, block->emb_layers_1_b, 1, SCALE_EMBED, CHANNELS * 2),
        "lup res_emb emb linear");
    // Out
    OP(h3_gpu_group_norm_bf16(gpu, input, output, block->out_norm_w, block->out_norm_b, b, c, t, h, w, NORM_GROUPS),
        "lup res_emb out norm");
    OP(h3_gpu_scale_embed_bf16(gpu, output, input, emb_out, b, c, t, h, w),
        "lup res_emb out scale_embed");
    OP(h3_gpu_silu_bf16(gpu, output, output, (uint32_t)elements),
        "lup res_emb out silu");
    OP(h3_gpu_conv3d_bf16(gpu, input, output,
                        block->out_layers_2_w, block->out_layers_2_b,
                        b, c, t, h, w, c, 1,
                        EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL,
                        1, 1, 1),
        "lup res_emb out conv3d");
    // Residual Add
    OP(h3_gpu_add_bf16(gpu, output, input, residual, (uint32_t)elements),
        "lup res_emb residual add");
end:
#undef OP
    return ok;
}

static int h3_lup_forward_temporal(h3_gpu* gpu, h3_lup_temporal_block* block,
                                    h3_gpu_tensor *output, h3_gpu_tensor *input, h3_gpu_tensor *residual,
                                    uint32_t b, uint32_t c, uint32_t t, uint32_t h, uint32_t w,
                                    char *error, size_t error_size) {
    int ok = 1;
#define OP(call, label) do {                                                    \
    if (!gpu_op(gpu, (call), error, error_size, label)) { ok = 0; goto end; }   \
} while (0)
    // Residual
    size_t elements = (size_t)b * c * t * h * w;
    OP(h3_gpu_copy_bf16(gpu, residual, 0, input, 0, elements),
        "lup temporal residual");
    // Norm
    OP(h3_gpu_group_norm_bf16(gpu, output, input, block->norm_w, block->norm_b, b, c, t, h, w, NORM_GROUPS),
        "lup temporal norm");
    // SiLU
    OP(h3_gpu_silu_bf16(gpu, input, output, (uint32_t)elements),
        "lup temporal silu");
    // DW Conv
    uint32_t padding = (uint32_t)TEMPORAL_KERNEL / 2;
    OP(h3_gpu_conv3d_bf16(gpu, output, input,
                        block->dwconv_w, block->dwconv_b,
                        b, c, t, h, w, c, CHANNELS,
                        TEMPORAL_KERNEL, 1, 1,
                        padding, 0, 0),
        "lup temporal dwconv");
    // PW Conv
    OP(h3_gpu_conv3d_bf16(gpu, input, output,
                        block->pwconv_w, block->pwconv_b,
                        b, c, t, h, w, c, 1,
                        1, 1, 1,
                        0, 0, 0),
        "lup temporal pwconv");
    // Residual Add
    OP(h3_gpu_add_bf16(gpu, output, input, residual, (uint32_t)elements),
        "lup temporal residual add");
end:
#undef OP
    return ok;
}

static int h3_lup_forward_chunk(h3_lup* lup, h3_gpu_tensor *output, h3_gpu_tensor *input,
                                h3_gpu_tensor *x_in, h3_gpu_tensor *x_in_out, h3_gpu_tensor *x_in_res,
                                h3_gpu_tensor *x_out, h3_gpu_tensor *x_out_out, h3_gpu_tensor *x_out_res,
                                h3_gpu_tensor *scale_embed_in, h3_gpu_tensor *scale_embed_out,
                                uint32_t b, uint32_t c, uint32_t t, uint32_t h, uint32_t w,
                                uint32_t h_out, uint32_t w_out,
                                char *error, size_t error_size) {
    h3_gpu *gpu = lup->gpu;
    int ok = 1;
#define OP(call, label) do {                                                            \
    if (!gpu_op(gpu, (call), error, error_size, label)) { ok = 0; goto end; }    \
} while (0)
    // Begin chunk ops
    OP(h3_gpu_begin(gpu),
        "lup chunk begin");

    // DEBUG
    //OP(h3_gpu_interpolate_bf16(gpu, output, input, (uint32_t)b * c * t, h, w, h_out, w_out),
    //    "lup chunk debug interpolate");

    // Conv In
    OP(h3_gpu_conv3d_bf16(gpu, x_in, input,
                        lup->conv_in_w, lup->conv_in_b,
                        b, c, t, h, w, CHANNELS, 1,
                        EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL,
                        1, 1, 1),
        "lup chunk conv_in");

    // In Blocks
    for (unsigned i = 0; i < IN_BLOCKS; i++) {
        // Res Emb In
        if (!h3_lup_forward_res_emb(gpu, &lup->in_blocks[i].res_emb_in, x_in_out, x_in, x_in_res,
                                scale_embed_out, scale_embed_in,
                                b, CHANNELS, t, h, w, error, error_size) ||
        // Temporal
            !h3_lup_forward_temporal(gpu, &lup->in_blocks[i].temporal, x_in, x_in_out, x_in_res,
                                b, CHANNELS, t, h, w, error, error_size) ||
        // Res Emb Out
            !h3_lup_forward_res_emb(gpu, &lup->in_blocks[i].res_emb_out, x_in_out, x_in, x_in_res,
                                scale_embed_out, scale_embed_in,
                                b, CHANNELS, t, h, w, error, error_size) ) {
            ok = 0; goto end;
        }
        // Swap input and output tensors
        //h3_gpu_copy_bf16(gpu, x_in, 0, x_in_out, 0, (size_t)b * CHANNELS * t * h * w);
        h3_gpu_tensor *swap = x_in; x_in = x_in_out; x_in_out = swap;
    }

    // Interpolate (Trilinear)
    OP(h3_gpu_interpolate_bf16(gpu, x_out, x_in, (uint32_t)b * CHANNELS * t, h, w, h_out, w_out),
        "lup chunk interpolate");

    // Out blocks
    for (unsigned i = 0; i < OUT_BLOCKS; i++) {
        // Res Emb In
        if (!h3_lup_forward_res_emb(gpu, &lup->out_blocks[i].res_emb_in, x_out_out, x_out, x_out_res,
                                scale_embed_out, scale_embed_in,
                                b, CHANNELS, t, h_out, w_out, error, error_size) ||
        // Temporal
            !h3_lup_forward_temporal(gpu, &lup->out_blocks[i].temporal, x_out, x_out_out, x_out_res,
                                b, CHANNELS, t, h_out, w_out, error, error_size) ||
        // Res Emb Out
            !h3_lup_forward_res_emb(gpu, &lup->out_blocks[i].res_emb_out, x_out_out, x_out, x_out_res,
                                scale_embed_out, scale_embed_in,
                                b, CHANNELS, t, h_out, w_out, error, error_size) ) {
            ok = 0; goto end;
        }
        // Swap input and output tensors
        //h3_gpu_copy_bf16(gpu, x_out, 0, x_out_out, 0, (size_t)b * CHANNELS * t * h_out * w_out);
        h3_gpu_tensor *swap = x_out; x_out = x_out_out; x_out_out = swap;
    }

    // Norm Out, SiLU, Conv Out
    OP(h3_gpu_group_norm_bf16(gpu, x_out_out, x_out, lup->norm_out_w, lup->norm_out_b,
                        b, CHANNELS, t, h_out, w_out, NORM_GROUPS),
        "lup chunk norm_out");
    uint32_t x_out_elements = b * CHANNELS * t * h_out * w_out;
    OP(h3_gpu_silu_bf16(gpu, x_out, x_out_out, x_out_elements),
        "lup chunk silu");
    OP(h3_gpu_conv3d_bf16(gpu, output, x_out,
                        lup->conv_out_w, lup->conv_out_b,
                        b, CHANNELS, t, h_out, w_out, c, 1,
                        EMBED_KERNEL, EMBED_KERNEL, EMBED_KERNEL,
                        1, 1, 1),
        "lup chunk conv_out");

    // Submit chunk ops
    OP(h3_gpu_submit(gpu),
        "lup chunk submit");
end:
#undef OP
    return ok;
}

int h3_lup_forward(h3_lup *lup, float **video_latent,
                    int *latent_t, int *latent_h, int *latent_w,
                    h3_lup_progress progress, void *progress_opaque,
                    char *error, size_t error_size) {
    h3_gpu *gpu = lup->gpu;
    int ok = 1;
    int b = 1;
    int c = IN_CHANNELS;
    int t = *latent_t;
    int h = *latent_h;
    int w = *latent_w;

    float scale = lup->ctx.scale;
    const h3_layout *layout = lup->ctx.layout;
    uint32_t h_out = layout->signature[2];
    uint32_t w_out = layout->signature[3];

    // TODO: Global var to enable chunking
    int t_chunk_enabled = 1;
    // Is chunking actually needed/used?
    int t_chunk_used = t_chunk_enabled && (t > TEMPORAL_CHUNK);

    // Latent dimensions
    //uint32_t h_raw = (uint32_t)roundf((float)h * scale);
    //uint32_t w_raw = (uint32_t)roundf((float)w * scale);
    //uint32_t h_out = align_up(h_raw, 2);
    //uint32_t w_out = align_up(w_raw, 2);
    // Pixel dimensions
    //uint32_t h_pixel_in = h * H3_VAE_SPATIAL_RATIO;
    //uint32_t w_pixel_in = w * H3_VAE_SPATIAL_RATIO;
    //uint32_t h_pixel_out = h_out * H3_VAE_SPATIAL_RATIO;
    //uint32_t w_pixel_out = w_out * H3_VAE_SPATIAL_RATIO;

    //printf("\n");
    //printf("lup in  h=%u, w=%u\n", (uint32_t)h, (uint32_t)w);
    //printf("lup out h=%u, w=%u\n", h_out, w_out);
    //fflush(stdout);

    int progress_count = t_chunk_used ? (t + TEMPORAL_CHUNK - 1) / TEMPORAL_CHUNK : 1;
    report(progress, progress_opaque, 0, progress_count);

#define OP(call, label) do {                                                            \
    if (!gpu_op(gpu, (call), error, error_size, label)) { ok = 0; goto free_input; }    \
} while (0)

    // Allocate tensors
    uint16_t scale_bf16 = bfloat16(scale - 1.0f); // Scale is offset by 1.0 for embed
    h3_gpu_tensor *scale_tensor = h3_gpu_tensor_load(gpu, &scale_bf16, 1, H3_GPU_BF16);
    h3_gpu_tensor *scale_embed = h3_gpu_tensor_new(gpu, SCALE_EMBED, H3_GPU_BF16);
    h3_gpu_tensor *scale_embed_in = h3_gpu_tensor_new(gpu, SCALE_EMBED, H3_GPU_BF16);
    h3_gpu_tensor *scale_embed_out = h3_gpu_tensor_new(gpu, CHANNELS * 2, H3_GPU_BF16);
    h3_gpu_tensor *mean = h3_gpu_tensor_load(lup->gpu, LATENTS_MEAN, IN_CHANNELS, H3_GPU_F32);
    h3_gpu_tensor *std = h3_gpu_tensor_load(lup->gpu, LATENTS_STD, IN_CHANNELS, H3_GPU_F32);
    size_t x_elements = (size_t)b * c * t * h * w;
    h3_gpu_tensor *x_f32 = h3_gpu_tensor_from_f32_nocopy(lup->gpu, *video_latent, x_elements);
    h3_gpu_tensor *x = h3_gpu_tensor_new(lup->gpu, x_elements, H3_GPU_BF16);
    if (!scale_tensor || !scale_embed || !scale_embed_out ||
        !mean || !std || !x_f32 || !x) { ok = 0; goto free_input; }

    // Begin input operations
    OP(h3_gpu_begin(gpu),
        "lup input begin");
    // Standardize
    OP(h3_gpu_standardize_f32(gpu, x_f32, x_f32, mean, std, (uint32_t)x_elements, IN_CHANNELS, (uint32_t)t * h * w),
        "lup input standardize");
    // Cast to bf16
    OP(h3_gpu_cast(gpu, x, x_f32, (uint32_t)x_elements),
        "lup input cast");
    // Scale Embed
    OP(h3_gpu_linear_bf16(gpu, scale_embed, scale_tensor, lup->embed_0_w, lup->embed_0_b, 1, 1, SCALE_EMBED),
        "lup input embed embed_0");
    OP(h3_gpu_silu_bf16(gpu, scale_embed, scale_embed, SCALE_EMBED),
        "lup input embed silu");
    OP(h3_gpu_linear_bf16(gpu, scale_embed_in, scale_embed, lup->embed_2_w, lup->embed_2_b, 1, SCALE_EMBED, SCALE_EMBED),
        "lup input embed embed_2");
    OP(h3_gpu_silu_bf16(gpu, scale_embed_in, scale_embed_in, SCALE_EMBED), // Moved upstream to save redundant SiLUs
        "lup input embed pre-silu");
    // Process input operations
    OP(h3_gpu_submit(gpu),
        "lup input submit");

    // Temporal Pad (if enabled)
    if (t_chunk_used) {
        // Allocate temporal padded input (if enabled)
        uint32_t t_padded = t + 2 * TEMPORAL_KERNEL;
        size_t x_padded_elements = (size_t)b * c * t_padded * h * w;
        h3_gpu_tensor *x_padded = h3_gpu_tensor_new(gpu, x_padded_elements, H3_GPU_BF16);
        if (!x_padded) { ok = 0; goto free_input; }
        // Apply pad
#define PAD_OP(call, label) gpu_op(gpu, (call), error, error_size, label)
        if (!PAD_OP(h3_gpu_begin(gpu),
                "lup pad begin") ||
            !PAD_OP(h3_gpu_temporal_pad_bf16(gpu, x_padded, x, b, c, t, h, w, TEMPORAL_KERNEL),
                "lup input pad") ||
            !PAD_OP(h3_gpu_submit(gpu),
                "lup pad submit")) {
            free_tensor(&x_padded);
            ok = 0;
            goto free_input;
        }
#undef PAD_OP
        // Replace x with x_padded
        free_tensor(&x);
        x = x_padded;
    }

free_input:
    free_tensor(&scale_tensor);
    free_tensor(&scale_embed);
    free_tensor(&x_f32);
    if (!ok) goto free_chunk;

    // Allocate activation tensors
    uint32_t t_chunk_max = t_chunk_used ? TEMPORAL_CHUNK + (TEMPORAL_KERNEL * 2) : t;
    size_t x_in_elements = (size_t)b * CHANNELS * t_chunk_max * h * w;
    h3_gpu_tensor *x_in = h3_gpu_tensor_new(gpu, x_in_elements, H3_GPU_BF16);
    h3_gpu_tensor *x_in_out = h3_gpu_tensor_new(gpu, x_in_elements, H3_GPU_BF16);
    h3_gpu_tensor *x_in_res = h3_gpu_tensor_new(gpu, x_in_elements, H3_GPU_BF16);
    size_t x_out_elements = (size_t)b * CHANNELS * t_chunk_max * h_out * w_out;
    h3_gpu_tensor *x_out = h3_gpu_tensor_new(gpu, x_out_elements, H3_GPU_BF16);
    h3_gpu_tensor *x_out_out = h3_gpu_tensor_new(gpu, x_out_elements, H3_GPU_BF16);
    h3_gpu_tensor *x_out_res = h3_gpu_tensor_new(gpu, x_out_elements, H3_GPU_BF16);
    size_t output_elements = (size_t)b * c * t * h_out * w_out;
    h3_gpu_tensor *output = h3_gpu_tensor_new(gpu, output_elements, H3_GPU_BF16);
    if (!x_in || !x_in_out || !x_in_res ||
        !x_out || !x_out_out || !x_out_res ||
        !output) { ok = 0; goto free_input; }

    // Forward Pass
    //printf("\nt_chunk_used %d\n", t_chunk_used);
    //fflush(stdout);
    if (!t_chunk_used) {
        ok = h3_lup_forward_chunk(lup, output, x,
                             x_in, x_in_out, x_in_res,
                             x_out, x_out_out, x_out_res,
                             scale_embed_in, scale_embed_out,
                             b, c, t, h, w,
                             h_out, w_out,
                             error, error_size);
        if (!ok) goto free_input;
        report(progress, progress_opaque, progress_count, progress_count);
    } else {
        // Allocate chunking tensors
        size_t chunk_in_elements = (size_t)b * c * t_chunk_max * h * w;
        h3_gpu_tensor *chunk_in = h3_gpu_tensor_new(gpu, chunk_in_elements, H3_GPU_BF16);
        size_t chunk_out_elements = (size_t)b * c * t_chunk_max * h_out * w_out;
        h3_gpu_tensor *chunk_out = h3_gpu_tensor_new(gpu, chunk_out_elements, H3_GPU_BF16);
        size_t slice_out_elements = (size_t)b * c * t_chunk_max * h_out * w_out;
        h3_gpu_tensor *slice_out = h3_gpu_tensor_new(gpu, slice_out_elements, H3_GPU_BF16);
        if (!output || !chunk_in || !chunk_out || !slice_out ) { ok = 0; goto free_input; }

#define CHUNK_OP(call, label) do {                                                            \
    if (!gpu_op(gpu, (call), error, error_size, label)) { ok = 0; goto chunk_end; }    \
} while (0)
        // Zero output accumulator: [B, C, T, H_out, W_out]
        CHUNK_OP(h3_gpu_zero(gpu, output, output_elements, sizeof(uint16_t)), "lup zero output");

        // Temporal Chunking
        for(int start=0; start < t; start += TEMPORAL_CHUNK) {
            //printf("\nlup chunk start=%u\n", (uint32_t)start);
            //fflush(stdout);
            // Chunk Split
            int end = min(t, start + TEMPORAL_CHUNK);
            // Indices for output chunk: Include start/end blend regions if needed
            int out_start = max(0, start - TEMPORAL_KERNEL);
            int out_end = min(t, end + TEMPORAL_KERNEL);
            int out_count = out_end - out_start;
            // Indices for input x chunk: Includes start/end blend regions if needed
            int in_start = max(0, out_start - TEMPORAL_KERNEL);
            int in_end = min(t + 2 * TEMPORAL_KERNEL, out_end + TEMPORAL_KERNEL);
            int in_count = in_end - in_start;

            // Slice chunk input (from x_padded)
            CHUNK_OP(h3_gpu_begin(gpu),
                "lup chunk slice in begin");
            uint32_t t_padded = t + 2 * TEMPORAL_KERNEL;
            CHUNK_OP(h3_gpu_temporal_slice_bf16(gpu, chunk_in, x, b, c, t_padded, h, w, in_start, in_end),
                "lup chunk slice in");
            CHUNK_OP(h3_gpu_submit(gpu),
                "lup chunk slice in submit");

            // Chunk forward
            ok = h3_lup_forward_chunk(lup, chunk_out, chunk_in,
                                 x_in, x_in_out, x_in_res,
                                 x_out, x_out_out, x_out_res,
                                 scale_embed, scale_embed_out,
                                 b, c, in_count, h, w,
                                 h_out, w_out,
                                 error, error_size);
            if (!ok) goto chunk_end;

            // Slice and blend chunk out
            CHUNK_OP(h3_gpu_begin(gpu),
                "lup chunk slice in begin");
            int slice_start = (out_start + TEMPORAL_KERNEL) - in_start;
            int slice_end = slice_start + out_count;
            CHUNK_OP(h3_gpu_temporal_slice_bf16(gpu, slice_out, chunk_out, b, c, in_count, h_out, w_out, slice_start, slice_end),
                "lup chunk slice out");
            uint32_t overlap_start = (start == 0) ? 0 : TEMPORAL_KERNEL;
            uint32_t overlap_end   = (end == t)   ? 0 : TEMPORAL_KERNEL;
            uint32_t offset = (uint32_t)start;
            CHUNK_OP(h3_gpu_temporal_blend_bf16(gpu, output, slice_out,
                                b, c, t, h_out, w_out, out_count,
                                overlap_start, overlap_end,
                                offset),
                "lup chunk blend");
            CHUNK_OP(h3_gpu_submit(gpu),
                "lup chunk submit");

            if (ok) report(progress, progress_opaque, (start / TEMPORAL_CHUNK) + 1, progress_count);
        }
    chunk_end:
        free_tensor(&chunk_in);
        free_tensor(&chunk_out);
        free_tensor(&slice_out);
        if (!ok) goto free_input;
    }

free_chunk:
    free_tensor(&scale_embed_in);
    free_tensor(&scale_embed_out);
    free_tensor(&x);

    free_tensor(&x_in);
    free_tensor(&x_in_out);
    free_tensor(&x_in_res);
    free_tensor(&x_out);
    free_tensor(&x_out_out);
    free_tensor(&x_out_res);

    if (!ok) goto free_output;

    // Allocate output tensors
    h3_gpu_tensor *output_f32 = h3_gpu_tensor_new(gpu, output_elements, H3_GPU_F32);
    if (!output_f32) { ok = 0; goto free_input; }

    // Begin output ops
    OP(h3_gpu_begin(gpu),
        "lup output begin");
    // Cast back to float for output
    OP(h3_gpu_cast(gpu, output_f32, output, (uint32_t)output_elements),
        "lup output cast");
    // Un-standardize
    OP(h3_gpu_unstandardize_f32(gpu, output_f32, output_f32, mean, std, (uint32_t)output_elements, IN_CHANNELS, (uint32_t)t * h_out * w_out),
        "lup output unstandardize");
    // Submit output ops
    OP(h3_gpu_submit(gpu),
        "lup output submit");

free_output:
    free_tensor(&output);
    free_tensor(&mean);
    free_tensor(&std);
    if (!ok) goto free_end;

    // Replace video latent
    // TODO: We really need to be passing h3_gpu_tensors to/from the dit/lup not float arrays, this could save quite a few copy/malloc ops
    free(*video_latent);
    *video_latent = malloc(output_elements * sizeof(float));
    if (!*video_latent) { ok = 0; goto free_input; }
    h3_gpu_tensor_read_f32(output_f32, *video_latent, output_elements);
free_end:
    free_tensor(&output_f32);
#undef OP
    if (!ok) {
        printf("\nh3_lup_forward failed\n");
        fflush(stdout);
    }
    return ok;
}
