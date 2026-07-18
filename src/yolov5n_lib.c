/**
 * Production YOLOv5n NPU wrapper for TH1520 (VIP9000) / LicheePi 4A.
 *
 * Design goals
 * ------------
 * 1. Zero malloc on the hot path (persistent tensors + prealloc f32 dequant).
 * 2. Maximize NPU duty cycle: hold device lock ONLY around session_run so
 *    preprocess/postprocess of other handles can overlap.
 * 3. Multi-model safe: one handle = one session; NPU is time-shared.
 * 4. Stable heap: never free() qinfo that aliases model.params.
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

#include "io.h"
#include "shl_c920.h"
#include "yolov5n_lib.h"

#ifndef YOLO_API
#if defined(__GNUC__)
#define YOLO_API __attribute__((visibility("default")))
#else
#define YOLO_API
#endif
#endif

#define INPUT_HEIGHT      YOLO_INPUT_H
#define INPUT_WIDTH       YOLO_INPUT_W
#define MAX_DETECT        1024
#define NUM_OUTPUTS       3
#define OUT_CHANNELS      255
#define CTX_MAGIC         0x59304E31u  /* 'Y0N1' */
#define CLIP_INT8(v)      ((v) > 127 ? 127 : ((v) < -128 ? -128 : (v)))

#if defined(__GNUC__)
#define HOT       __attribute__((hot))
#define RESTRICT  __restrict__
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PREFETCH(p) __builtin_prefetch((p), 0, 3)
#else
#define HOT
#define RESTRICT
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#define PREFETCH(p) ((void)0)
#endif

void *csinn_(char *params);
void csinn_update_input_and_run(struct csinn_tensor **input_tensors, void *sess);

/* VIP9000: one graph execution at a time per process. */
static pthread_mutex_t g_npu_lock = PTHREAD_MUTEX_INITIALIZER;

static const int k_out_h[NUM_OUTPUTS] = {80, 40, 20};
static const int k_out_w[NUM_OUTPUTS] = {80, 40, 20};

typedef struct {
    uint32_t magic;
    void *sess;
    char *params_blob;           /* must outlive sess (qinfo aliases into it) */

    struct csinn_tensor *input_tensor;
    int8_t *input_bufs[2];       /* ping-pong: prep next while NPU holds prev */
    int input_size;
    int buf_idx;

    int8_t lut[256] __attribute__((aligned(64)));

    struct csinn_tensor *raw_out[NUM_OUTPUTS];
    struct csinn_tensor *f32_out[NUM_OUTPUTS];
    float *f32_bufs[NUM_OUTPUTS];
    int f32_elems[NUM_OUTPUTS];
    int output_num;

    struct shl_yolov5_params yolo_params;
    struct shl_yolov5_box *boxes;

    uint64_t last_pre_ns;
    uint64_t last_npu_ns;
    uint64_t last_post_ns;

    pthread_mutex_t lock;        /* same-handle reentrancy guard */
} ModelContext;

static inline ModelContext *ctx_from(void *handle)
{
    ModelContext *ctx = (ModelContext *)handle;
    if (UNLIKELY(!ctx || ctx->magic != CTX_MAGIC || !ctx->sess)) return NULL;
    return ctx;
}

static void safe_free_owned_tensor(struct csinn_tensor *t)
{
    if (!t) return;
    /* f32 shells: data lives in ctx->f32_bufs[]; qinfo always NULL */
    t->data = NULL;
    t->qinfo = NULL;
    t->quant_channel = 0;
    csinn_free_tensor(t);
}

static void build_quant_lut(ModelContext *ctx)
{
    struct csinn_tensor *sess_input = ((struct csinn_session *)ctx->sess)->input[0];
    float scale = sess_input->qinfo[0].scale;
    int32_t zero_point = sess_input->qinfo[0].zero_point;
    float multiplier = 1.0f / (255.0f * scale);

    for (int v = 0; v < 256; v++) {
        int32_t val = (int32_t)roundf((float)v * multiplier) + zero_point;
        ctx->lut[v] = (int8_t)CLIP_INT8(val);
    }
}

static void init_yolo_params(struct shl_yolov5_params *params)
{
    memset(params, 0, sizeof(*params));
    params->conf_thres = 0.25f;
    params->iou_thres = 0.45f;
    params->strides[0] = 8;
    params->strides[1] = 16;
    params->strides[2] = 32;
    static const float anchors[18] = {
        10.f, 13.f, 16.f, 30.f, 33.f, 23.f,
        30.f, 61.f, 62.f, 45.f, 59.f, 119.f,
        116.f, 90.f, 156.f, 198.f, 373.f, 326.f
    };
    memcpy(params->anchors, anchors, sizeof(anchors));
}

/*
 * BGR HWC uint8 → RGB NCHW int8 via LUT.
 * Tight loop: prefetch next cache line, no branches beyond LUT load.
 */
static HOT void quantize_bgr_to_nchw(const uint8_t *RESTRICT bgr,
                                     int8_t *RESTRICT dst,
                                     const int8_t *RESTRICT lut)
{
    /* size_t avoids GCC -Waggressive-loop-optimizations on signed i*3 UB */
    const size_t hw = (size_t)INPUT_HEIGHT * (size_t)INPUT_WIDTH;
    int8_t *RESTRICT ch_r = dst;
    int8_t *RESTRICT ch_g = dst + hw;
    int8_t *RESTRICT ch_b = dst + 2 * hw;
    const uint8_t *RESTRICT p = bgr;

    size_t i = 0;
    for (; i + 8 <= hw; i += 8) {
        PREFETCH(p + 192);
        for (int k = 0; k < 8; k++, p += 3) {
            ch_b[i + (size_t)k] = lut[p[0]];
            ch_g[i + (size_t)k] = lut[p[1]];
            ch_r[i + (size_t)k] = lut[p[2]];
        }
    }
    for (; i < hw; i++, p += 3) {
        ch_b[i] = lut[p[0]];
        ch_g[i] = lut[p[1]];
        ch_r[i] = lut[p[2]];
    }
}

/* Manual INT8→F32 into a preallocated buffer (no shl_c920_* malloc). */
static HOT void dequant_i8_to_f32(const int8_t *RESTRICT src,
                                  float *RESTRICT dst,
                                  int n, float scale, int32_t zp)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        dst[i + 0] = ((int32_t)src[i + 0] - zp) * scale;
        dst[i + 1] = ((int32_t)src[i + 1] - zp) * scale;
        dst[i + 2] = ((int32_t)src[i + 2] - zp) * scale;
        dst[i + 3] = ((int32_t)src[i + 3] - zp) * scale;
    }
    for (; i < n; i++)
        dst[i] = ((int32_t)src[i] - zp) * scale;
}

static void *create_graph(const char *params_path, char **params_out)
{
    int binary_size = 0;
    char *params = get_binary_from_file((char *)params_path, &binary_size);
    if (!params) return NULL;
    *params_out = params;

    size_t len = strlen(params_path);
    if (len >= 7 && strcmp(params_path + len - 7, ".params") == 0)
        return csinn_(params);

    if (len >= 3 && strcmp(params_path + len - 3, ".bm") == 0) {
        struct shl_bm_sections *section = (struct shl_bm_sections *)(params + 4128);
        if (section->graph_offset)
            return csinn_import_binary_model(params);
        return csinn_(params + section->params_offset * 4096);
    }

    free(params);
    *params_out = NULL;
    return NULL;
}

static int alloc_io_buffers(ModelContext *ctx)
{
    ctx->input_size = csinn_tensor_byte_size(((struct csinn_session *)ctx->sess)->input[0]);

    for (int b = 0; b < 2; b++) {
        ctx->input_bufs[b] = (int8_t *)shl_mem_alloc_aligned(ctx->input_size, 0);
        if (!ctx->input_bufs[b]) return -1;
    }

    ctx->output_num = csinn_get_output_number(ctx->sess);
    if (ctx->output_num > NUM_OUTPUTS) ctx->output_num = NUM_OUTPUTS;
    if (ctx->output_num < 1) return -1;

    for (int i = 0; i < ctx->output_num; i++) {
        ctx->raw_out[i] = csinn_alloc_tensor(NULL);
        ctx->f32_out[i] = csinn_alloc_tensor(NULL);
        if (!ctx->raw_out[i] || !ctx->f32_out[i]) return -1;

        ctx->f32_elems[i] = OUT_CHANNELS * k_out_h[i] * k_out_w[i];
        ctx->f32_bufs[i] = (float *)shl_mem_alloc((size_t)ctx->f32_elems[i] * sizeof(float));
        if (!ctx->f32_bufs[i]) return -1;

        ctx->f32_out[i]->dim_count = 4;
        ctx->f32_out[i]->dim[0] = 1;
        ctx->f32_out[i]->dim[1] = OUT_CHANNELS;
        ctx->f32_out[i]->dim[2] = k_out_h[i];
        ctx->f32_out[i]->dim[3] = k_out_w[i];
        ctx->f32_out[i]->dtype = CSINN_DTYPE_FLOAT32;
        ctx->f32_out[i]->layout = CSINN_LAYOUT_NCHW;
        ctx->f32_out[i]->qinfo = NULL;
        ctx->f32_out[i]->quant_channel = 0;
        ctx->f32_out[i]->data = ctx->f32_bufs[i];
    }

    ctx->boxes = (struct shl_yolov5_box *)malloc(MAX_DETECT * sizeof(struct shl_yolov5_box));
    return ctx->boxes ? 0 : -1;
}

static void destroy_ctx_resources(ModelContext *ctx)
{
    if (!ctx) return;

    if (ctx->boxes) {
        free(ctx->boxes);
        ctx->boxes = NULL;
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        if (ctx->f32_bufs[i]) {
            shl_mem_free(ctx->f32_bufs[i]);
            ctx->f32_bufs[i] = NULL;
        }
        safe_free_owned_tensor(ctx->f32_out[i]);
        ctx->f32_out[i] = NULL;
        /*
         * raw_out: after a frame we NULL qinfo (may have aliased .params).
         * If never run, qinfo is still the heap block from alloc_tensor — leave
         * it for csinn_free_tensor. Only detach session-owned data pointer.
         */
        if (ctx->raw_out[i]) {
            ctx->raw_out[i]->data = NULL;
            csinn_free_tensor(ctx->raw_out[i]);
            ctx->raw_out[i] = NULL;
        }
    }

    if (ctx->input_tensor) {
        ctx->input_tensor->data = NULL;
        csinn_free_tensor(ctx->input_tensor);
        ctx->input_tensor = NULL;
    }

    for (int b = 0; b < 2; b++) {
        if (ctx->input_bufs[b]) {
            shl_mem_free(ctx->input_bufs[b]);
            ctx->input_bufs[b] = NULL;
        }
    }

    if (ctx->sess) {
        csinn_session_deinit(ctx->sess);
        csinn_free_session(ctx->sess);
        ctx->sess = NULL;
    }

    /* Free weight blob ONLY after session is gone (qinfo pointed into it). */
    if (ctx->params_blob) {
        free(ctx->params_blob);
        ctx->params_blob = NULL;
    }

    ctx->magic = 0;
    pthread_mutex_destroy(&ctx->lock);
    free(ctx);
}

YOLO_API void *init_model(const char *params_path)
{
    if (!params_path) return NULL;

    ModelContext *ctx = (ModelContext *)calloc(1, sizeof(ModelContext));
    if (!ctx) return NULL;

    pthread_mutex_init(&ctx->lock, NULL);
    ctx->magic = CTX_MAGIC;

    ctx->sess = create_graph(params_path, &ctx->params_blob);
    if (!ctx->sess) {
        destroy_ctx_resources(ctx);
        return NULL;
    }

    ctx->input_tensor = csinn_alloc_tensor(NULL);
    if (!ctx->input_tensor) {
        destroy_ctx_resources(ctx);
        return NULL;
    }
    ctx->input_tensor->dim_count = 4;
    ctx->input_tensor->dim[0] = 1;
    ctx->input_tensor->dim[1] = 3;
    ctx->input_tensor->dim[2] = INPUT_HEIGHT;
    ctx->input_tensor->dim[3] = INPUT_WIDTH;

    if (alloc_io_buffers(ctx) != 0) {
        destroy_ctx_resources(ctx);
        return NULL;
    }

    build_quant_lut(ctx);
    init_yolo_params(&ctx->yolo_params);
    ctx->buf_idx = 0;
    ctx->input_tensor->data = ctx->input_bufs[0];

    return ctx;
}

YOLO_API int yolo_set_thresholds(void *handle, float conf_thres, float iou_thres)
{
    ModelContext *ctx = ctx_from(handle);
    if (!ctx) return -1;
    if (conf_thres < 0.f || conf_thres > 1.f) return -1;
    if (iou_thres < 0.f || iou_thres > 1.f) return -1;

    pthread_mutex_lock(&ctx->lock);
    ctx->yolo_params.conf_thres = conf_thres;
    ctx->yolo_params.iou_thres = iou_thres;
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

YOLO_API int yolo_get_timings_us(void *handle, float *pre_us,
                                 float *npu_us, float *post_us)
{
    ModelContext *ctx = ctx_from(handle);
    if (!ctx) return -1;

    pthread_mutex_lock(&ctx->lock);
    if (pre_us)  *pre_us  = (float)ctx->last_pre_ns  / 1000.0f;
    if (npu_us)  *npu_us  = (float)ctx->last_npu_ns  / 1000.0f;
    if (post_us) *post_us = (float)ctx->last_post_ns / 1000.0f;
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

YOLO_API int run_inference(void *handle, uint8_t *bgr_pixels,
                           float *out_boxes, int max_boxes)
{
    ModelContext *ctx = ctx_from(handle);
    if (UNLIKELY(!ctx || !bgr_pixels || !out_boxes || max_boxes <= 0))
        return -1;

    pthread_mutex_lock(&ctx->lock);

    /* --- 1. Preprocess (CPU, no NPU lock) → ping-pong slot --- */
    uint64_t t0 = shl_get_timespec();
    int wr = ctx->buf_idx;
    quantize_bgr_to_nchw(bgr_pixels, ctx->input_bufs[wr], ctx->lut);
    ctx->input_tensor->data = ctx->input_bufs[wr];
    uint64_t t1 = shl_get_timespec();

    /* --- 2. NPU (device lock: other models wait here only) --- */
    struct csinn_tensor *input_tensors[1] = { ctx->input_tensor };
    pthread_mutex_lock(&g_npu_lock);
    uint64_t t2 = shl_get_timespec();
    csinn_update_input_and_run(input_tensors, ctx->sess);
    uint64_t t3 = shl_get_timespec();
    pthread_mutex_unlock(&g_npu_lock);

    /* Flip buffer so a concurrent prep on another thread pattern can use the other. */
    ctx->buf_idx ^= 1;

    /* --- 3. Dequant + NMS (CPU, NPU free for other handles) --- */
    uint64_t t4 = shl_get_timespec();
    struct csinn_tensor *output_ptrs[NUM_OUTPUTS];

    for (int i = 0; i < ctx->output_num; i++) {
        struct csinn_tensor *raw = ctx->raw_out[i];
        raw->data = NULL;
        /* get_output may set qinfo → model.params; never free that pointer */
        csinn_get_output(i, raw, ctx->sess);

        float scale = raw->qinfo[0].scale;
        int32_t zp = raw->qinfo[0].zero_point;
        int n = ctx->f32_elems[i];
        int got = csinn_tensor_size(raw);
        if (got > 0 && got < n) n = got;

        dequant_i8_to_f32((const int8_t *)raw->data, ctx->f32_bufs[i], n, scale, zp);

        /* Keep shell metadata in sync if driver adjusted dims */
        ctx->f32_out[i]->dim_count = raw->dim_count;
        {
            int dc = raw->dim_count;
            if (dc > 8) dc = 8; /* csinn_tensor.dim[] is typically 8 */
            for (int d = 0; d < dc; d++)
                ctx->f32_out[i]->dim[d] = raw->dim[d];
        }
        ctx->f32_out[i]->data = ctx->f32_bufs[i];
        ctx->f32_out[i]->qinfo = NULL;
        ctx->f32_out[i]->quant_channel = 0;
        ctx->f32_out[i]->dtype = CSINN_DTYPE_FLOAT32;

        output_ptrs[i] = ctx->f32_out[i];
    }

    int num = shl_c920_detect_yolov5_postprocess(output_ptrs, ctx->boxes, &ctx->yolo_params);

    int return_num = 0;
    if (num > 0) {
        if (num > max_boxes) num = max_boxes;
        if (num > MAX_DETECT) num = MAX_DETECT;
        for (int k = 0; k < num; k++) {
            out_boxes[k * 6 + 0] = ctx->boxes[k].x1;
            out_boxes[k * 6 + 1] = ctx->boxes[k].y1;
            out_boxes[k * 6 + 2] = ctx->boxes[k].x2;
            out_boxes[k * 6 + 3] = ctx->boxes[k].y2;
            out_boxes[k * 6 + 4] = ctx->boxes[k].score;
            out_boxes[k * 6 + 5] = (float)ctx->boxes[k].label;
            return_num++;
        }
    }

    /* Neuter raw qinfo aliases so a future free path cannot touch .params */
    for (int i = 0; i < ctx->output_num; i++) {
        ctx->raw_out[i]->data = NULL;
        ctx->raw_out[i]->qinfo = NULL;
        ctx->raw_out[i]->quant_channel = 0;
    }

    uint64_t t5 = shl_get_timespec();
    ctx->last_pre_ns  = t1 - t0;
    ctx->last_npu_ns  = t3 - t2;
    ctx->last_post_ns = t5 - t4;

    pthread_mutex_unlock(&ctx->lock);
    return return_num;
}

YOLO_API int yolo_warmup(void *handle, int runs)
{
    ModelContext *ctx = ctx_from(handle);
    if (!ctx || runs < 1) return -1;

    uint8_t *zeros = (uint8_t *)calloc((size_t)INPUT_HEIGHT * INPUT_WIDTH * 3, 1);
    float *tmp = (float *)malloc(16 * 6 * sizeof(float));
    if (!zeros || !tmp) {
        free(zeros);
        free(tmp);
        return -1;
    }

    int rc = 0;
    for (int i = 0; i < runs; i++) {
        if (run_inference(handle, zeros, tmp, 16) < 0) {
            rc = -1;
            break;
        }
    }

    free(tmp);
    free(zeros);
    return rc;
}

YOLO_API void release_model(void *handle)
{
    ModelContext *ctx = ctx_from(handle);
    if (!ctx) return;

    pthread_mutex_lock(&ctx->lock);
    ctx->magic = 0; /* invalidate before teardown */
    pthread_mutex_unlock(&ctx->lock);

    destroy_ctx_resources(ctx);
}
