/**
 * YOLOv8n NPU wrapper for TH1520 (VIP9000).
 *
 * Expects HHB graph with either:
 *   A) one decoded output  (1, 4+nc, N) e.g. (1, 84, 8400)  — DFL already in graph
 *   B) three raw heads     (1, 4*reg_max+nc, H, W) e.g. (1, 144, 80/40/20) — DFL on CPU
 *
 * Generate vendor/hhb_v8/{model.c,io.c,io.h} + model.params via HHB (see docs/YOLOV8_HHB.md).
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>

#include "io.h"
#include "shl_c920.h"
#include "yolov8_lib.h"

#ifndef YOLO_API
#if defined(__GNUC__)
#define YOLO_API __attribute__((visibility("default")))
#else
#define YOLO_API
#endif
#endif

#define INPUT_H        YOLO8_INPUT_H
#define INPUT_W        YOLO8_INPUT_W
#define NUM_CLASSES    80
#define REG_MAX        16
#define MAX_DETECT     1024
#define MAX_OUTPUTS    3
#define MAX_CAND       16384
#define CTX_MAGIC      0x59303832u  /* Y082 */
#define CLIP_INT8(v)   ((v) > 127 ? 127 : ((v) < -128 ? -128 : (v)))

#if defined(__GNUC__)
#define HOT       __attribute__((hot))
#define RESTRICT  __restrict__
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define HOT
#define RESTRICT
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif

void *csinn_(char *params);
void csinn_update_input_and_run(struct csinn_tensor **input_tensors, void *sess);

static pthread_mutex_t g_npu_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    float x1, y1, x2, y2, score;
    int label;
} DetBox;

typedef enum {
    OUT_DECODED = 0, /* (1, 4+nc, N) */
    OUT_RAW_DFL = 1  /* 3 x (1, 4*reg_max+nc, H, W) */
} OutLayout;

typedef struct {
    uint32_t magic;
    void *sess;
    char *params_blob;

    struct csinn_tensor *input_tensor;
    int8_t *input_bufs[2];
    int input_size;
    int buf_idx;
    int8_t lut[256] __attribute__((aligned(64)));

    int output_num;
    OutLayout layout;
    int nc;
    int reg_max;
    int strides[MAX_OUTPUTS];
    int out_h[MAX_OUTPUTS];
    int out_w[MAX_OUTPUTS];
    int out_c[MAX_OUTPUTS];
    int f32_elems[MAX_OUTPUTS];

    struct csinn_tensor *raw_out[MAX_OUTPUTS];
    struct csinn_tensor *f32_out[MAX_OUTPUTS];
    float *f32_bufs[MAX_OUTPUTS];

    float conf_thres;
    float iou_thres;
    DetBox *cands;
    DetBox *boxes;
    uint8_t *suppressed;

    uint64_t last_pre_ns, last_npu_ns, last_post_ns;
    pthread_mutex_t lock;
} ModelContext;

static inline ModelContext *ctx_from(void *h)
{
    ModelContext *c = (ModelContext *)h;
    if (UNLIKELY(!c || c->magic != CTX_MAGIC || !c->sess)) return NULL;
    return c;
}

static void safe_free_owned_tensor(struct csinn_tensor *t)
{
    if (!t) return;
    t->data = NULL;
    t->qinfo = NULL;
    t->quant_channel = 0;
    csinn_free_tensor(t);
}

static float sigmoidf_fast(float x)
{
    return 1.f / (1.f + expf(-x));
}

static HOT void quantize_bgr_to_nchw(const uint8_t *RESTRICT bgr,
                                     int8_t *RESTRICT dst,
                                     const int8_t *RESTRICT lut)
{
    const size_t hw = (size_t)INPUT_H * (size_t)INPUT_W;
    int8_t *RESTRICT ch_r = dst;
    int8_t *RESTRICT ch_g = dst + hw;
    int8_t *RESTRICT ch_b = dst + 2 * hw;
    const uint8_t *RESTRICT p = bgr;
    size_t i = 0;
    for (; i + 8 <= hw; i += 8) {
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

static HOT void dequant_i8_to_f32(const int8_t *RESTRICT src, float *RESTRICT dst,
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

/* Softmax expectation over reg_max bins → distance in "grid units". */
static float dfl_expect(const float *logits, int reg_max)
{
    float m = logits[0];
    for (int i = 1; i < reg_max; i++)
        if (logits[i] > m) m = logits[i];
    float sum = 0.f, acc = 0.f;
    for (int i = 0; i < reg_max; i++) {
        float e = expf(logits[i] - m);
        sum += e;
        acc += e * (float)i;
    }
    return acc / (sum + 1e-9f);
}

static float iou_xyxy(const DetBox *a, const DetBox *b)
{
    float xx1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float yy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float xx2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float yy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    float w = xx2 - xx1;
    float h = yy2 - yy1;
    if (w <= 0.f || h <= 0.f) return 0.f;
    float inter = w * h;
    float area_a = (a->x2 - a->x1) * (a->y2 - a->y1);
    float area_b = (b->x2 - b->x1) * (b->y2 - b->y1);
    return inter / (area_a + area_b - inter + 1e-9f);
}

static int cmp_score_desc(const void *a, const void *b)
{
    const DetBox *pa = (const DetBox *)a;
    const DetBox *pb = (const DetBox *)b;
    if (pa->score < pb->score) return 1;
    if (pa->score > pb->score) return -1;
    return 0;
}

static int nms_class_agnostic(DetBox *cands, int n, float iou_thres,
                              DetBox *out, int max_out, uint8_t *suppressed)
{
    if (n <= 0) return 0;
    qsort(cands, (size_t)n, sizeof(DetBox), cmp_score_desc);
    memset(suppressed, 0, (size_t)n);
    int kept = 0;
    for (int i = 0; i < n && kept < max_out; i++) {
        if (suppressed[i]) continue;
        out[kept++] = cands[i];
        for (int j = i + 1; j < n; j++) {
            if (suppressed[j]) continue;
            if (iou_xyxy(&cands[i], &cands[j]) > iou_thres)
                suppressed[j] = 1;
        }
    }
    return kept;
}

/*
 * Decoded head: data layout NCHW-ish as (1, C, N) contiguous:
 *   channel 0..3 = cx,cy,w,h (pixels); 4..4+nc = class scores (sigmoid applied if needed)
 */
static int decode_flat(const float *data, int c, int n, int nc, float conf_thres,
                       DetBox *cands, int max_cands, int apply_sigmoid)
{
    int count = 0;
    for (int i = 0; i < n && count < max_cands; i++) {
        float best = -1.f;
        int best_c = 0;
        for (int k = 0; k < nc; k++) {
            float s = data[(4 + k) * n + i];
            if (apply_sigmoid) s = sigmoidf_fast(s);
            if (s > best) {
                best = s;
                best_c = k;
            }
        }
        if (best < conf_thres) continue;
        float cx = data[0 * n + i];
        float cy = data[1 * n + i];
        float bw = data[2 * n + i];
        float bh = data[3 * n + i];
        DetBox *d = &cands[count++];
        d->x1 = cx - bw * 0.5f;
        d->y1 = cy - bh * 0.5f;
        d->x2 = cx + bw * 0.5f;
        d->y2 = cy + bh * 0.5f;
        d->score = best;
        d->label = best_c;
    }
    return count;
}

/* Raw DFL head: (1, 4*reg_max+nc, H, W) NCHW */
static int decode_raw_dfl(const float *data, int c, int h, int w, int stride,
                          int reg_max, int nc, float conf_thres,
                          DetBox *cands, int max_cands, int count)
{
    const int box_ch = 4 * reg_max;
    if (c < box_ch + nc) return count;
    float tmp[REG_MAX];
    if (reg_max > REG_MAX) return count;

    const int hw = h * w;
    for (int gy = 0; gy < h && count < max_cands; gy++) {
        for (int gx = 0; gx < w && count < max_cands; gx++) {
            int idx = gy * w + gx;
            /* max logit then one sigmoid — much cheaper than nc sigmoids */
            float best_logit = data[box_ch * hw + idx];
            int best_c = 0;
            for (int k = 1; k < nc; k++) {
                float v = data[(box_ch + k) * hw + idx];
                if (v > best_logit) {
                    best_logit = v;
                    best_c = k;
                }
            }
            float best = sigmoidf_fast(best_logit);
            if (best < conf_thres) continue;

            float dist[4];
            for (int t = 0; t < 4; t++) {
                for (int i = 0; i < reg_max; i++)
                    tmp[i] = data[(t * reg_max + i) * hw + idx];
                dist[t] = dfl_expect(tmp, reg_max);
            }
            float anchor_x = ((float)gx + 0.5f) * (float)stride;
            float anchor_y = ((float)gy + 0.5f) * (float)stride;
            float x1 = anchor_x - dist[0] * (float)stride;
            float y1 = anchor_y - dist[1] * (float)stride;
            float x2 = anchor_x + dist[2] * (float)stride;
            float y2 = anchor_y + dist[3] * (float)stride;

            DetBox *d = &cands[count++];
            d->x1 = x1;
            d->y1 = y1;
            d->x2 = x2;
            d->y2 = y2;
            d->score = best;
            d->label = best_c;
        }
    }
    return count;
}

static void build_quant_lut(ModelContext *ctx)
{
    struct csinn_tensor *sess_input = ((struct csinn_session *)ctx->sess)->input[0];
    float scale = sess_input->qinfo[0].scale;
    int32_t zp = sess_input->qinfo[0].zero_point;
    float mult = 1.0f / (255.0f * scale);
    for (int v = 0; v < 256; v++) {
        int32_t val = (int32_t)roundf((float)v * mult) + zp;
        ctx->lut[v] = (int8_t)CLIP_INT8(val);
    }
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

static int probe_outputs(ModelContext *ctx)
{
    struct csinn_session *sess = (struct csinn_session *)ctx->sess;
    ctx->output_num = csinn_get_output_number(sess);
    if (ctx->output_num < 1) return -1;
    if (ctx->output_num > MAX_OUTPUTS) ctx->output_num = MAX_OUTPUTS;

    ctx->nc = NUM_CLASSES;
    ctx->reg_max = REG_MAX;

    for (int i = 0; i < ctx->output_num; i++) {
        struct csinn_tensor *t = sess->output[i];
        ctx->out_c[i] = t->dim[1];
        ctx->out_h[i] = (t->dim_count >= 3) ? t->dim[2] : 1;
        ctx->out_w[i] = (t->dim_count >= 4) ? t->dim[3] : t->dim[2];
        if (t->dim_count == 3) {
            /* (1, C, N) */
            ctx->out_h[i] = 1;
            ctx->out_w[i] = t->dim[2];
        }
        ctx->f32_elems[i] = csinn_tensor_size(t);
        if (ctx->f32_elems[i] <= 0) {
            ctx->f32_elems[i] = ctx->out_c[i] * ctx->out_h[i] * ctx->out_w[i];
        }
    }

    /* Heuristic layout */
    if (ctx->output_num == 1 && ctx->out_c[0] >= 5) {
        ctx->layout = OUT_DECODED;
        ctx->nc = ctx->out_c[0] - 4;
        if (ctx->nc < 1) ctx->nc = NUM_CLASSES;
        ctx->strides[0] = 0;
    } else if (ctx->output_num >= 3) {
        ctx->layout = OUT_RAW_DFL;
        ctx->strides[0] = 8;
        ctx->strides[1] = 16;
        ctx->strides[2] = 32;
        /*
         * channels = 4 * reg_max + nc. Prefer reg_max=16 (Ultralytics default).
         * e.g. c=74 → nc=10; c=144 → nc=80.
         */
        int ch0 = ctx->out_c[0];
        if (ch0 >= 4 * 16 + 1 && (ch0 - 4 * 16) <= 1024) {
            ctx->reg_max = 16;
            ctx->nc = ch0 - 4 * 16;
        } else if (ch0 > 4 && ch0 % 4 == 0) {
            /* fallback: assume nc=80 */
            ctx->nc = 80;
            ctx->reg_max = (ch0 - 80) / 4;
            if (ctx->reg_max < 1) ctx->reg_max = REG_MAX;
        }
    } else {
        ctx->layout = OUT_DECODED;
        ctx->nc = ctx->out_c[0] - 4;
        if (ctx->nc < 1) ctx->nc = NUM_CLASSES;
    }

    printf("YOLOv8 outputs: num=%d layout=%s nc=%d reg_max=%d\n",
           ctx->output_num,
           ctx->layout == OUT_DECODED ? "DECODED" : "RAW_DFL",
           ctx->nc, ctx->reg_max);
    for (int i = 0; i < ctx->output_num; i++) {
        printf("  out[%d]: c=%d h=%d w=%d elems=%d\n",
               i, ctx->out_c[i], ctx->out_h[i], ctx->out_w[i], ctx->f32_elems[i]);
    }
    return 0;
}

static int alloc_io(ModelContext *ctx)
{
    ctx->input_size = csinn_tensor_byte_size(((struct csinn_session *)ctx->sess)->input[0]);
    for (int b = 0; b < 2; b++) {
        ctx->input_bufs[b] = (int8_t *)shl_mem_alloc_aligned(ctx->input_size, 0);
        if (!ctx->input_bufs[b]) return -1;
    }

    for (int i = 0; i < ctx->output_num; i++) {
        ctx->raw_out[i] = csinn_alloc_tensor(NULL);
        ctx->f32_out[i] = csinn_alloc_tensor(NULL);
        if (!ctx->raw_out[i] || !ctx->f32_out[i]) return -1;
        ctx->f32_bufs[i] = (float *)shl_mem_alloc((size_t)ctx->f32_elems[i] * sizeof(float));
        if (!ctx->f32_bufs[i]) return -1;
        ctx->f32_out[i]->dtype = CSINN_DTYPE_FLOAT32;
        ctx->f32_out[i]->layout = CSINN_LAYOUT_NCHW;
        ctx->f32_out[i]->qinfo = NULL;
        ctx->f32_out[i]->quant_channel = 0;
        ctx->f32_out[i]->data = ctx->f32_bufs[i];
    }

    ctx->cands = (DetBox *)malloc(MAX_CAND * sizeof(DetBox));
    ctx->boxes = (DetBox *)malloc(MAX_DETECT * sizeof(DetBox));
    ctx->suppressed = (uint8_t *)malloc(MAX_CAND);
    if (!ctx->cands || !ctx->boxes || !ctx->suppressed) return -1;
    return 0;
}

static void destroy_ctx(ModelContext *ctx)
{
    if (!ctx) return;
    free(ctx->cands);
    free(ctx->boxes);
    free(ctx->suppressed);
    for (int i = 0; i < MAX_OUTPUTS; i++) {
        if (ctx->f32_bufs[i]) shl_mem_free(ctx->f32_bufs[i]);
        safe_free_owned_tensor(ctx->f32_out[i]);
        if (ctx->raw_out[i]) {
            ctx->raw_out[i]->data = NULL;
            csinn_free_tensor(ctx->raw_out[i]);
        }
    }
    if (ctx->input_tensor) {
        ctx->input_tensor->data = NULL;
        csinn_free_tensor(ctx->input_tensor);
    }
    for (int b = 0; b < 2; b++) {
        if (ctx->input_bufs[b]) shl_mem_free(ctx->input_bufs[b]);
    }
    if (ctx->sess) {
        csinn_session_deinit(ctx->sess);
        csinn_free_session(ctx->sess);
    }
    if (ctx->params_blob) free(ctx->params_blob);
    ctx->magic = 0;
    pthread_mutex_destroy(&ctx->lock);
    free(ctx);
}

YOLO_API void *yolov8_init_model(const char *params_path)
{
    if (!params_path) return NULL;
    ModelContext *ctx = (ModelContext *)calloc(1, sizeof(ModelContext));
    if (!ctx) return NULL;
    pthread_mutex_init(&ctx->lock, NULL);
    ctx->magic = CTX_MAGIC;
    ctx->conf_thres = 0.25f;
    ctx->iou_thres = 0.45f;

    ctx->sess = create_graph(params_path, &ctx->params_blob);
    if (!ctx->sess) {
        destroy_ctx(ctx);
        return NULL;
    }

    ctx->input_tensor = csinn_alloc_tensor(NULL);
    if (!ctx->input_tensor) {
        destroy_ctx(ctx);
        return NULL;
    }
    ctx->input_tensor->dim_count = 4;
    ctx->input_tensor->dim[0] = 1;
    ctx->input_tensor->dim[1] = 3;
    ctx->input_tensor->dim[2] = INPUT_H;
    ctx->input_tensor->dim[3] = INPUT_W;

    if (probe_outputs(ctx) != 0 || alloc_io(ctx) != 0) {
        destroy_ctx(ctx);
        return NULL;
    }

    build_quant_lut(ctx);
    ctx->buf_idx = 0;
    ctx->input_tensor->data = ctx->input_bufs[0];

    if (ctx->layout == OUT_DECODED) {
        fprintf(stderr,
                "WARNING: DECODED layout (1x%dxN) usually contains Slice/StridedSlice "
                "and fails on VIP9000 (Could not create network object).\n"
                "Re-export raw heads: python3 scripts/export_yolov8_raw_heads.py "
                "--weights ppe.pt\n",
                ctx->out_c[0]);
    }

    /* Smoke-run: if NPU graph failed to build, this often returns without
     * creating network — catch obvious null input before first user frame. */
    {
        struct csinn_tensor *tin = ((struct csinn_session *)ctx->sess)->input[0];
        if (!tin) {
            fprintf(stderr, "ERROR: session has no input tensor (NPU graph not loaded)\n");
            destroy_ctx(ctx);
            return NULL;
        }
    }

    return ctx;
}

YOLO_API int yolov8_set_thresholds(void *handle, float conf, float iou)
{
    ModelContext *ctx = ctx_from(handle);
    if (!ctx || conf < 0.f || conf > 1.f || iou < 0.f || iou > 1.f) return -1;
    pthread_mutex_lock(&ctx->lock);
    ctx->conf_thres = conf;
    ctx->iou_thres = iou;
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

YOLO_API int yolov8_get_timings_us(void *handle, float *pre, float *npu, float *post)
{
    ModelContext *ctx = ctx_from(handle);
    if (!ctx) return -1;
    pthread_mutex_lock(&ctx->lock);
    if (pre)  *pre  = (float)ctx->last_pre_ns / 1000.f;
    if (npu)  *npu  = (float)ctx->last_npu_ns / 1000.f;
    if (post) *post = (float)ctx->last_post_ns / 1000.f;
    pthread_mutex_unlock(&ctx->lock);
    return 0;
}

YOLO_API int yolov8_run_inference(void *handle, uint8_t *bgr, float *out_boxes, int max_boxes)
{
    ModelContext *ctx = ctx_from(handle);
    if (UNLIKELY(!ctx || !bgr || !out_boxes || max_boxes <= 0)) return -1;

    pthread_mutex_lock(&ctx->lock);

    uint64_t t0 = shl_get_timespec();
    int wr = ctx->buf_idx;
    quantize_bgr_to_nchw(bgr, ctx->input_bufs[wr], ctx->lut);
    ctx->input_tensor->data = ctx->input_bufs[wr];
    uint64_t t1 = shl_get_timespec();

    struct csinn_tensor *inputs[1] = { ctx->input_tensor };
    pthread_mutex_lock(&g_npu_lock);
    uint64_t t2 = shl_get_timespec();
    csinn_update_input_and_run(inputs, ctx->sess);
    uint64_t t3 = shl_get_timespec();
    pthread_mutex_unlock(&g_npu_lock);
    ctx->buf_idx ^= 1;

    uint64_t t4 = shl_get_timespec();
    for (int i = 0; i < ctx->output_num; i++) {
        struct csinn_tensor *raw = ctx->raw_out[i];
        raw->data = NULL;
        csinn_get_output(i, raw, ctx->sess);
        float scale = raw->qinfo[0].scale;
        int32_t zp = raw->qinfo[0].zero_point;
        int n = ctx->f32_elems[i];
        int got = csinn_tensor_size(raw);
        if (got > 0 && got < n) n = got;
        dequant_i8_to_f32((const int8_t *)raw->data, ctx->f32_bufs[i], n, scale, zp);
        raw->data = NULL;
        raw->qinfo = NULL;
        raw->quant_channel = 0;
    }

    int ncand = 0;
    if (ctx->layout == OUT_DECODED) {
        int c = ctx->out_c[0];
        int n = (ctx->out_h[0] == 1) ? ctx->out_w[0] : (ctx->out_h[0] * ctx->out_w[0]);
        /* scores may already be sigmoid'd after HHB; try without first via threshold —
           apply_sigmoid=1 is safer for raw logits. */
        ncand = decode_flat(ctx->f32_bufs[0], c, n, ctx->nc, ctx->conf_thres,
                            ctx->cands, MAX_CAND, 1);
    } else {
        for (int i = 0; i < ctx->output_num; i++) {
            ncand = decode_raw_dfl(ctx->f32_bufs[i], ctx->out_c[i], ctx->out_h[i], ctx->out_w[i],
                                   ctx->strides[i], ctx->reg_max, ctx->nc, ctx->conf_thres,
                                   ctx->cands, MAX_CAND, ncand);
        }
    }

    int nout = nms_class_agnostic(ctx->cands, ncand, ctx->iou_thres,
                                  ctx->boxes, max_boxes < MAX_DETECT ? max_boxes : MAX_DETECT,
                                  ctx->suppressed);

    for (int k = 0; k < nout; k++) {
        out_boxes[k * 6 + 0] = ctx->boxes[k].x1;
        out_boxes[k * 6 + 1] = ctx->boxes[k].y1;
        out_boxes[k * 6 + 2] = ctx->boxes[k].x2;
        out_boxes[k * 6 + 3] = ctx->boxes[k].y2;
        out_boxes[k * 6 + 4] = ctx->boxes[k].score;
        out_boxes[k * 6 + 5] = (float)ctx->boxes[k].label;
    }

    uint64_t t5 = shl_get_timespec();
    ctx->last_pre_ns = t1 - t0;
    ctx->last_npu_ns = t3 - t2;
    ctx->last_post_ns = t5 - t4;

    pthread_mutex_unlock(&ctx->lock);
    return nout;
}

YOLO_API int yolov8_warmup(void *handle, int runs)
{
    if (runs < 1) return -1;
    uint8_t *z = (uint8_t *)calloc((size_t)INPUT_H * INPUT_W * 3, 1);
    float *tmp = (float *)malloc(16 * 6 * sizeof(float));
    if (!z || !tmp) {
        free(z);
        free(tmp);
        return -1;
    }
    int rc = 0;
    for (int i = 0; i < runs; i++) {
        if (yolov8_run_inference(handle, z, tmp, 16) < 0) {
            rc = -1;
            break;
        }
    }
    free(tmp);
    free(z);
    return rc;
}

YOLO_API void yolov8_release_model(void *handle)
{
    ModelContext *ctx = ctx_from(handle);
    if (!ctx) return;
    pthread_mutex_lock(&ctx->lock);
    ctx->magic = 0;
    pthread_mutex_unlock(&ctx->lock);
    destroy_ctx(ctx);
}

/* Drop-in aliases for check.py */
YOLO_API void *init_model(const char *p) { return yolov8_init_model(p); }
YOLO_API int run_inference(void *h, uint8_t *b, float *o, int m)
{
    return yolov8_run_inference(h, b, o, m);
}
YOLO_API void release_model(void *h) { yolov8_release_model(h); }
YOLO_API int yolo_set_thresholds(void *h, float c, float i)
{
    return yolov8_set_thresholds(h, c, i);
}
YOLO_API int yolo_warmup(void *h, int n) { return yolov8_warmup(h, n); }
YOLO_API int yolo_get_timings_us(void *h, float *a, float *b, float *c)
{
    return yolov8_get_timings_us(h, a, b, c);
}
