/**
 * yolov5n_lib.h — Production ctypes API for YOLOv5n on TH1520 NPU (VIP9000).
 *
 * Threading model
 * ---------------
 * - Each init_model() handle is an independent session + weight blob.
 * - VIP9000 runs one graph at a time: session_run is serialized by an internal
 *   process-wide NPU lock. Two models are safe; they time-share the NPU.
 * - Preprocess / postprocess run WITHOUT the NPU lock so model B can quantize
 *   while model A is on the NPU, and A can NMS while B is on the NPU.
 * - Do NOT call run_inference() concurrently on the SAME handle.
 *
 * Build: compile with -fvisibility=hidden (only YOLO_API symbols are exported).
 */
#ifndef YOLOV5N_LIB_H_
#define YOLOV5N_LIB_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define YOLO_API __attribute__((visibility("default")))
#else
#define YOLO_API
#endif

#define YOLO_INPUT_H 640
#define YOLO_INPUT_W 640

/** Load model.params / .bm, build LUT, prealloc buffers. NULL on failure. */
YOLO_API void *init_model(const char *params_path);

/**
 * Full frame path: LUT quantize (BGR HWC uint8) → NPU → NMS.
 * out_boxes: [N*6] = x1,y1,x2,y2,score,cls  (letterbox pixel space).
 * Returns number of boxes, or -1 on error.
 */
YOLO_API int run_inference(void *handle, uint8_t *bgr_hwc,
                           float *out_boxes, int max_boxes);

/** Release session, weight blob, and all buffers. */
YOLO_API void release_model(void *handle);

/** Tune NMS (defaults: conf=0.25, iou=0.45). Returns 0 / -1. */
YOLO_API int yolo_set_thresholds(void *handle, float conf_thres, float iou_thres);

/** Dry-run zeros to warm NPU/driver caches. Returns 0 / -1. */
YOLO_API int yolo_warmup(void *handle, int runs);

/**
 * Last-frame timings in microseconds (any pointer may be NULL).
 * pre = LUT+layout, npu = session_run only, post = dequant+NMS+copy.
 */
YOLO_API int yolo_get_timings_us(void *handle, float *pre_us,
                                 float *npu_us, float *post_us);

#ifdef __cplusplus
}
#endif

#endif /* YOLOV5N_LIB_H_ */
