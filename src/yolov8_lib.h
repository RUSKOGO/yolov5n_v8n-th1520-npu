/**
 * yolov8_lib.h — YOLOv8n on TH1520 NPU (VIP9000), ctypes API.
 *
 * Same threading rules as yolov5n_lib.h.
 * Build into a SEPARATE .so (libyolov8n.so) with HHB-generated vendor/hhb_v8/model.c.
 */
#ifndef YOLOV8_LIB_H_
#define YOLOV8_LIB_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define YOLO_API __attribute__((visibility("default")))
#else
#define YOLO_API
#endif

#define YOLO8_INPUT_H 640
#define YOLO8_INPUT_W 640

YOLO_API void *yolov8_init_model(const char *params_path);
YOLO_API int  yolov8_run_inference(void *handle, uint8_t *bgr_hwc,
                                   float *out_boxes, int max_boxes);
YOLO_API void yolov8_release_model(void *handle);
YOLO_API int  yolov8_set_thresholds(void *handle, float conf, float iou);
YOLO_API int  yolov8_warmup(void *handle, int runs);
YOLO_API int  yolov8_get_timings_us(void *handle, float *pre_us,
                                    float *npu_us, float *post_us);

/* Aliases matching the v5 demo names (for drop-in check.py with this .so only). */
YOLO_API void *init_model(const char *params_path);
YOLO_API int  run_inference(void *handle, uint8_t *bgr_hwc,
                            float *out_boxes, int max_boxes);
YOLO_API void release_model(void *handle);
YOLO_API int  yolo_set_thresholds(void *handle, float conf, float iou);
YOLO_API int  yolo_warmup(void *handle, int runs);
YOLO_API int  yolo_get_timings_us(void *handle, float *pre_us,
                                  float *npu_us, float *post_us);

#ifdef __cplusplus
}
#endif

#endif /* YOLOV8_LIB_H_ */
