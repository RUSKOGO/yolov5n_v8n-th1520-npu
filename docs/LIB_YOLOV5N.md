**Language / Язык:** [English](#english) · [Русский](#русский)

---

<a id="english"></a>

# Technical documentation: `libyolov5n` (`yolov5n_lib`)

Shared library for **YOLOv5n INT8** inference on **TH1520 VIP9000**.

| Item | Value |
|------|--------|
| Sources | `src/yolov5n_lib.c`, `src/yolov5n_lib.h` |
| HHB graph | `vendor/hhb/{model.c,io.c,io.h}` |
| Output binary | `libyolov5n.so` |
| Build | `scripts/build_so.sh` |
| Demo | `python/check.py` |
| Weights | `model.params` (HHB; place next to `.so`) |

---

## Role in the stack

```text
Camera / file
    → Python letterbox (BGR 640×640)
        → libyolov5n.so
              LUT quantize (C)
              csinn_update_input_and_run (NPU)
              INT8→F32 dequant (C)
              shl_c920 yolov5 NMS (C)
        ← boxes [x1,y1,x2,y2,score,cls]
    → draw / WebSocket preview
```

Python never touches INT8 tensors or NMS on the hot path.

---

## Public C API

Declared in `yolov5n_lib.h`. Compiled with `-fvisibility=hidden`; only `YOLO_API`
symbols are exported.

### `void *init_model(const char *params_path)`

- Loads `model.params` (or `.bm`) via HHB `io` / `csinn_`
- Creates session, builds **256-entry quant LUT** from input scale/zero-point
- Preallocates: input ping-pong buffers, raw output shells, f32 dequant buffers
- Returns opaque handle, or `NULL` on failure

### `int run_inference(void *handle, uint8_t *bgr_hwc, float *out_boxes, int max_boxes)`

| Argument | Meaning |
|----------|---------|
| `bgr_hwc` | Contiguous `uint8` BGR, **640×640×3**, letterboxed |
| `out_boxes` | Caller-owned `float` buffer, capacity `max_boxes * 6` |
| return | Number of detections `N`, or `-1` |

Each detection: `x1, y1, x2, y2, score, cls` in **letterbox pixel space**.

Pipeline inside one call:

1. **Pre (CPU):** LUT quantize into ping-pong slot (no NPU lock)
2. **NPU:** `csinn_update_input_and_run` under **global NPU lock**
3. **Post (CPU):** get outputs → manual dequant → SHL YOLOv5 detect/NMS → copy boxes

### `void release_model(void *handle)`

Destroys session and buffers. **Weight blob is freed only after the session**,
and output `qinfo` pointers that may alias into `model.params` are neutered
first (avoids heap corruption).

### `int yolo_set_thresholds(void *handle, float conf, float iou)`

Defaults typically `0.25` / `0.45`. Returns `0` / `-1`.

### `int yolo_warmup(void *handle, int runs)`

Runs inference on a zero image to warm driver / NPU caches.

### `int yolo_get_timings_us(void *handle, float *pre, float *npu, float *post)`

Last-frame timings in **microseconds**. Any pointer may be `NULL`.

| Field | Measures |
|-------|----------|
| `pre` | LUT + layout into input buffer |
| `npu` | `session_run` only (under NPU lock) |
| `post` | dequant + NMS + box copy |

---

## Threading model

| Rule | Detail |
|------|--------|
| One NPU | Process-wide `g_npu_lock` around `session_run` only |
| Multi-model | Two `init_model()` handles are OK; they time-share the NPU |
| Overlap | While model A is on NPU, model B may run LUT / NMS |
| Same handle | **Do not** call `run_inference` concurrently on one handle (per-handle mutex) |

---

## Expected HHB graph

- Input: `images`, shape `1×3×640×640`, INT8 asymmetric after HHB
- Outputs: three YOLOv5 detection heads (raw conv outputs), **no** NMS in graph
- Postprocess uses SHL helpers (`shl_c920_detect_yolov5_*` family) inside the `.so`

Regenerating the graph: [QUANTIZATION.md](QUANTIZATION.md) (YOLOv5 section).

---

## Memory / stability notes

1. **`model.params` must outlive the session** — quant info often points into the blob.
2. After each frame, raw output `qinfo` / `data` that alias session memory are
   cleared before any free path can touch them.
3. Hot path avoids `malloc` / `free` (persistent context).
4. Input uses **two** INT8 buffers (ping-pong) so the NPU can still reference
   the previous frame’s storage while the next is filled.

---

## Build

```bash
export HHB_NN2=.../hhb/install_nn2/th1520
export HHB_PB=.../hhb/prebuilt
export CC=riscv64-unknown-linux-gnu-gcc   # or gcc on-board
./scripts/build_so.sh
# → libyolov5n.so
```

Links: `yolov5n_lib.c` + `vendor/hhb/io.c` + `vendor/hhb/model.c` + `-lshl` (+ jpeg/png/z as in script).

---

## Python usage (ctypes sketch)

```python
import ctypes, numpy as np

lib = ctypes.CDLL("./libyolov5n.so")
lib.init_model.restype = ctypes.c_void_p
# ... set argtypes as in python/check.py ...

sess = lib.init_model(b"model.params")
lib.yolo_warmup(sess, 2)
boxes = np.zeros(100 * 6, np.float32)
n = lib.run_inference(sess, letterboxed_bgr, boxes, 100)
lib.release_model(sess)
```

Full demo: `python/check.py` (FastAPI + WebSocket JPEG preview, V4L2 webcam).

---

## Troubleshooting

| Symptom | Likely cause |
|---------|----------------|
| Segfault on exit / after minutes | Freeing aliased `qinfo` — use this lib, not raw sample `free` |
| Low FPS, high Python CPU | Letterbox/NMS still in Python — call `run_inference` only |
| `init_model` NULL | Wrong/missing `model.params`, or `.so` not matching graph |
| NPU busy / serialize | Expected with two models; check `npu_us` timings |

---

<a id="русский"></a>

# Техническая спецификация: `libyolov5n`

**[↑ English](#english)** · **Русский**

Разделяемая библиотека для инференса **YOLOv5n INT8** на **TH1520 VIP9000**.

| Параметр | Значение |
|----------|----------|
| Исходники | `src/yolov5n_lib.c`, `src/yolov5n_lib.h` |
| Граф HHB | `vendor/hhb/{model.c,io.c,io.h}` |
| Артефакт | `libyolov5n.so` |
| Сборка | `scripts/build_so.sh` |
| Демо | `python/check.py` |
| Веса | `model.params` (рядом с `.so`) |

## Место в стеке

```text
Камера / файл
    → letterbox в Python (BGR 640×640)
        → libyolov5n.so
              квантование LUT (C)
              csinn_update_input_and_run (NPU)
              INT8→F32 (C)
              NMS YOLOv5 SHL (C)
        ← рамки [x1, y1, x2, y2, score, cls]
    → отрисовка / WebSocket
```

На горячем пути Python не работает с INT8-тензорами и не выполняет NMS.

## Открытый C API

Компиляция с `-fvisibility=hidden`; наружу экспортируются только символы `YOLO_API`.

| Функция | Назначение |
|---------|------------|
| `init_model(path)` | Загрузка `model.params`, LUT, предаллокация; `NULL` при ошибке |
| `run_inference(h, bgr, boxes, max)` | Полный кадр; возврат числа детекций или `-1` |
| `release_model(h)` | Освобождение сессии и буферов (безопасно относительно aliases в `.params`) |
| `yolo_set_thresholds(h, conf, iou)` | Пороги NMS (обычно 0.25 / 0.45) |
| `yolo_warmup(h, n)` | Прогрев драйвера / NPU |
| `yolo_get_timings_us(h, pre, npu, post)` | Тайминги последнего кадра, мкс |

- **Вход:** непрерывный BGR `uint8`, letterbox **640×640×3**  
- **Выход:** координаты в пространстве letterbox; `cls` — идентификатор класса  
- **Потоки:** глобальная блокировка NPU вокруг `session_run`; mutex на handle;  
  два `init_model` допустимы; параллельный `run_inference` на одном handle — нет

Подробные правила памяти и сборки — в секции English (те же требования к графу HHB).

## Сборка

```bash
export HHB_NN2=.../hhb/install_nn2/th1520
export HHB_PB=.../hhb/prebuilt
export CC=riscv64-unknown-linux-gnu-gcc   # или gcc на плате
./scripts/build_so.sh
```

## Диагностика

| Симптом | Вероятная причина |
|---------|-------------------|
| Segfault при выходе / через время | Освобождение aliased `qinfo` — используйте эту библиотеку, не сырой пример |
| Низкий FPS при высокой загрузке Python | Квант/NMS всё ещё в Python |
| `init_model` возвращает `NULL` | Нет или чужой `model.params` / несовпадение с графом |
| NPU «занят» | Ожидаемо при двух моделях; смотрите `npu_us` |
