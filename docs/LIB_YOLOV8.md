**Language / Язык:** [English](#english) · [Русский](#русский)

---

<a id="english"></a>

# Technical documentation: `libyolov8n` (`yolov8_lib`)

Shared library for **YOLOv8** (nano or custom, e.g. **PPE 10-class**) INT8
inference on **TH1520 VIP9000**.

| Item | Value |
|------|--------|
| Sources | `src/yolov8_lib.c`, `src/yolov8_lib.h` |
| HHB graph | `vendor/hhb_v8/{model.c,io.c,io.h}` |
| Output binary | `libyolov8n.so` |
| Build | `scripts/build_so_v8.sh` |
| Demo | `python/check_v8.py` |
| Class names | `python/classes.txt` |
| Weights | `model.params` (same HHB run as `model.c`) |

---

## Role in the stack

Same product shape as v5, different postprocess:

```text
BGR 640×640
  → LUT INT8 (C)
  → NPU (HHB graph, preferably RAW heads, no Slice)
  → INT8→F32 dequant (C)
  → layout probe:
        RAW_DFL  → DFL decode + NMS (C)
        DECODED  → flat decode + NMS (C)
  ← boxes
```

ctypes aliases (`init_model`, `run_inference`, …) match the v5 demo names so
`check_v8.py` stays drop-in with a different `.so`.

---

## Public C API

Prefixed API and v5-compatible aliases (see `yolov8_lib.h`):

| Function | Alias |
|----------|--------|
| `yolov8_init_model` | `init_model` |
| `yolov8_run_inference` | `run_inference` |
| `yolov8_release_model` | `release_model` |
| `yolov8_set_thresholds` | `yolo_set_thresholds` |
| `yolov8_warmup` | `yolo_warmup` |
| `yolov8_get_timings_us` | `yolo_get_timings_us` |

Semantics match [LIB_YOLOV5N.md](LIB_YOLOV5N.md): BGR 640² in, `N×6` boxes out,
timings pre / npu / post, same threading rules (global NPU lock + per-handle mutex).

---

## Output layouts (auto-detected)

On `init_model`, `probe_outputs()` inspects HHB session outputs:

### A) `RAW_DFL` (recommended)

- **3** outputs, each `(1, C, H, W)` with `C = 4 * reg_max + nc`
- Ultralytics default `reg_max = 16`
- Example PPE: `nc = 10` → `C = 74` → maps `80×80`, `40×40`, `20×20`
- Library runs **Distribution Focal Loss decode** (softmax expectation over 16 bins)
  then class-agnostic NMS in C

Log line:

```text
YOLOv8 outputs: num=3 layout=RAW_DFL nc=10 reg_max=16
  out[0]: c=74 h=80 w=80 ...
```

### B) `DECODED`

- **1** output shaped like `(1, 4+nc, N)` (DFL already applied in graph)
- Only viable if Softmax/Slice survived HHB **and** VIP9000 accepts the network
- Often **fails** or is slow on TH1520 — prefer raw heads export

`nc` for raw heads: `out_c[0] - 4 * 16` when `reg_max` is 16 (e.g. 74 → 10).

---

## Why YOLOv8 needs special export

| Graph content | On VIP9000 |
|---------------|------------|
| Default Ultralytics ONNX (`output0` with DFL) | Softmax / Slice → create-network failure or CPU fallback |
| C2f `chunk` / `Split` | `csinn_split` JSON crash **or** `CusStridedSlice` on CPU (~240 ms) |
| Raw heads + C2f without Slice | NPU-friendly; DFL on CPU in this library |

Export tooling:

- `scripts/export_yolov8_raw_heads.py` — raw heads + C2f patch
- `scripts/onnx_slice_to_conv.py` — Slice → 1×1 Conv gather
- `scripts/onnx_split_to_slice.py` — Split → Slice (then convert to Conv)

Full pipeline: [QUANTIZATION.md](QUANTIZATION.md), [YOLOV8_HHB.md](YOLOV8_HHB.md).

---

## Postprocess details (`RAW_DFL`)

For each spatial cell on each scale (strides 8 / 16 / 32):

1. Take max class logit → one sigmoid → filter by `conf_thres`
2. For each of 4 sides: softmax over `reg_max` bins → expected distance
3. Decode box from anchor `(gx+0.5)*stride` ± distance × stride
4. Class-agnostic NMS with `iou_thres`

Optimizations vs naive decode:

- One sigmoid per cell (max logit), not `nc` sigmoids
- Preallocated candidate / suppressed buffers
- No heap traffic on the hot path

---

## Class names (application layer)

The `.so` returns **integer class ids**. Names live in Python:

```text
# python/classes.txt (PPE example)
Hardhat
Mask
NO-Hardhat
NO-Mask
NO-Safety Vest
Person
Safety Cone
Safety Vest
machinery
vehicle
```

```bash
python3 check_v8.py --names classes.txt
```

Order must match training / `YOLO("your.pt").names`.

---

## Build & deploy pair

```bash
# after HHB:
grep -c strided_slice hhb_out/model.c   # must be 0
cp -f hhb_out/model.c hhb_out/io.c hhb_out/io.h vendor/hhb_v8/
./scripts/build_so_v8.sh

# board MUST get both:
#   libyolov8n.so
#   hhb_out/model.params   (same run)
```

If the board still prints `strided_slice_/m/model.2/Slice_*`, it is running an
**old** `.so` or **old** `model.params`. Verify with:

```bash
strings libyolov8n.so | grep strided_slice   # empty on good build
md5sum libyolov8n.so model.params
```

---

## Threading / multi-model

Identical to v5:

- Global `g_npu_lock` around NPU run
- Safe to load `libyolov5n.so` and `libyolov8n.so` in one process (two locks are
  separate statics **per .so** — if both are loaded, each has its own lock;
  for true cross-library serialization you’d share one lock; in practice run
  one pipeline or accept time-sharing at Python level)

Recommendation: one infer thread per model, or a single queue feeding both.

---

## Demo checklist (`check_v8.py`)

| Step | Expectation |
|------|-------------|
| Load | Class names printed from `classes.txt` |
| Init | No `CusStridedSlice` INFO spam |
| Probe | `RAW_DFL`, correct `nc` |
| Stats | `npu_ms` ~15–40 (healthy), not ~250 |
| Overlay | PPE labels, not COCO `bus`/`car` |

Webcam: NPU init runs **before** V4L2 open; MJPG 320×240 by default to reduce
USB disconnects under NPU load.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `npu_ms` ~240, Slice warnings | Re-export without Slice; new HHB; redeploy **both** artifacts |
| `nc=80` with `c=74` | Old `.so` without probe fix — rebuild current `yolov8_lib.c` |
| Wrong labels | Fix `classes.txt` order |
| Cam `errno=19` | Soft reopen in demo; try `--source /dev/video0`; still-image mode for FPS |
| `csinn_split` abort at init | ONNX still has Split — run rewrite scripts before HHB |

---

<a id="русский"></a>

# Техническая спецификация: `libyolov8n`

**[↑ English](#english)** · **Русский**

Разделяемая библиотека для инференса **YOLOv8** (nano или своя модель, например
**PPE на 10 классов**) в INT8 на **TH1520 VIP9000**.

| Параметр | Значение |
|----------|----------|
| Исходники | `src/yolov8_lib.c`, `src/yolov8_lib.h` |
| Граф HHB | `vendor/hhb_v8/{model.c,io.c,io.h}` |
| Артефакт | `libyolov8n.so` |
| Сборка | `scripts/build_so_v8.sh` |
| Демо | `python/check_v8.py` |
| Имена классов | `python/classes.txt` |
| Веса | `model.params` (тот же прогон HHB, что и `model.c`) |

## Место в стеке

LUT → NPU → dequant → автоматический выбор раскладки выходов:

| Раскладка | Выходы HHB | Постпроцесс |
|-----------|------------|-------------|
| **RAW_DFL** (рекомендуется) | 3 тензора `(1, 4·reg_max+nc, H, W)` | DFL + NMS в C |
| **DECODED** | 1 тензор `(1, 4+nc, N)` | На VIP9000 часто нестабилен или медленен |

Пример PPE: `nc=10`, `reg_max=16` → `C=74`. В журнале:

```text
YOLOv8 outputs: num=3 layout=RAW_DFL nc=10 reg_max=16
```

API с префиксом `yolov8_*` и алиасы как у v5 (`init_model`, `run_inference`, …)
описаны в секции English.

## Почему нужен особый экспорт

| Содержимое графа | Поведение VIP9000 |
|------------------|-------------------|
| Стандартный Ultralytics ONNX с DFL | Softmax / Slice → сбой или откат на CPU |
| C2f `chunk` / `Split` | Авария `csinn_split` или `CusStridedSlice` (~240 мс) |
| Сырые головы + C2f без Slice | Граф пригоден для NPU; DFL выполняется в этой библиотеке |

Инструменты: `scripts/export_yolov8_raw_heads.py`, при необходимости
`onnx_split_to_slice.py` / `onnx_slice_to_conv.py`.  
Полный пайплайн: [QUANTIZATION.md](QUANTIZATION.md#русский).

## Деплой

Всегда выкладывайте пару из **одного** прогона HHB:

```text
libyolov8n.so + model.params
```

```bash
grep -c strided_slice vendor/hhb_v8/model.c   # 0
strings libyolov8n.so | grep strided_slice    # пусто
```

Имена классов живут только в приложении (`classes.txt`); библиотека возвращает
целочисленные идентификаторы.

## Критерии демо (`check_v8.py`)

| Шаг | Ожидание |
|-----|----------|
| Загрузка | Имена из `classes.txt` |
| Инициализация | Нет спама `CusStridedSlice` |
| Probe | `RAW_DFL`, корректный `nc` |
| Статистика | `npu_ms` ~15–40 (не ~250) |
| Подписи | PPE-имена, не COCO вроде `bus` / `car` |

Камера: NPU поднимается **до** V4L2; по умолчанию MJPG 320×240.

## Диагностика

| Симптом | Действие |
|---------|----------|
| `npu_ms` ~240, предупреждения Slice | Переэкспорт без Slice; новый HHB; оба артефакта на плату |
| `nc=80` при `c=74` | Пересобрать актуальный `yolov8_lib.c` |
| Неверные подписи | Порядок строк в `classes.txt` |
| Камера `errno=19` | Soft-reopen в демо; другой `/dev/video*`; режим JPEG |
| Обрыв на `csinn_split` | Убрать Split из ONNX до HHB |
