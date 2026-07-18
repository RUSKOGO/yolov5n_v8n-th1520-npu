# YOLOv5n / YOLOv8 на TH1520 NPU (LicheePi 4A)

Низкоуровневая C-библиотека + Python-демо для INT8-инференса YOLO на
**NPU VIP9000** платы **Sipeed LicheePi 4A** (чип XuanTie TH1520).

Цель: NPU не простаивает из‑за медленного Python на RISC-V.
Типично: **~15–40 ms** на граф (десятки FPS), LUT-препроцесс и NMS в C.

| | YOLOv5n | YOLOv8 (в т.ч. custom PPE) |
|--|---------|----------------------------|
| Библиотека | `libyolov5n.so` | `libyolov8n.so` |
| Исходники | `src/yolov5n_lib.*` | `src/yolov8_lib.*` |
| Граф HHB | `vendor/hhb/` | `vendor/hhb_v8/` |
| Демо | `python/check.py` | `python/check_v8.py` |
| Сборка | `scripts/build_so.sh` | `scripts/build_so_v8.sh` |
| Постпроцесс | SHL yolov5 NMS | свой DFL decode + NMS |

---

## Быстрый старт (уже есть `.so` + `model.params`)

На плате:

```bash
cd check_run8   # или любая папка с артефактами
# нужны: libyolov8n.so  model.params  check_v8.py  classes.txt

python3 -m venv venv_new && source venv_new/bin/activate
pip install -r requirements.txt   # из python/requirements.txt

python3 check_v8.py --source auto
# браузер: http://<IP-платы>:8000/
```

Имена классов PPE (порядок = id из обучения):

```text
Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest,
Person, Safety Cone, Safety Vest, machinery, vehicle
```

Файл: `python/classes.txt` (или `--names classes.txt`).

---

## С нуля (веса → плата)

Полная инструкция: **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**

Кратко:

1. Обучить / взять `.pt` (Ultralytics YOLOv8)
2. Экспорт raw-heads ONNX **без Slice/Split** (`scripts/export_yolov8_raw_heads.py`)
3. Квантование HHB → `model.c` + `model.params`
4. Сборка `libyolov8n.so` (`scripts/build_so_v8.sh`)
5. Деплой на LicheePi 4A + `check_v8.py`

Для YOLOv5: см. раздел в GETTING_STARTED и `vendor/hhb/`.

Детали HHB / графа v8: **[docs/YOLOV8_HHB.md](docs/YOLOV8_HHB.md)**

---

## Структура репозитория

```text
yolov5n-th1520-npu/
├── README.md
├── docs/
│   ├── GETTING_STARTED.md   ← с нуля: что скачать и как собрать
│   └── YOLOV8_HHB.md        ← экспорт, Slice→Conv, HHB флаги
├── src/
│   ├── yolov5n_lib.c / .h   ← ctypes API v5
│   └── yolov8_lib.c / .h    ← ctypes API v8 (DFL + NMS)
├── vendor/
│   ├── hhb/                 ← HHB model.c/io.* для YOLOv5n
│   └── hhb_v8/              ← то же для YOLOv8 (после вашего HHB)
├── python/
│   ├── check.py             ← демо v5
│   ├── check_v8.py          ← демо v8 + webcam + WebSocket
│   ├── classes.txt          ← имена классов (PPE)
│   └── requirements.txt
├── scripts/
│   ├── build_so.sh / build_so_v8.sh
│   ├── export_yolov8_raw_heads.py   ← обязательный экспорт для NPU
│   ├── onnx_slice_to_conv.py        ← если в ONNX остались Slice
│   ├── onnx_split_to_slice.py
│   ├── make_calib_images.py
│   └── export_yolov8_onnx.py
└── calib/                   ← картинки для калибровки HHB (свои!)
```

**Не коммитить** (см. `.gitignore`): `*.so`, `model.params`, `*.onnx`, `*.pt`, `hhb_out/`.

---

## C API (ctypes)

Одинаковый контракт для v5 и v8:

```c
void *init_model(const char *params_path);
int   run_inference(void *handle, uint8_t *bgr_640x640,
                    float *out_boxes, int max_boxes);
void  release_model(void *handle);
int   yolo_set_thresholds(void *handle, float conf, float iou);
int   yolo_warmup(void *handle, int runs);
int   yolo_get_timings_us(void *handle, float *pre, float *npu, float *post);
```

- Вход: contiguous **BGR** `uint8`, letterbox **640×640×3**
- Выход: `x1,y1,x2,y2,score,cls` в координатах letterbox
- Два `init_model()` можно; NPU сериализуется внутренним lock
- Не вызывать `run_inference` параллельно на **одном** handle

---

## Производительность

| Симптом | Причина | Что делать |
|---------|---------|------------|
| `npu_ms` ~200–270, спам `Strided_slice` | C2f Slice на CPU | переэкспорт без Slice, новый HHB + `.so` + `model.params` |
| `npu_ms` ~15–40, FPS ок | норма | — |
| камера `errno=19` | USB UVC + NPU | NPU init первым; MJPG; `--source /dev/video0` |
| подписи bus/car | COCO имена | свой `classes.txt` |

Всегда копируйте на плату **пару** `libyolov*.so` + `model.params` из **одного** HHB-прогона.

Проверка нового v8-графа:

```bash
grep -c strided_slice vendor/hhb_v8/model.c   # 0
strings libyolov8n.so | grep strided_slice    # пусто
```

---

## Требования

| Где | Что |
|-----|-----|
| Docker / ПК с HHB | пакет `hhb`, Python, ultralytics (экспорт) |
| Сборка `.so` | `riscv64-unknown-linux-gnu-gcc` **или** `gcc` на плате |
| Плата RevyOS | `libshl` (из HHB runtime), Python3, OpenCV, FastAPI |

---

## Лицензии

- `vendor/hhb*` — код, сгенерированный HHB (шаблоны T-Head)
- Обёртки `src/*` — ваша лицензия при публикации
- Ultralytics YOLO — GPL-3.0 (если тащите их train/export код)

---

## Связанные доки

- [С нуля](docs/GETTING_STARTED.md)
- [YOLOv8 + HHB подробно](docs/YOLOV8_HHB.md)
