**Language / Язык:** [English](#english) · [Русский](#русский)

---

<a id="english"></a>

# YOLO on TH1520 NPU

[![Target](https://img.shields.io/badge/board-LicheePi%204A%20%2F%20TH1520-blue)](#quick-start)
[![NPU](https://img.shields.io/badge/NPU-VIP9000%20INT8-green)](#quick-start)

C shared libraries and Python demos for **INT8 YOLO** on the
**XuanTie TH1520 VIP9000** (Sipeed **LicheePi 4A**, RevyOS).

Preprocess, NPU run, and postprocess live in a small `.so` so you get
**tens of FPS** instead of the usual **1–4 FPS** from naive Python + HHB samples.

| Library | Binary | Demo |
|---------|--------|------|
| YOLOv5n | `libyolov5n.so` | [`examples/yolov5n`](examples/yolov5n) |
| YOLOv8 / PPE | `libyolov8n.so` | [`examples/yolov8n`](examples/yolov8n) |

## Quick start

Prebuilt `.so` + `model.params` are in the repo. **Run on the board** (RISC-V), not on a PC.

```bash
git clone <this-repo>.git
cd yolov5n-th1520-npu/examples/yolov8n   # or examples/yolov5n
chmod +x run.sh
./run.sh
```

Open `http://<board-ip>:8000/`.

Same without the helper script:

```bash
cd examples/yolov8n
pip install -r requirements.txt
python3 check_v8.py --source auto --names classes.txt
```

## What you get

- Zero-overhead C runtime (`init_model` / `run_inference` via ctypes)
- Webcam → WebSocket preview demos
- Export + HHB scripts to quantize **your own** weights

## Documentation

| Doc | Contents |
|-----|----------|
| [examples/](examples/) | Ready-to-run bundles |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#english) | Board / webcam notes |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md#english) | `.pt` → ONNX → HHB → `.so` |
| [docs/LIB_YOLOV5N.md](docs/LIB_YOLOV5N.md#english) | `libyolov5n` API |
| [docs/LIB_YOLOV8.md](docs/LIB_YOLOV8.md#english) | `libyolov8n` API |
| [docs/YOLOV8_HHB.md](docs/YOLOV8_HHB.md#english) | YOLOv8 graph checklist |
| [docs/README.md](docs/README.md#english) | Full index |

**Why this library**, vs alternatives, and optimizations: see
[docs/LIB_YOLOV5N.md](docs/LIB_YOLOV5N.md#english) / [docs/LIB_YOLOV8.md](docs/LIB_YOLOV8.md#english)
and the project motivation in [docs/README.md](docs/README.md#english).
(Short version: Python+HHB samples starve the NPU; this `.so` keeps it fed.)

### C API (sketch)

```c
void *init_model(const char *params_path);
int   run_inference(void *handle, uint8_t *bgr_640x640,
                    float *out_boxes, int max_boxes);
void  release_model(void *handle);
```

See [LIB_YOLOV5N](docs/LIB_YOLOV5N.md#english) / [LIB_YOLOV8](docs/LIB_YOLOV8.md#english) for the full surface.

## License notes

MIT — see [LICENSE](LICENSE).

`vendor/hhb*` is HHB-generated (T-Head). Ultralytics train/export is GPL-3.0 if
redistributed; this repo focuses on the **runtime**.

---

<a id="русский"></a>

# YOLO на NPU TH1520

**[↑ English](#english)** · **Русский**

[![Плата](https://img.shields.io/badge/плата-LicheePi%204A%20%2F%20TH1520-blue)](#быстрый-старт)
[![NPU](https://img.shields.io/badge/NPU-VIP9000%20INT8-green)](#быстрый-старт)

C-библиотеки и Python-демо для **INT8 YOLO** на **VIP9000** (Sipeed **LicheePi 4A**, RevyOS).

Препроцесс, прогон на NPU и постпроцесс — в `.so`, чтобы получать **десятки FPS**,
а не типичные **1–4 FPS** у наивного Python и примеров HHB.

| Библиотека | Бинарь | Демо |
|------------|--------|------|
| YOLOv5n | `libyolov5n.so` | [`examples/yolov5n`](examples/yolov5n) |
| YOLOv8 / PPE | `libyolov8n.so` | [`examples/yolov8n`](examples/yolov8n) |

## Быстрый старт

В репозитории уже лежат собранные `.so` и `model.params`.
Запускать нужно **на плате** (RISC-V), не на ПК.

```bash
git clone <этот-репо>.git
cd yolov5n-th1520-npu/examples/yolov8n   # или examples/yolov5n
chmod +x run.sh
./run.sh
```

Откройте `http://<IP-платы>:8000/`.

Без скрипта:

```bash
cd examples/yolov8n
pip install -r requirements.txt
python3 check_v8.py --source auto --names classes.txt
```

## Что внутри

- Runtime на C (`init_model` / `run_inference` через ctypes)
- Демо: камера → предпросмотр по WebSocket
- Скрипты экспорта и HHB для **своих** весов

## Документация

| Документ | Содержание |
|----------|------------|
| [examples/](examples/) | Готовые бандлы «скачал — запустил» |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md#русский) | Плата, камера |
| [docs/QUANTIZATION.md](docs/QUANTIZATION.md#русский) | `.pt` → ONNX → HHB → `.so` |
| [docs/LIB_YOLOV5N.md](docs/LIB_YOLOV5N.md#русский) | API `libyolov5n` |
| [docs/LIB_YOLOV8.md](docs/LIB_YOLOV8.md#русский) | API `libyolov8n` |
| [docs/YOLOV8_HHB.md](docs/YOLOV8_HHB.md#русский) | Чеклист графа YOLOv8 |
| [docs/README.md](docs/README.md#русский) | Полное оглавление |

**Зачем библиотека**, сравнение с аналогами и оптимизации — в
[спецификациях](docs/LIB_YOLOV8.md#русский) и [оглавлении](docs/README.md#русский).
Кратко: Python + сэмплы HHB не успевают кормить NPU; этот `.so` держит его загруженным.

### C API (кратко)

```c
void *init_model(const char *params_path);
int   run_inference(void *handle, uint8_t *bgr_640x640,
                    float *out_boxes, int max_boxes);
void  release_model(void *handle);
```

Полностью: [LIB_YOLOV5N](docs/LIB_YOLOV5N.md#русский) / [LIB_YOLOV8](docs/LIB_YOLOV8.md#русский).

## Лицензии

MIT — см. [LICENSE](LICENSE).

`vendor/hhb*` — код HHB (T-Head). Ultralytics (train/export) — GPL-3.0 при
распространении; этот репозиторий сосредоточен на **runtime**.
