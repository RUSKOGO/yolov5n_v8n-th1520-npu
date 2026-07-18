**Language / Язык:** [English](#english) · [Русский](#русский)

---

<a id="english"></a>

# Quantization guide: custom YOLO → TH1520 NPU

Complete pipeline from a trained checkpoint to a board-ready
`libyolov*.so` + `model.params`.

```text
.pt / .weights
    → ONNX (opset 12, NPU-friendly graph)
        → HHB INT8 (board=th1520, calibrate)
            → model.c + model.params
                → build_so*.sh → libyolov*.so
                    → LicheePi 4A + check_*.py
```

---

## 0. Prerequisites

| Component | Purpose |
|-----------|---------|
| Trained weights | Ultralytics `.pt` (v5/v8) or exportable PyTorch module |
| Linux x86_64 (or vendor Docker) | Run **HHB** |
| Package `hhb` + SHL for `th1520` | Quantize + codegen C |
| `ultralytics`, `torch`, `onnx` | Export |
| `riscv64-unknown-linux-gnu-gcc` or on-board `gcc` | Build `.so` |
| This repository | Export scripts, libs, demos |
| Calibration images | 20–100 JPEGs from **your domain** |

Install export deps (build/PC machine):

```bash
pip install ultralytics onnx onnxruntime torch
```

Confirm HHB:

```bash
hhb --help | head
```

---

## 1. Prepare calibration data

HHB fits INT8 scales on real tensors. Random noise works technically but
**hurts accuracy**.

```bash
mkdir -p calib
cp /path/to/domain_images/*.jpg calib/
# optional fallback:
# python3 scripts/make_calib_images.py --out calib --n 24
```

Use the same letterbox / color distribution you will run at inference when possible.

---

## Part A — YOLOv8 / custom PPE (recommended path)

### A.1 Export NPU-friendly ONNX

Do **not** rely on plain `yolo export` for VIP9000 if you care about FPS.

```bash
python3 scripts/export_yolov8_raw_heads.py \
  --weights ppe.pt \
  --out yolov8n_raw.onnx \
  --imgsz 640 \
  --opset 12
```

What the script does:

1. Loads Ultralytics model
2. **Patches every C2f**: `cv1(x).chunk(2)` → two half-convs (`cv1a` / `cv1b`)  
   → **no channel Slice** in ONNX
3. Replaces Detect forward with **three raw heads** (concat box+cls, **no DFL** in graph)
4. Writes `classes.txt` from `model.names`

Verify:

```text
ONNX ops: Slice=0 Split=0 (want both 0)
head[i]: (1, 4*reg_max+nc, H, W)   # e.g. (1, 74, 80/40/20) for nc=10
```

If Slice/Split remain:

```bash
python3 scripts/onnx_split_to_slice.py --in yolov8n_raw.onnx --out mid.onnx
python3 scripts/onnx_slice_to_conv.py  --in mid.onnx --out yolov8n_raw.onnx
```

### A.2 Run HHB

One command (no mid-line `#` comments):

```bash
hhb -D \
  --model-file yolov8n_raw.onnx \
  --model-format onnx \
  --data-scale-div 255 \
  --board th1520 \
  --input-name "images" \
  --output-name "output0;output1;output2" \
  --input-shape "1 3 640 640" \
  --calibrate-dataset calib \
  --quantization-scheme "int8_asym"
```

Notes:

| Flag | Meaning |
|------|---------|
| `--data-scale-div 255` | Input assumed 0..255 → divide by 255 (matches our LUT side) |
| `--board th1520` | VIP9000 codegen |
| `--quantization-scheme int8_asym` | Asymmetric INT8 |
| `--calibrate-dataset calib` | Directory of images |

### A.3 Validate generated graph

```bash
grep -c strided_slice hhb_out/model.c   # → 0
grep -c csinn_split hhb_out/model.c     # → 0
```

Non-zero `strided_slice` ⇒ you will see ~4 FPS and `CusStridedSlice` on device.
Fix ONNX and re-run HHB.

### A.4 Install graph into the repo & build

```bash
cp -f hhb_out/model.c hhb_out/io.c hhb_out/io.h vendor/hhb_v8/

export HHB_NN2=/usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520
export HHB_PB=/usr/local/lib/python3.8/dist-packages/hhb/prebuilt
export CC=riscv64-unknown-linux-gnu-gcc

./scripts/build_so_v8.sh
# → libyolov8n.so

strings libyolov8n.so | grep strided_slice || echo "OK"
```

### A.5 Deploy

```bash
./scripts/pack_board_v8.sh
# or manually copy:
#   libyolov8n.so
#   hhb_out/model.params   →  model.params on board
#   python/check_v8.py
#   python/classes.txt
#   python/requirements.txt
```

On the board:

```bash
python3 check_v8.py --names classes.txt --source auto
```

Success criteria: no Slice spam, `layout=RAW_DFL nc=…`, `npu_ms` tens of ms.

---

## Part B — YOLOv5n

### B.1 Export ONNX

Requirements aligned with Sipeed / T-Head samples:

- Opset **12**
- Fixed size (often `640×640`; some samples use `384×640` — keep **one** size end-to-end)
- Outputs = **three detect heads** (last convs), **not** a single NMS’d tensor
- Avoid heavy postprocess nodes inside ONNX

Example with Ultralytics v5-style export (adjust to your tree):

```bash
# illustrative — match your yolov5 export tooling
python export.py --weights yolov5n.pt --include onnx --opset 12 --imgsz 640
# then trim/rename outputs to the three head tensors HHB expects
```

Inspect:

```bash
python3 - <<'PY'
import onnx
m = onnx.load("yolov5n.onnx")
print([(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print([(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
PY
```

### B.2 HHB

```bash
hhb -D \
  --model-file yolov5n.onnx \
  --model-format onnx \
  --data-scale-div 255 \
  --board th1520 \
  --input-name "images" \
  --output-name "<head0>;<head1>;<head2>" \
  --input-shape "1 3 640 640" \
  --calibrate-dataset calib \
  --quantization-scheme "int8_asym"
```

Replace `<head*>` with actual ONNX output names (Netron or the script above).

### B.3 Build

```bash
cp -f hhb_out/model.c hhb_out/io.c hhb_out/io.h vendor/hhb/
./scripts/build_so.sh
# deploy libyolov5n.so + model.params + python/check.py
```

Postprocess is SHL YOLOv5 NMS inside `yolov5n_lib` — keep heads compatible with that path.

---

## Part C — Accuracy & calibration tips

1. **Domain calib** beats generic COCO crops for PPE / factory scenes.
2. Recalibrate after any architecture / export change.
3. Compare FP32 ONNX vs INT8 on a fixed image set before board bring-up.
4. If INT8 drops recall: more calib images, check `--data-scale-div` vs training
   preprocess, confirm letterbox `114`/`128` padding matches what you use in demos
   (`check_*.py` uses 128 gray).

---

## Part D — Common HHB / board failures

| Symptom | Cause | Action |
|---------|--------|--------|
| `failed to infer the model format` | Missing `--model-format onnx` | Add the flag |
| `bash: --data-scale-div: command not found` | Broken line continuations / comments | Paste full `hhb -D \` block |
| `Could not create network` / SegFault | Softmax/Slice/DFL in graph | Raw-heads export |
| `csinn_split` / `Expect number` | Split ops | `onnx_split_to_slice` (+ Conv) |
| `Strided_slice ... memory leak`, `npu_ms~240` | Slice on CPU | C2f patch / `onnx_slice_to_conv`; **new** `.so`+`.params` |
| Board still slow after “good” HHB | Old artifacts on device | `md5sum` both files; `strings` for `strided_slice` |

---

## Part E — Checklist before calling it done

- [ ] ONNX: `Slice=0`, `Split=0` (v8)
- [ ] HHB finished without error
- [ ] `grep strided_slice hhb_out/model.c` → `0`
- [ ] `vendor/hhb` or `vendor/hhb_v8` updated from **this** `hhb_out`
- [ ] `.so` rebuilt after copying `model.c`
- [ ] Board has **matching** `.so` + `model.params`
- [ ] Demo: healthy `npu_ms`, correct class names
- [ ] Optional: `scripts/pack_board_v8.sh` for a clean bundle

---

## Related docs

- [LIB_YOLOV5N.md](LIB_YOLOV5N.md) — runtime API v5  
- [LIB_YOLOV8.md](LIB_YOLOV8.md) — runtime API v8  
- [YOLOV8_HHB.md](YOLOV8_HHB.md) — short v8 graph reference  
- [GETTING_STARTED.md](GETTING_STARTED.md) — board / webcam  
- [../README.md](../README.md) — project overview  

---

<a id="русский"></a>

# Квантование модели: YOLO → NPU TH1520

**[↑ English](#english)** · **Русский**

Полный путь от обученного чекпоинта до готовых к плате `libyolov*.so` и `model.params`.

```text
.pt / веса
    → ONNX (opset 12, граф без Slice/Split)
        → HHB INT8 (--board th1520, калибровка)
            → model.c + model.params
                → build_so*.sh → libyolov*.so
                    → LicheePi 4A + check_*.py
```

## 0. Что потребуется

| Компонент | Назначение |
|-----------|------------|
| Обученные веса | Ultralytics `.pt` (v5/v8) |
| Linux x86_64 или Docker вендора | Запуск **HHB** |
| Пакет `hhb` + SHL для `th1520` | Квантование и генерация C |
| `ultralytics`, `torch`, `onnx` | Экспорт |
| `riscv64-unknown-linux-gnu-gcc` или `gcc` на плате | Сборка `.so` |
| Этот репозиторий | Скрипты экспорта, библиотеки, демо |
| Калибровочные кадры | 20–100 JPEG из **вашего** домена |

```bash
pip install ultralytics onnx onnxruntime torch
hhb --help | head
```

## 1. Калибровка

```bash
mkdir -p calib
cp /path/to/domain_images/*.jpg calib/
# запасной вариант (хуже по точности):
# python3 scripts/make_calib_images.py --out calib --n 24
```

Желательно совпадение letterbox и цветового пайплайна с инференсом.

---

## Часть A — YOLOv8 / custom PPE

### A.1 Экспорт ONNX под NPU

Обычный `yolo export` для VIP9000 часто даёт низкий FPS или аварии. Используйте:

```bash
python3 scripts/export_yolov8_raw_heads.py \
  --weights ppe.pt \
  --out yolov8n_raw.onnx \
  --imgsz 640 \
  --opset 12
```

Скрипт:

1. Загружает модель Ultralytics  
2. Патчит каждый **C2f**: `chunk(2)` → два полу-свёртки (`cv1a` / `cv1b`) — **без channel Slice**  
3. Заменяет Detect на **три сырые головы** (без DFL в графе)  
4. Пишет `classes.txt` из `model.names`

Ожидаемо:

```text
ONNX ops: Slice=0 Split=0
head[i]: (1, 4*reg_max+nc, H, W)   # например (1, 74, 80/40/20) при nc=10
```

Если Slice/Split остались:

```bash
python3 scripts/onnx_split_to_slice.py --in yolov8n_raw.onnx --out mid.onnx
python3 scripts/onnx_slice_to_conv.py  --in mid.onnx --out yolov8n_raw.onnx
```

### A.2 HHB

Одна команда (**без** комментариев `#` внутри продолжений строк):

```bash
hhb -D \
  --model-file yolov8n_raw.onnx \
  --model-format onnx \
  --data-scale-div 255 \
  --board th1520 \
  --input-name "images" \
  --output-name "output0;output1;output2" \
  --input-shape "1 3 640 640" \
  --calibrate-dataset calib \
  --quantization-scheme "int8_asym"
```

| Флаг | Смысл |
|------|--------|
| `--data-scale-div 255` | Вход 0…255, деление на 255 (согласовано с LUT) |
| `--board th1520` | Кодогенерация под VIP9000 |
| `--quantization-scheme int8_asym` | Асимметричный INT8 |
| `--calibrate-dataset calib` | Каталог изображений |

### A.3 Проверка графа

```bash
grep -c strided_slice hhb_out/model.c   # → 0
grep -c csinn_split hhb_out/model.c     # → 0
```

Ненулевой `strided_slice` на плате даст ~4 FPS и `CusStridedSlice`. Исправьте ONNX и повторите HHB.

### A.4 Сборка

```bash
cp -f hhb_out/model.c hhb_out/io.c hhb_out/io.h vendor/hhb_v8/

export HHB_NN2=/usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520
export HHB_PB=/usr/local/lib/python3.8/dist-packages/hhb/prebuilt
export CC=riscv64-unknown-linux-gnu-gcc

./scripts/build_so_v8.sh
strings libyolov8n.so | grep strided_slice || echo "OK"
```

### A.5 Деплой

```bash
./scripts/pack_board_v8.sh
# либо вручную на плату:
#   libyolov8n.so
#   hhb_out/model.params  →  model.params
#   python/check_v8.py, classes.txt, requirements.txt
```

На плате:

```bash
python3 check_v8.py --names classes.txt --source auto
```

Критерий успеха: нет спама Slice, `layout=RAW_DFL nc=…`, `npu_ms` — десятки миллисекунд.

---

## Часть B — YOLOv5n

Требования: opset **12**, фиксированный размер (как правило 640×640), **три головы**
детекции без NMS в графе.

```bash
hhb -D \
  --model-file yolov5n.onnx \
  --model-format onnx \
  --data-scale-div 255 \
  --board th1520 \
  --input-name "images" \
  --output-name "<голова0>;<голова1>;<голова2>" \
  --input-shape "1 3 640 640" \
  --calibrate-dataset calib \
  --quantization-scheme "int8_asym"

cp -f hhb_out/model.c hhb_out/io.c hhb_out/io.h vendor/hhb/
./scripts/build_so.sh
```

Имена выходов возьмите из Netron / `onnx.load`. Постпроцесс — NMS SHL внутри `yolov5n_lib`.

---

## Точность

1. Калибруйте на кадрах своего домена, а не на случайном шуме.  
2. После смены архитектуры или экспорта — повторная калибровка.  
3. Сравните FP32 ONNX и INT8 на фиксированном наборе до выезда на плату.  
4. При просадке recall: больше calib-кадров, проверка `--data-scale-div`, padding letterbox (**128** в демо).

---

## Типичные сбои

| Симптом | Причина | Действие |
|---------|---------|----------|
| `failed to infer the model format` | Нет `--model-format onnx` | Добавить флаг |
| `bash: --data-scale-div: command not found` | Оборвана многострочная команда | Вставить блок `hhb -D \` целиком |
| `Could not create network` / SegFault | Softmax/Slice/DFL в графе | Raw-heads экспорт |
| `csinn_split` / `Expect number` | Операции Split | `onnx_split_to_slice` (+ Conv) |
| `Strided_slice...`, `npu_ms~240` | Slice на CPU | Патч C2f / `onnx_slice_to_conv`; **новые** `.so` и `.params` |
| После «удачного» HHB всё ещё медленно | На плате старые файлы | `md5sum`; `strings … \| grep strided_slice` |

---

## Чеклист перед релизом

- [ ] ONNX: `Slice=0`, `Split=0` (для v8)
- [ ] HHB завершился без ошибки
- [ ] `grep strided_slice hhb_out/model.c` → `0`
- [ ] `vendor/hhb` или `vendor/hhb_v8` обновлены из **этого** `hhb_out`
- [ ] `.so` пересобран после копирования `model.c`
- [ ] На плате согласованная пара `.so` + `model.params`
- [ ] Демо: нормальный `npu_ms`, верные имена классов

Связанные документы: [LIB_YOLOV5N](LIB_YOLOV5N.md#русский) · [LIB_YOLOV8](LIB_YOLOV8.md#русский) · [YOLOV8_HHB](YOLOV8_HHB.md#русский) · [GETTING_STARTED](GETTING_STARTED.md#русский) · [README](../README.md#русский)
