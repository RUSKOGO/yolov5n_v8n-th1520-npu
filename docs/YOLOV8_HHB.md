# YOLOv8n → TH1520 (HHB) → libyolov8n.so

Техническая шпаргалка. Полный путь «с нуля»: [GETTING_STARTED.md](GETTING_STARTED.md).

## 1. Экспорт ONNX

```bash
pip install ultralytics onnx
python3 scripts/export_yolov8_raw_heads.py --weights ppe.pt --out yolov8n_raw.onnx
```

Требования VIP9000:

| Требование | Почему |
|------------|--------|
| Opset **12** | стабильнее для HHB |
| **Нет NMS** в графе | NMS в нашей C-либе |
| Fixed `imgsz=640` | фиксированный shape |
| **3 raw heads** `(1, 4*reg_max+nc, H, W)` | без DFL Softmax в графе |
| **0 Slice / 0 Split** | иначе CPU `CusStridedSlice` → ~4 FPS или краш |

Скрипт патчит C2f (`chunk` → два half-Conv) и пишет `classes.txt`.

### Если в ONNX остались Slice / Split

```bash
python3 scripts/onnx_split_to_slice.py --in in.onnx --out mid.onnx
python3 scripts/onnx_slice_to_conv.py  --in mid.onnx --out yolov8n_raw.onnx
```

`onnx_slice_to_conv.py` заменяет channel-Slice на 1×1 Conv (на NPU).

### Если HHB падает на DECODED / Softmax

Симптом: `Could not create network object`, `MBS parser`, SegFault.  
Решение: только raw-heads экспорт выше (не `yolo export` с одним `output0`).

### Если краш на `csinn_split`

Симптом: `shl_pnna_create_split_internal`, `Expect number`.  
Решение: Split→Slice→Conv (команды выше), затем HHB.

---

## 2. HHB

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

Проверка:

```bash
grep -c strided_slice hhb_out/model.c   # 0
grep -c csinn_split hhb_out/model.c     # 0
cp -f hhb_out/model.c hhb_out/io.c hhb_out/io.h vendor/hhb_v8/
```

---

## 3. Сборка

```bash
./scripts/build_so_v8.sh
# → libyolov8n.so
```

На плату: **`libyolov8n.so` + `hhb_out/model.params`** из этого же прогона.

---

## 4. Демо

```bash
python3 python/check_v8.py --names python/classes.txt --source auto
```

Ожидаемый лог:

```text
YOLOv8 outputs: num=3 layout=RAW_DFL nc=10 reg_max=16
```

Без спама `Strided_slice ... memory leak`. `npu_ms` ~15–40.

---

## Layouts в `yolov8_lib.c`

| Layout | Выходы HHB | Постпроцесс |
|--------|------------|-------------|
| `RAW_DFL` | 3× `(1, 4*reg_max+nc, H, W)` | DFL + NMS в C (рекомендуется) |
| `DECODED` | 1× `(1, 4+nc, N)` | без DFL (если Softmax влез в NPU) |

`nc` для raw heads: `channels - 4*16` (пример: 74 → 10).

---

## Типичные PPE-имена (пример)

```text
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

Берите из своего `.pt`: `YOLO("ppe.pt").names`.
