**Language / Язык:** [English](#english) · [Русский](#русский)

---

<a id="english"></a>

# YOLOv8 × HHB — graph checklist

Short reference. Full pipeline: [QUANTIZATION.md](QUANTIZATION.md#english).  
Runtime: [LIB_YOLOV8.md](LIB_YOLOV8.md#english).

## Required ONNX

| Item | Value |
|------|--------|
| Opset | 12 |
| Input | `images`, shape `1×3×640×640` |
| Outputs | `output0;output1;output2` (raw heads) |
| Per head | `(1, 4*reg_max+nc, H, W)`, e.g. `(1, 74, 80)` |
| `Slice` / `Split` | **0** |

```bash
python3 scripts/export_yolov8_raw_heads.py --weights ppe.pt --out yolov8n_raw.onnx

# only if Slice/Split remain:
python3 scripts/onnx_split_to_slice.py --in yolov8n_raw.onnx --out mid.onnx
python3 scripts/onnx_slice_to_conv.py  --in mid.onnx --out yolov8n_raw.onnx
```

## HHB

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

grep -c strided_slice hhb_out/model.c   # must be 0
grep -c csinn_split hhb_out/model.c     # must be 0
cp -f hhb_out/{model.c,io.c,io.h} vendor/hhb_v8/
./scripts/build_so_v8.sh
```

## Failure modes

| Symptom | Meaning |
|---------|---------|
| `Strided_slice ... memory leak` and `npu_ms~240` | Slice on CPU — bad graph or stale deploy |
| `csinn_split` / `Expect number` | Split still present |
| Softmax / create-network failure | Use raw-heads export, not DECODED Ultralytics ONNX |

---

<a id="русский"></a>

# YOLOv8 × HHB — чеклист графа

**[↑ English](#english)** · **Русский**

Краткий справочник. Полный пайплайн: [QUANTIZATION.md](QUANTIZATION.md#русский).  
Runtime: [LIB_YOLOV8.md](LIB_YOLOV8.md#русский).

## Требования к ONNX

| Параметр | Значение |
|----------|----------|
| Opset | 12 |
| Вход | `images`, форма `1×3×640×640` |
| Выходы | `output0;output1;output2` (сырые головы) |
| На голову | `(1, 4*reg_max+nc, H, W)`, напр. `(1, 74, 80)` |
| `Slice` / `Split` | **0** |

```bash
python3 scripts/export_yolov8_raw_heads.py --weights ppe.pt --out yolov8n_raw.onnx

# только если Slice/Split остались:
python3 scripts/onnx_split_to_slice.py --in yolov8n_raw.onnx --out mid.onnx
python3 scripts/onnx_slice_to_conv.py  --in mid.onnx --out yolov8n_raw.onnx
```

## HHB

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

grep -c strided_slice hhb_out/model.c   # должно быть 0
grep -c csinn_split hhb_out/model.c     # должно быть 0
cp -f hhb_out/{model.c,io.c,io.h} vendor/hhb_v8/
./scripts/build_so_v8.sh
```

## Типичные сбои

| Симптом | Смысл |
|---------|--------|
| `Strided_slice ... memory leak` и `npu_ms~240` | Slice на CPU — плохой граф или устаревший деплой |
| `csinn_split` / `Expect number` | В графе остался Split |
| Softmax / ошибка создания сети | Нужен raw-heads экспорт, не DECODED ONNX Ultralytics |
