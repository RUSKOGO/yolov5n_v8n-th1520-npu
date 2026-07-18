# С нуля: YOLO → TH1520 NPU → LicheePi 4A

Пошаговый гайд: что скачать, как экспортировать, прогнать HHB,
собрать `.so` и запустить веб-демо на плате.

Целевая плата: **Sipeed LicheePi 4A** (TH1520, NPU VIP9000), ОС **RevyOS**.
Ниже путь для **YOLOv8** (в т.ч. custom PPE). Блок про **YOLOv5** — в конце.

---

## 0. Что понадобится

### Железо / машины

| Роль | Где |
|------|-----|
| Экспорт ONNX + HHB | ПК или Docker с **HHB** (часто x86_64 Linux) |
| Сборка `libyolov8n.so` | тот же Docker HHB **или** кросс-gcc, **или** gcc на плате |
| Запуск | LicheePi 4A по SSH |

### Софт

1. **Этот репозиторий** (`yolov5n-th1520-npu`)
2. **Веса** Ultralytics: `ppe.pt` / `yolov8n.pt` / свой `.pt`
3. **HHB** (T-Head Heterogeneous Honey Badger) + SHL для `th1520`  
   Обычно ставится как Python-пакет `hhb` внутри официального Docker/образа Sipeed/T-Head.
4. На машине экспорта: `python3`, `pip install ultralytics onnx onnxruntime` (и torch)
5. На плате: Python 3.9+, OpenCV, FastAPI, uvicorn; драйвер NPU / `libshl` уже в RevyOS или из HHB runtime

### Артефакты, которые в итоге лежат на плате

```text
check_run8/
  libyolov8n.so      # собранная библиотека
  model.params       # веса INT8 из HHB (тот же прогон!)
  check_v8.py        # из python/
  classes.txt        # имена классов
  requirements.txt   # опционально
```

---

## 1. Клонировать репозиторий

```bash
git clone <URL-вашего-репо>.git
cd yolov5n-th1520-npu   # или как назовёте корень пакета
```

Положите веса рядом, например:

```bash
cp /path/to/ppe.pt .
```

---

## 2. Калибровочные картинки для HHB

HHB считает шкалы INT8 по датасету. Лучше **ваши** PPE/сцены, не random noise.

```bash
mkdir -p calib
# 20–50 jpg из домена модели:
cp /path/to/ppe_images/*.jpg calib/

# или синтетика (хуже по точности):
python3 scripts/make_calib_images.py --out calib --n 24
```

---

## 3. Экспорт ONNX под NPU (критично)

Обычный `yolo export` для v8 на VIP9000 **плохой выбор**:
в графе остаются DFL Softmax / Slice → либо краш, либо **~4 FPS** (`CusStridedSlice` на CPU).

Используйте скрипт из репо (raw heads + патч C2f без `chunk`):

```bash
pip install ultralytics onnx
python3 scripts/export_yolov8_raw_heads.py \
  --weights ppe.pt \
  --out yolov8n_raw.onnx \
  --imgsz 640 \
  --opset 12
```

Ожидаемый вывод:

```text
Patched C2f modules (no chunk/Slice): 8
Detect: nl=3 nc=10 reg_max=16 → channels/head=74
ONNX ops: Slice=0 Split=0 (want both 0)
Wrote classes.txt
```

Если `Slice>0` или `Split>0`:

```bash
python3 scripts/onnx_split_to_slice.py --in yolov8n_raw.onnx --out yolov8n_nosplit.onnx
python3 scripts/onnx_slice_to_conv.py  --in yolov8n_nosplit.onnx --out yolov8n_raw.onnx
```

Выходы ONNX: `output0/1/2` формы `(1, 74, 80|40|20)` при `nc=10`.

---

## 4. HHB: квантование и codegen

В окружении, где есть команда `hhb`:

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

Важно:
- не вставляйте комментарии `# ...` в середину многострочной команды;
- всё одной командой `hhb -D ...`.

После успеха появится каталог `hhb_out/` (имя может отличаться — смотрите лог HHB):

```bash
# проверка: Slice/Split не должны попасть в C-граф
grep -c strided_slice hhb_out/model.c   # → 0
grep -c csinn_split hhb_out/model.c     # → 0

cp -f hhb_out/model.c hhb_out/io.c hhb_out/io.h vendor/hhb_v8/
```

На плату позже копируете **`hhb_out/model.params`** (не старый!).

Подробности и troubleshooting: [YOLOV8_HHB.md](YOLOV8_HHB.md).

---

## 5. Сборка `libyolov8n.so`

В Docker HHB обычно уже есть кросс-компилятор и пути к SHL.

```bash
chmod +x scripts/build_so_v8.sh

# при необходимости поправьте пути:
export HHB_NN2=/usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520
export HHB_PB=/usr/local/lib/python3.8/dist-packages/hhb/prebuilt
export CC=riscv64-unknown-linux-gnu-gcc   # или gcc — если собираете на плате

./scripts/build_so_v8.sh
# → ./libyolov8n.so
```

Проверка, что в `.so` нет старого Slice:

```bash
strings libyolov8n.so | grep strided_slice || echo "OK: no strided_slice"
```

---

## 6. Деплой на LicheePi 4A

С хоста:

```bash
BOARD=sipeed@192.168.x.x   # IP платы
scp libyolov8n.so \
    hhb_out/model.params \
    python/check_v8.py \
    python/classes.txt \
    python/requirements.txt \
    ${BOARD}:~/ruskogo/check_run8/
```

На плате:

```bash
cd ~/ruskogo/check_run8
# model.params должен называться именно так (или поправьте путь в скрипте)
ls -l libyolov8n.so model.params check_v8.py classes.txt

python3 -m venv venv_new && source venv_new/bin/activate
pip install -r requirements.txt

# убедиться, что md5 совпадает с HHB-машиной:
md5sum libyolov8n.so model.params
```

Запуск:

```bash
python3 check_v8.py --source auto
# или фото без камеры (замер FPS):
# python3 check_v8.py --source /path/to/test.jpg
```

Откройте в браузере: `http://<IP-платы>:8000/`

### Признаки «всё правильно»

- **Нет** спама `Strided_slice software implementation...`
- `YOLOv8 outputs: ... layout=RAW_DFL nc=10 reg_max=16`
- `npu_ms` порядка **15–40** (не 200+)
- Подписи из `classes.txt` (`Hardhat`, `Person`, …), не COCO

### Если снова ~4 FPS и Slice в логе

На плате **старый** `.so` или **старый** `model.params`. Скопируйте **оба** заново из одного HHB-прогона и проверьте `md5sum` / `strings`.

---

## 7. Имена классов

Порядок = id из датасета (`ppe.pt`):

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

Экспорт сам пишет `classes.txt`. Можно так:

```bash
python3 check_v8.py --names classes.txt
```

---

## 8. Webcam на LPi4A

- Init NPU **до** открытия камеры (так сделано в `check_v8.py`)
- Часто работают `/dev/video0` (MJPG) или `auto`
- При `VIDIOC_REQBUFS errno=19` — USB отвалился; soft-reopen уже в демо; можно уменьшить разрешение (в скрипте 320×240)

---

## YOLOv5n (кратко)

1. ONNX YOLOv5n, opset 12, три головы (как в гайдах Sipeed), **без** тяжёлого постпроцесса в графе  
2. HHB `--board th1520` → `vendor/hhb/{model.c,io.c,io.h}` + `model.params`  
3. `./scripts/build_so.sh` → `libyolov5n.so`  
4. На плате: `python/check.py`

Граф v5 в `vendor/hhb/` уже может быть в репо (без `model.params` — его кладёте рядом с `.so`).

---

## Чеклист «релиз на плату»

- [ ] ONNX: `Slice=0`, `Split=0`
- [ ] HHB: `grep strided_slice hhb_out/model.c` → 0
- [ ] `vendor/hhb_v8/` обновлён, `.so` пересобран
- [ ] На плате **пара** `.so` + `model.params` с одинаковым md5, что на билде
- [ ] `classes.txt` на месте
- [ ] В логе нет `CusStridedSlice`, `npu_ms` адекватный

---

## Полезные команды

```bash
# классы из .pt
python3 - <<'PY'
from ultralytics import YOLO
print(YOLO("ppe.pt").names)
PY

# устройства камеры
ls -l /dev/video*
v4l2-ctl --list-devices   # если установлен
```

Дальше по графу v8: [YOLOV8_HHB.md](YOLOV8_HHB.md).  
Обзор репо: [../README.md](../README.md).
