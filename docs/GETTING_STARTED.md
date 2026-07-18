**Language / Язык:** [English](#english) · [Русский](#русский)

---

<a id="english"></a>

# Getting started on the board

Project overview: [README](../README.md#english).  
Quantization: [QUANTIZATION.md](QUANTIZATION.md#english).  
APIs: [LIB_YOLOV5N](LIB_YOLOV5N.md#english) · [LIB_YOLOV8](LIB_YOLOV8.md#english).

## Runtime package on the device

| File | Role |
|------|------|
| `libyolov5n.so` or `libyolov8n.so` | Native runtime |
| `model.params` | INT8 weights (**same HHB run** as the linked `model.c`) |
| `check.py` / `check_v8.py` | Smoke test / webcam demo |
| `classes.txt` | Class names for custom models (v8) |
| `requirements.txt` | Python dependencies |

```text
~/check_run/      libyolov5n.so  model.params  check.py
~/check_run8/     libyolov8n.so  model.params  check_v8.py  classes.txt
```

Always deploy **`.so` and `model.params` as a matched pair**.

## Install and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 check.py --source auto
python3 check_v8.py --source auto --names classes.txt
python3 check_v8.py --source ./test.jpg    # measure FPS without USB camera
# http://<board-ip>:8000/
```

### Acceptance criteria (YOLOv8)

| Pass | Fail |
|------|------|
| No `Strided_slice ... memory leak` flood | CusStridedSlice INFO spam |
| `layout=RAW_DFL nc=…`, `npu_ms` ~15–40 | `npu_ms` ~200–270, ~4 FPS |

If it fails: copy both artifacts again from one HHB build (`md5sum`, `strings libyolov8n.so | grep strided_slice`).

## Webcam notes (LicheePi 4A)

- Initialize **NPU before** opening V4L2 (demos already do this)
- Prefer MJPG; `check_v8.py` defaults to 320×240 capture
- `errno=19` → try another `/dev/video*`, or still-image mode
- Trust overlay **`npu_ms` / `compute_ms`**, not browser FPS alone

```bash
./scripts/pack_board_v8.sh
scp -r dist/check_run8 sipeed@<board-ip>:~/ruskogo/
```

## Next steps

1. Call `init_model` / `run_inference` from your application (see library docs).  
2. Replace weights via [QUANTIZATION.md](QUANTIZATION.md#english).  
3. Version `.so` + `model.params` together in releases.

---

<a id="русский"></a>

# Запуск на плате

**[↑ English](#english)** · **Русский**

Обзор проекта: [README](../README.md#русский).  
Квантование: [QUANTIZATION.md](QUANTIZATION.md#русский).  
API: [LIB_YOLOV5N](LIB_YOLOV5N.md#русский) · [LIB_YOLOV8](LIB_YOLOV8.md#русский).

## Состав runtime на устройстве

| Файл | Назначение |
|------|------------|
| `libyolov5n.so` или `libyolov8n.so` | Нативная библиотека |
| `model.params` | Веса INT8 (**тот же** прогон HHB, что и связанный `model.c`) |
| `check.py` / `check_v8.py` | Проверка / демо с камерой |
| `classes.txt` | Имена классов для своей модели (v8) |
| `requirements.txt` | Зависимости Python |

```text
~/check_run/      libyolov5n.so  model.params  check.py
~/check_run8/     libyolov8n.so  model.params  check_v8.py  classes.txt
```

Всегда выкладывайте **согласованную пару** `.so` и `model.params`.

## Установка и запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 check.py --source auto
python3 check_v8.py --source auto --names classes.txt
python3 check_v8.py --source ./test.jpg    # замер FPS без USB-камеры
# http://<IP-платы>:8000/
```

### Критерии приёмки (YOLOv8)

| Успех | Провал |
|-------|--------|
| Нет потока `Strided_slice ... memory leak` | Спам CusStridedSlice |
| `layout=RAW_DFL nc=…`, `npu_ms` ~15–40 | `npu_ms` ~200–270, ~4 FPS |

При провале: снова скопируйте оба файла из одного прогона HHB  
(`md5sum`, `strings libyolov8n.so | grep strided_slice`).

## Камера (LicheePi 4A)

- Сначала инициализация **NPU**, затем V4L2 (в демо уже так)
- Предпочтителен MJPG; в `check_v8.py` захват по умолчанию 320×240
- `errno=19` → другой `/dev/video*` или режим статичного кадра
- Ориентируйтесь на **`npu_ms` / `compute_ms`** в оверлее, а не только на FPS в браузере

```bash
./scripts/pack_board_v8.sh
scp -r dist/check_run8 sipeed@<IP-платы>:~/ruskogo/
```

## Дальнейшие шаги

1. Вызов `init_model` / `run_inference` из вашего приложения (см. спецификации библиотек).  
2. Смена весов — по [QUANTIZATION.md](QUANTIZATION.md#русский).  
3. В релизах версионируйте `.so` и `model.params` **вместе**.
