# Документация

| Файл | Содержание |
|------|------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | С нуля: скачать, экспорт, HHB, сборка, плата |
| [YOLOV8_HHB.md](YOLOV8_HHB.md) | Детали графа YOLOv8, Slice/Split, флаги HHB |
| [../README.md](../README.md) | Обзор репозитория |

## Типовой поток (YOLOv8 PPE)

```mermaid
flowchart LR
  PT["ppe.pt"] --> EXP["export_yolov8_raw_heads.py"]
  EXP --> ONNX["yolov8n_raw.onnx\nSlice=0 Split=0"]
  ONNX --> HHB["hhb -D --board th1520"]
  HHB --> MC["model.c + model.params"]
  MC --> SO["build_so_v8.sh\nlibyolov8n.so"]
  SO --> BOARD["LicheePi 4A\ncheck_v8.py"]
```
