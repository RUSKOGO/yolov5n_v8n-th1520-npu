# Ready-to-run demos for LicheePi 4A (prebuilt `.so` + `model.params`).

**Requires the board** (RISC-V + VIP9000). These binaries will not run on x86/Windows.

| Folder | Model | One command |
|--------|-------|-------------|
| [`yolov5n/`](yolov5n/) | YOLOv5n COCO | `./run.sh` |
| [`yolov8n/`](yolov8n/) | YOLOv8 PPE (10 classes) | `./run.sh` |

```bash
cd examples/yolov8n
chmod +x run.sh
./run.sh
# open http://<board-ip>:8000/
```

Each folder contains: `libyolov*.so`, `model.params`, check script, `requirements.txt`
(and `classes.txt` for v8).
