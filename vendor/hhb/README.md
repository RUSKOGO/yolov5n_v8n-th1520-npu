# HHB-generated YOLOv5n graph

Files here are produced by HHB (`--board th1520`) for YOLOv5n INT8.

- `model.c` / `io.c` / `io.h` — linked into `libyolov5n.so`
- `model.params` — **not** stored in git; copy next to the `.so` on the board

Rebuild:

```bash
./scripts/build_so.sh
```
