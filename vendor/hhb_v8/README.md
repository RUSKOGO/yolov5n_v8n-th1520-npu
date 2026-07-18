# HHB-generated YOLOv8 graph (copy from hhb_out after quantize)

Required files in this directory:

- `model.c`
- `io.c`
- `io.h`

Board also needs `model.params` next to `libyolov8n.so` (same HHB run).

```bash
# after HHB:
grep -c strided_slice ../../hhb_out/model.c   # must be 0
cp -f ../../hhb_out/model.c ../../hhb_out/io.c ../../hhb_out/io.h .
cd ../.. && ./scripts/build_so_v8.sh
```

See [docs/GETTING_STARTED.md](../../docs/GETTING_STARTED.md).
