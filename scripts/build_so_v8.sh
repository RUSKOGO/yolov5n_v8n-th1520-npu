#!/usr/bin/env bash
# Build libyolov8n.so — requires vendor/hhb_v8/{model.c,io.c,io.h} from HHB.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CC="${CC:-riscv64-unknown-linux-gnu-gcc}"
HHB_NN2="${HHB_NN2:-/usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520}"
HHB_PB="${HHB_PB:-/usr/local/lib/python3.8/dist-packages/hhb/prebuilt}"
OUT="${OUT:-$ROOT/libyolov8n.so}"
BUILD_TYPE="${BUILD_TYPE:-prod}"
HHB_DIR="${HHB_DIR:-$ROOT/vendor/hhb_v8}"

for f in model.c io.c io.h; do
  if [[ ! -f "$HHB_DIR/$f" ]]; then
    echo "ERROR: missing $HHB_DIR/$f"
    echo "Export YOLOv8n ONNX → run HHB for TH1520 → copy generated files here."
    echo "See docs/YOLOV8_HHB.md"
    exit 1
  fi
done

COMMON=(
  -shared -fPIC
  src/yolov8_lib.c
  "$HHB_DIR/io.c"
  "$HHB_DIR/model.c"
  -o "$OUT"
  -I src -I "$HHB_DIR"
  -I "${HHB_NN2}/include/"
  -L "${HHB_NN2}/lib/"
  -L "${HHB_PB}/decode/install/lib/rv"
  -L "${HHB_PB}/runtime/riscv_linux"
  -lshl -lprebuilt_runtime -ljpeg -lpng -lz -lstdc++ -lm -lpthread
  -mabi=lp64d -march=rv64gcv0p7_zfh_xtheadc
  -Wl,-unresolved-symbols=ignore-in-shared-libs
)

if [[ "$BUILD_TYPE" == "debug" ]]; then
  FLAGS=(-O2 -g -fvisibility=hidden -ffunction-sections -fdata-sections -Wl,--gc-sections)
else
  FLAGS=(
    -O3 -DNDEBUG -fvisibility=hidden
    -ffunction-sections -fdata-sections
    -fno-plt -fno-semantic-interposition
    -ftree-vectorize -fno-math-errno -fno-trapping-math
    -flto=auto
    -Wl,--gc-sections -Wl,-O1 -Wl,--as-needed -Wl,--strip-debug
  )
fi

echo "Building YOLOv8 → $OUT"
"$CC" "${COMMON[@]}" "${FLAGS[@]}"
ls -lh "$OUT"
