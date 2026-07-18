#!/usr/bin/env bash
# Build libyolov5n.so for TH1520 / LicheePi 4A
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CC="${CC:-riscv64-unknown-linux-gnu-gcc}"
# On-board native build:  CC=gcc ./scripts/build_so.sh

HHB_NN2="${HHB_NN2:-/usr/local/lib/python3.8/dist-packages/hhb/install_nn2/th1520}"
HHB_PB="${HHB_PB:-/usr/local/lib/python3.8/dist-packages/hhb/prebuilt}"

if [[ ! -d "$HHB_NN2/include" ]]; then
  echo "ERROR: HHB headers not found at: $HHB_NN2/include"
  echo "Set HHB_NN2=... to your install_nn2/th1520 path"
  exit 1
fi

OUT="${OUT:-$ROOT/libyolov5n.so}"
BUILD_TYPE="${BUILD_TYPE:-prod}"   # prod | debug

COMMON=(
  -shared -fPIC
  src/yolov5n_lib.c
  vendor/hhb/io.c
  vendor/hhb/model.c
  -o "$OUT"
  -I src -I vendor/hhb
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
  echo "==> DEBUG build"
else
  FLAGS=(
    -O3 -DNDEBUG
    -fvisibility=hidden
    -ffunction-sections -fdata-sections
    -fno-plt -fno-semantic-interposition
    -ftree-vectorize -fno-math-errno -fno-trapping-math
    -flto=auto
    -Wl,--gc-sections -Wl,-O1 -Wl,--as-needed
    -Wl,--strip-debug
  )
  echo "==> PROD build"
fi

echo "CC=$CC"
echo "HHB_NN2=$HHB_NN2"
echo "OUT=$OUT"

"$CC" "${COMMON[@]}" "${FLAGS[@]}"

echo "OK: $OUT"
file "$OUT" 2>/dev/null || true
ls -lh "$OUT"
