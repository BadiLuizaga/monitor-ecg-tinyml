#!/usr/bin/env sh
# Prueba T1: compila TFLM para PC, genera los vectores de referencia en Python,
# compila test_tflm.cpp con el modelo v1 del firmware y lo ejecuta.
# Uso (desde la raíz del repositorio): sh device-firmware/test/tflm/run_tflm_test.sh
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FW="$ROOT/device-firmware"
T="$FW/test/tflm"
CXX="${CXX:-g++}"
PY="${PYTHON:-python}"
TFLM_COMMIT="b8d75cbf6b8009762bc1e3ee4817bc33e5dcee47"
SRC="$ROOT/.tools/tflm/src/tflite-micro-$TFLM_COMMIT"
FB="$ROOT/.tools/tflm/deps/flatbuffers-25.9.23/include"

sh "$T/build_tflm_lib.sh"
"$PY" "$T/make_vectors.py"
mkdir -p "$T/bin"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -fno-exceptions -fno-rtti \
  -DFLATBUFFERS_LOCALE_INDEPENDENT=0 -DFLATBUFFERS_PREFER_PRINTF=1 \
  -I"$FW/include" -isystem "$SRC" -isystem "$FB" \
  -DECG_HAVE_TFLM=1 \
  "$T/test_tflm.cpp" "$FW/src/inference_tflm.cpp" "$FW/src/signal_processing.cpp" "$FW/src/generated/model_data.cc" \
  "$ROOT/.tools/tflm/build/libtflm.a" -o "$T/bin/test_tflm"
"$T/bin/test_tflm" "$T/vectors"
