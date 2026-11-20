#!/usr/bin/env sh
# Compila TensorFlow Lite Micro (núcleo + kernels de referencia) como librería
# estática para PC, replicando la lista de fuentes de su CMake
# (cmake/tflm_sources.cmake: TFLM_MICRO_CORE_SRCS + TFLM_MICRO_KERNEL_SRCS +
# TFLM_SYSTEM_DEFAULT_SRCS), sin kernels optimizados, sin "signal" y sin pruebas.
#
# Todo queda dentro de .tools/tflm (fuera de git):
#   fuentes   .tools/tflm/src/tflite-micro-<commit>/
#   flatbuffers (solo cabeceras) .tools/tflm/deps/flatbuffers-25.9.23/include
#   salida    .tools/tflm/build/libtflm.a
# Uso (desde la raíz del repositorio): sh device-firmware/test/tflm/build_tflm_lib.sh
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TFLM_COMMIT="b8d75cbf6b8009762bc1e3ee4817bc33e5dcee47"
SRC="$ROOT/.tools/tflm/src/tflite-micro-$TFLM_COMMIT"
FB="$ROOT/.tools/tflm/deps/flatbuffers-25.9.23/include"
# TFLM_BUILD_DIR / TFLM_EXTRA_FLAGS: solo para variantes de diagnóstico (p. ej. -DTFLITE_SINGLE_ROUNDING=1).
OUT="${TFLM_BUILD_DIR:-$ROOT/.tools/tflm/build}"
CXX="${CXX:-g++}"
AR="${AR:-ar}"
[ -d "$SRC" ] || { echo "No existe $SRC"; exit 1; }
[ -f "$FB/flatbuffers/flatbuffers.h" ] || { echo "Falta flatbuffers en $FB"; exit 1; }
mkdir -p "$OUT/obj"

M="$SRC/tensorflow/lite/micro"
LIST="$OUT/sources.txt"
{
  ls "$M"/*.cc "$M"/arena_allocator/*.cc "$M"/memory_planner/*.cc "$M"/kernels/internal/*.cc \
     "$M"/kernels/kernel_util.cc "$M"/kernels/*.cc 2>/dev/null
} | grep -v -E '_test\.cc$|_flexbuffers_generated_data\.cc$' \
  | grep -v -E '/(fake_micro_context|mock_micro_graph|test_helper_custom_ops|test_helpers)\.cc$' \
  | grep -v -E '/kernels/(conv_test_common|kernel_runner)\.cc$' \
  | sort -u > "$LIST"
echo "fuentes TFLM: $(wc -l < "$LIST")"

FLAGS="-std=c++17 -O2 -fno-exceptions -fno-rtti -fno-threadsafe-statics -ffunction-sections -fdata-sections \
 -Wall -Wsign-compare -Wdouble-promotion -Wvla -Wno-unused-parameter -Wnon-virtual-dtor \
 -DFLATBUFFERS_LOCALE_INDEPENDENT=0 -DFLATBUFFERS_PREFER_PRINTF=1 -DKERNELS_OPTIMIZED_FOR_SPEED -DTF_LITE_USE_CTIME \
 -I$SRC -I$FB ${TFLM_EXTRA_FLAGS:-}"
LOG="$OUT/build.log"
: > "$LOG"
OBJS=""
while read -r f; do
  rel="${f#$SRC/}"
  o="$OUT/obj/$(echo "$rel" | tr '/' '_' | sed 's/\.cc$/.o/')"
  if [ ! -f "$o" ] || [ "$f" -nt "$o" ]; then
    # shellcheck disable=SC2086
    "$CXX" $FLAGS -c "$f" -o "$o" >> "$LOG" 2>&1 || { echo "ERROR compilando $rel (ver $LOG)"; tail -30 "$LOG"; exit 1; }
  fi
  OBJS="$OBJS $o"
done < "$LIST"
rm -f "$OUT/libtflm.a"
# shellcheck disable=SC2086
"$AR" rcs "$OUT/libtflm.a" $OBJS
echo "librería: $OUT/libtflm.a ($(wc -c < "$OUT/libtflm.a") bytes); avisos del compilador: $(grep -c 'warning:' "$LOG" || true)"
