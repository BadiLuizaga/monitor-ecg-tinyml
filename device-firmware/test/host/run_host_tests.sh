#!/usr/bin/env sh
# Compila y ejecuta las pruebas del firmware en PC (sin ESP32).
# Requiere: Python del proyecto (para generar vectores) y g++/clang++ C++17.
# Uso (desde la raíz del repositorio):  sh device-firmware/test/host/run_host_tests.sh
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
FW="$ROOT/device-firmware"
CXX="${CXX:-g++}"
PY="${PYTHON:-python}"

mkdir -p "$FW/build"
"$PY" "$FW/test/host/make_vectors.py"
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror \
  -I"$FW/include" -I"$FW/build" \
  "$FW/test/host/test_host.cpp" \
  "$FW/src/ble_protocol.cpp" "$FW/src/ble_transport.cpp" "$FW/src/signal_processing.cpp" \
  "$FW/src/alert_manager.cpp" "$FW/src/ecg_source.cpp" "$FW/src/ecg_pipeline.cpp" \
  -o "$FW/build/test_host"
"$FW/build/test_host"

# Demo de main.cpp (modo PC)
"$CXX" -std=c++17 -O2 -Wall -Wextra -Werror -I"$FW/include" \
  "$FW/src/main.cpp" "$FW/src/ble_protocol.cpp" "$FW/src/ble_transport.cpp" "$FW/src/signal_processing.cpp" \
  "$FW/src/alert_manager.cpp" "$FW/src/ecg_source.cpp" "$FW/src/ecg_pipeline.cpp" \
  -o "$FW/build/demo_host"
"$FW/build/demo_host"
