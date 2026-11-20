// test_tflm.cpp - Prueba T1: TensorFlow Lite Micro en PC con el modelo v1.
//
// Carga el modelo compilado en el firmware (src/generated/model_data.cc), con
// los mismos 8 operadores que registra src/inference_tflm.cpp, y compara con
// los vectores de Python (make_vectors.py):
//   1. entradas/salidas: número, tipo, forma y cuantización frente a
//      generated/model_config.h y meta del intérprete de Python;
//   2. operadores del modelo (todos registrados);
//   3. cuantización de la entrada con ecg::quantizeInt8 (código del firmware)
//      frente a la de Python, byte a byte;
//   4. salida int8 de TFLM frente al intérprete TFLite de Python con kernels de
//      referencia (BUILTIN_REF) y con los kernels por defecto.
// Criterio fijado antes de ejecutar: 0 fallos de inferencia, entrada int8
// idéntica, |salida TFLM - salida REF| <= 1 LSB en todas las clases y misma
// clase predicha en el 100 % de los latidos (salvo empates exactos).
//
// Después ejecuta los mismos latidos con el motor real del firmware
// (ecg::TflmInferenceEngine, src/inference_tflm.cpp, arena de
// ECG_TFLM_ARENA_BYTES) y lo compara con el intérprete de esta prueba.
// Uso: sh device-firmware/test/tflm/run_tflm_test.sh

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "generated/model_config.h"
#include "generated/model_data.h"
#include "inference.h"
#include "integration_config.h"
#include "signal_processing.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace {

int g_fail = 0;
#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      std::printf("  FALLO: " __VA_ARGS__);      \
      std::printf("\n");                         \
      ++g_fail;                                  \
    }                                            \
  } while (0)

std::vector<char> readFile(const std::string& path) {
  std::vector<char> d;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return d;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  d.resize(static_cast<size_t>(n));
  if (std::fread(d.data(), 1, d.size(), f) != d.size()) d.clear();
  std::fclose(f);
  return d;
}

int argmaxInt8(const int8_t* v, int n) {
  int b = 0;
  for (int k = 1; k < n; ++k)
    if (v[k] > v[b]) b = k;
  return b;
}

bool hasTie(const int8_t* v, int n) {
  int b = argmaxInt8(v, n), c = 0;
  for (int k = 0; k < n; ++k) c += v[k] == v[b];
  return c > 1;
}

// Arena: el tamaño previsto para el ESP32 (integration_config.h) y otra mayor
// para poder medir el uso real si la prevista no bastara.
alignas(16) uint8_t g_arena[ECG_TFLM_ARENA_BYTES];
alignas(16) uint8_t g_big_arena[256 * 1024];

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "device-firmware/test/tflm/vectors";
  std::printf("== T1: TFLM en PC, modelo v1 (%u bytes)\n", static_cast<unsigned>(g_ecg_model_data_len));

  const tflite::Model* model = tflite::GetModel(g_ecg_model_data);
  CHECK(model->version() == TFLITE_SCHEMA_VERSION, "versión de esquema %u != %d",
        static_cast<unsigned>(model->version()), TFLITE_SCHEMA_VERSION);

  // --- Operadores del modelo ---
  std::set<std::string> ops;
  const auto* sg = model->subgraphs()->Get(0);
  CHECK(model->subgraphs()->size() == 1, "subgrafos = %u", static_cast<unsigned>(model->subgraphs()->size()));
  for (unsigned i = 0; i < sg->operators()->size(); ++i) {
    const auto* oc = model->operator_codes()->Get(sg->operators()->Get(i)->opcode_index());
    ops.insert(tflite::EnumNameBuiltinOperator(tflite::micro::GetBuiltinCode(oc)));
  }
  std::printf("operadores del modelo (%u nodos):", static_cast<unsigned>(sg->operators()->size()));
  for (const auto& o : ops) std::printf(" %s", o.c_str());
  std::printf("\n");

  // Mismo resolver que src/inference_tflm.cpp.
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddExpandDims();
  resolver.AddConcatenation();

  // --- Arena prevista para el ESP32 ---
  {
    tflite::MicroInterpreter small(model, resolver, g_arena, sizeof g_arena);
    const bool ok = small.AllocateTensors() == kTfLiteOk;
    std::printf("arena prevista ECG_TFLM_ARENA_BYTES=%d: %s\n", ECG_TFLM_ARENA_BYTES,
                ok ? "suficiente" : "INSUFICIENTE");
    CHECK(ok, "AllocateTensors falla con la arena prevista de %d bytes", ECG_TFLM_ARENA_BYTES);
  }

  tflite::MicroInterpreter interp(model, resolver, g_big_arena, sizeof g_big_arena);
  if (interp.AllocateTensors() != kTfLiteOk) {
    std::printf("FALLO: AllocateTensors (operador no registrado o arena insuficiente)\n");
    return 1;
  }
  std::printf("arena usada: %u bytes\n", static_cast<unsigned>(interp.arena_used_bytes()));

  // --- Entradas y salidas ---
  CHECK(interp.inputs_size() == 1, "entradas = %u (v1 tiene 1)", static_cast<unsigned>(interp.inputs_size()));
  CHECK(interp.outputs_size() == 1, "salidas = %u", static_cast<unsigned>(interp.outputs_size()));
  TfLiteTensor* in = interp.input(0);
  TfLiteTensor* out = interp.output(0);
  CHECK(in->type == kTfLiteInt8 && out->type == kTfLiteInt8, "tipos de entrada/salida no int8");
  CHECK(in->bytes == ECG_MODEL_INPUT_LENGTH, "bytes de entrada %u != %d", static_cast<unsigned>(in->bytes),
        ECG_MODEL_INPUT_LENGTH);
  CHECK(out->bytes == ECG_MODEL_NUM_CLASSES, "bytes de salida %u", static_cast<unsigned>(out->bytes));
  CHECK(in->dims->size == 3 && in->dims->data[0] == 1 && in->dims->data[1] == ECG_MODEL_INPUT_LENGTH &&
            in->dims->data[2] == 1,
        "forma de entrada inesperada");
  CHECK(out->dims->size == 2 && out->dims->data[1] == ECG_MODEL_NUM_CLASSES, "forma de salida inesperada");
  std::printf("entrada: int8 [%d,%d,%d] scale=%.9g zp=%d | salida: int8 [%d,%d] scale=%.9g zp=%d\n",
              in->dims->data[0], in->dims->data[1], in->dims->data[2], static_cast<double>(in->params.scale),
              static_cast<int>(in->params.zero_point), out->dims->data[0], out->dims->data[1],
              static_cast<double>(out->params.scale), static_cast<int>(out->params.zero_point));
  CHECK(std::fabs(in->params.scale - ECG_MODEL_INPUT_SCALE) <= 1e-6f * ECG_MODEL_INPUT_SCALE,
        "escala de entrada != model_config.h");
  CHECK(in->params.zero_point == ECG_MODEL_INPUT_ZERO_POINT, "punto cero de entrada != model_config.h");
  CHECK(std::fabs(out->params.scale - ECG_MODEL_OUTPUT_SCALE) <= 1e-6f * ECG_MODEL_OUTPUT_SCALE,
        "escala de salida != model_config.h");
  CHECK(out->params.zero_point == ECG_MODEL_OUTPUT_ZERO_POINT, "punto cero de salida != model_config.h");

  // --- Vectores ---
  const std::vector<char> beats = readFile(dir + "/beats.bin");
  const std::vector<char> exp = readFile(dir + "/expected.bin");
  if (beats.size() < 8) {
    std::printf("FALLO: no se pueden leer los vectores en %s\n", dir.c_str());
    return 1;
  }
  int32_t n = 0, L = 0;
  std::memcpy(&n, beats.data(), 4);
  std::memcpy(&L, beats.data() + 4, 4);
  const int C = ECG_MODEL_NUM_CLASSES;
  const size_t rec = static_cast<size_t>(L) + 2 * C;
  if (L != ECG_MODEL_INPUT_LENGTH || beats.size() != 8 + static_cast<size_t>(n) * L * 4 ||
      exp.size() != static_cast<size_t>(n) * rec) {
    std::printf("FALLO: tamaño de vectores inesperado\n");
    return 1;
  }
  const float* X = reinterpret_cast<const float*>(beats.data() + 8);

  int invokeFail = 0, inMismatch = 0, refExact = 0, optExact = 0, refArg = 0, optArg = 0, ties = 0;
  int maxDiffRef = 0, maxDiffOpt = 0, confMismatch = 0, beatsWithin1 = 0;
  int elemHist[9] = {0};  // |TFLM - REF| por salida (8 = 8 o más)
  std::vector<int8_t> tflmOut(static_cast<size_t>(n) * C, 0);
  double invokeSeconds = 0.0;
  FILE* csv = std::fopen((dir + "/tflm_outputs.csv").c_str(), "w");
  if (csv) std::fprintf(csv, "i,tflm0,tflm1,tflm2,tflm3,ref0,ref1,ref2,ref3,class_tflm,class_ref\n");
  for (int i = 0; i < n; ++i) {
    const float* x = X + static_cast<size_t>(i) * L;
    const int8_t* eIn = reinterpret_cast<const int8_t*>(exp.data() + i * rec);
    const int8_t* eRef = eIn + L;
    const int8_t* eOpt = eRef + C;
    // Cuantización con el código del firmware (signal_processing.cpp).
    bool same = true;
    for (int k = 0; k < L; ++k) {
      in->data.int8[k] = ecg::quantizeInt8(x[k], ECG_MODEL_INPUT_SCALE, ECG_MODEL_INPUT_ZERO_POINT);
      same = same && in->data.int8[k] == eIn[k];
    }
    inMismatch += !same;
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = interp.Invoke() == kTfLiteOk;
    invokeSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
      ++invokeFail;
      continue;
    }
    const int8_t* o = out->data.int8;
    bool eqR = true, eqO = true, within1 = true;
    std::memcpy(&tflmOut[static_cast<size_t>(i) * C], o, static_cast<size_t>(C));
    for (int k = 0; k < C; ++k) {
      const int d = std::abs(o[k] - eRef[k]);
      ++elemHist[std::min(d, 8)];
      within1 = within1 && d <= 1;
      maxDiffRef = std::max(maxDiffRef, std::abs(o[k] - eRef[k]));
      maxDiffOpt = std::max(maxDiffOpt, std::abs(o[k] - eOpt[k]));
      eqR = eqR && o[k] == eRef[k];
      eqO = eqO && o[k] == eOpt[k];
    }
    refExact += eqR;
    beatsWithin1 += within1;
    optExact += eqO;
    const int cT = argmaxInt8(o, C), cR = argmaxInt8(eRef, C), cO = argmaxInt8(eOpt, C);
    const bool tie = hasTie(o, C) || hasTie(eRef, C);
    ties += tie;
    refArg += cT == cR || tie;
    optArg += cT == cO || hasTie(o, C) || hasTie(eOpt, C);
    // Confianza como la calcula inference_tflm.cpp frente a la de Python.
    const float confT = (o[cT] - ECG_MODEL_OUTPUT_ZERO_POINT) * ECG_MODEL_OUTPUT_SCALE;
    const float confR = (eRef[cR] - ECG_MODEL_OUTPUT_ZERO_POINT) * ECG_MODEL_OUTPUT_SCALE;
    confMismatch += std::fabs(confT - confR) > ECG_MODEL_OUTPUT_SCALE + 1e-7f;
    if (csv)
      std::fprintf(csv, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%c,%c\n", i, o[0], o[1], o[2], o[3], eRef[0], eRef[1],
                   eRef[2], eRef[3], ECG_MODEL_CLASSES[cT], ECG_MODEL_CLASSES[cR]);
  }
  if (csv) std::fclose(csv);

  std::printf("inferencias: %d (fallidas %d), %.1f us/inferencia en PC (orientativo)\n", n, invokeFail,
              1e6 * invokeSeconds / (n ? n : 1));
  std::printf("entrada int8 C++ == Python: %d/%d\n", n - inMismatch, n);
  std::printf("salida idéntica a TFLite REF: %d/%d (máx |diff| %d LSB); misma clase: %d/%d (empates %d)\n",
              refExact, n, maxDiffRef, refArg, n, ties);
  std::printf("salida idéntica a TFLite por defecto: %d/%d (máx |diff| %d LSB); misma clase: %d/%d\n", optExact,
              n, maxDiffOpt, optArg, n);
  std::printf("confianza (como inference_tflm.cpp) distinta en más de 1 LSB: %d\n", confMismatch);

  std::printf("distribución |TFLM - REF| por salida (%d salidas):", n * C);
  for (int d = 0; d < 9; ++d)
    if (elemHist[d]) std::printf(" %d LSB: %d;", d, elemHist[d]);
  std::printf("\nsalidas dentro de ±1 LSB: %d/%d; latidos con las 4 salidas dentro de ±1 LSB: %d/%d\n",
              elemHist[0] + elemHist[1], n * C, beatsWithin1, n);

  // --- Motor real del firmware (src/inference_tflm.cpp) ---
  ecg::TflmInferenceEngine engine;
  const bool engineReady = engine.begin();
  std::printf("motor real TflmInferenceEngine: begin()=%s, arena usada %u de %d bytes, inputLength=%u, "
              "rrFeatures=%u\n",
              engineReady ? "OK" : "FALLO", static_cast<unsigned>(engine.arenaUsedBytes()), ECG_TFLM_ARENA_BYTES,
              static_cast<unsigned>(engine.inputLength()), static_cast<unsigned>(engine.rrFeatures()));
  CHECK(engineReady, "TflmInferenceEngine::begin() falla");
  int engFail = 0, engClassVsTest = 0, engConfVsTest = 0, engClassVsRef = 0;
  if (engineReady) {
    for (int i = 0; i < n; ++i) {
      ecg::InferenceResult r;
      if (!engine.classify(X + static_cast<size_t>(i) * L, static_cast<size_t>(L), nullptr, 0, &r)) {
        ++engFail;
        continue;
      }
      const int8_t* o = &tflmOut[static_cast<size_t>(i) * C];
      const int8_t* eRef = reinterpret_cast<const int8_t*>(exp.data() + i * rec) + L;
      const int cT = argmaxInt8(o, C);
      const float confT = (o[cT] - ECG_MODEL_OUTPUT_ZERO_POINT) * ECG_MODEL_OUTPUT_SCALE;
      engClassVsTest += r.classIndex == cT && r.beatClass == ECG_MODEL_CLASSES[cT];
      engConfVsTest += r.confidence == confT;
      engClassVsRef += r.classIndex == argmaxInt8(eRef, C) || hasTie(eRef, C);
    }
  }
  std::printf("motor real: %d inferencias (fallidas %d); clase = intérprete de prueba %d/%d; confianza idéntica "
              "%d/%d; clase = TFLite REF %d/%d\n",
              engineReady ? n : 0, engFail, engClassVsTest, n, engConfVsTest, n, engClassVsRef, n);
  CHECK(engFail == 0, "%d inferencias fallidas en el motor real", engFail);
  CHECK(engClassVsTest == n && engConfVsTest == n, "el motor real difiere del intérprete de prueba");
  CHECK(engClassVsRef == n, "clase del motor real distinta de REF en %d latidos", n - engClassVsRef);

  CHECK(invokeFail == 0, "%d inferencias fallidas", invokeFail);
  CHECK(inMismatch == 0, "%d entradas cuantizadas distintas de Python", inMismatch);
  CHECK(maxDiffRef <= 1, "diferencia máxima con REF %d LSB > 1", maxDiffRef);
  CHECK(refArg == n, "clase distinta de REF en %d latidos", n - refArg);
  CHECK(confMismatch == 0, "%d confianzas distintas", confMismatch);

  std::printf(g_fail ? "== T1: NO APROBADA (%d fallos)\n" : "== T1: APROBADA\n", g_fail);
  return g_fail ? 1 : 0;
}
