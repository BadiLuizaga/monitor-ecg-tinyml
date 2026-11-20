// diag_tflm.cpp - Diagnóstico de T1: vuelca la salida de cada nodo del modelo v1
// ejecutado con TFLM (preserve_all_tensors = true) para compararla capa a capa
// con el intérprete TFLite de Python (diag_layers.py).
//
// Entrada: vectors/beats.bin y vectors/expected.bin (make_vectors.py); se usa la
// entrada int8 de Python tal cual, para aislar los kernels de la cuantización.
// Salida: vectors/tflm_layers.bin
//   int32 n_beats, int32 n_nodes, por nodo: int32 tensor, int32 bytes
//   después, por latido, los bytes de la salida de cada nodo en orden.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "generated/model_config.h"
#include "generated/model_data.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/schema/schema_generated.h"

namespace {
alignas(16) uint8_t g_arena[512 * 1024];

std::vector<char> readFile(const std::string& path) {
  std::vector<char> d;
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return d;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  d.resize(static_cast<size_t>(n));
  if (std::fread(d.data(), 1, d.size(), f) != d.size()) d.clear();
  std::fclose(f);
  return d;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "device-firmware/test/tflm/vectors";
  const tflite::Model* model = tflite::GetModel(g_ecg_model_data);
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddExpandDims();
  resolver.AddConcatenation();
  tflite::MicroInterpreter interp(model, resolver, g_arena, sizeof g_arena, nullptr, nullptr,
                                  /*preserve_all_tensors=*/true);
  if (interp.AllocateTensors() != kTfLiteOk) {
    std::printf("AllocateTensors falló\n");
    return 1;
  }
  const auto* sg = model->subgraphs()->Get(0);
  const int nNodes = static_cast<int>(sg->operators()->size());
  std::vector<int32_t> outIdx(nNodes), outBytes(nNodes);
  for (int i = 0; i < nNodes; ++i) outIdx[i] = sg->operators()->Get(i)->outputs()->Get(0);

  const std::vector<char> beats = readFile(dir + "/beats.bin");
  const std::vector<char> exp = readFile(dir + "/expected.bin");
  int32_t n = 0, L = 0;
  if (beats.size() < 8) return 1;
  std::memcpy(&n, beats.data(), 4);
  std::memcpy(&L, beats.data() + 4, 4);
  const size_t rec = static_cast<size_t>(L) + 2 * ECG_MODEL_NUM_CLASSES;
  if (exp.size() != static_cast<size_t>(n) * rec) return 1;

  FILE* f = std::fopen((dir + "/tflm_layers.bin").c_str(), "wb");
  if (!f) return 1;
  TfLiteTensor* in = interp.input(0);
  bool header = false;
  for (int b = 0; b < n; ++b) {
    std::memcpy(in->data.int8, exp.data() + b * rec, static_cast<size_t>(L));
    if (interp.Invoke() != kTfLiteOk) {
      std::printf("Invoke falló en %d\n", b);
      return 1;
    }
    if (!header) {
      for (int i = 0; i < nNodes; ++i) {
        size_t bytes = 0;
        tflite::TfLiteEvalTensorByteLength(interp.GetTensor(outIdx[i]), &bytes);
        outBytes[i] = static_cast<int32_t>(bytes);
      }
      std::fwrite(&n, 4, 1, f);
      std::fwrite(&nNodes, 4, 1, f);
      for (int i = 0; i < nNodes; ++i) {
        std::fwrite(&outIdx[i], 4, 1, f);
        std::fwrite(&outBytes[i], 4, 1, f);
      }
      header = true;
    }
    for (int i = 0; i < nNodes; ++i)
      std::fwrite(interp.GetTensor(outIdx[i])->data.raw, 1, static_cast<size_t>(outBytes[i]), f);
  }
  std::fclose(f);
  std::printf("volcado: %d latidos x %d nodos -> %s/tflm_layers.bin\n", n, nNodes, dir.c_str());
  return 0;
}
