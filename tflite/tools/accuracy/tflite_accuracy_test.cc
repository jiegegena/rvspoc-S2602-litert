// Copyright 2025 Google LLC.
// Copyright 2026 Hu Jie <2810295945@qq.com>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TFLite ImageNet Top-1 Accuracy Test Tool
//
// Required directory structure:
//   data_dir/
//     n01440764/
//       xxx1.JPEG
//       xxx2.JPEG
//     n01443537/
//       xxx1.JPEG
//     ...
//
// Usage:
//   ./tflite_accuracy_test \
//     --model=mobilenet_v1_1.0_224.tflite \
//     --data_dir=/path/to/ILSVRC2012_img_val \
//     --synset_file=synset_words.txt \
//     --num_images=0

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tflite/interpreter.h"
#include "tflite/kernels/register.h"
#include "tflite/model.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// ============================================================
// Command line flags parsing
// ============================================================

struct Flags {
  std::string model;
  std::string data_dir;
  std::string synset_file;
  int num_images = 0;
  int num_threads = -1;  // Default -1, use TFLite default (consistent with benchmark)
  bool use_xnnpack = true;  // Default enabled (consistent with benchmark)
};

bool ParseFlags(int argc, char** argv, Flags& flags) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 8) == "--model=") {
      flags.model = arg.substr(8);
    } else if (arg.substr(0, 11) == "--data_dir=") {
      flags.data_dir = arg.substr(11);
    } else if (arg.substr(0, 14) == "--synset_file=") {
      flags.synset_file = arg.substr(14);
    } else if (arg.substr(0, 13) == "--num_images=") {
      flags.num_images = std::stoi(arg.substr(13));
    } else if (arg.substr(0, 14) == "--num_threads=") {
      flags.num_threads = std::stoi(arg.substr(14));
    } else if (arg == "--use_xnnpack" || arg == "--use_xnnpack=true") {
      flags.use_xnnpack = true;
    } else if (arg == "--no_use_xnnpack" || arg == "--use_xnnpack=false") {
      flags.use_xnnpack = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: tflite_accuracy_test [options]\n"
                << "  --model=FILE         TFLite model file path\n"
                << "  --data_dir=DIR       ImageNet validation dataset directory\n"
                << "  --synset_file=FILE   synset_words.txt path\n"
                << "  --num_images=N       Number of test images (0=all)\n"
                << "  --num_threads=N      Number of threads (default -1, use TFLite default)\n"
                << "  --use_xnnpack        Enable XNNPACK acceleration (enabled by default)\n"
                << "  --use_xnnpack=false  Disable XNNPACK acceleration\n";
      return false;
    }
  }
  return true;
}

// ============================================================
// Image loading and preprocessing
// ============================================================

struct ImageData {
  std::vector<float> data;
  int width;
  int height;
  int channels;
};

ImageData LoadAndPreprocessImage(const std::string& image_path,
                                 int target_w = 224, int target_h = 224) {
  int img_w, img_h, img_c;
  unsigned char* pixels =
      stbi_load(image_path.c_str(), &img_w, &img_h, &img_c, 3);
  if (!pixels) {
    return ImageData{{}, 0, 0, 0};
  }

  std::vector<float> input_data(target_w * target_h * 3);
  for (int y = 0; y < target_h; ++y) {
    for (int x = 0; x < target_w; ++x) {
      int src_x = std::min(x * img_w / target_w, img_w - 1);
      int src_y = std::min(y * img_h / target_h, img_h - 1);
      for (int c = 0; c < 3; ++c) {
        uint8_t val = pixels[(src_y * img_w + src_x) * 3 + c];
        input_data[(y * target_w + x) * 3 + c] = (val / 127.5f) - 1.0f;
      }
    }
  }

  stbi_image_free(pixels);
  return ImageData{std::move(input_data), target_w, target_h, 3};
}

// ============================================================
// Top-1 prediction
// ============================================================

int GetTop1(const float* output, int num_classes) {
  int max_idx = 0;
  float max_val = output[0];
  for (int i = 1; i < num_classes; ++i) {
    if (output[i] > max_val) {
      max_val = output[i];
      max_idx = i;
    }
  }
  return max_idx;
}

int GetTop1(const uint8_t* output, int num_classes) {
  int max_idx = 0;
  uint8_t max_val = output[0];
  for (int i = 1; i < num_classes; ++i) {
    if (output[i] > max_val) {
      max_val = output[i];
      max_idx = i;
    }
  }
  return max_idx;
}

int GetTop1(const int8_t* output, int num_classes) {
  int max_idx = 0;
  int8_t max_val = output[0];
  for (int i = 1; i < num_classes; ++i) {
    if (output[i] > max_val) {
      max_val = output[i];
      max_idx = i;
    }
  }
  return max_idx;
}

// ============================================================
// Load synset_words.txt and build WNID -> index mapping
// ============================================================

std::map<std::string, int> LoadSynsetMapping(const std::string& path) {
  std::map<std::string, int> mapping;
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open synset file: " << path << std::endl;
    return mapping;
  }
  std::string line;
  int index = 0;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    size_t space_pos = line.find(' ');
    std::string wnid = (space_pos != std::string::npos)
                           ? line.substr(0, space_pos)
                           : line;
    mapping[wnid] = index++;
  }
  return mapping;
}

// ============================================================
// Scan WNID subdirectories under data_dir
// ============================================================

struct SampleInfo {
  std::string image_path;
  std::string wnid;
  int class_index;
};

std::vector<SampleInfo> ScanDataset(const std::string& data_dir,
                                    const std::map<std::string, int>& synset_map,
                                    int max_images) {
  std::vector<SampleInfo> samples;

  // Collect all WNID subdirectories and sort alphabetically
  std::vector<std::string> wnid_dirs;
  DIR* root = opendir(data_dir.c_str());
  if (!root) {
    std::cerr << "Error: Cannot open directory: " << data_dir << std::endl;
    return samples;
  }

  struct dirent* wnid_entry;
  while ((wnid_entry = readdir(root)) != nullptr) {
    std::string wnid = wnid_entry->d_name;
    if (wnid == "." || wnid == "..") continue;
    if (synset_map.find(wnid) == synset_map.end()) continue;
    wnid_dirs.push_back(wnid);
  }
  closedir(root);
  std::sort(wnid_dirs.begin(), wnid_dirs.end());

  // Iterate WNID subdirectories in alphabetical order
  for (const auto& wnid : wnid_dirs) {
    if (max_images > 0 && static_cast<int>(samples.size()) >= max_images) break;

    int class_idx = synset_map.at(wnid);
    std::string wnid_dir = data_dir + "/" + wnid;

    DIR* subdir = opendir(wnid_dir.c_str());
    if (!subdir) continue;

    struct dirent* img_entry;
    while ((img_entry = readdir(subdir)) != nullptr) {
      if (max_images > 0 && static_cast<int>(samples.size()) >= max_images) break;

      std::string filename = img_entry->d_name;
      if (filename == "." || filename == "..") continue;

      // Only process image files
      if (filename.size() < 5) continue;
      std::string ext = filename.substr(filename.size() - 5);
      if (ext.find(".JPEG") == std::string::npos &&
          ext.find(".jpeg") == std::string::npos &&
          ext.find(".jpg") == std::string::npos &&
          ext.find(".JPG") == std::string::npos) {
        continue;
      }

      samples.push_back({wnid_dir + "/" + filename, wnid, class_idx});
    }
    closedir(subdir);
  }

  // Limit the number of returned samples
  if (max_images > 0 && static_cast<int>(samples.size()) > max_images) {
    samples.resize(max_images);
  }

  return samples;
}

// ============================================================
// Main flow
// ============================================================

int RunAccuracyTest(const Flags& flags) {
  // ---- 1. Load synset mapping ----
  auto synset_map = LoadSynsetMapping(flags.synset_file);
  if (synset_map.empty()) {
    std::cerr << "Error: Synset mapping is empty" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Loaded " << synset_map.size() << " synset mappings" << std::endl;

  // ---- 2. Scan dataset ----
  auto samples = ScanDataset(flags.data_dir, synset_map, flags.num_images);
  if (samples.empty()) {
    std::cerr << "Error: No valid images found, please check directory structure" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Found " << samples.size() << " images" << std::endl;

  // ---- 3. Load model ----
  std::cout << "Loading model: " << flags.model << std::endl;
  auto model = tflite::FlatBufferModel::BuildFromFile(flags.model.c_str());
  if (!model) {
    std::cerr << "Error: Cannot load model" << std::endl;
    return EXIT_FAILURE;
  }

  // ---- 4. Create Interpreter ----
  std::unique_ptr<tflite::OpResolver> resolver;
  if (flags.use_xnnpack) {
    resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolver>();
  } else {
    resolver = std::make_unique<tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates>();
  }
  tflite::InterpreterBuilder builder(*model, *resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  if (!interpreter) {
    std::cerr << "Error: Cannot create Interpreter" << std::endl;
    return EXIT_FAILURE;
  }

  // Set number of threads
  interpreter->SetNumThreads(flags.num_threads);

  // Allocate tensor memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cerr << "Error: Cannot allocate tensor memory" << std::endl;
    return EXIT_FAILURE;
  }

  // Get input/output information
  auto input_indices = interpreter->inputs();
  auto output_indices = interpreter->outputs();
  if (input_indices.empty() || output_indices.empty()) {
    std::cerr << "Error: Model has no inputs or outputs" << std::endl;
    return EXIT_FAILURE;
  }

  TfLiteTensor* input_tensor = interpreter->tensor(input_indices[0]);
  TfLiteTensor* output_tensor = interpreter->tensor(output_indices[0]);

  // Check model type
  enum class ModelType { kFloat32, kUInt8, kInt8 };
  ModelType model_type = ModelType::kFloat32;
  if (input_tensor->type == kTfLiteUInt8) {
    model_type = ModelType::kUInt8;
  } else if (input_tensor->type == kTfLiteInt8) {
    model_type = ModelType::kInt8;
  }

  const char* type_str = "Float32";
  if (model_type == ModelType::kUInt8) type_str = "Quantized (uint8)";
  else if (model_type == ModelType::kInt8) type_str = "Quantized (int8)";
  std::cout << "Model type: " << type_str << std::endl;
  if (model_type != ModelType::kFloat32) {
    std::cout << "Input  quantization: scale=" << input_tensor->params.scale
              << " zero_point=" << (int)input_tensor->params.zero_point
              << std::endl;
    std::cout << "Output quantization: scale=" << output_tensor->params.scale
              << " zero_point=" << (int)output_tensor->params.zero_point
              << std::endl;
  }

  // Get number of output classes
  int num_classes = 1;
  for (int d = 0; d < output_tensor->dims->size; ++d) {
    num_classes *= output_tensor->dims->data[d];
  }
  std::cout << "Model output classes: " << num_classes << std::endl;

  // Some models have 1001 outputs (with background class), skip index 0
  int class_offset = 0;
  if (num_classes == 1001 && synset_map.size() == 1000) {
    class_offset = 1;
    std::cout << "Detected 1001-class model (with background), skipping index 0" << std::endl;
  }

  std::cout << "Model loaded successfully" << std::endl;

  // ---- 5. Iterate images and test ----
  int total = static_cast<int>(samples.size());
  if (flags.num_images > 0 && flags.num_images < total) {
    total = flags.num_images;
  }
  int correct = 0;
  int processed = 0;
  int errors = 0;
  float total_time_ms = 0.0f;

  std::cout << "\nStarting test with " << total << " images..." << std::endl;
  std::cout << "========================================" << std::endl;

  std::string current_wnid;
  for (int i = 0; i < total; ++i) {
    const auto& sample = samples[i];

    // Print current folder being processed
    if (sample.wnid != current_wnid) {
      // Print stats for previous folder (if any)
      if (!current_wnid.empty() && processed > 0) {
        float acc = 100.0f * correct / processed;
        float avg_ms = total_time_ms / processed;
        std::cout << "[" << current_wnid << "] [" << processed << "/" << total << "]"
                  << " Top-1: " << std::fixed << std::setprecision(2) << acc
                  << "%  Avg inference: " << std::setprecision(1) << avg_ms
                  << " ms" << std::endl;
      }
      current_wnid = sample.wnid;
    }

    // Load and preprocess image
    auto image = LoadAndPreprocessImage(sample.image_path);
    if (image.data.empty()) {
      std::cerr << "Skip: Cannot load " << sample.image_path << std::endl;
      errors++;
      continue;
    }

    // Fill input based on model type
    if (model_type == ModelType::kUInt8) {
      // Quantized uint8 model: convert [-1, 1] to [0, 255]
      uint8_t* input_data = interpreter->typed_input_tensor<uint8_t>(0);
      for (size_t j = 0; j < image.data.size(); ++j) {
        float val = (image.data[j] + 1.0f) * 127.5f;
        input_data[j] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val)));
      }
    } else if (model_type == ModelType::kInt8) {
      // Quantized int8 model: convert [-1, 1] to [-128, 127]
      int8_t* input_data = interpreter->typed_input_tensor<int8_t>(0);
      for (size_t j = 0; j < image.data.size(); ++j) {
        float val = image.data[j] * 127.0f;
        input_data[j] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, val)));
      }
    } else {
      // Float model: copy directly
      float* input_data = interpreter->typed_input_tensor<float>(0);
      memcpy(input_data, image.data.data(), image.data.size() * sizeof(float));
    }

    // Inference and timing
    auto start = std::chrono::high_resolution_clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
      std::cerr << "Error: Inference failed " << sample.image_path << std::endl;
      errors++;
      continue;
    }
    auto end = std::chrono::high_resolution_clock::now();
    float infer_ms =
        std::chrono::duration<float, std::milli>(end - start).count();
    total_time_ms += infer_ms;

    // Get output based on model type
    int pred;
    if (model_type == ModelType::kUInt8) {
      uint8_t* output_data = interpreter->typed_output_tensor<uint8_t>(0);
      pred = GetTop1(output_data, num_classes) - class_offset;
    } else if (model_type == ModelType::kInt8) {
      int8_t* output_data = interpreter->typed_output_tensor<int8_t>(0);
      pred = GetTop1(output_data, num_classes) - class_offset;
    } else {
      float* output_data = interpreter->typed_output_tensor<float>(0);
      pred = GetTop1(output_data, num_classes) - class_offset;
    }
    int truth = sample.class_index;
    bool is_correct = (pred == truth);
    if (is_correct) correct++;
    processed++;

    // Print first 10 error examples
    if (!is_correct && (correct + errors) <= 10) {
      std::cout << "  Error: " << sample.image_path
                << " truth=" << truth << " pred=" << pred << std::endl;
    }
  }

  // Print stats for last folder
  if (!current_wnid.empty() && processed > 0) {
    float acc = 100.0f * correct / processed;
    float avg_ms = total_time_ms / processed;
    std::cout << "[" << current_wnid << "] [" << processed << "/" << total << "]"
              << " Top-1: " << std::fixed << std::setprecision(2) << acc
              << "%  Avg inference: " << std::setprecision(1) << avg_ms
              << " ms" << std::endl;
  }

  // ---- 6. Output final results ----
  std::cout << "\n========================================" << std::endl;
  std::cout << "Accuracy Test Results" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Model:          " << flags.model << std::endl;
  std::cout << "Test samples:   " << processed << std::endl;
  std::cout << "Load failures:  " << errors << std::endl;
  std::cout << "Top-1 correct:  " << correct << std::endl;
  if (processed > 0) {
    float acc = 100.0f * correct / processed;
    float avg_ms = total_time_ms / processed;
    std::cout << "Top-1 accuracy: " << std::fixed << std::setprecision(2)
              << acc << "%" << std::endl;
    std::cout << "Avg inference:  " << std::setprecision(1) << avg_ms
              << " ms" << std::endl;
    std::cout << "Throughput:     " << std::setprecision(1)
              << 1000.0f / avg_ms << " FPS" << std::endl;
  }
  std::cout << "========================================" << std::endl;

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  Flags flags;
  if (!ParseFlags(argc, argv, flags)) {
    return EXIT_SUCCESS;  // --help
  }

  if (flags.model.empty() || flags.data_dir.empty() || flags.synset_file.empty()) {
    std::cerr << "Error: Must specify --model, --data_dir, --synset_file" << std::endl;
    std::cerr << "Use --help for usage" << std::endl;
    return EXIT_FAILURE;
  }

  // Display parameters (consistent with benchmark)
  std::cout << "INFO: Graph: [" << flags.model << "]" << std::endl;
  std::cout << "INFO: Use xnnpack: [" << (flags.use_xnnpack ? 1 : 0) << "]" << std::endl;
  std::cout << "INFO: Num images: [" << flags.num_images << "]" << std::endl;
  std::cout << "INFO: Num threads: [" << flags.num_threads << "]" << std::endl;

  return RunAccuracyTest(flags);
}
