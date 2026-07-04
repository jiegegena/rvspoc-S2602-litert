# TFLite ImageNet Accuracy Test Tool

An accuracy test tool based on TensorFlow Lite for evaluating model accuracy on the ImageNet validation dataset.

## Required Directory Structure

The validation dataset directory should be organized with WNID subfolders:

```
ILSVRC2012_img_val/
├── n01440764/
│   ├── ILSVRC2012_val_xxx1.JPEG
│   ├── ILSVRC2012_val_xxx2.JPEG
│   └── ...
├── n01443537/
│   ├── ILSVRC2012_val_xxx1.JPEG
│   └── ...
└── ... (1000 WNID subfolders total)
```

## File Description

| File | Description |
|------|-------------|
| `tflite_accuracy_test.cc` | Main accuracy test program |
| `CMakeLists.txt` | CMake build configuration |
| `synset_words.txt` | WNID to class index mapping file (sorted alphabetically) |
| `imagenet_labels.h` | ImageNet class names (for reference only, program uses synset_words.txt) |
| `ILSVRC2012_validation_ground_truth.txt` | Validation ground truth (for reference only, program uses directory structure) |

## Build

### x86 Host Build

```bash
cd litert
cmake --preset default
cmake --build cmake_build --target tflite_accuracy_test -j8
```

### RISC-V Cross Compilation

```bash
cd litert
cmake --preset linux-riscv64-rvv
cmake --build cmake_build_riscv64_rvv --target tflite_accuracy_test -j8
```

## Usage

### Basic Usage

```bash
./tflite_accuracy_test \
  --model=mobilenet_v1_1.0_224.tflite \
  --data_dir=/path/to/ILSVRC2012_img_val \
  --synset_file=synset_words.txt
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model=FILE` | TFLite model file path | Required |
| `--data_dir=DIR` | ImageNet validation dataset directory (WNID subfolders) | Required |
| `--synset_file=FILE` | synset_words.txt path | Required |
| `--num_images=N` | Number of test images (0=all) | 0 |
| `--num_threads=N` | Number of inference threads (-1=use TFLite default) | -1 |
| `--use_xnnpack` | Enable XNNPACK acceleration | Enabled by default |
| `--use_xnnpack=false` | Disable XNNPACK acceleration | - |
| `--help` | Show help information | - |

### Examples

#### Test all images (50000)

```bash
./tflite_accuracy_test \
  --model=mobilenet_v1_1.0_224.tflite \
  --data_dir=val \
  --synset_file=synset_words.txt
```

#### Test first 1000 images

```bash
./tflite_accuracy_test \
  --model=mobilenet_v1_1.0_224.tflite \
  --data_dir=val \
  --synset_file=synset_words.txt \
  --num_images=1000
```

#### Disable XNNPACK (use default CPU operators)

```bash
./tflite_accuracy_test \
  --model=mobilenet_v1_1.0_224.tflite \
  --data_dir=val \
  --synset_file=synset_words.txt \
  --use_xnnpack=false
```

#### Specify number of threads

```bash
./tflite_accuracy_test \
  --model=mobilenet_v1_1.0_224.tflite \
  --data_dir=val \
  --synset_file=synset_words.txt \
  --num_threads=4
```

## Output Example

```
INFO: Graph: [mobilenet_v1_1.0_224.tflite]
INFO: Use xnnpack: [1]
INFO: Num images: [1000]
INFO: Num threads: [-1]
Loaded 1000 synset mappings
Found 1000 images
Loading model: mobilenet_v1_1.0_224.tflite
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
Model output classes: 1001
Detected 1001-class model (with background), skipping index 0
Model loaded successfully

Starting test with 1000 images...
========================================
  Error: val/n01440764/ILSVRC2012_val_00037375.JPEG truth=0 pred=389
[n01440764] [50/1000] Top-1: 90.00%  Avg inference: 11.5 ms
[n01443537] [100/1000] Top-1: 86.00%  Avg inference: 11.7 ms
...
[n01531178] [1000/1000] Top-1: 79.33%  Avg inference: 12.0 ms

========================================
Accuracy Test Results
========================================
Model:          mobilenet_v1_1.0_224.tflite
Test samples:   1000
Load failures:  0
Top-1 correct:  793
Top-1 accuracy: 79.33%
Avg inference:  12.0 ms
Throughput:     83.3 FPS
========================================
```

## Generate synset_words.txt

To generate `synset_words.txt` from `meta.mat`, use the following Python script:

```python
import scipy.io

meta = scipy.io.loadmat('meta.mat')
synsets = meta['synsets']

# Generate alphabetically sorted synset_words.txt
wnids = []
for i in range(1000):
    wnid = str(synsets[i][0][1][0])
    words = str(synsets[i][0][2][0])
    wnids.append((wnid, words))

wnids.sort(key=lambda x: x[0])

with open('synset_words.txt', 'w') as f:
    for wnid, words in wnids:
        f.write(f"{wnid} {words}\n")
```

## Notes

1. **Model output classes**: Supports both 1000-class and 1001-class (with background) models, automatically detected
2. **XNNPACK**: Enabled by default, consistent with benchmark tool behavior
3. **Image preprocessing**: Normalized to [-1, 1] using `(pixel / 127.5) - 1.0`, consistent with MobileNet and other models
4. **Sorting order**: Traversed in WNID alphabetical order (n01440764, n01443537, ...), consistent with synset_words.txt

## Differences from LiteRT Version

This tool uses the native TFLite Interpreter, while `litert/examples/accuracy_test/` uses the LiteRT runtime.

| Feature | TFLite Version | LiteRT Version |
|---------|---------------|----------------|
| Dependencies | TFLite | LiteRT |
| XNNPACK Control | `--use_xnnpack` parameter | Compile-time decision |
| Use Cases | RISC-V, Embedded | Android, GPU, NPU |
