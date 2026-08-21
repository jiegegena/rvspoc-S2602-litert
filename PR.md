

# **一、验证平台说明**：

A210实机，（此外还包括进迭时空K1，但不考虑放入此次提交）

# **二、依赖库说明**

依赖库版本：

```bash
ubuntu24.04.4（测试了ubuntu22、18等版本也行，但建议采用高版本）
riscv-gcc版本：14.2.0（硬性要求>=14）
riscv64-linux-gnu-gcc
riscv64-linux-gnu-g++
cmake版本：4.0.1
```

```bash
cmake4.0.1安装可以参考下面方法：
1、wget https://github.com/Kitware/CMake/releases/download/v4.0.1/cmake-4.0.1-linux-x86_64.tar.gz
2、tar -zxvf cmake-4.0.1-linux-x86_64.tar.gz
3、sudo mv cmake-4.0.1-linux-x86_64 /usr/local/cmake-4.0.1
4、echo 'export PATH=/usr/local/cmake-4.0.1/bin:$PATH' >> ~/.bashrc
5、source ~/.bashrc
```

# **三、程序编译及安装步骤**

1. **拉取代码**

2. **下载依赖源码**（litert会依赖之前的tensorflowlite的部分代码）。

​	手动下载和指定，如果不指定的话编译的时候会自动下载，这样每次构建都会重复下载.
脚本内已指定，即 `litert/CMakePresets.json里的"TENSORFLOW_SOURCE_DIR": "${sourceDir}/tensorflow-src"`，具体操作如下：

```bash
cd litert
git clone https://github.com/tensorflow/tensorflow.git tensorflow-src
```

3. **构建host_flatc_build流程：**

```bash
在根目录下运行：
mkdir -p host_flatc_build && cd host_flatc_build

目录下创建CMakeLists.txt文件，内容如下：
cmake_minimum_required(VERSION 3.16)
project(HostFlatc LANGUAGES C CXX)
include(FetchContent)
FetchContent_Declare(flatbuffers GIT_REPOSITORY https://github.com/google/flatbuffers.git GIT_TAG v25.9.23)
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(flatbuffers)

然后构建和编译：
cmake -S . -B . -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build . --target flatc -- -j8
```

4. **构建和编译**

可直接复现示例程序：编译成果物位于代码根目录的 `S2602_test目录`，如下所示：

```bash
S2602_test/
--------riscv64  	  // riscv标量可运行程序
--------riscv64-rvv   // rvv优化后的可运行程序
--------x86 		  // x86可运行程序（用于对比）
```

5. **手动构建请参考下面示例**

（为保证代码的兼容性和可移植性，本代码构建方法贴合官方构建方法，使用cmake构建，因此代码内使用的脚本代码较少）

**构建riscv标量：**

```bash
cd litert
cmake --preset linux-riscv64
cmake --build cmake_build_riscv64 --target benchmark_model tflite_accuracy_test -j6
```

**构建riscv-rvv：**

```bash
cd litert
cmake --preset linux-riscv64-rvv
cmake --build cmake_build_riscv64_rvv --target benchmark_model tflite_accuracy_test -j6
```

**如需要构建x86版本：**

```bash
cd litert
cmake --preset default
cmake --build cmake_build --target benchmark_model tflite_accuracy_test -j6
```

**构建生成的成果物位于：**

```bash
位于rvspoc-S2602-litert/litert/cmake_build_**/tflite_build/tools目录下，
可执行程序为：benchmark_model（用于性能测试）和tflite_accuracy_test（用于真实图片的精度测试）
```

# **四、程序运行步骤**

测试如下六类模型，你可以在s2602_test/models目录中找到

```bash
efficientnet-tflite-lite0-fp32-v2.tflite efficientnet-tflite-lite0-int8-v2.tflite
mobilenet_v1_1.0_224.tflite  mobilenet_v1_1.0_224_quant.tflite
mobilenet_v2_1.0_224.tflite  mobilenet_v2_1.0_224_quant.tflite
```

## 4.1、性能测试

测试方法：

```bash
./benchmark_model --graph=<模型路径> --num_runs=<测试轮数> --warmup_runs=<模型预热次数> --num_threads=<线程数>
```

这里统一使用下面命令进行测试：

```bash
./benchmark_model --graph=../models/mobilenet_v1_1.0_224.tflite --num_runs=50 --warmup_runs=5 --num_threads=<可选1、2、4、8>

# 模型测试50轮，预热5轮
```

示例：

```bash
./benchmark_model --graph=../models/mobilenet_v1_1.0_224.tflite --num_runs=50 --warmup_runs=5 --num_threads=4
./benchmark_model --graph=../models/mobilenet_v1_1.0_224_quant.tflite --num_runs=50 --warmup_runs=5 --num_threads=4
./benchmark_model --graph=../models/efficientnet-tflite-lite0-fp32-v2.tflite --num_runs=50 --warmup_runs=5 --num_threads=4
./benchmark_model --graph=../models/efficientnet-tflite-lite0-int8-v2.tflite --num_runs=50 --warmup_runs=5 --num_threads=4 
./benchmark_model --graph=../models/mobilenet_v2_1.0_224.tflite --num_runs=50 --warmup_runs=5 --num_threads=4 
./benchmark_model --graph=../models/mobilenet_v2_1.0_224_quant.tflite --num_runs=50 --warmup_runs=5 --num_threads=4 
```

## **4.2、精度测试**

测试方法：

```bash
./tflite_accuracy_test \
 --data_dir=<划分好的测试集路径val_picture_10k> \
 --synset_file=<synset_words.txt文件路径> \
 --num_images=<测试图片数，0为测试全部> \
 --num_threads=8 \
 --model=<模型路径>
```

​	其中：
​	synset_words.txt：标签文件，它包含了 `ImageNet` 数据集中 1000 个类别。你可以在代码路径：`s2602_test/synset_words.txt` 或 `tflite/tools/accuracy/ILSVRC2012_img_val/synset_words.txt` 中找到。
​   val_picture_10k：测试集合，按测试代码格式处理过的测试集，一共10000张数据，你可以直接在如下网盘链接下载：`https://pan.quark.cn/s/1356e3eea5e7` 找到（建议）。如果你需要手动划分数据集，请参考4.3。

riscv精度测试示例：

```bash
./tflite_accuracy_test \
 --model=../models/mobilenet_v1_1.0_224.tflite \
 --data_dir=/tmp/test/val_picture_10k \
 --synset_file=/tmp/test/synset_words.txt \
 --num_images=0 \
 --use_xnnpack=false \
 --num_threads=8
```

x86测试示例：

```bash
./tflite_accuracy_test \
 --model=models/mobilenet_v1_1.0_224.tflite \
 --data_dir=val_picture_10k \
 --synset_file=synset_words.txt \
 --num_images=0 \
 --use_xnnpack=false \
 --num_threads=8
```

## **4.3、数据集处理（可选）**

下载完整的验证集到目录：`tflite/tools/accuracy/ILSVRC2012_img_val`：

该目录下执行：
```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```

数据处理：

```bash
解压到val
mkdir val
tar xvf ILSVRC2012_img_val.tar -C ./val

运行脚本生成测试格式：
python3 -m venv .venv
source .venv/bin/activate
pip install scipy
python3 organize_val.py
```

生成数据格式如下：

```bash
val/
├── n01440764/
│  ├── ILSVRC2012_val_xxx1.JPEG
│  ├── ILSVRC2012_val_xxx2.JPEG
│  └── ...
├── n01443537/
│  ├── ILSVRC2012_val_xxx1.JPEG
│  └── ...
└── ... (1000 WNID subfolders total)
```

验证数据集取前10000张图片（建议），或你上传所有图片，但需要指定--num_images=10000

# 五、程序运行结果：

## **5.1、性能测试结果**

​	riscv标量构建方法可能采用不同的后端，开启或关闭不同的宏有不同的效果，这里riscv的构建方法尽可能接近riscv-rvv，以次保证更好的进行对比，具体配置您可以参考目录：litert/CMakePresets.json的配置。
​	备注：在a210上，对于高速推理场景下，开启线程高于或等于8，某些场景可能反而会比4更慢，可能和大小核调度和缓存命中有关，这在riscv芯片-进迭时空K1上测试就没有这种情况（不影响实际测试）。此外，a210远程机在我通过scp传输大量数据后（中断），再使用4、2、1线程推理可能会变慢，这应该和远程机器相关（因此建议先测性能再测精度，或重启机器）。
​	下面为具体测试结果：对于FPS计算公式throughput = FPS = batch_size x 1000 / latency_ms，其中 `batch_size = 1`。（注：受机器环境等影响，测试结果可能会出现细微差异）

### **5.1.1、mobilenet_v1_1.0_224.tflite**

./benchmark_model --graph=../models/mobilenet_v1_1.0_224.tflite --num_runs=50 --warmup_runs=5 --num_threads=4

| 参数     |          | Scalar ms |         |         |         | RVV  ms |         |         |
| -------- | -------- | --------- | ------- | ------- | ------- | ------- | ------- | ------- |
| Threads  | 1        | 2         | 4       | 8       | 1       | 2       | 4       | 8       |
| 启动时间 | 1.75     | 1.666     | 1.652   | 2.133   | 2.105   | 2.036   | 2.033   | 2.117   |
| avg      | 2493.380 | 1251.620  | 635.840 | 489.728 | 186.713 | 102.209 | 67.608  | 80.5679 |
| p50      | 2489.548 | 1251.431  | 631.755 | 488.900 | 186.621 | 102.209 | 67.612  | 80.198  |
| p95      | 2508.892 | 1253.674  | 632.628 | 496.761 | 186.987 | 102.358 | 67.821  | 84.579  |
| FPS      | 0.401766 | 0.798.966 | 1.58283 | 2.04195 | 5.35581 | 9.78387 | 14.7912 | 12.4119 |
| std      | 828      | 765       | 460     | 3271    | 401     | 94      | 124     | 2083    |
| 内存占用 | 28.6094  | 28.3047   | 28.2812 | 27.7773 | 24.375  | 24      | 23.875  | 24.125  |

### **5.1.2、mobilenet_v1_1.0_224_quant.tflite**

./benchmark_model --graph=../models/mobilenet_v1_1.0_224_quant.tflite --num_runs=50 --warmup_runs=5 --num_threads=4

| 参数     |          |          | Scalar ms |         |         |         | RVV  ms |         |
| -------- | -------- | -------- | --------- | ------- | ------- | ------- | ------- | ------- |
| Threads  | 1        | 2        | 4         | 8       | 1       | 2       | 4       | 8       |
| 启动时间 | 3.286    | 3.308    | 3.313     | 2.579   | 3.276   | 3.321   | 3.305   | 3.329   |
| avg      | 3104.820 | 1554.630 | 783.802   | 622.293 | 188.390 | 98.1663 | 51.7645 | 70.0691 |
| p50      | 3104.745 | 1552.675 | 782.179   | 622.310 | 188.279 | 97.003  | 49.657  | 69.938  |
| p95      | 3105.843 | 1562.510 | 790.473   | 625.950 | 189.366 | 104.218 | 70.685  | 72.117  |
| FPS      | 0.32208  | 0.644031 | 1.27841   | 1.60696 | 5.30813 | 10.3224 | 20.141  | 14.2716 |
| std      | 301      | 3941     | 201       | 2064    | 456     | 152     | 80      | 1187    |
| 内存占用 | 8.82031  | 8.98828  | 8.87109   | 8.125   | 8.125   | 7.875   | 7.875   | 7.25    |

### **5.1.3、mobilenet_v2_1.0_224.tflite**

./benchmark_model --graph=../models/mobilenet_v2_1.0_224.tflite --num_runs=50 --warmup_runs=5 --num_threads=4

| 参数     |          |         | Scalar ms |         |         |         | RVV  ms |         |
| -------- | -------- | ------- | --------- | ------- | ------- | ------- | ------- | ------- |
| Threads  | 1        | 2       | 4         | 8       | 1       | 2       | 4       | 8       |
| 启动时间 | 2.696    | 2.68    | 2.661     | 2.722   | 2.087   | 2.101   | 2.094   | 2.682   |
| avg      | 1391.390 | 699.553 | 355.058   | 320.347 | 136.464 | 73.4187 | 48.1393 | 56.9886 |
| p50      | 1385.785 | 699.411 | 355.002   | 320.799 | 136.453 | 73.423  | 48136   | 56.389  |
| p95      | 1409.418 | 701.046 | 355.912   | 326.833 | 136.553 | 73.511  | 48.220  | 57.874  |
| FPS      | 0.721973 | 1.42948 | 2.81644   | 3.16546 | 7.32794 | 13.6205 | 20.773  | 17.8217 |
| std      | 965      | 642     | 389       | 3881    | 42      | 56      | 48      | 1355    |
| 内存占用 | 28.9297  | 28.9531 | 28.8828   | 8.9258  | 22.625  | 22.75   | 22.375  | 22.375  |

 

### **5.1.4mobilenet_v2_1.0_224_quant.tflite**

./benchmark_model --graph=../models/mobilenet_v2_1.0_224_quant.tflite --num_runs=50 --warmup_runs=5 --num_threads=4

| 参数     |          |         | Scalar ms |         |         |         | RVV  ms |         |
| -------- | -------- | ------- | --------- | ------- | ------- | ------- | ------- | ------- |
| Threads  | 1        | 2       | 4         | 8       | 1       | 2       | 4       | 8       |
| 启动时间 | 3.382    | 3.388   | 3.421     | 4.485   | 4.464   | 4.492   | 3.387   | 3.444   |
| avg      | 1741.760 | 877.289 | 444.769   | 410.885 | 144.616 | 74.777  | 39.9525 | 52.550  |
| p50      | 1741.578 | 877.285 | 444.732   | 411.499 | 143.257 | 74.776  | 39.941  | 52.161  |
| p95      | 1742.778 | 877.545 | 445.097   | 416.347 | 150.105 | 74.888  | 40.063  | 54.443  |
| FPS      | 0.574133 | 1.13987 | 2.24836   | 2.43377 | 6.98411 | 13.3731 | 25.0297 | 19.0587 |
| std      | 596      | 182     | 229       | 3544    | 2744    | 46      | 44      | 911     |
| 内存占用 | 9.38672  | 9.26172 | 8.97266   | 8.64453 | 7.625   | 7.625   | 7.5     | 7.5     |

 

### **5.1.5、efficientnet-tflite-lite0-fp32-v2.tflite**

./benchmark_model --graph=../models/efficientnet-tflite-lite0-fp32-v2.tflite --num_runs=50 --warmup_runs=5 --num_threads=4

| 参数     |          |         | Scalar ms |         |         |         | RVV  ms |         |
| -------- | -------- | ------- | --------- | ------- | ------- | ------- | ------- | ------- |
| Threads  | 1        | 2       | 4         | 8       | 1       | 2       | 4       | 8       |
| 启动时间 | 2.251    | 2.281   | 2.196     | 2.139   | 2.869   | 2.819   | 2.822   | 2.862   |
| avg      | 1740.830 | 881.471 | 4482.44   | 411696  | 162.544 | 88.0346 | 56.280  | 71.6511 |
| p50      | 1740.727 | 881.477 | 448.192   | 411944  | 162.534 | 88.026  | 56.274  | 71.914  |
| p95      | 1742.319 | 882.235 | 449.016   | 418323  | 162.618 | 88.142  | 56.404  | 75.048  |
| FPS      | 0.574439 | 1.13447 | 2.23093   | 2.42898 | 6.15219 | 11.3592 | 17.7683 | 13.9565 |
| std      | 750      | 490     | 417       | 3929    | 42      | 53      | 64      | 2085    |
| 内存占用 | 33.5156  | 33.8711 | 33.1719   | 32.8086 | 27      | 26.875  | 27      | 26.5    |


### **5.1.6、efficientnet-tflite-lite0-int8-v2.tflite**

./benchmark_model --graph=../models/efficientnet-tflite-lite0-int8-v2.tflite --num_runs=50 --warmup_runs=5 --num_threads=4

 | 参数     |          |          | Scalar ms |         |         |         | RVV  ms |         |
| -------- | -------- | -------- | --------- | ------- | ------- | ------- | ------- | ------- |
| Threads  | 1        | 2        | 4         | 8       | 1       | 2       | 4       | 8       |
| 启动时间 | 6.643    | 6.535    | 6.539     | 6.634   | 6.519   | 6.541   | 6.668   | 6.555   |
| avg      | 4311.740 | 2173.080 | 1104.930  | 705.604 | 282.566 | 144.591 | 76.7534 | 54.7198 |
| p50      | 4311.714 | 2172.978 | 1104.944  | 704.787 | 282.491 | 144.597 | 76.750  | 53.211  |
| p95      | 4312.853 | 2173.812 | 1105.454  | 725.892 | 282.912 | 144.763 | 76.888  | 53.777  |
| FPS      | 0.231925 | 0.460176 | 0.905036  | 1.41723 | 3.539   | 6.91608 | 13.0287 | 18.7525 |
| std      | 447      | 402      | 305       | 8124    | 410     | 98      | 74      | 280     |
| 内存占用 | 11.4805  | 11.418   | 11.1914   | 11.168  | 11.0195 | 11.125  | 11.0391 | 10.668  |



## 5.2、模型级 Top-1 精度测试结果

测试了业内比较标准的数据集 ImageNet 的前10000 张验证集：

测试示例：
```bash
./tflite_accuracy_test \
 --model=../models/mobilenet_v1_1.0_224.tflite \
 --data_dir=/tmp/test/val_picture_10k \
 --synset_file=/tmp/test/synset_words.txt \
 --num_images=0 \
 --use_xnnpack=false \
 --num_threads=8
```


| 模型                              | x86    | RVV Top-1 |   Δ    |
| --------------------------------- | ------ | --------- | ------ |
| mobilenet_v1_1.0_224              | 71.97% | 71.97%    | +0.00% |
| mobilenet_v1_1.0_224_quant        | 71.34% | 71.51%    | +0.17% |
| mobilenet_v2_1.0_224              | 73.06% | 73.06%    | +0.00% |
| mobilenet_v2_1.0_224_quant        | 71.27% | 71.40%    | +0.13% |
| efficientnet-tflite-lite0-fp32-v2 | 76.33% | 76.33%    | +0.00% |
| efficientnet-tflite-lite0-int8-v2 | 70.84% | 70.96%    | +0.12% |

结论：rvv优化在fp32模型中和x86完全保持一致，无任何精度丢失，量化模型精度有所提升。

# 六、AI 辅助说明

若使用 AI 辅助编写代码，需在提交报告中说明使用方式及占比:

70% of the code including scripts and partial RVV optimizations was AI‑generated; optimization directions were human‑decided, and bugs were debugged and fixed manually.

---