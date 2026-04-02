# 方案 B：Decoder TVN 加载失败排查 — 三段拆分测试

## 背景

EdgeSAM 的合并 decoder TVN 模型在 TV 设备上无法加载。三段 decoder 子模块的 ONNX 均已通过 NPU 编译，但合并后的 TVN 加载失败。需要逐段排查是哪一段出了问题。

## 策略

将 decoder 拆分为 3 个独立的 TVN 模型，分别在 TV 上加载和运行，定位失败的具体子模块。

```
Decoder 拆分为 3 段:

  Part 1: Prompt Encoding
    输入: point_embedding_pe [1,5,256] + point_labels [1,5]
    输出: sparse_embedding [1,5,256]

  Part 2: Transformer
    输入: image_embeddings [1,256,64,64] + sparse_embedding [1,5,256]
    输出: hs [1,10,256] + src [1,4096,256]

  Part 3: Mask Head
    输入: hs [1,10,256] + src [1,4096,256]
    输出: scores [1,4] + masks [1,4,256,256]
```

## 1. 导出 3 段 ONNX 并编译为 TVN

```bash
# 导出 3 段 ONNX（已带 NPU 修复）
python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./npu_diag --num-points 5

# 输出:
#   npu_diag/part1_prompt_encoding.onnx
#   npu_diag/part2_transformer.onnx
#   npu_diag/part3_mask_head.onnx

# 分别用 NPU 编译器编译为 TVN（命令根据具体编译器调整）
# npu_compile part1_prompt_encoding.onnx → part1.tvn
# npu_compile part2_transformer.onnx    → part2.tvn
# npu_compile part3_mask_head.onnx      → part3.tvn
```

---

## 2. 项目结构

编写 3 个独立的测试程序，分别测试每段模型的 loadModel + inference。

```
decoder_diag/
├── CMakeLists.txt
├── include/
│   └── npu_test_utils.h      // 通用测试工具函数
├── src/
│   ├── test_part1.cpp         // 测试 Part 1: Prompt Encoding
│   ├── test_part2.cpp         // 测试 Part 2: Transformer
│   └── test_part3.cpp         // 测试 Part 3: Mask Head
└── models/
    ├── part1.tvn
    ├── part2.tvn
    └── part3.tvn
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.14)
project(decoder_diag)
set(CMAKE_CXX_STANDARD 17)

find_library(NPU_LIB npu_inference PATHS /path/to/npu/lib)

# 三个独立的可执行文件
add_executable(test_part1 src/test_part1.cpp)
add_executable(test_part2 src/test_part2.cpp)
add_executable(test_part3 src/test_part3.cpp)

target_link_libraries(test_part1 ${NPU_LIB})
target_link_libraries(test_part2 ${NPU_LIB})
target_link_libraries(test_part3 ${NPU_LIB})
```

---

## 3. 通用测试工具

```cpp
// npu_test_utils.h
#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// 库函数声明（根据实际头文件调整）
bool loadModel(std::string path, int width, int height);
void releaseModel();

struct model_input {
    const unsigned char* data;
};

struct classifier_output {
    std::vector<float> data;
};

void inference(model_input* model_in, classifier_output* res);

// ---- 工具函数 ----

// 创建全零 float buffer
std::vector<float> make_zeros(int count) {
    return std::vector<float>(count, 0.0f);
}

// 创建随机 float buffer（固定种子，可复现）
std::vector<float> make_random(int count, unsigned seed = 42) {
    std::vector<float> v(count);
    srand(seed);
    for (int i = 0; i < count; i++) {
        v[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }
    return v;
}

// 打印 float buffer 的前 N 个值和统计信息
void print_stats(const char* name, const float* data, int count) {
    float min_v = data[0], max_v = data[0], sum = 0;
    for (int i = 0; i < count; i++) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
        sum += data[i];
    }
    float mean = sum / count;

    printf("  %s: count=%d, min=%.6f, max=%.6f, mean=%.6f\n",
           name, count, min_v, max_v, mean);
    printf("    first 8 values: ");
    for (int i = 0; i < 8 && i < count; i++) {
        printf("%.4f ", data[i]);
    }
    printf("\n");
}

// 检查输出是否全零（通常表示推理失败）
bool is_all_zero(const float* data, int count) {
    for (int i = 0; i < count; i++) {
        if (data[i] != 0.0f) return false;
    }
    return true;
}

// 检查输出是否含 NaN / Inf
bool has_nan_inf(const float* data, int count) {
    for (int i = 0; i < count; i++) {
        if (std::isnan(data[i]) || std::isinf(data[i])) return true;
    }
    return false;
}
```

---

## 4. 三个测试程序

### 4.1 test_part1.cpp — 测试 Prompt Encoding

```cpp
// test_part1.cpp
//
// Part 1: Prompt Encoding (label embedding selection)
//
// 模型输入:
//   input 0: point_embedding_pe  float[1][5][256]    — 位置编码（CPU预计算）
//   input 1: point_labels        float[1][5]         — 标签值
//
// 模型输出:
//   output 0: sparse_embedding   float[1][5][256]    — prompt embedding
//
// 输入总大小: 5*256 + 5 = 1285 floats = 5140 bytes
// 输出总大小: 5*256 = 1280 floats = 5120 bytes
//
// 注意: 此模型非常小（约 8 种算子，几十个节点），如果这个都加载失败，
// 说明问题出在 TVN 格式或库函数本身，不是模型复杂度问题。

#include "npu_test_utils.h"

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "models/part1.tvn";

    printf("========================================\n");
    printf("  Test Part 1: Prompt Encoding\n");
    printf("========================================\n");

    // --- Step 1: Load ---
    printf("\n[1] Loading model: %s\n", model_path);
    // 说明: loadModel 的 width/height 参数含义需根据库的实际行为确定。
    // 如果库用这两个参数设定输入 tensor 的空间尺寸，Part 1 没有空间维度，
    // 可能需要传 0 或 1。如果库从 TVN 文件自动读取输入 shape，
    // 这两个参数可以传任意值。需要根据实际测试调整。
    bool loaded = loadModel(model_path, 0, 0);
    if (!loaded) {
        printf("  FAIL: loadModel returned false\n");
        printf("  RESULT: Part 1 LOAD FAILED\n");
        return 1;
    }
    printf("  OK: model loaded\n");

    // --- Step 2: Prepare input ---
    printf("\n[2] Preparing input\n");

    const int N = 5;           // prompt 点数量
    const int EMBED_DIM = 256;

    // 模拟输入: 随机 PE + 典型标签
    auto pe_data = make_random(N * EMBED_DIM, 42);  // [5][256]

    // 标签: 1个前景点 + 4个padding点
    float labels[N] = {1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

    // 拼接为连续 buffer: [pe_data | labels]
    // 如果库支持多输入，分别传递；否则按偏移拼接
    int input_floats = N * EMBED_DIM + N;  // 1285
    std::vector<float> input_buf(input_floats);
    memcpy(input_buf.data(), pe_data.data(), N * EMBED_DIM * sizeof(float));
    memcpy(input_buf.data() + N * EMBED_DIM, labels, N * sizeof(float));

    print_stats("point_embedding_pe", pe_data.data(), N * EMBED_DIM);
    print_stats("point_labels", labels, N);

    // --- Step 3: Inference ---
    printf("\n[3] Running inference\n");

    model_input min;
    min.data = reinterpret_cast<const unsigned char*>(input_buf.data());

    classifier_output mout;

    auto t0 = std::chrono::high_resolution_clock::now();
    inference(&min, &mout);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Inference time: %.2f ms\n", ms);

    // --- Step 4: Validate output ---
    printf("\n[4] Validating output\n");

    int expected_output_size = N * EMBED_DIM;  // 1280
    printf("  Output size: %d floats (expected %d)\n",
           (int)mout.data.size(), expected_output_size);

    if ((int)mout.data.size() != expected_output_size) {
        printf("  WARNING: output size mismatch!\n");
    }

    if (mout.data.empty()) {
        printf("  FAIL: output is empty\n");
        printf("  RESULT: Part 1 INFERENCE FAILED\n");
        releaseModel();
        return 1;
    }

    print_stats("sparse_embedding", mout.data.data(), mout.data.size());

    if (is_all_zero(mout.data.data(), mout.data.size())) {
        printf("  WARNING: output is all zeros\n");
    }
    if (has_nan_inf(mout.data.data(), mout.data.size())) {
        printf("  FAIL: output contains NaN or Inf\n");
        releaseModel();
        return 1;
    }

    // --- Done ---
    releaseModel();
    printf("\n========================================\n");
    printf("  RESULT: Part 1 PASS\n");
    printf("========================================\n");
    return 0;
}
```

---

### 4.2 test_part2.cpp — 测试 Transformer

```cpp
// test_part2.cpp
//
// Part 2: Transformer (2x TwoWayAttentionBlock + final attention)
//
// 模型输入:
//   input 0: image_embeddings   float[1][256][64][64]  — encoder 输出
//   input 1: sparse_embedding   float[1][5][256]       — Part 1 输出
//
// 模型输出:
//   output 0: hs               float[1][10][256]       — transformer tokens
//   output 1: src              float[1][4096][256]     — 处理后的 image tokens
//
// 输入总大小: 256*64*64 + 5*256 = 1,049,856 floats = 4,199,424 bytes
// 输出总大小: 10*256 + 4096*256 = 1,051,136 floats = 4,204,544 bytes
//
// 这是计算量最大的部分，包含 46 个 MatMul 和 7 个 Softmax。

#include "npu_test_utils.h"

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "models/part2.tvn";

    printf("========================================\n");
    printf("  Test Part 2: Transformer\n");
    printf("========================================\n");

    // --- Step 1: Load ---
    printf("\n[1] Loading model: %s\n", model_path);
    bool loaded = loadModel(model_path, 0, 0);
    if (!loaded) {
        printf("  FAIL: loadModel returned false\n");
        printf("  RESULT: Part 2 LOAD FAILED\n");
        return 1;
    }
    printf("  OK: model loaded\n");

    // --- Step 2: Prepare input ---
    printf("\n[2] Preparing input\n");

    const int EMBED_DIM = 256;
    const int SPATIAL = 64 * 64;  // 4096
    const int N = 5;

    // image_embeddings: [1][256][64][64]
    auto img_embed = make_random(EMBED_DIM * SPATIAL, 42);  // 1,048,576 floats

    // sparse_embedding: [1][5][256]
    auto sparse_embed = make_random(N * EMBED_DIM, 123);  // 1,280 floats

    // 拼接
    int input_floats = EMBED_DIM * SPATIAL + N * EMBED_DIM;  // 1,049,856
    std::vector<float> input_buf(input_floats);
    memcpy(input_buf.data(), img_embed.data(), EMBED_DIM * SPATIAL * sizeof(float));
    memcpy(input_buf.data() + EMBED_DIM * SPATIAL,
           sparse_embed.data(), N * EMBED_DIM * sizeof(float));

    print_stats("image_embeddings", img_embed.data(), EMBED_DIM * SPATIAL);
    print_stats("sparse_embedding", sparse_embed.data(), N * EMBED_DIM);

    // --- Step 3: Inference ---
    printf("\n[3] Running inference\n");

    model_input min;
    min.data = reinterpret_cast<const unsigned char*>(input_buf.data());

    classifier_output mout;

    auto t0 = std::chrono::high_resolution_clock::now();
    inference(&min, &mout);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Inference time: %.2f ms\n", ms);

    // --- Step 4: Validate output ---
    printf("\n[4] Validating output\n");

    int num_tokens = 1 + 4 + N;  // iou_token(1) + mask_tokens(4) + prompt(5) = 10
    int hs_size = num_tokens * EMBED_DIM;    // 10 * 256 = 2,560
    int src_size = SPATIAL * EMBED_DIM;       // 4096 * 256 = 1,048,576
    int expected_total = hs_size + src_size;  // 1,051,136

    printf("  Output size: %d floats (expected %d)\n",
           (int)mout.data.size(), expected_total);
    printf("    hs expected:  %d floats [1][%d][%d]\n", hs_size, num_tokens, EMBED_DIM);
    printf("    src expected: %d floats [1][%d][%d]\n", src_size, SPATIAL, EMBED_DIM);

    if ((int)mout.data.size() != expected_total) {
        printf("  WARNING: output size mismatch!\n");
    }

    if (mout.data.empty()) {
        printf("  FAIL: output is empty\n");
        printf("  RESULT: Part 2 INFERENCE FAILED\n");
        releaseModel();
        return 1;
    }

    // 拆分输出
    print_stats("hs", mout.data.data(), std::min(hs_size, (int)mout.data.size()));
    if ((int)mout.data.size() > hs_size) {
        print_stats("src", mout.data.data() + hs_size,
                    std::min(src_size, (int)mout.data.size() - hs_size));
    }

    if (has_nan_inf(mout.data.data(), mout.data.size())) {
        printf("  FAIL: output contains NaN or Inf\n");
        releaseModel();
        return 1;
    }

    // --- Done ---
    releaseModel();
    printf("\n========================================\n");
    printf("  RESULT: Part 2 PASS\n");
    printf("========================================\n");
    return 0;
}
```

---

### 4.3 test_part3.cpp — 测试 Mask Head

```cpp
// test_part3.cpp
//
// Part 3: Mask Head (upscaling + hypernetwork MLPs + stability score)
//
// 模型输入:
//   input 0: hs    float[1][10][256]       — transformer token 输出
//   input 1: src   float[1][4096][256]     — 处理后的 image tokens
//
// 模型输出:
//   output 0: scores  float[1][4]              — 4 个候选 mask 的质量分数
//   output 1: masks   float[1][4][256][256]    — 4 个候选 mask
//
// 输入总大小: 10*256 + 4096*256 = 1,051,136 floats = 4,204,544 bytes
// 输出总大小: 4 + 4*256*256 = 262,148 floats = 1,048,592 bytes
//
// 包含 ConvTranspose（上采样）、Gemm（MLP）、Erf（GELU）、
// Sigmoid（stability score）等算子。

#include "npu_test_utils.h"

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "models/part3.tvn";

    printf("========================================\n");
    printf("  Test Part 3: Mask Head\n");
    printf("========================================\n");

    // --- Step 1: Load ---
    printf("\n[1] Loading model: %s\n", model_path);
    bool loaded = loadModel(model_path, 0, 0);
    if (!loaded) {
        printf("  FAIL: loadModel returned false\n");
        printf("  RESULT: Part 3 LOAD FAILED\n");
        return 1;
    }
    printf("  OK: model loaded\n");

    // --- Step 2: Prepare input ---
    printf("\n[2] Preparing input\n");

    const int EMBED_DIM = 256;
    const int SPATIAL = 4096;
    const int NUM_TOKENS = 10;  // 1 + 4 + 5

    // hs: [1][10][256]
    auto hs_data = make_random(NUM_TOKENS * EMBED_DIM, 42);  // 2,560 floats

    // src: [1][4096][256]
    auto src_data = make_random(SPATIAL * EMBED_DIM, 123);  // 1,048,576 floats

    // 拼接
    int input_floats = NUM_TOKENS * EMBED_DIM + SPATIAL * EMBED_DIM;  // 1,051,136
    std::vector<float> input_buf(input_floats);
    memcpy(input_buf.data(), hs_data.data(),
           NUM_TOKENS * EMBED_DIM * sizeof(float));
    memcpy(input_buf.data() + NUM_TOKENS * EMBED_DIM,
           src_data.data(), SPATIAL * EMBED_DIM * sizeof(float));

    print_stats("hs", hs_data.data(), NUM_TOKENS * EMBED_DIM);
    print_stats("src", src_data.data(), SPATIAL * EMBED_DIM);

    // --- Step 3: Inference ---
    printf("\n[3] Running inference\n");

    model_input min;
    min.data = reinterpret_cast<const unsigned char*>(input_buf.data());

    classifier_output mout;

    auto t0 = std::chrono::high_resolution_clock::now();
    inference(&min, &mout);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Inference time: %.2f ms\n", ms);

    // --- Step 4: Validate output ---
    printf("\n[4] Validating output\n");

    int scores_size = 4;
    int masks_size = 4 * 256 * 256;  // 262,144
    int expected_total = scores_size + masks_size;  // 262,148

    printf("  Output size: %d floats (expected %d)\n",
           (int)mout.data.size(), expected_total);
    printf("    scores expected: %d floats [1][4]\n", scores_size);
    printf("    masks expected:  %d floats [1][4][256][256]\n", masks_size);

    if ((int)mout.data.size() != expected_total) {
        printf("  WARNING: output size mismatch!\n");
    }

    if (mout.data.empty()) {
        printf("  FAIL: output is empty\n");
        printf("  RESULT: Part 3 INFERENCE FAILED\n");
        releaseModel();
        return 1;
    }

    // 拆分输出
    printf("\n  Scores (first 4 values = quality of 4 candidate masks):\n    ");
    for (int i = 0; i < 4 && i < (int)mout.data.size(); i++) {
        printf("%.6f ", mout.data[i]);
    }
    printf("\n");

    if ((int)mout.data.size() > scores_size) {
        print_stats("masks", mout.data.data() + scores_size,
                    std::min(masks_size, (int)mout.data.size() - scores_size));
    }

    if (has_nan_inf(mout.data.data(), mout.data.size())) {
        printf("  FAIL: output contains NaN or Inf\n");
        releaseModel();
        return 1;
    }

    // 额外检查: scores 应该在 [0, 1] 范围内（stability score）
    bool scores_valid = true;
    for (int i = 0; i < 4 && i < (int)mout.data.size(); i++) {
        if (mout.data[i] < 0.0f || mout.data[i] > 1.0f) {
            scores_valid = false;
        }
    }
    if (!scores_valid) {
        printf("  WARNING: scores outside [0, 1] range\n");
    }

    // --- Done ---
    releaseModel();
    printf("\n========================================\n");
    printf("  RESULT: Part 3 PASS\n");
    printf("========================================\n");
    return 0;
}
```

---

## 5. 执行步骤

```bash
# 编译
mkdir build && cd build
cmake .. && make

# 逐个测试
./test_part1 ../models/part1.tvn
./test_part2 ../models/part2.tvn
./test_part3 ../models/part3.tvn
```

## 6. 结果判读

| Part 1 | Part 2 | Part 3 | 结论 |
|--------|--------|--------|------|
| PASS | PASS | PASS | 三段均可用，问题出在合并模型的 TVN 大小或内存限制 |
| FAIL | PASS | PASS | Part 1 有问题，检查 loadModel 的 width/height 参数含义 |
| PASS | FAIL | PASS | Part 2 有问题，可能是模型太大（约 4MB 输入 + 4MB 输出） |
| PASS | PASS | FAIL | Part 3 有问题，检查 ConvTranspose/Gemm 算子的 NPU 支持 |
| FAIL | FAIL | FAIL | 库函数本身有问题，或 TVN 格式不兼容 |
| LOAD FAIL | — | — | loadModel 失败，检查 TVN 文件路径、格式、内存 |

### LOAD FAILED 排查方向

1. **TVN 文件大小**：合并 decoder TVN 可能超过 TV 设备的 NPU 内存限制。对比单段 TVN 文件大小。
2. **loadModel 参数**：`width` 和 `height` 参数可能需要与模型输入尺寸匹配。不同模型的输入尺寸不同，见下表。
3. **多模型并发**：TV 的 NPU 可能不支持同时加载多个模型。测试时先 releaseModel 再加载下一个。

### loadModel 的 width / height 参数参考

库函数 `loadModel(path, width, height)` 的 width/height 可能指输入 tensor 的空间维度。每段模型的情况不同：

| 模型 | 输入 tensor | 可能的 width | 可能的 height | 备注 |
|------|-----------|------------|-------------|------|
| Encoder | `[1,3,1024,1024]` | 1024 | 1024 | 图像空间尺寸 |
| Part 1 | `[1,5,256]` + `[1,5]` | 256 或 0 | 5 或 0 | 无空间维度 |
| Part 2 | `[1,256,64,64]` + `[1,5,256]` | 64 | 64 | image_embeddings 空间 |
| Part 3 | `[1,10,256]` + `[1,4096,256]` | 256 或 0 | 10 或 0 | 无空间维度 |

如果 loadModel 不依赖 width/height（自动从 TVN 读取 shape），可以统一传 0 或 1。

---

## 7. 关于库函数改造的说明

测试代码中，输入数据的传递方式（拼接为单个 buffer）是基于现有库函数只接受单个 `const unsigned char* data` 的限制。如果库函数已被改造为支持多输入（参见 `npu_tv_deployment_guide.md` 第 5 节方案 B），则每个输入应分别传递，不需要拼接。

对于多输出同理：如果库函数的 `classifier_output` 只有一个 `vector<float>`，则两个输出会被拼接在一起，需要按已知的 size 拆分。如果已支持多输出，则直接分别获取。

**关键**：拆分/拼接的顺序必须与 ONNX 模型定义的输入/输出顺序一致。输入输出顺序在本文档第 4 节每个测试程序的头部注释中有标注。
