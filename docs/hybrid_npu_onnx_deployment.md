# 方案 A：Encoder NPU + Decoder ONNX Runtime 混合部署

## 背景

EdgeSAM decoder 的 TVN 模型虽然能通过 NPU 编译，但在 TV 设备上无法加载。作为 fallback 方案，encoder 仍在 NPU 上运行（TVN），decoder 改用 ONNX Runtime 在 CPU 上运行。

## 架构

```
输入: image (HxW, uint8), point_coords (N 个点), point_labels (N 个标签)

Step 1. 图像预处理 (CPU)         → float[1, 3, 1024, 1024]
Step 2. Encoder 推理 (NPU/TVN)   → float[1, 256, 64, 64]
Step 3. 坐标变换 (CPU)           → float[1, N, 2]
Step 4. Decoder 推理 (CPU/ONNX)  → scores float[1, 4] + masks float[1, 4, 256, 256]
Step 5. 后处理 (CPU)             → 二值 mask (原图尺寸)
```

**与全 NPU 方案的差异**：decoder 在 CPU 上跑 ONNX Runtime，不需要 PE 预计算（sin/cos 在 CPU 上没问题），不需要 pe_gaussian_matrix.bin，使用原始 decoder ONNX（非 NPU 版本）。

---

## 1. 模型文件

| 文件 | 运行环境 | 来源 | 导出命令 |
|------|---------|------|---------|
| `encoder.tvn` | NPU | `edge_sam_3x_encoder_npu.onnx` 编译 | `python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth` |
| `edge_sam_3x_decoder.onnx` | CPU (ONNX Runtime) | 直接使用 ONNX 文件 | `python scripts/export_onnx_model.py weights/edge_sam_3x.pth --decoder --use-stability-score` |

注意 decoder 用的是 **原始导出脚本** `export_onnx_model.py`（不是 `_npu` 版本），因为 ONNX Runtime 支持所有标准算子，不需要 NPU 的算子替换。

---

## 2. 依赖

需要在 TV 设备上编译安装 ONNX Runtime C++ 库。

### 获取 ONNX Runtime

- 官方预编译包（推荐 aarch64 Linux 版本）：从 https://github.com/microsoft/onnxruntime/releases 下载
- 或者交叉编译 ONNX Runtime for TV 平台

### CMakeLists.txt 示例

```cmake
cmake_minimum_required(VERSION 3.14)
project(edgesam_hybrid)

set(CMAKE_CXX_STANDARD 17)

# NPU 库（已有）
find_library(NPU_LIB npu_inference PATHS /path/to/npu/lib)

# ONNX Runtime
set(ONNXRUNTIME_DIR "/path/to/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(edgesam_hybrid
    src/main.cpp
    src/image_preprocess.cpp
    src/npu_encoder.cpp
    src/onnx_decoder.cpp
    src/postprocess.cpp
)

target_link_libraries(edgesam_hybrid
    ${NPU_LIB}
    onnxruntime
)
```

---

## 3. 项目结构

```
edgesam_hybrid/
├── CMakeLists.txt
├── include/
│   ├── npu_encoder.h       // NPU encoder 封装
│   ├── onnx_decoder.h      // ONNX Runtime decoder 封装
│   ├── image_preprocess.h  // 图像预处理
│   └── postprocess.h       // 后处理
├── src/
│   ├── main.cpp
│   ├── npu_encoder.cpp
│   ├── onnx_decoder.cpp
│   ├── image_preprocess.cpp
│   └── postprocess.cpp
└── models/
    ├── encoder.tvn                  // NPU encoder
    └── edge_sam_3x_decoder.onnx     // ONNX decoder
```

---

## 4. 各模块实现

### 4.1 NPU Encoder

使用现有 TVN 库函数加载和推理 encoder，接口无需改动。

```cpp
// npu_encoder.h
class NpuEncoder {
public:
    bool init(const std::string& tvn_path);
    // 输入: float[1][3][1024][1024] 预处理后的图像
    // 输出: float[1][256][64][64] image embeddings
    bool run(const float* input_image, std::vector<float>& output_embeddings);
    void release();
};
```

**输入**：`float[1][3][1024][1024]`，共 3,145,728 个 float，12,582,912 bytes。
传入库函数时 reinterpret_cast 为 `const unsigned char*`。

**输出**：`float[1][256][64][64]`，共 1,048,576 个 float，4,194,304 bytes。
从 `vector<float>` 中获取，size 应为 1,048,576。

### 4.2 图像预处理

与全 NPU 方案完全相同，详见 `docs/npu_tv_deployment_guide.md` 的 Step 1。

```cpp
// image_preprocess.h
struct PreprocessResult {
    std::vector<float> tensor;  // float[1][3][1024][1024]
    int orig_h, orig_w;         // 原始图像尺寸
    int new_h, new_w;           // resize 后的尺寸（pad 前）
};

// 输入: BGR uint8 图像
// 输出: normalized + padded float tensor
PreprocessResult preprocess_image(const uint8_t* image_bgr, int height, int width);
```

实现步骤：
1. **Resize**：`scale = 1024.0 / max(h, w)`，双线性插值到 `(new_h, new_w)`
2. **Normalize**（RGB 顺序）：`pixel = (pixel - mean) / std`
   - mean = `[123.675, 116.28, 103.53]`
   - std = `[58.395, 57.12, 57.375]`
   - 如果输入是 BGR，需要交换 R 和 B 通道
3. **Pad**：右侧和底部补 0 到 `1024x1024`
4. **转为 NCHW**：`output[c][y][x] = normalized_pixel`

### 4.3 ONNX Runtime Decoder

这是本方案的核心模块。使用 ONNX Runtime C++ API 加载和运行 decoder。

```cpp
// onnx_decoder.h
#include <onnxruntime_cxx_api.h>

class OnnxDecoder {
public:
    bool init(const std::string& onnx_path);

    // 输入:
    //   image_embeddings: float[1][256][64][64]  — 来自 NPU encoder
    //   point_coords:     float[1][N][2]         — 变换后的坐标
    //   point_labels:     float[1][N]            — 标签
    // 输出:
    //   scores: float[1][4]
    //   masks:  float[1][4][256][256]
    bool run(const float* image_embeddings,
             const float* point_coords,
             const float* point_labels,
             int num_points,
             std::vector<float>& out_scores,
             std::vector<float>& out_masks);

    void release();

private:
    Ort::Env env_;
    Ort::Session* session_ = nullptr;
    Ort::AllocatorWithDefaultOptions allocator_;
};
```

#### ONNX Runtime C++ API 核心用法

```cpp
#include <onnxruntime_cxx_api.h>

bool OnnxDecoder::init(const std::string& onnx_path) {
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "edgesam_decoder");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(4);  // CPU 线程数，根据 TV 平台调整
    session_ = new Ort::Session(env_, onnx_path.c_str(), opts);
    return true;
}

bool OnnxDecoder::run(
    const float* image_embeddings,
    const float* point_coords,
    const float* point_labels,
    int num_points,
    std::vector<float>& out_scores,
    std::vector<float>& out_masks
) {
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // --- 创建输入 tensors ---
    // 输入 0: image_embeddings [1, 256, 64, 64]
    std::vector<int64_t> embed_shape = {1, 256, 64, 64};
    Ort::Value embed_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(image_embeddings),
        1 * 256 * 64 * 64,
        embed_shape.data(), embed_shape.size()
    );

    // 输入 1: point_coords [1, N, 2]
    std::vector<int64_t> coords_shape = {1, (int64_t)num_points, 2};
    Ort::Value coords_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(point_coords),
        1 * num_points * 2,
        coords_shape.data(), coords_shape.size()
    );

    // 输入 2: point_labels [1, N]
    std::vector<int64_t> labels_shape = {1, (int64_t)num_points};
    Ort::Value labels_tensor = Ort::Value::CreateTensor<float>(
        mem_info, const_cast<float*>(point_labels),
        1 * num_points,
        labels_shape.data(), labels_shape.size()
    );

    // --- 输入/输出名称 ---
    const char* input_names[] = {"image_embeddings", "point_coords", "point_labels"};
    const char* output_names[] = {"scores", "masks"};

    // --- 推理 ---
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(embed_tensor));
    input_tensors.push_back(std::move(coords_tensor));
    input_tensors.push_back(std::move(labels_tensor));

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 3,
        output_names, 2
    );

    // --- 取出输出 ---
    // scores: [1, 4]
    float* scores_ptr = outputs[0].GetTensorMutableData<float>();
    out_scores.assign(scores_ptr, scores_ptr + 4);

    // masks: [1, 4, 256, 256]
    float* masks_ptr = outputs[1].GetTensorMutableData<float>();
    out_masks.assign(masks_ptr, masks_ptr + 4 * 256 * 256);

    return true;
}
```

### 4.4 坐标变换

```cpp
// 原始坐标 (x, y) → 模型空间坐标
// point_coords 格式: float[N][2]，每个点是 (x, y)
void transform_coords(
    const float* raw_coords,    // [N][2] 原始图像像素坐标
    int num_points,
    int orig_h, int orig_w,
    float* out_coords           // [N][2] 模型空间坐标
) {
    float scale = 1024.0f / std::max(orig_h, orig_w);
    int new_w = (int)(orig_w * scale + 0.5f);
    int new_h = (int)(orig_h * scale + 0.5f);

    for (int i = 0; i < num_points; i++) {
        out_coords[i * 2]     = raw_coords[i * 2]     * ((float)new_w / orig_w);  // x
        out_coords[i * 2 + 1] = raw_coords[i * 2 + 1] * ((float)new_h / orig_h);  // y
    }
}
```

### 4.5 后处理

```cpp
// 输入: scores[4], masks[4][256][256]
// 输出: binary_mask[orig_h][orig_w]
void postprocess(
    const float* scores,
    const float* masks,
    int new_h, int new_w,      // resize 后的尺寸（Step 1 中记录）
    int orig_h, int orig_w,
    uint8_t* output_mask       // 输出二值 mask
) {
    // 1. 选最佳 mask
    int best = 0;
    for (int i = 1; i < 4; i++) {
        if (scores[i] > scores[best]) best = i;
    }

    // 2. 取对应 mask: float[256][256]
    const float* best_mask = masks + best * 256 * 256;

    // 3. 双线性插值到 1024x1024
    // 4. 裁剪到 [new_h][new_w]（去掉 padding）
    // 5. 双线性插值到 [orig_h][orig_w]
    // 6. 二值化: > 0.0 为前景 (1), 否则为背景 (0)
}
```

### 4.6 主程序

```cpp
// main.cpp
int main(int argc, char** argv) {
    // 参数: <encoder.tvn> <decoder.onnx> <image_path> <point_x> <point_y> <label>
    const char* encoder_path = argv[1];
    const char* decoder_path = argv[2];
    const char* image_path = argv[3];

    // 解析 prompt points（支持多个点）
    int num_points = (argc - 4) / 3;  // 每个点: x y label
    std::vector<float> raw_coords(num_points * 2);
    std::vector<float> raw_labels(num_points);
    for (int i = 0; i < num_points; i++) {
        raw_coords[i * 2]     = atof(argv[4 + i * 3]);      // x
        raw_coords[i * 2 + 1] = atof(argv[4 + i * 3 + 1]);  // y
        raw_labels[i]          = atof(argv[4 + i * 3 + 2]);  // label
    }

    // 初始化
    NpuEncoder encoder;
    encoder.init(encoder_path);

    OnnxDecoder decoder;
    decoder.init(decoder_path);

    // 读取图像（用 stb_image 或 OpenCV）
    int orig_h, orig_w;
    uint8_t* image_bgr = load_image(image_path, &orig_h, &orig_w);

    // Step 1: 预处理
    auto prep = preprocess_image(image_bgr, orig_h, orig_w);

    // Step 2: Encoder (NPU)
    std::vector<float> image_embeddings;
    encoder.run(prep.tensor.data(), image_embeddings);

    // Step 3: 坐标变换
    std::vector<float> transformed_coords(num_points * 2);
    transform_coords(raw_coords.data(), num_points,
                     orig_h, orig_w, transformed_coords.data());

    // Step 4: Decoder (ONNX Runtime CPU)
    std::vector<float> scores, masks;
    decoder.run(image_embeddings.data(),
                transformed_coords.data(),
                raw_labels.data(),
                num_points,
                scores, masks);

    // Step 5: 后处理
    std::vector<uint8_t> output_mask(orig_h * orig_w);
    postprocess(scores.data(), masks.data(),
                prep.new_h, prep.new_w,
                orig_h, orig_w,
                output_mask.data());

    // 保存结果
    save_mask(output_mask.data(), orig_h, orig_w, "output_mask.bin");

    encoder.release();
    decoder.release();
    return 0;
}
```

---

## 5. Point Labels 说明

```
label  含义
 -1    padding 点（无效）
  0    背景点（负例）
  1    前景点（正例）
  2    box 左上角
  3    box 右下角
```

**ONNX Runtime 版本的 decoder 支持动态点数**（导出时设了 `dynamic_axes`），不需要固定 N=5 或 padding。传几个点就是几个点。

---

## 6. 与全 NPU 方案的关键区别

| 项目 | 全 NPU 方案 | 混合方案（本文档） |
|------|-----------|----------------|
| Encoder | NPU (TVN) | NPU (TVN) |
| Decoder | NPU (TVN) | **CPU (ONNX Runtime)** |
| Decoder ONNX 文件 | `_decoder_npu.onnx`（NPU 优化版） | **`_decoder.onnx`（原始版）** |
| PE 预计算 | 需要（CPU 端 sin/cos） | **不需要**（ONNX 内部包含） |
| pe_gaussian_matrix.bin | 需要 | **不需要** |
| 点数 N | 固定（静态 shape） | **动态**（可变点数） |
| Decoder 速度 | 快（NPU 加速） | 较慢（CPU） |
| 输入接口 | `(image_embeddings, point_embedding_pe, point_labels)` | **`(image_embeddings, point_coords, point_labels)`** |

最大简化点：不需要 PE 预计算，不需要高斯矩阵文件，不需要 padding 到固定点数。

---

## 7. 调试建议

1. **先单独测试 ONNX Runtime decoder**：用 Python onnxruntime 跑同一个 ONNX 文件，记录输入输出，作为 C++ 的参考基准。

2. **先硬编码测试**：在 main.cpp 中先用固定的 dummy 输入（全零或随机），确认 ONNX Runtime 能正常 load 和 run，再接入真实数据。

3. **确认 ONNX Runtime 链接正确**：TV 平台可能需要特定的 ABI 版本。运行时检查 `Ort::GetApiBase()->GetApi(ORT_API_VERSION)` 是否成功。

4. **性能调优**：
   - `SetIntraOpNumThreads(N)` 控制 CPU 并行度
   - 如果 TV 有 GPU，可以尝试 ONNX Runtime CUDA/OpenCL EP
   - Encoder 是计算瓶颈（CNN），decoder 在 CPU 上的延迟通常可接受
