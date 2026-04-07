# EdgeSAM NPU TV 部署指南

本文档指导在 TV 设备上用 C++ 部署 EdgeSAM NPU 模型，实现输入一张图片和 point prompt，输出 segmentation mask。

## 1. 模型文件

需要两个 TVN 模型文件（由 ONNX 编译得到）：

| 文件 | 来源 ONNX | 功能 |
|------|----------|------|
| `encoder.tvn` | `edge_sam_3x_encoder_npu.onnx` | 图像编码器（RepViT CNN） |
| `decoder.tvn` | `edge_sam_3x_decoder_npu.onnx` | 合并 Decoder（Prompt Embedding + Transformer + Mask Head） |

导出命令：
```bash
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth --decoder --use-stability-score
```

---

## 2. 端到端推理流水线

```
输入: image (HxW BGR/RGB uint8), point_coords (N个点的x,y), point_labels (N个标签)

Step 1. 图像预处理 (CPU)         → float[1, 3, 1024, 1024]
Step 2. Encoder 推理 (NPU)       → float[1, 256, 64, 64]
Step 3. 坐标变换 (CPU)           → float[1, N, 2]
Step 4. PE 位置编码计算 (CPU)     → float[1, N, 256]
Step 5. Decoder 推理 (NPU)       → scores float[1, 4] + masks float[1, 4, 256, 256]
Step 6. 后处理 (CPU)             → 最终二值 mask (原图尺寸)
```

**关键点**：Step 4 的 PE 计算必须在 CPU 端完成，因为 NPU 不支持 Sin/Cos 算子，这部分已从 ONNX 模型中移出。

---

## 3. 各步骤详细实现

### Step 1: 图像预处理

输入：`uint8 image[H][W][3]`（BGR 或 RGB 均可，但需与 normalize 的 mean/std 通道顺序匹配）

**1a. Resize（保持长边比例缩放到 1024）**

```
target_length = 1024
scale = 1024.0 / max(orig_h, orig_w)
new_h = round(orig_h * scale)
new_w = round(orig_w * scale)
// 使用双线性插值 resize 到 (new_h, new_w)
```

注意：`new_h` 和 `new_w` 需要记录下来，后续 Step 3 坐标变换和 Step 6 后处理都要用。

**1b. Normalize（ImageNet 标准化，RGB 通道顺序）**

```
// RGB 通道的 mean 和 std
pixel_mean[3] = {123.675, 116.28, 103.53}
pixel_std[3]  = {58.395,  57.12,  57.375}

// 对每个像素
for each pixel (y, x):
    for c in 0..2:
        output[c][y][x] = (float(input[y][x][c]) - pixel_mean[c]) / pixel_std[c]
```

如果输入是 BGR，需要先转换为 RGB（交换第 0 和第 2 通道），或者将 mean/std 的顺序调整为 BGR：
```
// BGR 通道的 mean 和 std
pixel_mean_bgr[3] = {103.53, 116.28, 123.675}
pixel_std_bgr[3]  = {57.375, 57.12,  58.395}
```

**1c. Pad（右侧和底部补零到 1024x1024）**

```
// 输出 tensor: float[1][3][1024][1024]，先全部填 0
// 将 normalized 图像放在左上角 [0:new_h, 0:new_w]
// 右侧 pad (1024 - new_w) 列，底部 pad (1024 - new_h) 行，值为 0.0
```

**最终输出**：`float[1][3][1024][1024]`，NCHW 格式，内存连续。

总计 `1 * 3 * 1024 * 1024 * 4 = 12,582,912 bytes`。

---

### Step 2: Encoder 推理 (NPU)

| | 名称 | 类型 | 形状 | 字节数 |
|--|------|------|------|--------|
| 输入 | `image` | FLOAT32 | [1, 3, 1024, 1024] | 12,582,912 |
| 输出 | `image_embeddings` | FLOAT32 | [1, 256, 64, 64] | 4,194,304 |

将 Step 1 输出的 float 数据 reinterpret_cast 为 `const unsigned char*` 传给 `inference()`。

输出的 `image_embeddings` 需要保存，Step 5 Decoder 推理时使用。

---

### Step 3: 坐标变换

将原始图像坐标转换为模型坐标空间（即 resize 后的坐标）。

```
// 输入: point_coords[N][2]，每个点是 (x, y)，原图像素坐标
// 输出: transformed_coords[N][2]，模型空间坐标

scale = 1024.0 / max(orig_h, orig_w)
new_h = round(orig_h * scale)
new_w = round(orig_w * scale)

for i in 0..N-1:
    transformed_coords[i][0] = point_coords[i][0] * (new_w / (float)orig_w)  // x
    transformed_coords[i][1] = point_coords[i][1] * (new_h / (float)orig_h)  // y
```

---

### Step 4: PE 位置编码计算 (CPU)

**这是最关键的一步**，必须在 CPU 端实现，NPU 模型不包含这部分计算。

#### 4a. 提取高斯矩阵

需要从 PyTorch checkpoint 中提取一个常量矩阵 `positional_encoding_gaussian_matrix`，形状为 `float[2][128]`。

提取方法（Python，只需执行一次）：

```python
import torch
import numpy as np
from edge_sam import sam_model_registry

sam = sam_model_registry["edge_sam"](checkpoint="weights/edge_sam_3x.pth")
matrix = sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.numpy()
# matrix.shape = (2, 128)

# 保存为二进制文件供 C++ 加载
matrix.astype(np.float32).tofile("pe_gaussian_matrix.bin")
# 文件大小: 2 * 128 * 4 = 1024 bytes
```

#### 4b. C++ 端 PE 计算逻辑

```
输入:
  transformed_coords[N][2]  — Step 3 的输出（float，模型空间坐标）
  gaussian_matrix[2][128]   — 从 checkpoint 提取的常量

输出:
  point_embedding_pe[1][N][256]  — 位置编码

计算过程（对每个点 i = 0..N-1）:

  // 1. 归一化到 [0, 1]
  norm_x = (transformed_coords[i][0] + 0.5) / 1024.0
  norm_y = (transformed_coords[i][1] + 0.5) / 1024.0

  // 2. 映射到 [-1, 1]
  norm_x = 2.0 * norm_x - 1.0
  norm_y = 2.0 * norm_y - 1.0

  // 3. 矩阵乘法: [1, 2] @ [2, 128] = [1, 128]
  for j in 0..127:
      proj[j] = norm_x * gaussian_matrix[0][j] + norm_y * gaussian_matrix[1][j]

  // 4. 缩放
  for j in 0..127:
      proj[j] = proj[j] * 2.0 * PI    // PI = 3.14159265358979323846

  // 5. Sin/Cos 拼接得到 256 维向量
  for j in 0..127:
      point_embedding_pe[0][i][j]       = sin(proj[j])   // 前 128 维
      point_embedding_pe[0][i][j + 128] = cos(proj[j])   // 后 128 维
```

注意 `point_embedding_pe` 的内存布局为 `float[1][N][256]`，连续存储。

---

### Step 5: Decoder 推理 (NPU)

Decoder 有 3 个输入和 2 个输出，需要修改库函数以支持多输入多输出（见第 5 节）。

#### 输入

| 序号 | 名称 | 类型 | 形状 | 字节数 | 来源 |
|------|------|------|------|--------|------|
| 0 | `image_embeddings` | FLOAT32 | [1, 256, 64, 64] | 4,194,304 | Step 2 输出 |
| 1 | `point_embedding_pe` | FLOAT32 | [1, N, 256] | N * 1024 | Step 4 输出 |
| 2 | `point_labels` | FLOAT32 | [1, N] | N * 4 | 用户输入 |

**N 的默认值为 5**（导出时 `--num-points 5`），这是静态 shape，不能动态改变。如果实际 prompt 点数少于 5，需要 padding（见下文）。

#### Point Labels 编码

```
label 值含义:
  -1 = padding 点（无效，填充用）
   0 = 背景点（负例）
   1 = 前景点（正例）
   2 = box 左上角
   3 = box 右下角
```

**Padding 规则**：如果实际点数 < N，用 `label = -1` 填充剩余位置，对应的坐标填 `(0.0, 0.0)`。

示例：用户点击了 1 个前景点 (500, 375)，N=5 时：
```
transformed_coords = [(500*scale, 375*scale), (0,0), (0,0), (0,0), (0,0)]
point_labels        = [1.0, -1.0, -1.0, -1.0, -1.0]
```

注意 `point_labels` 的数据类型是 **FLOAT32**，不是整数。

#### 输出

| 序号 | 名称 | 类型 | 形状 | 字节数 | 含义 |
|------|------|------|------|--------|------|
| 0 | `scores` | FLOAT32 | [1, 4] | 16 | 4 个候选 mask 的质量分数 |
| 1 | `masks` | FLOAT32 | [1, 4, 256, 256] | 1,048,576 | 4 个候选 mask（低分辨率） |

---

### Step 6: 后处理

```
// 1. 选择最佳 mask
best_idx = argmax(scores[0..3])

// 2. 提取对应的 mask: float[256][256]
best_mask = masks[0][best_idx]  // 偏移: best_idx * 256 * 256

// 3. 双线性插值放大到 1024x1024
mask_1024 = bilinear_resize(best_mask, 1024, 1024)

// 4. 裁剪 padding 区域（只保留实际图像区域）
mask_cropped = mask_1024[0:new_h][0:new_w]   // new_h, new_w 来自 Step 1

// 5. 双线性插值到原图尺寸
mask_final = bilinear_resize(mask_cropped, orig_h, orig_w)

// 6. 二值化
for each pixel:
    binary_mask[y][x] = (mask_final[y][x] > 0.0) ? 1 : 0
```

阈值为 `0.0`（mask 值 > 0 表示前景）。

---

## 4. 数据内存布局

所有 tensor 均为 **NCHW** 格式（batch, channel, height, width），**行优先**（C-contiguous），FLOAT32。

内存中的元素顺序示例（`[1, 256, 64, 64]`）：
```
// tensor[n][c][h][w] 的地址 = base + ((n * C + c) * H + h) * W + w
// 连续遍历: n=0 → c=0..255 → h=0..63 → w=0..63
```

传给 `inference()` 时，将 `float*` 强制转换为 `const unsigned char*`，按总字节数传递。

---

## 5. 库函数改造建议

现有库函数接口过于简单，无法满足多输入多输出需求。建议做以下改造：

### 现有接口

```cpp
bool loadModel(string path, int width, int height);
void releaseModel();
void inference(model_input *model_in, classifier_output *res);
// model_input: const unsigned char* data
// classifier_output: vector<float>
```

### 建议改造方案

**方案 A：最小改动 — 保持单 input/output，拼接数据**

Encoder 可以直接使用现有接口（单输入单输出）。

Decoder 需要将 3 个输入拼接为一个连续 buffer，在库内部按偏移量拆分：

```cpp
// Decoder 输入拼接（调用端）
// 总大小 = image_embeddings + point_embedding_pe + point_labels
// N=5 时: 4194304 + 5120 + 20 = 4,199,444 bytes

float* decoder_input = malloc(total_size);
memcpy(decoder_input + 0,              image_embeddings, 4194304);  // [1,256,64,64]
memcpy(decoder_input + 4194304/4,      point_embedding_pe, 5120);   // [1,5,256]
memcpy(decoder_input + 4194304/4+5120/4, point_labels, 20);        // [1,5]

// 库内部需要知道如何拆分这个 buffer 绑定到模型的 3 个输入 tensor
```

输出类似：`vector<float>` 前 4 个元素是 scores，后 `4*256*256=262144` 个元素是 masks。

```cpp
// 解析输出
float scores[4] = {res[0], res[1], res[2], res[3]};
float* masks = &res[4];  // masks[mask_idx * 256 * 256 + h * 256 + w]
```

**方案 B：推荐 — 扩展为多输入多输出**

```cpp
struct tensor_data {
    const float* data;     // 数据指针
    int dims[4];           // 各维度大小，如 {1, 256, 64, 64}
    int ndim;              // 维度数量
};

bool loadModel(const string& path);
void releaseModel();

// 多输入多输出推理
void inference(
    const vector<tensor_data>& inputs,
    vector<vector<float>>& outputs
);
```

调用示例：
```cpp
// Encoder
vector<tensor_data> enc_inputs = {
    {image_data, {1, 3, 1024, 1024}, 4}
};
vector<vector<float>> enc_outputs;
inference(enc_inputs, enc_outputs);
// enc_outputs[0] = image_embeddings, size = 1*256*64*64 = 1048576

// Decoder
vector<tensor_data> dec_inputs = {
    {image_embeddings,   {1, 256, 64, 64}, 4},
    {point_embedding_pe, {1, 5, 256},      3},
    {point_labels,       {1, 5},           2}
};
vector<vector<float>> dec_outputs;
inference(dec_inputs, dec_outputs);
// dec_outputs[0] = scores, size = 4
// dec_outputs[1] = masks,  size = 4*256*256 = 262144
```

---

## 6. 完整代码结构建议

```
edgesam_npu/
├── CMakeLists.txt
├── include/
│   ├── npu_model.h           // 模型加载/推理封装（改造后的库）
│   ├── edgesam_pipeline.h    // EdgeSAM 端到端 pipeline
│   └── pe_encoding.h         // CPU 端 PE 位置编码计算
├── src/
│   ├── npu_model.cpp          // 库函数实现（需改造）
│   ├── edgesam_pipeline.cpp   // 主 pipeline 实现
│   ├── pe_encoding.cpp        // PE 计算实现
│   ├── image_preprocess.cpp   // 图像预处理（resize, normalize, pad）
│   └── main.cpp               // 入口
└── assets/
    ├── encoder.tvn            // Encoder 模型
    ├── decoder.tvn            // Decoder 模型
    └── pe_gaussian_matrix.bin // PE 高斯矩阵（2x128 float, 1024 bytes）
```

### edgesam_pipeline 伪代码

```cpp
class EdgeSAMPipeline {
private:
    NpuModel encoder_;
    NpuModel decoder_;
    float gaussian_matrix_[2][128];  // 从 .bin 文件加载
    int num_points_ = 5;            // 必须与 ONNX 导出时的 --num-points 一致

    vector<float> image_embeddings_; // 缓存 encoder 输出

public:
    bool init(string encoder_path, string decoder_path, string pe_matrix_path) {
        encoder_.loadModel(encoder_path);
        decoder_.loadModel(decoder_path);
        load_pe_matrix(pe_matrix_path, gaussian_matrix_);
        return true;
    }

    // 设置图像（只需调用一次，可对同一张图做多次 prompt 推理）
    void setImage(const uint8_t* image_bgr, int orig_h, int orig_w) {
        // Step 1: 预处理
        float preprocessed[1 * 3 * 1024 * 1024];
        preprocess(image_bgr, orig_h, orig_w, preprocessed);

        // Step 2: Encoder 推理
        image_embeddings_.resize(1 * 256 * 64 * 64);
        encoder_.inference(preprocessed, image_embeddings_.data());
    }

    // 执行 prompt 推理
    void predict(
        const float* point_coords_xy,  // [num_actual_points][2], 原图坐标
        const float* point_labels,      // [num_actual_points]
        int num_actual_points,
        int orig_h, int orig_w,
        float* output_mask,             // [orig_h][orig_w] 输出
        float* output_score             // 最佳分数
    ) {
        // Step 3: 坐标变换
        float transformed[5][2] = {};
        float labels_padded[5];
        fill(labels_padded, labels_padded + 5, -1.0f);  // 默认 padding

        float scale = 1024.0f / max(orig_h, orig_w);
        int new_h = round(orig_h * scale);
        int new_w = round(orig_w * scale);

        for (int i = 0; i < num_actual_points && i < 5; i++) {
            transformed[i][0] = point_coords_xy[i * 2]     * new_w / (float)orig_w;
            transformed[i][1] = point_coords_xy[i * 2 + 1] * new_h / (float)orig_h;
            labels_padded[i] = point_labels[i];
        }

        // Step 4: PE 位置编码
        float pe[1][5][256];
        compute_pe(transformed, 5, gaussian_matrix_, pe);

        // Step 5: Decoder 推理
        float scores[4];
        float masks[4 * 256 * 256];
        decoder_.inference(image_embeddings_, pe, labels_padded, scores, masks);

        // Step 6: 后处理
        int best = argmax(scores, 4);
        *output_score = scores[best];
        postprocess(&masks[best * 256 * 256], new_h, new_w, orig_h, orig_w, output_mask);
    }
};
```

---

## 7. 常量速查表

| 常量 | 值 | 用途 |
|------|----|------|
| 模型输入图像尺寸 | 1024 x 1024 | Resize + Pad 目标 |
| Encoder 输出空间尺寸 | 64 x 64 | image_embeddings 空间维度 |
| Embedding 维度 | 256 | 所有 embedding 的通道数 |
| Mask 输出尺寸 | 256 x 256 | 低分辨率 mask（需放大到原图） |
| 候选 Mask 数量 | 4 | scores 和 masks 的第一维 |
| PE 高斯矩阵形状 | [2, 128] | positional_encoding_gaussian_matrix |
| 默认 Prompt 点数 (N) | 5 | 静态 shape，编译时固定 |
| Mask 二值化阈值 | 0.0 | mask 值 > 0 为前景 |
| pixel_mean (RGB) | [123.675, 116.28, 103.53] | ImageNet 均值 |
| pixel_std (RGB) | [58.395, 57.12, 57.375] | ImageNet 标准差 |
| PI | 3.14159265358979323846 | PE 计算中的 2*PI 系数 |

---

## 8. 调试建议

1. **验证 Encoder 输出**：可用 Python 加载同一张图，对比 PyTorch encoder 输出和 NPU encoder 输出，误差应 < 0.01。

2. **验证 PE 计算**：用 Python 对比：
   ```python
   from scripts.export_onnx_model_npu import compute_point_pe
   pe_python = compute_point_pe(sam, torch.tensor([[[500.0, 375.0]]]))
   # 与 C++ 实现的结果逐元素对比
   ```

3. **验证 Decoder 输出**：先用 onnxruntime 在 CPU 上跑 decoder ONNX，对比 NPU 输出。

4. **常见错误**：
   - 坐标顺序：point_coords 是 **(x, y)** 格式，不是 (row, col)
   - PE 计算遗漏 `+0.5` 偏移或 `2*coords - 1` 映射
   - point_labels 忘记用 float 类型
   - Padding 点的 label 没有设为 -1.0
   - 输出 mask 的 resize 顺序错误（应先放大到 1024x1024，再裁剪，再 resize 到原图）
