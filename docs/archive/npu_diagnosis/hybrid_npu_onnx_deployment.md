# 方案 A：Encoder NPU + Decoder ONNX Runtime 混合部署

Decoder TVN 在 TV 上无法加载。Encoder 仍用 NPU (TVN)，Decoder 改用 ONNX Runtime 在 CPU 上运行。

## 流水线

```
Step 1. 图像预处理 (CPU)         → float[1, 3, 1024, 1024]
Step 2. Encoder 推理 (NPU/TVN)   → float[1, 256, 64, 64]
Step 3. 坐标变换 (CPU)           → float[1, N, 2]
Step 4. Decoder 推理 (CPU/ONNX)  → scores float[1,4] + masks float[1,4,256,256]
Step 5. 后处理 (CPU)             → 二值 mask (原图尺寸)
```

## 模型文件

| 文件 | 运行环境 | 导出命令 |
|------|---------|---------|
| `encoder.tvn` | NPU | `python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth` → 编译为 TVN |
| `edge_sam_3x_decoder.onnx` | CPU | `python scripts/export_onnx_model.py weights/edge_sam_3x.pth --decoder --use-stability-score` |

注意用的是**原始** `export_onnx_model.py`（不是 `_npu` 版本）。ONNX Runtime 支持所有标准算子，不需要 NPU 的算子替换。

## 与全 NPU 方案的关键区别

| | 全 NPU | 混合方案 |
|--|--------|---------|
| Decoder 输入 | `image_embeddings` + `point_embedding_pe` + `point_labels` | `image_embeddings` + **`point_coords`** + `point_labels` |
| PE 预计算 | 需要（CPU 端 sin/cos） | **不需要**（ONNX 内部包含） |
| pe_gaussian_matrix.bin | 需要 | **不需要** |
| 点数 N | 固定 5（静态 shape） | **动态**（传几个是几个） |

**最大简化点**：不需要 PE 预计算，不需要高斯矩阵，不需要 padding。

## Decoder ONNX 的输入输出规格

**输入**（3 个 tensor）：

| 名称 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `image_embeddings` | FLOAT32 | [1, 256, 64, 64] | Step 2 encoder 输出 |
| `point_coords` | FLOAT32 | [1, N, 2] | 变换后坐标 (x,y)，N 可变 |
| `point_labels` | FLOAT32 | [1, N] | -1=padding, 0=背景, 1=前景, 2=box左上, 3=box右下 |

**输出**（2 个 tensor）：

| 名称 | 类型 | 形状 |
|------|------|------|
| `scores` | FLOAT32 | [1, 4] |
| `masks` | FLOAT32 | [1, 4, 256, 256] |

## ONNX Runtime C++ 关键 API

```cpp
#include <onnxruntime_cxx_api.h>

// 初始化
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "decoder");
Ort::SessionOptions opts;
opts.SetIntraOpNumThreads(4);
Ort::Session session(env, "decoder.onnx", opts);

// 创建输入 tensor
auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
std::vector<int64_t> embed_shape = {1, 256, 64, 64};
auto t0 = Ort::Value::CreateTensor<float>(mem, embed_ptr, 1*256*64*64, embed_shape.data(), 4);
// point_coords 和 point_labels 同理

// 推理
const char* in_names[]  = {"image_embeddings", "point_coords", "point_labels"};
const char* out_names[] = {"scores", "masks"};
std::vector<Ort::Value> inputs;
inputs.push_back(std::move(t0));
inputs.push_back(std::move(t1));
inputs.push_back(std::move(t2));
auto outputs = session.Run(Ort::RunOptions{}, in_names, inputs.data(), 3, out_names, 2);

// 取输出
float* scores = outputs[0].GetTensorMutableData<float>();  // [4]
float* masks  = outputs[1].GetTensorMutableData<float>();  // [4*256*256]
```

## 预处理和后处理

与全 NPU 方案完全相同，详见 `npu_tv_deployment_guide.md` 的 Step 1（预处理）、Step 3（坐标变换）、Step 6（后处理）。
