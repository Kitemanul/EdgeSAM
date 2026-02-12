# EdgeSAM NPU 兼容性改动总结

## 概述

EdgeSAM 的 ONNX 模型在 NPU 编译器上无法直接编译，原因是模型中包含多种 NPU 不支持的 ONNX 算子。本文档详细记录了所有不支持的算子、对应的修复方案、以及修复后的验证结果。

**最终结果**：Encoder + Decoder 全部通过 NPU 编译。

---

## NPU 不支持的算子汇总

在排查过程中，共发现以下 ONNX 算子在目标 NPU 编译器上不被支持：

| 不支持的算子 | 来源 | 所在模块 |
|-------------|------|---------|
| `Sin` | `torch.sin()` — 位置编码 (PE) | Decoder (Prompt Encoding) |
| `Cos` | `torch.cos()` — 位置编码 (PE) | Decoder (Prompt Encoding) |
| `Abs` | `torch.abs()` — 标签相等判断 | Decoder (Prompt Encoding) |
| `Gather` | `tensor[:, i, :]` 整数索引 — PyTorch 导出为 Gather | Decoder (Mask Head) |
| `Erf` | `nn.GELU` — GELU 激活函数的精确实现 | Decoder (Mask Head) |
| bool `ReduceSum` | `(masks > threshold).sum()` — 布尔张量求和 | Decoder (Stability Score) |
| `int64` 数据类型 | ONNX 规范要求 Reshape/Expand 等算子使用 int64 shape | 全局 |
| `int16` 数据类型 | stability score 中间计算使用 int16 | Decoder (Stability Score) |

---

## 修复方案汇总

共 7 项修复，分为 5 项模型级修复和 2 项后处理修复：

| 编号 | 修复内容 | 消除的算子/问题 | 引入的替代算子 | 精度影响 |
|------|---------|---------------|--------------|---------|
| Fix 1 | PE 编码移至 CPU 预计算 | `Sin`, `Cos` | 无（CPU 端处理） | 零损失 |
| Fix 2 | 标签比较改用 float 算术 | `Abs`, `Equal`, `Cast(bool)` | `Sub`, `Mul`, `Relu` | 零损失 |
| Fix 3 | 索引改用 Slice+Reshape | `Gather` | `Slice`, `Reshape` | 零损失 |
| Fix 4 | GELU 替换为 tanh 近似 | `Erf` | `Tanh`, `Mul`, `Add` | mask < 0.014, score < 0.001 |
| Fix 5 | Stability score 改用 sigmoid | bool `ReduceSum`, `Greater`, `Cast` | `Sigmoid`, `Mul`, `Sub`, float `ReduceSum` | score < 0.004 |
| Fix 6 | ONNX 图简化（常量折叠） | 冗余 `Shape`→`Gather`→`Concat` 子图 | `Constant` | 零损失 |
| Fix 7 | int64/int16 → int32 转换 | `INT64`, `INT16` 数据类型 | `INT32` | 零损失 |

---

## 各修复详细说明

### Fix 1：Sin/Cos → PE 移至 CPU 预计算

**问题**：`PositionEmbeddingRandom._pe_encoding()` 使用 `torch.sin()` 和 `torch.cos()` 对点坐标做位置编码，导出为 ONNX 的 `Sin` 和 `Cos` 算子。

**原始代码**（`edge_sam/modeling/prompt_encoder.py`）：
```python
def _pe_encoding(self, coords):
    coords = 2 * coords - 1
    coords = coords @ self.positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
```

**无法用简单替换消除的原因**：
- Sin/Cos 是超越函数，不能用有限的加减乘除替代
- 多项式近似（Taylor 展开）需要先做周期约简（modulo 2π），但 `Floor` 算子 NPU 也不支持
- 查找表方案需要 `Gather` 算子来按坐标索引，`Gather` NPU 也不支持

**解决方案**：将 PE 编码从 ONNX 模型中完全移除，在 CPU/主机端预计算。

```python
# CPU 端预计算
def compute_point_pe(sam, point_coords):
    with torch.no_grad():
        coords = point_coords + 0.5
        coords = coords / sam.image_encoder.img_size
        pe = sam.prompt_encoder.pe_layer._pe_encoding(coords)
    return pe  # [1, N, 256]
```

**接口变化**：Decoder 输入从 `point_coords [1,N,2]` 改为 `point_embedding_pe [1,N,256]`。

**性能影响**：对 5 个点计算 PE 只需 1 次 [1,5,2] × [2,128] 矩阵乘法 + sin/cos，在 CPU 上耗时微秒级。

---

### Fix 2：Equal/Cast/Abs → float 算术

**问题**：标签比较 `point_labels == i` 导出为 `Equal` + `Cast(bool→float)` 算子。第一次修复尝试用 `torch.abs()` 替代 Equal，但 `Abs` 算子 NPU 也不支持。

**最终方案**：利用标签是整数值 float 的特性，用 `relu(1 - (x-val)²)` 替代相等判断：

```python
def _float_eq(x, val):
    diff = x - val
    return torch.relu(1.0 - diff * diff)
```

| 条件 | diff | diff² | 1-diff² | relu(...) |
|------|------|-------|---------|-----------|
| x == val | 0 | 0 | 1 | **1.0** |
| x != val | ≥1 整数 | ≥1 | ≤0 | **0.0** |

ONNX 算子：`Sub` → `Mul` → `Sub` → `Relu`，全部在 PASS 模型中已验证。

---

### Fix 3：Gather → Slice + Reshape

**问题**：Python 整数索引 `tensor[:, i, :]` 导出为 ONNX `Gather(axis=1, indices=i)` 算子。NPU 编译器报告 4 个 Gather 节点不支持。

**原始代码**：
```python
iou_token_out = hs[:, 0, :]                    # → Gather
token_i = mask_tokens_out[:, i, :]              # → Gather ×4
b, c, h, w = upscaled.shape                     # → Shape + Gather
masks = (hyper_in @ upscaled.view(b, c, h*w))   # 动态 shape
```

**修复后**：
```python
iou_token_out = hs[:, 0:1, :].reshape(1, 256)          # → Slice + Reshape
token_i = mask_tokens_out[:, i:i+1, :].reshape(1, 256)  # → Slice + Reshape ×4
masks = (hyper_in @ upscaled.reshape(1, 32, 65536))      # 静态常量 shape
```

**关键技巧**：
- 切片 `[:, i:i+1, :]` 导出为 `Slice`（不是 `Gather`）
- 用 `reshape` 而非 `squeeze` 降维（`Squeeze` 不在已验证的算子集中）
- 所有 shape 参数使用 Python int 常量，避免动态 `Shape` 算子

---

### Fix 4：Erf (GELU) → tanh 近似

**问题**：`nn.GELU` 的精确实现导出为 `Erf` ONNX 算子。虽然 Encoder 中的 `Erf` 能通过编译，但 Decoder 的 `output_upscaling` 中的 2 个 `nn.GELU` 实例在隔离编译时报错。

**修复方案**：替换为 Hendrycks (2016) 的 tanh 近似：

```python
class _GELUTanh(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))
```

**精度影响**：
- mask 值最大偏差 < 0.014
- score 最大偏差 < 0.001
- 二值化 mask 完全相同（IoU = 1.0）

**注意**：Decoder 的 Transformer 部分使用的是 `nn.ReLU`（不是 GELU），不需要替换。只有 `output_upscaling` 中的 2 个 GELU 需要处理。

---

### Fix 5：bool ReduceSum → sigmoid 阶跃近似

**问题**：Stability score 的原始实现：
```python
intersections = (masks > (threshold + offset)).sum(-1, dtype=torch.int16).sum(-1)
unions = (masks > (threshold - offset)).sum(-1, dtype=torch.int16).sum(-1)
```
- `masks > threshold` → `Greater` + `Cast(bool)` 算子
- `.sum(dtype=torch.int16)` → `ReduceSum` 在 bool/int16 类型上，NPU 不支持

**修复方案**：用 sigmoid 函数近似阶跃函数：

```python
def _stability_score_npu(masks, mask_threshold, threshold_offset):
    k = 50.0  # 大斜率使 sigmoid 近似 step function
    high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
    low = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
    intersections = high.sum(-1).sum(-1)
    unions = low.sum(-1).sum(-1)
    return intersections / unions
```

只使用 `Sub`, `Mul`, `Sigmoid`, float `ReduceSum`, `Div`，全部是 float32 运算。

---

### Fix 6：ONNX 图简化

**问题**：PyTorch ONNX 导出器对静态 shape 也会生成 `Shape → Gather → Concat → Reshape` 子图链。

**修复方案**：使用 onnxruntime 的 `ORT_ENABLE_BASIC` 优化级别做常量折叠，将这些子图折叠为简单的 `Constant` 节点。

---

### Fix 7：int64/int16 → int32

**问题**：ONNX 规范要求 `Reshape`、`Expand`、`Tile` 等算子的 shape 参数为 int64。大部分 NPU 编译器只支持 int32。

**修复方案**：遍历 ONNX 模型的 7 个位置（Cast 节点、ConstantOfShape、Constant、Initializer、value_info、Input、Output），将所有 int64/int16 替换为 int32。

**注意**：修改后的模型不符合 ONNX 规范（int64 是 spec 要求），不能在 onnxruntime 中加载。仅供 NPU 编译器使用。

---

## 接口变化

### 原始接口（标准 ONNX）

```
Decoder 输入:
  image_embeddings:  FLOAT [1, 256, 64, 64]
  point_coords:      FLOAT [1, N, 2]          ← 像素坐标
  point_labels:      FLOAT [1, N]

Decoder 输出:
  scores: FLOAT [1, 4]
  masks:  FLOAT [1, 4, 256, 256]
```

### NPU 兼容接口

```
CPU 预计算:
  point_embedding_pe = compute_point_pe(sam, point_coords)

Decoder 输入:
  image_embeddings:   FLOAT [1, 256, 64, 64]
  point_embedding_pe: FLOAT [1, N, 256]       ← CPU 预计算的 PE
  point_labels:       FLOAT [1, N]

Decoder 输出:
  scores: FLOAT [1, 4]
  masks:  FLOAT [1, 4, 256, 256]
```

### 完整推理流水线

```
1. 预处理 (CPU):  ResizeLongestSide(1024) → normalize → pad → [1, 3, 1024, 1024]
2. Encoder (NPU):  image → image_embeddings [1, 256, 64, 64]
3. 坐标变换 (CPU): apply_coords(point_coords, original_size) → transformed_coords
4. PE 编码 (CPU):  compute_point_pe(sam, transformed_coords) → point_embedding_pe [1, N, 256]
5. Decoder (NPU):  (image_embeddings, point_embedding_pe, point_labels) → (scores, masks)
6. 后处理 (CPU):   best = argmax(scores); mask = interpolate(masks[best]) → 原图尺寸
```

---

## 算子验证

### Encoder（PASS）— 13 种算子

```
Add(104) Constant(87) Conv(130) Div(29) Erf(27) Mul(66) Pow(2)
ReduceMean(14) Relu(10) Resize(1) Sigmoid(10) Sqrt(2) Sub(2)
```

### Decoder 修复后 — 预期算子（所有子模块合并）

消除的算子：`Sin`, `Cos`, `Abs`, `Gather`, `Erf`, `Greater`, `Cast(bool)`, bool `ReduceSum`

保留/引入的算子（全部在 PASS 模型中已验证）：
```
Add, Concat, Constant, ConstantOfShape, ConvTranspose, Div, Equal,
Expand, Gemm, MatMul, Mul, Pow, ReduceMean, ReduceSum(float),
Relu, Reshape, Shape, Sigmoid, Slice, Softmax, Sqrt, Sub, Tanh,
Transpose, Unsqueeze, Where
```

---

## 精度验证

在 truck.jpg 上测试 4 种 prompt 配置（参考：SamCoreMLModel PyTorch）：

| Prompt | Score 最大差 | Best Mask IoU | Best Mask 选择 |
|--------|-------------|---------------|---------------|
| 单前景点 (500,375) | 0.000844 | 1.000000 | MATCH |
| 单前景点 (600,500) | 0.001031 | 1.000000 | MATCH |
| 前景+背景点 | 0.000266 | 1.000000 | MATCH |
| Box prompt (label 2+3) | 0.003569 | 1.000000 | MATCH |

所有 prompt 的二值化 mask 完全相同（IoU = 1.0），score 差异 < 0.004。

---

## 使用方法

### 导出 NPU 兼容模型

```bash
# Encoder
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

# Decoder（推荐加 --use-stability-score）
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score

# 查看算子
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score --check-ops-only
```

### 诊断工具（三段拆分）

如果合并后的模型编译失败，可使用三段拆分工具定位问题：

```bash
python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./npu_diag
```

### 端到端测试

```bash
python scripts/test_3part_pipeline.py weights/edge_sam_3x.pth truck.jpg
```
