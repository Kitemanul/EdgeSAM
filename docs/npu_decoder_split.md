# EdgeSAM Decoder 三段拆分 — NPU 兼容方案

## 背景

EdgeSAM 的 ONNX decoder 作为整体无法通过 NPU 编译器编译。Encoder（RepViT，纯 CNN）可以正常编译。

**策略**：将 decoder 拆分为 3 个独立子模型，逐个定位编译失败的原因并针对性修复。

## 拆分结构

```
Encoder (RepViT)                    ── 已有，NPU 编译通过
    │
    │  image_embeddings [1, 256, 64, 64]
    ▼
┌─────────────────────────────────────────────────────┐
│              Decoder（原为整体，现拆为 3 段）           │
│                                                     │
│  Part 1: Prompt Encoding                            │
│    输入: point_coords [1, N, 2], point_labels [1, N]│
│    输出: sparse_embedding [1, N, 256]               │
│                                                     │
│  Part 2: Transformer                                │
│    输入: image_embeddings [1,256,64,64],            │
│          sparse_embedding [1, N, 256]               │
│    输出: hs [1, 5+N, 256], src [1, 4096, 256]      │
│                                                     │
│  Part 3: Mask Head                                  │
│    输入: hs [1, 5+N, 256], src [1, 4096, 256]      │
│    输出: scores [1, 4], masks [1, 4, 256, 256]     │
└─────────────────────────────────────────────────────┘
```

默认 N=5（prompt 点数量），可通过 `--num-points` 自定义。所有模型使用静态 shape，无动态维度。

## 编译结果与修复

| 模块 | 首次编译 | 报错 | 修复后 |
|------|---------|------|--------|
| Encoder | PASS | — | — |
| Part 1: Prompt Encoding | FAIL | `failed to legalize operation onnx.Abs` | 消除 Abs → PASS |
| Part 2: Transformer | PASS | — | — |
| Part 3: Mask Head | FAIL | `unsupported nodes: Gather` (4 个) | 消除 Gather → PASS |

---

## 修复详情

### 修复 1：Part 1 消除 `Abs` 算子

**问题**：`Abs` 算子 NPU 不支持。来源是 `_float_eq()` 中的 `torch.abs()`，用于实现 NPU-safe 的标签相等判断。

**原实现**：
```python
def _float_eq(x, val):
    return torch.clamp(1.0 - torch.abs(x - val), min=0.0, max=1.0)
    # ONNX 算子: Sub, Abs, Sub, Clip → Abs 不支持
```

**修复后**：
```python
def _float_eq(x, val):
    diff = x - val
    return torch.relu(1.0 - diff * diff)
    # ONNX 算子: Sub, Mul, Sub, Relu → 全部支持
```

**数学原理**：

point labels 是整数值 float（-1, 0, 1, 2, 3），所以 `diff = x - val` 只有两种情况：

| 条件 | diff | diff² | 1 - diff² | relu(...) |
|------|------|-------|-----------|-----------|
| x == val | 0 | 0 | 1 | **1.0** |
| x != val | ≥1 整数 | ≥1 | ≤0 | **0.0** |

输出与原实现完全等价，零精度损失。

**算子变化**：

| | 去掉 | 引入 | 引入的算子是否在 PASS 模型中 |
|--|------|------|--------------------------|
| 修复 1 | `Abs`, `Clip` | `Relu`, `Mul` | Relu ∈ Encoder, Mul ∈ Encoder |

---

### 修复 2：Part 3 消除 `Gather` 算子

**问题**：`Gather` 算子 NPU 不支持。编译器报告 4 个 Gather 节点。来源是 Python 整数索引操作 `tensor[:, i, :]`，ONNX 导出为 `Gather(axis=1, indices=i)`。

**原实现**：
```python
iou_token_out = hs[:, 0, :]                           # → Gather
mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]  # → Slice (OK)

for i in range(self.num_mask_tokens):  # 4 次
    token_i = mask_tokens_out[:, i, :]                 # → Gather ×4
    hyper_in_list.append(self.output_hypernetworks_mlps[i](token_i))

b, c, h, w = upscaled.shape                           # → Shape + Gather (动态)
masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
```

**修复后**：
```python
iou_token_out = hs[:, 0:1, :].reshape(1, 256)         # → Slice + Reshape
mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]  # → Slice (OK)

for i in range(self.num_mask_tokens):  # 4 次
    token_i = mask_tokens_out[:, i:i+1, :].reshape(1, 256) # → Slice + Reshape ×4
    hyper_in_list.append(self.output_hypernetworks_mlps[i](token_i))

masks = (hyper_in @ upscaled.reshape(1, 32, 65536)).reshape(1, -1, 256, 256)
#                                      ^ 静态常量，无 Shape 算子
```

**关键变化**：

| 操作 | 原 ONNX 算子 | 修复后 ONNX 算子 |
|------|-------------|----------------|
| `tensor[:, i, :]` 整数索引 | `Gather` | — |
| `tensor[:, i:i+1, :]` 切片索引 | — | `Slice` |
| `.squeeze(1)` 去掉维度 | — | 不使用（`Squeeze` 不在 PASS 集合中） |
| `.reshape(1, 256)` 去掉维度 | — | `Reshape` |
| `view(b, c, h*w)` 动态 shape | `Shape` + `Gather` | — |
| `reshape(1, 32, 65536)` 静态 shape | — | `Reshape`（常量 shape） |

**为什么用 `reshape` 而不是 `squeeze`**：

`squeeze` 导出为 ONNX `Squeeze` 算子，而 `Squeeze` 不在已知可通过编译的算子集合中（Encoder 和 Part 2 均未使用）。`reshape` 导出为 `Reshape` 算子，Part 2 中有 29 个 `Reshape` 实例已确认编译通过。

**算子变化**：

| | 去掉 | 引入 | 引入的算子是否在 PASS 模型中 |
|--|------|------|--------------------------|
| 修复 2 | `Gather` | `Slice`, `Reshape` | Slice ∈ Part 2, Reshape ∈ Part 2 |

---

### Part 3 其他 NPU 兼容处理（修复前已有）

这些改动在拆分之初就已内置，不是本次修改引入的：

**a) GELU → tanh 近似**

`nn.GELU` 导出为 `Erf` 算子。尽管 Encoder 中 `Erf` 能通过编译，但 decoder 的 `output_upscaling` 中 2 个 `nn.GELU` 被替换为 tanh 近似：

```python
class _GELUTanh(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))
```

精度影响：mask 值最大偏差 < 0.014，score 最大偏差 < 0.001。

**b) Stability Score sigmoid 近似**

原始实现使用 `masks > threshold`（导出为 `Greater` → `Cast(bool)` → `ReduceSum`），NPU 不支持 bool 类型的 `ReduceSum`。

替换为连续 sigmoid 阶跃近似：

```python
def _stability_score_npu(masks, mask_threshold, threshold_offset):
    k = 50.0  # 大斜率使 sigmoid 近似 step function
    high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
    low  = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
    intersections = high.sum(-1).sum(-1)
    unions = low.sum(-1).sum(-1)
    return intersections / unions
```

只使用 `Mul`, `Sub`, `Sigmoid`, `ReduceSum`, `Div`，全部是 float 运算。

**c) 标签比较的 bool-free 实现**

Part 1 的标签比较避免了 `Equal`/`Cast(bool)` 模式，使用纯 float 算术（见修复 1）。

---

## 算子对比总表

### Encoder（PASS）— 13 种算子

```
Add(104) Constant(87) Conv(130) Div(29) Erf(27) Mul(66) Pow(2)
ReduceMean(14) Relu(10) Resize(1) Sigmoid(10) Sqrt(2) Sub(2)
```
数据类型：仅 FLOAT

### Part 2: Transformer（PASS）— 20 种算子

```
Add(66) Concat(2) Constant(60) ConstantOfShape(1) Div(16) Equal(1)
Expand(1) MatMul(46) Mul(10) Pow(9) ReduceMean(18) Relu(2)
Reshape(29) Shape(1) Slice(1) Softmax(7) Sqrt(9) Sub(9)
Transpose(29) Where(1)
```
数据类型：FLOAT + INT64

### Part 1: Prompt Encoding（修复后）— 13 种算子

```
Add(6) Concat(1) Constant(16) Cos(1) Div(1) Expand(1) MatMul(1)
Mul(13) Relu(5) Shape(1) Sin(1) Sub(12) Unsqueeze(1)
```
数据类型：仅 FLOAT

相比 PASS 模型的新增算子：`Sin`, `Cos`, `Unsqueeze`（来自位置编码，非修改引入）

### Part 3: Mask Head（修复后）— 20 种算子

```
Add(6) Concat(1) Constant(37) ConvTranspose(2) Div(2) Gemm(12)
MatMul(1) Mul(15) Pow(1) ReduceMean(2) ReduceSum(4) Relu(8)
Reshape(3) Sigmoid(2) Slice(5) Sqrt(1) Sub(3) Tanh(2)
Transpose(1) Unsqueeze(4)
```
数据类型：仅 FLOAT

相比 PASS 模型的新增算子：`ConvTranspose`, `Gemm`, `ReduceSum`, `Tanh`, `Unsqueeze`（均为模型原始结构，非修改引入）

---

## 端到端验证

使用 `scripts/test_3part_pipeline.py` 在 truck.jpg 上测试 4 种 prompt 配置：

| Prompt | Score 最大差 | Best Mask IoU | Best Mask 选择 |
|--------|-------------|---------------|---------------|
| 单前景点 (500,375) | 0.000844 | 1.000000 | MATCH |
| 单前景点 (600,500) | 0.001031 | 1.000000 | MATCH |
| 前景+背景点 | 0.000266 | 1.000000 | MATCH |
| Box prompt (label 2+3) | 0.003569 | 1.000000 | MATCH |

参考对象：`SamCoreMLModel`（PyTorch），与 3-part ONNX 流水线行为一致（均不添加 padding point）。

所有 prompt 的二值化 mask 完全相同（IoU = 1.0），score 差异 < 0.004（来自 GELU tanh 近似和 stability score sigmoid 近似）。

---

## 使用方法

### 导出三段模型

```bash
python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./npu_diag

# 自定义 prompt 点数量
python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./npu_diag --num-points 2
```

### 验证端到端正确性

```bash
python scripts/test_3part_pipeline.py weights/edge_sam_3x.pth truck.jpg
```

### 推理流水线

```
输入: image (BGR), point_coords (N×2, 原图像素坐标), point_labels (N,)

1. 预处理: ResizeLongestSide(1024) → normalize → pad → [1, 3, 1024, 1024]
2. Encoder:  image → image_embeddings [1, 256, 64, 64]
3. 坐标变换: apply_coords(point_coords, original_size) → transformed_coords
4. Part 1:   (transformed_coords, point_labels) → sparse_embedding [1, N, 256]
5. Part 2:   (image_embeddings, sparse_embedding) → (hs [1, 5+N, 256], src [1, 4096, 256])
6. Part 3:   (hs, src) → (scores [1, 4], masks [1, 4, 256, 256])
7. 后处理:   best = argmax(scores); mask = interpolate(masks[best]) → 原图尺寸
```

注意：坐标变换（步骤 3）和后处理（步骤 7）在模型外部完成，不包含在 ONNX 中。

---

## 排查建议

如果 NPU 编译仍有失败：

1. 检查 `Sin`/`Cos` 是否支持 — 来自位置编码，若不支持可预计算为常量嵌入模型
2. 检查 `ConvTranspose` 是否支持 — 来自 mask upscaling，若不支持可替换为 `Upsample` + `Conv`
3. 检查 `Gemm` 是否支持 — 来自 MLP 全连接层，若不支持可替换为 `MatMul` + `Add`
4. 检查 `ReduceSum` 是否支持 — 来自 stability score，若不支持可预计算或改用其他聚合方式
5. 检查 `Tanh` 是否支持 — 来自 GELU tanh 近似，若不支持需要寻找其他 GELU 近似方案
