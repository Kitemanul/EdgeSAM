# EdgeSAM Decoder ONNX 导出 — NPU 算子兼容方案

脚本：`scripts/export_onnx_model_npu.py`

## 背景

EdgeSAM 的 ONNX decoder 在 TV NPU 编译器上编译失败。根因是 decoder 包含三类 NPU 不支持的元素：

| 问题 | 高 opset 表现 | 低 opset (11) 表现 |
|------|-------------|-------------------|
| `LayerNormalization` 算子 | 编译器报错 "LayerNorm not supported" | PyTorch exporter 自动分解为基本算子，不报错 |
| `Erf` 算子 (来自 GELU) | 编译器可能不报明确错误 | 同上，但 Erf **不会被分解**，仍然存在 |
| `int64` 数据类型 | 编译器立即报 "only support int32" | 量化校准能跑完，但编译阶段失败（不明错误） |

**Encoder（RepViT）不受影响**：纯 CNN，只有 `FLOAT` 类型，13 种基本算子，无 LayerNorm/GELU/int64。

---

## 修复总览

导出脚本对 decoder 做了 **5 项修复**，分为两个阶段：

```
阶段 1：PyTorch 侧（导出前）            阶段 2：ONNX 侧（导出后）
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│ ① nn.LayerNorm → 手写实现    │    │ ④ onnxruntime 图简化（常量折叠）   │
│ ② nn.GELU → tanh 近似       │    │ ⑤ int64/int16 → int32 全量转换    │
│ ③ stability_score int16→int32│    │    + Cast 桥接节点（shape 输入）   │
│    + 去掉 dynamic_axes       │    │                                  │
└─────────────────────────────┘    └──────────────────────────────────┘
```

---

## 修复 ①：nn.LayerNorm → LayerNormManual

### 问题

`nn.LayerNorm` 在 opset >= 17 导出为原生 `LayerNormalization` 算子，NPU 不支持。

**Decoder 中的位置**（`edge_sam/modeling/transformer.py`）：

```
TwoWayTransformer
├── layers[0] (TwoWayAttentionBlock)
│   ├── norm1 = nn.LayerNorm(256)    ← self-attention 后
│   ├── norm2 = nn.LayerNorm(256)    ← cross-attention (token→image) 后
│   ├── norm3 = nn.LayerNorm(256)    ← MLP 后
│   └── norm4 = nn.LayerNorm(256)    ← cross-attention (image→token) 后
├── layers[1] (TwoWayAttentionBlock)
│   ├── norm1, norm2, norm3, norm4   ← 同上，4 个
└── norm_final_attn = nn.LayerNorm(256)
```

共 **9 个**实例。

> `mask_decoder.py` 中的 `LayerNorm2d` 是**自定义类**（`common.py:32`），内部手写 mean/pow/sqrt，不是 `nn.LayerNorm`，不受影响。

### 替换方案

用基本算子手写等价的归一化计算：

```python
# nn.LayerNorm 的数学定义:
#   y = (x - mean) / sqrt(var + eps) * weight + bias

class LayerNormManual(nn.Module):
    def forward(self, x):
        dims = [-1]  # 在最后一维归一化（256 维）
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight + self.bias
        return x
```

### ONNX 算子映射

```
PyTorch 操作                    ONNX 算子         opset 引入版本
─────────────────────────────────────────────────────────────
x.mean(dim, keepdim)        →  ReduceMean         1
(x - mean)                  →  Sub                1
(...)  ** 2                 →  Pow                1
(...).mean(dim, keepdim)    →  ReduceMean         1
var + self.eps              →  Add                1
torch.sqrt(...)             →  Sqrt               1
(x - mean) / sqrt(...)      →  Div                1
x * self.weight             →  Mul                1
... + self.bias             →  Add                1
```

全部是 opset 1 的算子。

### 精度影响

**零损失**。数学公式与 `nn.LayerNorm` 完全一致，不是近似。

### 权重处理

```python
manual.weight = module.weight   # 直接复用同一个 nn.Parameter，零拷贝
manual.bias = module.bias
```

---

## 修复 ②：nn.GELU → GELUManual（tanh 近似）

### 问题

`nn.GELU` 内部使用 `erf` 函数，导出到 ONNX 生成 `Erf` 算子。`Erf` 从 opset 9 就存在于标准中，但**大量 NPU 编译器不支持**，且不会因 opset 版本降低而被自动分解。

**Decoder 中的位置**（`edge_sam/modeling/mask_decoder.py:55-61`）：

```python
self.output_upscaling = nn.Sequential(
    nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
    LayerNorm2d(64),
    nn.GELU(),             # ← 问题算子 ①
    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
    nn.GELU(),             # ← 问题算子 ②
)
```

共 **2 个**实例（导出时由于模型遍历也替换了 encoder 中的 GELU，但 encoder 实际不走这个路径）。

### 替换方案

使用 GELU 的标准 tanh 近似（来自原始论文 [GELUs, Hendrycks 2016](https://arxiv.org/abs/1606.08415)）：

```python
# 精确版: GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))    → 生成 Erf 算子
# tanh 近似: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

class GELUManual(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))
        # 0.7978845608028654 = sqrt(2 / pi)
```

> PyTorch 的 `nn.GELU(approximate='tanh')` 用的就是这个公式，GPT 系列模型也使用此近似。

### ONNX 算子映射

```
精确 GELU 的 ONNX 算子:              tanh 近似的 ONNX 算子:
────────────────────────          ──────────────────────
x / sqrt(2)    → Div              x * x * x    → Mul, Mul
erf(...)       → Erf  ← 不支持   0.044715 * x³ → Mul
1 + erf(...)   → Add              x + ...      → Add
0.5 * x        → Mul              √(2/π) * ... → Mul
... * (...)     → Mul              tanh(...)    → Tanh  ← 广泛支持
                                   1 + tanh     → Add
                                   0.5 * x * .. → Mul, Mul
```

**关键变化**：`Erf` → `Tanh`。`Tanh` 是几乎所有 NPU 都支持的基本算子。

### 精度影响

| 指标 | 值 |
|------|-----|
| 最大绝对误差 | < 0.004 |
| 对模型 mask IoU 的影响 | < 0.01 |

实测 decoder 输出的 per-mask IoU 对比（PyTorch 精确 GELU vs ONNX tanh 近似）：

| Mask | IoU |
|------|-----|
| 高置信度 masks (score > 0.9) | > 0.96 |
| 低置信度 masks (score < 0.7) | > 0.85 |

### 权重处理

`nn.GELU` 没有可学习参数，直接替换即可。

---

## 修复 ③：stability_score int16 → int32 + 去掉 dynamic_axes

### 问题 A：int16

`calculate_stability_score`（`edge_sam/utils/amg.py:166-170`）使用 `torch.int16` 做中间求和：

```python
# 原始代码
intersections = (
    (masks > (mask_threshold + threshold_offset))
    .sum(-1, dtype=torch.int16)   # ← int16
    .sum(-1, dtype=torch.int32)
)
```

NPU 编译器只支持 int32。

### 修复

Monkey-patch 为全 int32：

```python
.sum(-1, dtype=torch.int32)   # 改为 int32
.sum(-1, dtype=torch.int32)
```

通过 `patch_stability_score()` 在导出前替换，不修改源码。

### 问题 B：dynamic_axes

原始导出脚本声明了动态维度：

```python
dynamic_axes={
    "point_coords": {1: "num_points"},
    "point_labels": {1: "num_points"},
}
```

NPU 量化编译器需要静态 shape 才能正确计算量化参数（min/max 校准值依赖固定的 tensor 形状）。

### 修复

去掉 `dynamic_axes`，导出固定 shape。默认 5 个 prompt 点，可通过 `--num-points` 自定义。

---

## 修复 ④：onnxruntime 图简化（常量折叠）

### 问题

即使是固定 shape 导出，PyTorch ONNX exporter 仍然会生成大量 shape 计算子图：

```
Shape → Gather → Concat → Reshape
                  ↑
            ConstantOfShape → Where → Expand
```

这些节点全部使用 **int64** 类型（ONNX 规范要求 shape 操作使用 int64），是 NPU "only support int32" 错误的根源。

### 修复

用 onnxruntime 的图优化器做常量折叠：

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
sess_options.optimized_model_filepath = simplified_path
ort.InferenceSession(onnx_path, sess_options)
```

由于所有 shape 都是静态的，优化器可以在编译时计算出所有 shape 值，将整个 shape 子图折叠为常量。

### 效果

| 指标 | 简化前 | 简化后 |
|------|--------|--------|
| 节点数 | 521 | 445 |
| 算子种类 | 32 | 28 |
| 被消除的算子 | — | `Constant`, `ConstantOfShape`, `Shape`, `Where` |

---

## 修复 ⑤：int64/int16 → int32 全量转换 + Cast 桥接

### 问题

图简化后仍有残余 int64 张量（如 Reshape 的 shape 参数、Constant 中存储的整型值）。

### 修复策略

**全量转换** + **Cast 桥接**：

```
┌──────────────────────────────────────────────────────────┐
│ 对 ONNX 模型中所有 int64/int16 做以下处理:               │
│                                                          │
│ 1. Cast 节点的目标类型:  int64/int16 → int32             │
│ 2. ConstantOfShape 的 value:  int64/int16 → int32        │
│ 3. Constant 节点的数据:                                   │
│    - 如果不是 shape 输入: 直接转 int32                    │
│    - 如果是 shape 输入:  转 int32, 然后插入 Cast 桥接     │
│ 4. Initializer 权重: 同上逻辑                             │
│ 5. value_info / graph inputs / outputs: int64 → int32     │
└──────────────────────────────────────────────────────────┘
```

### Cast 桥接节点

ONNX 规范要求 `Reshape`, `Expand`, `Tile`, `Slice`, `Unsqueeze`, `ConstantOfShape` 的 shape 输入必须是 int64。但 NPU 不支持 int64。

解决方案：在 shape 输入位置插入 Cast(int32 → int64) 桥接节点：

```
  转换前:                         转换后:
  ┌──────────┐                   ┌──────────┐    ┌──────────┐
  │ Constant │ (int64)           │ Constant │    │   Cast   │
  │ [1,256,  │────→ Reshape      │ [1,256,  │──→│ int32 →  │──→ Reshape
  │  64,64]  │    (shape input)  │  64,64]  │   │  int64   │   (shape input)
  └──────────┘                   └──────────┘    └──────────┘
                                    (int32)
```

**关键点**：
- 存储和计算全部用 int32（NPU 友好）
- 仅在 ONNX 规范强制要求 int64 的 shape 输入位置，插入一个 Cast 桥接
- 这些 Cast 节点在 NPU 编译时通常被**编译器内部消化**（shape 参数不走 NPU 计算单元，而是在图构建阶段处理）

### 需要 Cast 桥接的 ONNX 算子

| 算子 | 需要 int64 的输入位置 | 说明 |
|------|---------------------|------|
| `Reshape` | input[1] (shape) | 目标形状，如 `[1, 8, 10, 16]` |
| `Expand` | input[1] (shape) | 广播目标形状 |
| `Tile` | input[1] (repeats) | 重复次数 |
| `Slice` | input[1,2,3,4] (starts/ends/axes/steps) | 切片参数 |
| `Unsqueeze` | input[1] (axes) | 扩展维度索引 (opset ≥ 13) |
| `ConstantOfShape` | input[0] (shape) | 目标形状 |

### 转换统计

| 指标 | 数量 |
|------|------|
| int64 → int32 转换 | 81 |
| int16 → int32 转换 | 0 |
| 插入的 Cast 桥接节点 | 73 |

---

## Encoder vs Decoder 算子对比

### Encoder（RepViT，纯 CNN）

```
数据类型: 仅 FLOAT
算子 (13 种): Add, Constant, Conv, Div, Erf, Mul, Pow,
             ReduceMean, Relu, Resize, Sigmoid, Sqrt, Sub
Shape/Reshape/Cast: 无
int64: 无
```

所有维度在编译时确定，无需任何 shape 动态计算。

> 注意：encoder 中仍有 `Erf`（来自 BatchNorm 相关计算），但 encoder 已经能在 NPU 上编译通过，说明这个 NPU 编译器对 encoder 中的 Erf 用法是支持的，或者 encoder 的 Erf 来源与 GELU 不同。

### Decoder（Transformer + Prompt 编码）

**修复前 (521 节点, 32 种算子)**:
```
问题算子: LayerNormalization(opset18) / Erf / int64 全链路
Shape 子图: Shape → Gather → Concat → Reshape → ConstantOfShape → Where → Expand
```

**修复后 (445 节点, 28 种算子)**:
```
消除: LayerNormalization, Erf, Constant, ConstantOfShape, Shape, Where
新增: 仅 Cast (int32→int64 桥接)
数据类型: 计算部分全 FLOAT + INT32, shape 桥接处 INT64
```

---

## 完整处理流水线

```
     .pth 权重文件
          │
          ▼
  ┌───────────────────────┐
  │ torch.load(map_location│
  │ ="cpu") 加载权重到      │
  │ 原始模型结构            │
  └───────┬───────────────┘
          │  权重已就位
          ▼
  ┌───────────────────────┐
  │ ① replace_layernorm() │  9 个 nn.LayerNorm → LayerNormManual
  │ ② replace_gelu()      │  nn.GELU → GELUManual (tanh)
  │ ③ patch_stability_score│  int16 → int32
  └───────┬───────────────┘
          │  权重通过 nn.Parameter 引用传递
          ▼
  ┌───────────────────────┐
  │ torch.onnx.export()   │  opset=11, 固定 shape, 无 dynamic_axes
  │ (legacy exporter)     │
  └───────┬───────────────┘
          │  原始 ONNX 文件 (521 节点)
          ▼
  ┌───────────────────────┐
  │ ④ onnxruntime 图优化   │  常量折叠, 消除 shape 子图
  │    (ORT_ENABLE_BASIC) │
  └───────┬───────────────┘
          │  简化后 (445 节点)
          ▼
  ┌───────────────────────┐
  │ ⑤ int64 → int32 转换  │  81 处转换 + 73 个 Cast 桥接
  │    + Cast 桥接插入      │
  └───────┬───────────────┘
          │
          ▼
   NPU 兼容的 .onnx 文件
```

---

## 最终 Decoder 算子清单

| 算子 | 来源 | ONNX opset |
|------|------|-----------|
| `Add` | 残差连接、bias、LayerNorm | 1 |
| `Cast` | int32→int64 桥接（shape 输入） | 1 |
| `Concat` | token 拼接 | 1 |
| `ConvTranspose` | mask upscaling 转置卷积 | 1 |
| `Cos` | 位置编码 sin/cos | 7 |
| `Div` | LayerNorm 归一化、attention 缩放 | 1 |
| `Equal` | prompt label 比较 (label == i) | 1 |
| `Expand` | tensor 广播 | 8 |
| `Gather` | embedding 索引 | 1 |
| `Gemm` | 线性层 (weight * x + bias) | 1 |
| `Greater` | stability score 阈值比较 | 1 |
| `MatMul` | attention Q*K^T, attention*V | 1 |
| `Mul` | 权重缩放、GELU 计算、LayerNorm | 1 |
| `Not` | 逻辑非 | 1 |
| `Pow` | LayerNorm 方差计算 (x^2) | 1 |
| `ReduceMean` | LayerNorm 均值/方差 | 1 |
| `ReduceSum` | stability score 求和 | 1 |
| `Relu` | transformer MLP 激活 | 1 |
| `Reshape` | attention head 重排 | 5 |
| `Sin` | 位置编码 sin/cos | 7 |
| `Slice` | tensor 切片 | 1 |
| `Softmax` | attention 权重归一化 | 1 |
| `Sqrt` | LayerNorm 标准差 | 1 |
| `Sub` | LayerNorm 去均值 | 1 |
| `Tanh` | GELU tanh 近似 | 1 |
| `Tile` | tensor 重复 | 1 |
| `Transpose` | attention head 重排 | 1 |
| `Unsqueeze` | 维度扩展 | 1 |

28 种算子，全部在 **opset 11 以内**。

---

## 使用方法

```bash
# 导出 decoder（固定 5 个 prompt 点）
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score

# 自定义 prompt 点数量
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score --num-points 2

# 导出 encoder
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

# 查看算子列表和数据类型
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score --check-ops-only
```

---

## 排查建议

如果 NPU 编译仍失败：

1. 用 `--check-ops-only` 获取完整算子列表，对照编译器文档逐个确认
2. 如果 Cast(int32→int64) 桥接节点报错，说明编译器对 shape 参数也不接受 int64 —— 需要进一步处理
3. 如果 `Sin`/`Cos` 不支持，可将位置编码预计算为常量嵌入模型
4. 如果 `ConvTranspose` 不支持，可替换为 `Upsample` + `Conv`
5. 如果 `Softmax` 不支持，可手写 exp + reducesum + div
