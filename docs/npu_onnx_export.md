# EdgeSAM NPU ONNX 导出 — 完整改动说明

## 背景

EdgeSAM 的 ONNX decoder 在 TV NPU 编译器上编译失败。根因是 decoder 包含三类 NPU 不支持的元素：

| 问题 | 高 opset 表现 | 低 opset (11) 表现 |
|------|-------------|-------------------|
| `LayerNormalization` 算子 | 编译器报错 "LayerNorm not supported" | PyTorch exporter 自动分解为基本算子，不报错 |
| `Erf` 算子 (来自 GELU) | 编译器可能不报明确错误 | 同上，但 Erf **不会被分解**，仍然存在 |
| `int64` 数据类型 | 编译器立即报 "only support int32" | 量化校准能跑完，但编译阶段失败（不明错误） |

**Encoder（RepViT）不受影响**：纯 CNN，只有 `FLOAT` 类型，13 种基本算子，无 LayerNorm/GELU/int64。

---

## 改动文件总览

本次共涉及 4 个文件的修改/新增：

| 文件 | 操作 | 说明 |
|------|------|------|
| `scripts/export_onnx_model_npu.py` | **新增** | NPU 兼容的 ONNX 导出脚本（核心） |
| `edge_sam/modeling/sam.py` | **修改** | mmdet/mmengine 改为 lazy import |
| `edge_sam/build_sam.py` | **修改** | torch.load 添加 map_location="cpu" |
| `docs/npu_onnx_export.md` | **新增** | 本文档 |

---

## 一、`scripts/export_onnx_model_npu.py`（核心脚本）

### 概述

针对 NPU 编译器做了 **4 项修复**，分为两个阶段：

```
阶段 1：PyTorch 侧（导出前）            阶段 2：ONNX 侧（导出后）
┌─────────────────────────────┐    ┌──────────────────────────────────┐
│ ① nn.GELU → tanh 近似       │    │ ③ onnxruntime 图简化（常量折叠）   │
│ ② stability_score int16→int32│    │ ④ int64/int16 → int32 全量转换    │
│    + 去掉 dynamic_axes       │    │    （7 处位置，无 Cast 桥接）       │
└─────────────────────────────┘    └──────────────────────────────────┘
```

> **关于 nn.LayerNorm**：在 opset >= 17 时会导出为 `LayerNormalization` 原生算子，
> NPU 不支持。但本脚本使用 **opset 11**，PyTorch exporter 会自动将 LayerNorm 分解为
> ReduceMean/Sub/Pow/Sqrt/Div/Mul/Add 基本算子，因此**无需手动替换**。
> 如果未来需要更高 opset，参考下方「修复（参考）」一节。

### 修复（参考）：nn.LayerNorm → LayerNormManual

> **注意**：当前脚本使用 opset 11，PyTorch 会自动分解 LayerNorm，因此**无需此修复**。
> 此节仅作参考，供未来使用更高 opset 时使用。

**问题**：`nn.LayerNorm` 在 opset >= 17 导出为原生 `LayerNormalization` 算子，NPU 不支持。

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

**替换方案**：用基本算子手写等价的归一化计算：

```python
class LayerNormManual(nn.Module):
    def forward(self, x):
        dims = [-1]  # 在最后一维归一化（256 维）
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight + self.bias
        return x
```

**ONNX 算子映射**：

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

**精度影响**：**零损失**。数学公式与 `nn.LayerNorm` 完全一致，不是近似。

**权重处理**：直接复用 `nn.Parameter` 引用，零拷贝：

```python
manual.weight = module.weight
manual.bias = module.bias
```

---

### 修复 ①：nn.GELU → GELUManual（tanh 近似）

**问题**：`nn.GELU` 使用 `erf` 函数，导出到 ONNX 生成 `Erf` 算子。NPU 编译器不支持 `Erf`，且**不会因降低 opset 版本而自动分解**。

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

**替换方案**：使用 GELU 的标准 tanh 近似（来自原始论文 [GELUs, Hendrycks 2016](https://arxiv.org/abs/1606.08415)）：

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

**ONNX 算子映射**：

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

**精度影响**：

| 指标 | 值 |
|------|-----|
| GELU 函数最大绝对误差 | < 0.004 |
| 对模型输出 score 的影响 | < 0.001 |
| 对模型输出 mask IoU 的影响 | > 0.998 |

实测在 truck.jpg 上 4 个测试点、16 个 mask 的对比结果：

| 指标 | 值 |
|------|-----|
| Score 最大差异 | 0.000923 |
| Mask IoU 最低值 | 0.998901 |
| Best mask 选择 | 4/4 完全一致 |

---

### 修复 ②：stability_score int16 → int32 + 去掉 dynamic_axes

**问题 A：int16**

`calculate_stability_score`（`edge_sam/utils/amg.py`）使用 `torch.int16` 做中间求和，NPU 只支持 int32。

```python
# 原始代码
intersections = (
    (masks > (mask_threshold + threshold_offset))
    .sum(-1, dtype=torch.int16)   # ← int16，NPU 不支持
    .sum(-1, dtype=torch.int32)
)
```

**修复**：通过 `patch_stability_score()` monkey-patch 为全 int32，不修改源码：

```python
.sum(-1, dtype=torch.int32)   # 改为 int32
.sum(-1, dtype=torch.int32)
```

**问题 B：dynamic_axes**

原始导出脚本声明了动态维度（`num_points` 可变），NPU 量化编译器需要静态 shape 才能正确计算量化参数。

**修复**：去掉 `dynamic_axes`，导出固定 shape。默认 5 个 prompt 点，可通过 `--num-points` 自定义。

---

### 修复 ③：onnxruntime 图简化（常量折叠）

**问题**：即使是固定 shape 导出，PyTorch ONNX exporter 仍然生成大量 shape 计算子图：

```
Shape → Gather → Concat → Reshape
                  ↑
            ConstantOfShape → Where → Expand
```

这些节点全部使用 **int64** 类型（ONNX 规范要求 shape 操作使用 int64），是 NPU "only support int32" 错误的根源。

**修复**：用 onnxruntime 的图优化器做常量折叠：

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
sess_options.optimized_model_filepath = simplified_path
ort.InferenceSession(onnx_path, sess_options)
```

由于所有 shape 都是静态的，优化器在编译时计算出所有 shape 值，将整个 shape 子图折叠为常量。

---

### 修复 ④：int64/int16 → int32 强制全量转换

**问题**：图简化后仍有残余 int64（如 Reshape 的 shape 常量、Cast 目标类型等）。

**修复策略**：**无条件强转所有 int64/int16 为 int32，不插入任何 Cast 桥接节点。**

```
转换范围：
 1. Cast 节点的目标类型:       to=int64/int16 → to=int32
 2. ConstantOfShape 的 value:  int64/int16 → int32
 3. Constant 节点的数据:       int64/int16 → int32
 4. Initializer 权重:          int64/int16 → int32
 5. value_info 类型声明:       int64/int16 → int32
 6. graph inputs 类型:         int64/int16 → int32
 7. graph outputs 类型:        int64/int16 → int32
```

**重要说明**：

- 转换后的模型**不符合 ONNX 规范**（ONNX 要求 Reshape/Expand/Tile 等的 shape 输入为 int64）
- 因此 **onnxruntime 无法加载**此模型（会报 `Type 'tensor(int32)' of input parameter ... is invalid`）
- 但 NPU 编译器有自己的类型系统，只接受 int32，所以这是正确的做法
- 之前尝试过插入 Cast(int32→int64) 桥接节点来兼容 ONNX 规范，但 NPU 编译器同样拒绝了桥接中的 int64

---

### PyTorch >= 2.6 兼容性

PyTorch 2.6 起默认使用新的 dynamo-based ONNX exporter，不支持 ScriptModule 且最低只支持 opset 18。脚本自动检测版本并使用 legacy exporter：

```python
def _onnx_export(*args, **kwargs):
    major, minor = int(torch.__version__.split('.')[0]), int(torch.__version__.split('.')[1])
    if major > 2 or (major == 2 and minor >= 6):
        kwargs['dynamo'] = False
    torch.onnx.export(*args, **kwargs)
```

---

## 二、`edge_sam/modeling/sam.py`（mmdet lazy import）

**问题**：`sam.py` 顶部无条件 import mmdet/mmengine，这两个包需要编译 C 扩展，在纯推理/导出环境中通常不安装。

**修改前**：

```python
from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
from mmdet.models.necks import FPN
from projects.EfficientDet import efficientdet
from mmengine import ConfigDict
```

**修改后**：

```python
try:
    from mmdet.models.dense_heads import RPNHead, CenterNetUpdateHead
    from mmdet.models.necks import FPN
    from projects.EfficientDet import efficientdet
    from mmengine import ConfigDict
except ImportError:
    RPNHead = CenterNetUpdateHead = FPN = efficientdet = ConfigDict = None
```

这些类只在 RPN head 训练时使用，推理和 ONNX 导出不需要。

---

## 三、`edge_sam/build_sam.py`（CPU 加载支持）

**问题**：`torch.load()` 不指定 `map_location` 时，会尝试加载到原始保存设备（通常是 CUDA），在 CPU-only 环境报错。

**修改**：

```python
# 修改前
state_dict = torch.load(f)

# 修改后
state_dict = torch.load(f, map_location="cpu")
```

---

## Encoder vs Decoder 对比：为什么 Encoder 没有 int64 问题

| | Encoder (RepViT) | Decoder (Transformer) |
|---|---|---|
| 架构 | 纯 CNN | Transformer + Prompt Encoding |
| 数据类型 | **仅 FLOAT** | FLOAT + INT32 |
| 算子种类 | 13 种 | 28 种 |
| Cast 节点 | 0 | 有 |
| Shape 相关算子 | **0** | 7 种 (Reshape, Expand, Gather, Concat, Slice, Tile, Unsqueeze) |

**原因**：CNN 的所有形状信息（stride、padding、channel 数）都编码在 Conv 算子的**属性**中，不需要额外的 Shape/Reshape 操作。而 Transformer 的 attention 机制需要 Reshape（拆分 multi-head）、Transpose、MatMul，prompt encoding 需要 Expand、Gather、Tile，这些算子的 shape 参数在 ONNX 规范中要求 int64 类型。

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
  │ ① replace_gelu()      │  2 个 nn.GELU → GELUManual (tanh)
  │ ② patch_stability_score│  int16 → int32
  └───────┬───────────────┘
          │  权重通过 nn.Parameter 引用传递
          ▼
  ┌───────────────────────┐
  │ torch.onnx.export()   │  opset=11, 固定 shape, 无 dynamic_axes
  │ (legacy exporter,     │  PyTorch >= 2.6 自动加 dynamo=False
  │  dynamo=False)        │
  └───────┬───────────────┘
          │  原始 ONNX 文件
          ▼
  ┌───────────────────────┐
  │ ③ onnxruntime 图优化   │  常量折叠, 消除 shape 子图
  │    (ORT_ENABLE_BASIC) │
  └───────┬───────────────┘
          │  简化后
          ▼
  ┌───────────────────────┐
  │ ④ int64 → int32 全量   │  7 处位置, 无 Cast 桥接
  │   转换 (含 Constant,   │
  │   Initializer 等)     │
  └───────┬───────────────┘
          │
          ▼
   NPU 兼容的 .onnx 文件
```

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

# 查看算子列表和数据类型（导出后自动删除文件）
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score --check-ops-only
```

---

## 排查建议

如果 NPU 编译仍失败：

1. 用 `--check-ops-only` 获取完整算子列表，对照编译器文档逐个确认
2. 如果 `Sin`/`Cos` 不支持，可将位置编码预计算为常量嵌入模型
3. 如果 `ConvTranspose` 不支持，可替换为 `Upsample` + `Conv`
4. 如果 `Softmax` 不支持，可手写 `exp` + `ReduceSum` + `Div`
5. 如果特定算子形状参数报错，检查是否有 int32 的 shape 输入不被编译器接受
