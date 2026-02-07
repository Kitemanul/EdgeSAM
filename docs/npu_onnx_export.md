# EdgeSAM Decoder ONNX 导出 — NPU 算子兼容方案

## 背景

EdgeSAM 的 ONNX decoder 模型在 TV NPU 编译器上编译失败。NPU 编译器推荐 opset 11，对高版本 opset 引入的算子支持有限。

| 导出 opset | 现象 |
|-----------|------|
| opset 18 | 编译器报错：`LayerNorm not supported` |
| opset 11 | 无 LayerNorm 报错，但编译仍然失败（推测是 `Erf` 算子不支持） |

encoder 编译通过，decoder 不通过，原因是 decoder 包含了 encoder 不使用的算子。

---

## 问题算子定位

### 1. `nn.LayerNorm` → ONNX `LayerNormalization` 算子

**位置**：`edge_sam/modeling/transformer.py` 中的 `TwoWayTransformer` 和 `TwoWayAttentionBlock`

```
TwoWayTransformer
├── layers (x2 TwoWayAttentionBlock)
│   ├── norm1 = nn.LayerNorm(256)    ← self-attention 后
│   ├── norm2 = nn.LayerNorm(256)    ← cross-attention (token→image) 后
│   ├── norm3 = nn.LayerNorm(256)    ← MLP 后
│   └── norm4 = nn.LayerNorm(256)    ← cross-attention (image→token) 后
└── norm_final_attn = nn.LayerNorm(256)  ← 最终 attention 后
```

共 **9 个** `nn.LayerNorm` 实例。

**ONNX 导出行为**：
- opset >= 17：导出为原生 `LayerNormalization` 算子（单个 op 节点）
- opset < 17：PyTorch exporter 自动分解为基本算子序列：
  ```
  ReduceMean → Sub → Pow → ReduceMean → Add(eps) → Sqrt → Div → Mul(weight) → Add(bias)
  ```

**为什么 encoder 没这个问题**：EdgeSAM 的 encoder 是 RepViT（纯 CNN 架构），使用 `BatchNorm`，不包含 `nn.LayerNorm`。

> **注意**：`mask_decoder.py:57` 的 `output_upscaling` 中的 `LayerNorm2d` 是自定义类（`edge_sam/modeling/common.py:32`），内部用 `mean` + `pow` + `sqrt` 手写实现，**不是** `nn.LayerNorm`，不受影响。

### 2. `nn.GELU` → ONNX `Erf` 算子

**位置**：`edge_sam/modeling/mask_decoder.py:55-61` 的 `output_upscaling`

```python
self.output_upscaling = nn.Sequential(
    nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),
    LayerNorm2d(64),       # ← 自定义实现，OK
    nn.GELU(),             # ← 问题算子
    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
    nn.GELU(),             # ← 问题算子
)
```

共 **2 个** `nn.GELU` 实例。

**ONNX 导出行为**：

`nn.GELU` 的标准数学定义：
```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

无论哪个 opset 版本，都会导出为 `Erf` 算子。`Erf` 虽然从 opset 9 就存在于 ONNX 标准中，但**大量 NPU 编译器不支持 `Erf`**。

**这就是 opset 11 下"没有报 LayerNorm 错误但编译仍失败"的最可能原因。**

**为什么 encoder 没这个问题**：RepViT encoder 内部使用的激活函数是 `nn.ReLU` 和 `nn.Hardswish`，不包含 `nn.GELU`。

---

## 解决方案

核心思路：在导出 ONNX **之前**，在 PyTorch 侧将不兼容的算子替换为用基本运算手写的等价实现。

脚本：`scripts/export_onnx_model_npu.py`

### 替换 1：`nn.LayerNorm` → `LayerNormManual`

**原始 `nn.LayerNorm` 计算**：
```
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

**替换实现**：
```python
class LayerNormManual(nn.Module):
    def forward(self, x):
        dims = list(range(-len(self.normalized_shape), 0))  # 归一化维度，如 [-1]
        mean = x.mean(dim=dims, keepdim=True)                # ReduceMean
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True) # Sub → Pow → ReduceMean
        x = (x - mean) / torch.sqrt(var + self.eps)          # Sub → Add → Sqrt → Div
        x = x * self.weight + self.bias                       # Mul → Add
        return x
```

**生成的 ONNX 算子**：`ReduceMean`, `Sub`, `Pow`, `Add`, `Sqrt`, `Div`, `Mul` — 全部是 opset 1-7 就有的基本算子。

**数值精度**：与 `nn.LayerNorm` **完全一致**（数学公式相同，不是近似）。

**权重处理**：直接复用原始 `nn.LayerNorm` 的 `weight` 和 `bias` 参数：
```python
manual.weight = module.weight   # 同一个 nn.Parameter 对象，零拷贝
manual.bias = module.bias
```

### 替换 2：`nn.GELU` → `GELUManual`（tanh 近似）

**原始 `nn.GELU` 计算**：
```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```
导出为 ONNX 时产生 `Erf` 算子。

**替换实现（tanh 近似）**：
```python
class GELUManual(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608028654 * (x + 0.044715 * x * x * x)
        ))
```

其中 `0.7978845608028654 = sqrt(2/pi)`。

这是 GELU 的标准 tanh 近似形式，来自论文 [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)。

**生成的 ONNX 算子**：`Mul`, `Add`, `Tanh` — 全部是最基本的算子。

**数值精度**：与精确 GELU 的最大绝对误差 < 0.004，对模型推理效果的影响可忽略。

> 实际上 PyTorch 的 `nn.GELU(approximate='tanh')` 用的就是这个公式，GPT 系列模型也使用 tanh 近似版本的 GELU。

**权重处理**：`nn.GELU` 没有可学习参数，直接替换即可。

---

## 替换前后 ONNX 算子对比

| 原始 PyTorch 模块 | 原始 ONNX 算子 (opset 18) | 原始 ONNX 算子 (opset 11) | 替换后 ONNX 算子 |
|---|---|---|---|
| `nn.LayerNorm` | `LayerNormalization` | `ReduceMean` + `Sub` + `Pow` + `ReduceMean` + `Add` + `Sqrt` + `Div` + `Mul` + `Add` | `ReduceMean` + `Sub` + `Pow` + `ReduceMean` + `Add` + `Sqrt` + `Div` + `Mul` + `Add` |
| `nn.GELU` | `Erf` + `Mul` + `Add` + `Mul` | `Erf` + `Mul` + `Add` + `Mul` | `Mul` + `Add` + `Tanh` + `Add` + `Mul` + `Mul` |

关键变化：**消除了 `LayerNormalization` 和 `Erf` 两个 NPU 不支持的算子**。

---

## 执行流程

```
加载 .pth 权重到原始模型结构
            │
            ▼
    权重加载完成（所有参数就位）
            │
            ▼
  replace_layernorm(): 替换 9 个 nn.LayerNorm
  replace_gelu():      替换 2 个 nn.GELU
            │
            ▼     权重通过 nn.Parameter 引用传递，零拷贝
    torch.onnx.export() 以 opset 11 导出
            │
            ▼
     生成 NPU 兼容的 .onnx 文件
```

不需要修改 .pth 文件，不需要重新训练。

---

## 使用方法

```bash
# 导出 NPU 兼容的 decoder
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score

# 导出 encoder（通常不需要，原始导出脚本就能编译通过）
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth

# 仅查看导出模型中的所有 ONNX 算子（排查兼容性）
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --use-stability-score --check-ops-only

# 指定输出路径
python scripts/export_onnx_model_npu.py weights/edge_sam_3x.pth \
    --decoder --output my_decoder.onnx

# 使用 onnx-simplifier 进一步优化（推荐）
pip install onnx-simplifier
python -m onnxsim edge_sam_3x_decoder_npu.onnx edge_sam_3x_decoder_npu_sim.onnx
```

---

## Decoder 完整算子清单（替换后预期）

以下是 opset 11 导出、替换 LayerNorm/GELU 后，decoder ONNX 模型中预期出现的算子：

| 算子 | 来源 | opset 版本 |
|------|------|-----------|
| `Add` | 各处加法、残差连接、bias | 1 |
| `Cast` | 类型转换（stability score 中的 int16/int32） | 1 |
| `Concat` | tensor 拼接 | 1 |
| `Constant` | 常量 | 1 |
| `ConvTranspose` | mask upscaling 的转置卷积 | 1 |
| `Cos` | 位置编码 | 7 |
| `Div` | LayerNorm 中的归一化、attention 缩放 | 1 |
| `Equal` | prompt label 比较 | 1 |
| `Expand` | tensor 广播 | 8 |
| `Gather` | embedding 索引 | 1 |
| `Gemm` / `MatMul` | 线性层、attention 计算 | 1 |
| `Mul` | 权重缩放、GELU 计算 | 1 |
| `Pow` | LayerNorm 中的方差计算 | 1 |
| `ReduceMean` | LayerNorm 中的均值/方差 | 1 |
| `ReduceSum` | stability score 求和 | 1 |
| `Reshape` | tensor 形状变换 | 5 |
| `Shape` | 获取 tensor 形状 | 1 |
| `Sin` | 位置编码 | 7 |
| `Softmax` | attention 权重归一化 | 1 |
| `Sqrt` | LayerNorm 中的标准差 | 1 |
| `Sub` | LayerNorm 中的去均值 | 1 |
| `Tanh` | GELU tanh 近似 | 1 |
| `Transpose` | attention head 重排 | 1 |
| `Unsqueeze` | 维度扩展 | 1 |

所有算子均在 **opset 11 以内**，且都是基本运算算子。

---

## 排查建议

如果替换后 NPU 编译仍失败：

1. 用 `--check-ops-only` 获取实际算子列表
2. 对照 NPU 编译器文档，找出不支持的算子
3. 用 `onnx-simplifier` 简化模型（可能消除冗余节点）
4. 如果 `Sin`/`Cos`（位置编码）不支持，可以考虑将位置编码预计算为常量
5. 如果 `ConvTranspose` 不支持，可以用 `Upsample` + `Conv` 替代
