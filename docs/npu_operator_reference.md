# NPU ONNX 算子兼容性参考手册

本文档基于 EdgeSAM 在 NPU 上的实际编译测试结果，记录所有已验证的可用算子（白名单）和不可用算子（黑名单），并说明每个算子的功能原理。黑名单算子附有通用的替代方案，可供其他模型的 NPU 适配参考。

验证方法：将模型拆分为 4 个子模块（Encoder、Part 1 Prompt Encoding、Part 2 Transformer、Part 3 Mask Head），逐个编译并汇总算子通过情况。白名单中的每个算子至少在一个通过编译的子模块中出现过。

---

## 白名单算子（24 种，已确认 NPU 编译通过）

### 算术运算

| 算子 | 功能 | 验证来源 |
|------|------|---------|
| **Add** | 逐元素加法 `a + b` | Encoder, Part 2, Part 3 |
| **Sub** | 逐元素减法 `a - b` | Encoder, Part 2, Part 3 |
| **Mul** | 逐元素乘法 `a * b` | Encoder, Part 2, Part 3 |
| **Div** | 逐元素除法 `a / b` | Encoder, Part 2, Part 3 |
| **Sqrt** | 逐元素平方根 `√x` | Encoder, Part 2 |
| **Pow** | 逐元素幂运算 `x^n`，LayerNorm 中用于计算方差（`(x - mean)²`） | Encoder, Part 2 |
| **Erf** | 误差函数 `erf(x) = (2/√π)∫₀ˣ e^{-t²}dt`，`nn.GELU` 激活函数的核心算子：`GELU(x) = 0.5x(1 + erf(x/√2))` | Encoder (27 个) |

### 激活函数

| 算子 | 功能 | 验证来源 |
|------|------|---------|
| **Relu** | 修正线性单元 `max(0, x)` | Encoder, Part 2, Part 3 |
| **Sigmoid** | S 型激活 `1/(1+e^{-x})`，值域 (0, 1) | Encoder, Part 3 |
| **Softmax** | 归一化指数函数 `softmax(x)ᵢ = eˣⁱ / Σeˣʲ`，attention 中将 score 转为概率分布 | Part 2 |
| **Tanh** | 双曲正切 `(eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)`，值域 (-1, 1) | Part 3 |

### 归约运算

| 算子 | 功能 | 验证来源 |
|------|------|---------|
| **ReduceMean** | 沿指定轴求均值。LayerNorm 的核心操作：`mean = ReduceMean(x, axis)` | Encoder, Part 2 |
| **ReduceSum** | 沿指定轴求和（**仅 float 类型**，int16/bool 类型不支持） | Part 3 |

### 矩阵运算

| 算子 | 功能 | 验证来源 |
|------|------|---------|
| **MatMul** | 矩阵乘法 `C = A @ B`。attention 中的 Q·K^T 和 attn·V 均使用此算子 | Part 2, Part 3 |
| **Gemm** | 通用矩阵乘法 `Y = αAB + βC`，`nn.Linear` 全连接层的 ONNX 表示 | Part 3 |
| **Conv** | 卷积运算。Encoder (RepViT) 的核心算子，包括 depthwise conv、pointwise conv 等 | Encoder (130 个) |
| **ConvTranspose** | 转置卷积（反卷积），用于上采样。Mask Head 的 `output_upscaling` 中使用 | Part 3 |

### 形状与索引

| 算子 | 功能 | 验证来源 |
|------|------|---------|
| **Reshape** | 改变 tensor 形状但不改变数据。如 `[1, 4096, 256]` → `[1, 256, 64, 64]` | Part 2 (29 个), Part 3 |
| **Transpose** | 转置 tensor 的维度顺序。如 `[B, HW, C]` → `[B, C, HW]` | Part 2, Part 3 |
| **Slice** | 切片操作 `tensor[:, start:end, :]`。提取 tensor 的子区间 | Part 2, Part 3 |
| **Concat** | 沿指定轴拼接多个 tensor。如将 `iou_token` 和 `mask_tokens` 拼接 | Part 2, Part 3 |
| **Expand** | 按广播规则扩展 tensor 维度（不复制数据）。如 `[1, 5, 256]` → `[B, 5, 256]` | Part 2 |
| **Resize** | 插值缩放（双线性/最近邻），用于特征图上采样 | Encoder |

### 其他

| 算子 | 功能 | 验证来源 |
|------|------|---------|
| **Constant** | 存储常量值（标量或 tensor），嵌入在 ONNX 图中 | 所有模块 |
| **ConstantOfShape** | 创建指定形状的 tensor 并填充常量值 | Part 2 |
| **Shape** | 获取 tensor 的形状信息，返回 int64 tensor。通常与 Reshape 配合使用 | Part 2 |
| **Equal** | 逐元素相等比较 `a == b`，返回 bool tensor | Part 2 (1 个) |
| **Where** | 条件选择 `Where(cond, x, y)` — cond 为 true 取 x，否则取 y | Part 2 |

---

## 黑名单算子（9 种，已确认 NPU 编译失败）

### Sin / Cos

**功能**：三角函数 `sin(x)` 和 `cos(x)`，将角度映射为 [-1, 1] 的周期函数。

**Transformer 中的来源**：
- Sinusoidal 位置编码：`PE(pos, 2i) = sin(pos / 10000^{2i/d})`
- RoPE 旋转位置编码（LLM 中广泛使用）

**为什么难以在 ONNX 内替换**：
- Taylor 展开需要先做周期约简（mod 2π），而 `Floor` 算子也不被支持
- 查找表需要 `Gather` 算子（也不被支持）
- 分段线性近似需要大量节点（约 50K），不实用

**解决方案 — 移至 CPU 预计算**：

将 sin/cos 计算从 ONNX 模型中移出，在 CPU 端完成。ONNX 模型接收预计算好的结果作为输入。

```python
# 原始（ONNX 内部，NPU 不支持）
coords = coords @ gaussian_matrix
coords = 2 * PI * coords
pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

# 修改后（CPU 端预计算，ONNX 模型不再包含此部分）
def compute_pe_on_cpu(coords, gaussian_matrix):
    coords = coords @ gaussian_matrix
    coords = 2 * PI * coords
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

# ONNX 模型输入从 raw_coords 改为 precomputed_pe
```

**适用范围**：任何使用 sinusoidal PE 或 RoPE 的 Transformer 模型。位置编码通常只依赖坐标/位置索引，计算量小（微秒级），移到 CPU 不影响总延迟。

**精度影响**：零损失。

---

### Gather

**功能**：按索引从 tensor 中提取元素。`Gather(data, indices, axis)` 沿指定轴用 indices 选取 data 的切片。

**Transformer 中的来源**：
- Python 整数索引 `tensor[:, i, :]` 导出为 `Gather(axis=1, indices=i)`
- Embedding lookup `embedding_table[token_ids]`
- 动态 shape 解包 `b, c, h, w = tensor.shape` → `Shape` + `Gather`

**解决方案 — Slice + Reshape 替代**：

```python
# 原始（生成 Gather 算子）
token = hs[:, 0, :]           # shape: [B, dim]

# 修改后（生成 Slice + Reshape，均在白名单中）
token = hs[:, 0:1, :].reshape(1, 256)  # Slice + Reshape
```

**关键细节**：
- `[:, i, :]` 整数索引 → Gather（不支持）
- `[:, i:i+1, :]` 切片索引 → Slice（支持）
- 不用 `squeeze()` 去掉多出的维度（会生成 `Squeeze` 算子，未验证），改用 `reshape()` 到目标形状

对于动态 shape，改用静态常量：

```python
# 原始（生成 Shape + Gather）
b, c, h, w = tensor.shape
output = tensor.view(b, c, h * w)

# 修改后（静态常量，无 Shape/Gather）
output = tensor.reshape(1, 256, 4096)  # 直接写死维度值
```

**适用范围**：所有 Transformer 中提取特定 token 的操作（如 CLS token、mask token）。

**精度影响**：零损失。

---

### GatherND / ScatterND / NonZero

**功能**：
- `GatherND`：N 维索引取值，按坐标列表从 tensor 中提取元素
- `ScatterND`：N 维索引赋值，按坐标列表向 tensor 中写入元素
- `NonZero`：返回所有非零元素的索引坐标

这三个算子通常一起出现，来源于 PyTorch 的 bool 索引赋值模式。

**Transformer 中的来源**：

```python
# Python 的 bool 索引赋值
tensor[mask] = value

# PyTorch ONNX 导出器的翻译：
# 1. mask 是 bool tensor（来自 Equal/Greater 比较）
# 2. NonZero(mask) → 找到 True 的坐标
# 3. GatherND(tensor, coords) → 取出对应元素
# 4. 修改值
# 5. ScatterND(tensor, coords, new_values) → 写回
```

常见于：attention mask 应用、条件赋值、label 索引选择等。

**解决方案 — 用乘法 mask 替代 bool 索引赋值**：

将 `tensor[condition] = value` 改写为 `tensor = tensor * (1 - mask) + value * mask`，其中 mask 通过纯 float 算术计算。

```python
# 原始（生成 Equal + NonZero + GatherND + ScatterND）
embedding[labels == -1] = 0.0
embedding[labels == -1] += not_a_point_embed

# 修改后（纯 float 算术，仅生成 Sub, Mul, Relu, Add）
def _float_eq(x, val):
    """整数值 float 的相等判断"""
    diff = x - val
    return torch.relu(1.0 - diff * diff)  # x==val → 1.0, x!=val → 0.0

mask = _float_eq(labels, -1.0)
embedding = embedding * (1.0 - mask) + not_a_point_embed * mask
```

**`_float_eq` 的数学原理**：

对于整数值 float（如 -1, 0, 1, 2, 3），`diff = x - val` 要么为 0（相等），要么 |diff| >= 1（不相等）：
- 相等时：`diff² = 0`，`1 - 0 = 1`，`relu(1) = 1.0`
- 不相等时：`diff² >= 1`，`1 - diff² <= 0`，`relu(≤0) = 0.0`

**限制**：此方法仅适用于比较值为整数的场景。对于连续值比较（如 `tensor[x > 0.5]`），需要使用 sigmoid 近似（见 Greater 的解决方案）。

**精度影响**：零损失（对整数值 float 精确等价）。

---

### Abs

**功能**：逐元素绝对值 `|x|`。

**来源**：在本项目中出现于 `_float_eq` 的早期实现版本 `clamp(1 - abs(x - val), 0, 1)`。也常见于 L1 损失、梯度裁剪等场景。

**解决方案 — 用平方替代**：

在可以使用平方的场景下（如比较距离）：

```python
# 原始（生成 Abs）
result = 1.0 - torch.abs(diff)

# 修改后（Abs → Mul，利用 x² >= 0 的性质）
result = 1.0 - diff * diff
```

注意：`abs(x)` 和 `x²` 并非通用等价，只在 `abs` 用于判断"是否为零"或作为距离度量时可替换。

**精度影响**：取决于具体场景。整数差值场景下零损失。

---

### Greater

**功能**：逐元素大于比较 `a > b`，返回 bool tensor。

**Transformer 中的来源**：
- Attention mask：`mask = (positions > threshold)`
- 阈值筛选：`valid = (scores > min_score)`
- 二值化：`binary_mask = (logits > 0)`

**解决方案 — sigmoid 阶跃近似**：

用 `sigmoid(k * (x - threshold))` 近似 `x > threshold` 的 0/1 阶跃函数。k 越大越接近真正的阶跃。

```python
# 原始（生成 Greater + Cast(bool→float)）
high_count = (masks > threshold).sum()

# 修改后（纯 float，生成 Sigmoid + Mul + ReduceSum）
k = 50.0  # 斜率，越大越接近阶跃
high_count = torch.sigmoid(k * (masks - threshold)).sum()
```

**sigmoid 近似的数学原理**：

```
sigmoid(k * x) ≈ step(x)  当 k 足够大时

当 x > 0:  sigmoid(50 * x) ≈ 1.0  (如 x=0.1 时 sigmoid(5) = 0.9933)
当 x < 0:  sigmoid(50 * x) ≈ 0.0  (如 x=-0.1 时 sigmoid(-5) = 0.0067)
当 x = 0:  sigmoid(0) = 0.5        (边界值，通常可接受)
```

**适用范围**：任何需要 `tensor > threshold` 后求和、求均值的场景。不适用于需要精确 bool 索引的场景（那种情况应使用乘法 mask 方案）。

**精度影响**：极小。k=50 时，在距离阈值 0.1 以外的区域误差 < 0.007。

---

### Cast(bool→int16)

**功能**：类型转换，将 bool tensor 转为 int16。通常伴随 Greater/Equal 出现，为了对比较结果做求和等数值运算。

**来源**：`(masks > threshold).sum(-1, dtype=torch.int16)` — 先比较得到 bool，再 cast 为 int16，最后 ReduceSum。NPU 不支持 bool/int16 类型的 ReduceSum。

**解决方案**：与 Greater 的解决方案配合，用 sigmoid 将比较结果直接表示为 float，跳过 bool 和 Cast 环节。见上文 Greater 的解决方案。

**精度影响**：同 Greater。

---

## 通用替代方案速查表

| 原始模式 | 不支持算子 | 替代方案 | 使用的白名单算子 | 精度 |
|---------|-----------|---------|----------------|------|
| `sin(x)`, `cos(x)` | Sin, Cos | 移至 CPU 预计算 | 无（不在 ONNX 中） | 无损 |
| `tensor[:, i, :]` | Gather | `tensor[:, i:i+1, :].reshape(...)` | Slice, Reshape | 无损 |
| `b,c,h,w = t.shape; t.view(b,c,h*w)` | Shape, Gather | `t.reshape(1, 256, 4096)` 静态常量 | Reshape | 无损 |
| `tensor[bool_mask] = value` | NonZero, GatherND, ScatterND | `tensor * (1-mask) + value * mask` | Mul, Sub, Add | 无损 |
| `x == val`（整数 float） | Equal + Cast | `relu(1 - (x-val)²)` | Sub, Mul, Relu | 无损 |
| `abs(x)` | Abs | `x * x`（距离/判零场景） | Mul | 视场景 |
| `tensor > threshold` | Greater | `sigmoid(50 * (tensor - threshold))` | Sigmoid, Mul, Sub | 极小 |
| `(t > th).sum(dtype=int16)` | Greater, Cast, ReduceSum(int16) | `sigmoid(50*(t-th)).sum()` | Sigmoid, ReduceSum(float) | 极小 |
| `squeeze(dim)` | Squeeze（未验证） | `reshape(target_shape)` | Reshape | 无损 |

---

## 附录 A：白名单算子的验证来源

```
Encoder (PASS) — 13 种:
  Add(104) Constant(87) Conv(130) Div(29) Erf(27) Mul(66) Pow(2)
  ReduceMean(14) Relu(10) Resize(1) Sigmoid(10) Sqrt(2) Sub(2)

Part 2: Transformer (PASS) — 20 种:
  Add(66) Concat(2) Constant(60) ConstantOfShape(1) Div(16) Equal(1)
  Expand(1) MatMul(46) Mul(10) Pow(9) ReduceMean(18) Relu(2)
  Reshape(29) Shape(1) Slice(1) Softmax(7) Sqrt(9) Sub(9)
  Transpose(29) Where(1)

Part 3: Mask Head — 修复后通过 (PASS) — 20 种:
  Add(6) Concat(1) Constant(37) ConvTranspose(2) Div(2) Erf(2) Gemm(12)
  MatMul(1) Mul(15) Pow(1) ReduceMean(2) ReduceSum(4) Relu(8)
  Reshape(3) Sigmoid(2) Slice(5) Sqrt(1) Sub(3) Tanh(2)
  Transpose(1) Unsqueeze(4)
```

## 附录 B：黑名单算子在 EdgeSAM 中的分布

| 算子 | 出现位置 | 出现原因 | 由哪个 Fix 消除 |
|------|---------|---------|---------------|
| Sin | Part 1 | 位置编码 `_pe_encoding()` | Fix A: PE 移至 CPU |
| Cos | Part 1 | 位置编码 `_pe_encoding()` | Fix A: PE 移至 CPU |
| Abs | Part 1 | 标签比较 `_float_eq()` 初版 | Fix B: `relu(1-diff²)` |
| GatherND | Part 1 | bool 索引 `embedding[labels==v]` | Fix B: 乘法 mask |
| ScatterND | Part 1 | bool 索引赋值 | Fix B: 乘法 mask |
| NonZero | Part 1 | bool 索引内部 | Fix B: 乘法 mask |
| Gather | Part 3 | 整数索引 `hs[:, 0, :]` | Fix C: Slice+Reshape |
| Greater | Part 3 | 阈值比较 `masks > th` | Fix E: sigmoid 近似 |
| Cast(bool→int16) | Part 3 | `sum(dtype=int16)` | Fix E: float ReduceSum |
