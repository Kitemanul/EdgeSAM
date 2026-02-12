# EdgeSAM NPU 编译排查过程记录

## 背景

EdgeSAM 需要部署到移动端 NPU（Qualcomm/MediaTek）。模型包含两个 ONNX 文件：
- **Encoder**（RepViT CNN）：纯卷积网络，约 130 个 Conv 算子
- **Decoder**：包含 Transformer 注意力、MLP、转置卷积等复杂结构

Encoder 直接通过 NPU 编译，Decoder 整体编译失败。

**核心挑战**：NPU 编译器的报错信息往往只指出第一个不支持的算子，但模型中可能有多个不同类型的不支持算子。需要系统性地定位并修复所有问题。

---

## 排查方法论

### 核心思路：分治法（Divide & Conquer）

Decoder 整体编译失败时，报错信息不足以定位所有问题。采用**拆分-隔离-修复**的策略：

1. **按功能边界拆分**：将 Decoder 按数据流拆分为独立的子模块
2. **逐个编译**：每个子模块单独导出 ONNX 并尝试 NPU 编译
3. **定位失败模块**：PASS/FAIL 二分法精确定位问题所在
4. **分析算子差异**：对比 PASS 模块和 FAIL 模块的算子集，找出不支持的算子
5. **最小化修改**：只替换不支持的算子，优先使用已在 PASS 模块中验证过的算子
6. **修复后重新编译**：验证修复是否有效，如有新问题继续迭代

### 辅助原则

- **算子白名单推理**：已通过编译的模块中出现的算子视为"安全"，新引入的算子应尽量来自白名单
- **精度优先**：优先选择零精度损失的替换方案；只在必要时接受微小精度损失
- **静态 shape**：NPU 编译器通常要求静态 shape，避免动态维度

---

## 排查过程

### 第一步：确定问题范围

将 Decoder 用原有脚本 `export_onnx_model.py` 导出为单个 ONNX 文件，提交 NPU 编译。

**结果**：编译失败，报错信息指向某个不支持的算子。

**问题**：Decoder 有约 200+ 个 ONNX 节点，涉及 20+ 种算子类型，单个报错无法反映全部问题。

**决定**：需要拆分 Decoder 以精确定位。

---

### 第二步：按功能边界拆分 Decoder

分析 Decoder 的数据流，确定 3 个天然的功能边界：

```
Part 1: Prompt Encoding
  输入: point_coords [1,N,2], point_labels [1,N]
  输出: sparse_embedding [1,N,256]
  功能: 位置编码 + 标签 embedding 选择

Part 2: Transformer
  输入: image_embeddings [1,256,64,64], sparse_embedding [1,N,256]
  输出: hs [1,5+N,256], src [1,4096,256]
  功能: token 准备 + 两层双向注意力

Part 3: Mask Head
  输入: hs [1,5+N,256], src [1,4096,256]
  输出: scores [1,4], masks [1,4,256,256]
  功能: 上采样 + 超网络 MLP + stability score
```

编写 `scripts/diagnose_npu_ops.py`，将每个 Part 作为独立的 `nn.Module` 导出。

**设计决策**：
- 所有模型使用静态 shape（batch=1, N=5），无动态维度
- 中间张量（dense_embedding, image_pe）作为 `register_buffer` 嵌入模型常量
- 使用 opset 11（NPU 编译器的常见要求）

---

### 第三步：逐段编译，建立算子白名单

编译结果：

| 模块 | 结果 | 算子种类 |
|------|------|---------|
| Encoder | PASS | 13 种 |
| Part 1: Prompt Encoding | FAIL | 13 种 |
| Part 2: Transformer | PASS | 20 种 |
| Part 3: Mask Head | FAIL | ~20 种 |

**建立算子白名单**：汇总所有 PASS 模块的算子集合

```
PASS 算子集（Encoder ∪ Part 2）:
Add, Concat, Constant, ConstantOfShape, Conv, Div, Equal, Erf, Expand,
MatMul, Mul, Pow, ReduceMean, Relu, Reshape, Resize, Shape, Sigmoid,
Slice, Softmax, Sqrt, Sub, Transpose, Where
```

---

### 第四步：修复 Part 3 — 消除 Gather

**报错**：`unsupported nodes: Gather`（4 个）

**分析 Gather 来源**：

| Python 代码 | ONNX 算子 | 个数 |
|-------------|----------|------|
| `hs[:, 0, :]` | `Gather(axis=1)` | 1 |
| `mask_tokens_out[:, i, :]` | `Gather(axis=1)` | 4 |
| `b, c, h, w = upscaled.shape` | `Shape + Gather` | 动态 |

**修复策略**：
- 整数索引 `tensor[:, i, :]` → 切片 `tensor[:, i:i+1, :]` + `reshape`
  - 切片导出为 `Slice`（在 Part 2 白名单中）
  - reshape 导出为 `Reshape`（在 Part 2 白名单中）
- 动态 shape `view(b, c, h*w)` → 静态 shape `reshape(1, 32, 65536)`

**为什么用 reshape 而不用 squeeze**：
- `squeeze` 导出为 `Squeeze` 算子
- `Squeeze` 不在任何 PASS 模块的算子集中
- `Reshape` 在 Part 2 中有 29 个实例，已确认安全

**修复后编译**：Part 3 PASS

---

### 第五步：修复 Part 1（第一轮）— 消除 Abs

**报错**：`failed to legalize operation onnx.Abs`

**分析 Abs 来源**：`_float_eq()` 函数中使用 `torch.abs()` 实现标签相等判断。

**原始实现**：
```python
def _float_eq(x, val):
    return torch.clamp(1.0 - torch.abs(x - val), min=0.0, max=1.0)
    # ONNX: Sub, Abs, Sub, Clip
```

**修复方案**：利用标签是整数值 float 的数学特性：

```python
def _float_eq(x, val):
    diff = x - val
    return torch.relu(1.0 - diff * diff)
    # ONNX: Sub, Mul, Sub, Relu
```

**数学证明**：
- 当 x == val 时：diff=0, diff²=0, 1-0=1, relu(1)=1.0 ✓
- 当 x != val 时（整数差 ≥ 1）：diff²≥1, 1-diff²≤0, relu(≤0)=0.0 ✓

与原实现完全等价，零精度损失。

**修复后编译**：Part 1 仍然 FAIL（新的报错）

---

### 第六步：修复 Part 1（第二轮）— 消除 Sin/Cos

**问题**：修复 Abs 后重新编译，Part 1 仍然失败。

**排查方法**：对比 Part 1 的算子集和所有 PASS 模块的算子白名单：

```
Part 1 算子: Add, Concat, Constant, Cos, Div, Expand, MatMul,
             Mul, Relu, Shape, Sin, Sub, Unsqueeze
```

逐一检查：

| 算子 | Encoder | Part 2 | Part 3 | 结论 |
|------|---------|--------|--------|------|
| Add | ✓ | ✓ | ✓ | 安全 |
| Concat | - | ✓ | ✓ | 安全 |
| Constant | ✓ | ✓ | ✓ | 安全 |
| **Cos** | - | - | - | **不在白名单** |
| Div | ✓ | ✓ | ✓ | 安全 |
| Expand | - | ✓ | - | 安全 |
| MatMul | - | ✓ | ✓ | 安全 |
| Mul | ✓ | ✓ | ✓ | 安全 |
| Relu | ✓ | ✓ | ✓ | 安全 |
| Shape | - | ✓ | - | 安全 |
| **Sin** | - | - | - | **不在白名单** |
| Sub | ✓ | ✓ | ✓ | 安全 |
| Unsqueeze | - | - | ✓ | 安全 |

**确认**：只有 `Sin` 和 `Cos` 不在任何 PASS 模型的算子集中。

**Sin/Cos 来源**：`PositionEmbeddingRandom._pe_encoding()` 中的 `torch.sin(coords)` 和 `torch.cos(coords)`。

**评估可能的替代方案**：

| 方案 | 可行性 | 原因 |
|------|--------|------|
| Taylor 多项式近似 | ✗ | 输入范围 [-12π, 12π] 太大，需要先做周期约简 |
| 周期约简 (mod 2π) | ✗ | 需要 `Floor` 算子，NPU 也不支持 |
| 查找表 + 索引 | ✗ | 需要 `Gather` 算子（已知不支持） |
| Padé 有理逼近 | ✗ | 同样需要周期约简 |
| ReLU 分段线性近似 | ✗ | 需要大量节点（~50K），不实用 |
| **移至 CPU 预计算** | **✓** | **零精度损失，微秒级开销** |

**最终方案**：将 PE 编码从 ONNX 模型中移出，在 CPU 端预计算。

Part 1 的 ONNX 模型不再包含 PE 计算，只做标签 embedding 选择。接口从 `(point_coords, point_labels)` 改为 `(point_embedding_pe, point_labels)`。

**修复后算子集**：
```
Add, Constant, Expand, Mul, Relu, Sub, Unsqueeze
```

全部在白名单中。

**修复后编译**：Part 1 PASS

---

### 第七步：全部通过

修复后的编译结果：

| 模块 | 结果 |
|------|------|
| Encoder | PASS |
| Part 1: Prompt Encoding | PASS |
| Part 2: Transformer | PASS |
| Part 3: Mask Head | PASS |

---

### 第八步：合并为单一模型

三段模型全部通过编译后，将修复合并回单一 Decoder 模型（`NpuSafeDecoder`），写入 `scripts/export_onnx_model_npu.py`。

合并过程：
1. 将 Part1 的标签 embedding 选择、Part2 的 Transformer、Part3 的 Mask Head 合并为单一 `nn.Module`
2. 保留所有 NPU 修复（float 标签比较、Slice+Reshape 索引、tanh GELU、sigmoid stability score）
3. PE 计算保持在 CPU 端（`compute_point_pe()` 函数）
4. 添加 ONNX 后处理（图简化 + int64→int32 转换）

---

## 经验总结

### 1. 分治法是排查 NPU 编译问题的最有效策略

NPU 编译器的报错信息有限，往往只给出第一个失败点。将模型按功能边界拆分后：
- 快速定位问题模块（从 200+ 节点缩小到 ~30 节点）
- 建立算子白名单，指导修复方向
- 可以并行修复不同模块的问题

### 2. 算子白名单推理比猜测更可靠

不要猜测某个算子是否支持，而是从已编译通过的模块中提取"白名单"。如果一个算子在 PASS 模块中出现过，它在相同上下文中大概率也是安全的。

### 3. 替换算子时优先选择白名单中的算子

每次修复时，引入的替代算子应来自白名单。例如：
- 用 `Relu`（Encoder 中有）替代 `Abs`
- 用 `Slice` + `Reshape`（Part 2 中有）替代 `Gather`
- 用 `Tanh` + `Mul`（Encoder 中有）替代 `Erf`

### 4. 不是所有算子都能在 ONNX 内替换

Sin/Cos 无法用 NPU 支持的算子组合替代（需要 Floor 做周期约简，但 Floor 也不支持）。这种情况下，将计算移到模型外部（CPU 端）是最干净的解决方案。

### 5. 精度验证不可省略

每次修复后都需要验证精度：
- 对比原始 PyTorch 模型（SamCoreMLModel）和修复后的 ONNX 流水线
- 使用多种 prompt 配置（单点、多点、box）
- 同时检查 score 差异和 mask IoU

### 6. 静态 shape 是 NPU 编译的基本要求

- 避免 `dynamic_axes`
- 用 Python int 常量替代动态 `tensor.shape` 解包
- Reshape 的目标 shape 使用字面量常数

### 7. 注意 ONNX 导出器的隐式行为

PyTorch → ONNX 导出器的一些隐式行为会引入意外算子：
- `tensor[:, i, :]`（整数索引）→ `Gather`
- `tensor.squeeze()`→ `Squeeze`
- `b, c, h, w = tensor.shape` → `Shape` + `Gather`
- `nn.GELU()` → `Erf`
- `tensor == value` → `Equal` + `Cast(bool)`

了解这些映射规则有助于提前避免问题。
