# NPU 不支持的 ONNX 算子

通过 `scripts/ablate_npu_fixes.py` 消融实验确认，以下 9 个 ONNX 算子无法通过 NPU 编译器编译。

实验方法：将 Decoder 拆分为 3 段（Part 1 Prompt Encoding / Part 2 Transformer / Part 3 Mask Head），对 Part 1 的 2 个 Fix（A/B）和 Part 3 的 3 个 Fix（C/D/E）进行全组合消融，逐个提交 NPU 编译，记录 PASS/FAIL。

## 不支持算子一览

| 算子 | 模型位置 | PyTorch 来源 | 所属 Fix |
|------|---------|-------------|---------|
| `Sin` | Part 1: Prompt Encoding | `PositionEmbeddingRandom._pe_encoding()` 中的 `torch.sin(coords)` | Fix A |
| `Cos` | Part 1: Prompt Encoding | `PositionEmbeddingRandom._pe_encoding()` 中的 `torch.cos(coords)` | Fix A |
| `Abs` | Part 1: Prompt Encoding | `_float_eq()` 初版中的 `torch.abs(x - val)` | Fix B |
| `GatherND` | Part 1: Prompt Encoding | bool 索引赋值 `point_embedding[point_labels == -1] = 0.0` | Fix B |
| `ScatterND` | Part 1: Prompt Encoding | bool 索引赋值 `point_embedding[point_labels == -1] += embed` | Fix B |
| `NonZero` | Part 1: Prompt Encoding | bool 索引赋值（PyTorch 先用 NonZero 找到 True 的索引） | Fix B |
| `Gather` | Part 3: Mask Head | 整数索引 `hs[:, 0, :]`、`mask_tokens_out[:, i, :]`、`b, c, h, w = tensor.shape` | Fix C |
| `Greater` | Part 3: Mask Head | stability score 中的 `masks > threshold` | Fix E |
| `Cast(bool→int16)` | Part 3: Mask Head | stability score 中的 `.sum(-1, dtype=torch.int16)` | Fix E |

注：`Equal` 算子本身可通过编译（Part 2 Transformer 中有 1 个 Equal 且 PASS），但 Equal 配合 bool 索引赋值会生成 `NonZero`/`GatherND`/`ScatterND`，这些不支持。

注：`Erf` 算子可通过编译（Encoder 中有 27 个 Erf 且 PASS）。消融实验中 `part3_fixCE`（保留 Erf）编译通过，确认 Fix D（GELU→tanh）不必要。

### 关于 opset 11 支持的说明

NPU 编译器文档声称支持 ONNX opset 11，但实际测试表明上述 9 个算子（Sin、Cos、Abs、GatherND、ScatterND、NonZero、Gather、Greater、Cast(bool→int16)）均属于 opset 11 的标准算子，理应被支持但实际编译失败。这说明 NPU 编译器对 opset 11 的支持是不完整的，仅覆盖了部分算子子集。实际可用的算子以本文档的白名单（见附录）为准，不能依赖 opset 版本号来判断算子是否支持。

---

## 解决方法

### Fix A — 消除 Sin/Cos

**位置**：`edge_sam/modeling/prompt_encoder.py` → `PositionEmbeddingRandom._pe_encoding()`

**原始代码**（ONNX 内部计算 PE）：
```python
# _pe_encoding() 内部
coords = coords @ self.positional_encoding_gaussian_matrix
coords = 2 * np.pi * coords
return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
```

**解决方法**：将 PE 编码从 ONNX 模型中移出，在 CPU 端预计算。

```python
# CPU 端预计算（推理前调用）
def compute_point_pe(sam, point_coords):
    with torch.no_grad():
        coords = point_coords + 0.5
        coords = coords / sam.image_encoder.img_size
        pe = sam.prompt_encoder.pe_layer._pe_encoding(coords)
    return pe  # [1, N, 256]
```

ONNX 模型输入从 `point_coords [1,N,2]` 改为 `point_embedding_pe [1,N,256]`。

**精度影响**：零损失。计算完全相同，只是执行位置从 NPU 移到 CPU。

---

### Fix B — 消除 GatherND/ScatterND/NonZero/Abs

**位置**：`edge_sam/utils/coreml.py` → `SamCoreMLModel._embed_points()`

**原始代码**（bool 索引赋值）：
```python
point_embedding = point_embedding * (point_labels != -1)
point_embedding = point_embedding + self.not_a_point_embed.weight * (point_labels == -1)
for i in range(self.num_point_embeddings):
    point_embedding = point_embedding + self.point_embeddings[i].weight * (point_labels == i)
```

ONNX 导出后生成 `Equal → NonZero → GatherND → ScatterND` 算子链。

**解决方法**：用纯 float 算术替代 bool 比较和索引赋值。

```python
@staticmethod
def _float_eq(x, val):
    """整数值 float 的相等判断：relu(1 - (x-val)^2)"""
    diff = x - val
    return torch.relu(1.0 - diff * diff)
    # ONNX 算子: Sub, Mul, Sub, Relu（全在白名单中）

# 用法
point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding_pe)
mask_neg1 = _float_eq(point_labels, -1.0)
sparse = point_embedding_pe * (1.0 - mask_neg1)
sparse = sparse + self.not_a_point_embed.weight * mask_neg1
for i in range(num_point_embeddings):
    mask_i = _float_eq(point_labels, float(i))
    sparse = sparse + self.point_embeddings[i].weight * mask_i
```

**数学原理**：point labels 是整数值 float（-1, 0, 1, 2, 3），`diff = x - val` 只有两种情况：

| 条件 | diff | diff^2 | 1 - diff^2 | relu(...) |
|------|------|--------|------------|-----------|
| x == val | 0 | 0 | 1 | **1.0** |
| x != val | >= 1 整数 | >= 1 | <= 0 | **0.0** |

**精度影响**：零损失。与原实现完全等价。

---

### Fix C — 消除 Gather

**位置**：`edge_sam/modeling/mask_decoder.py` → `MaskDecoder.predict_masks()`

**原始代码**（整数索引 + 动态 shape）：
```python
iou_token_out = hs[:, 0, :]                             # → Gather(axis=1)
mask_tokens_out[:, i, :]                                 # → Gather(axis=1) x4
b, c, h, w = upscaled_embedding.shape                   # → Shape + Gather
masks = (hyper_in @ upscaled_embedding.view(b, c, h*w))  # 动态 shape
```

**解决方法**：用 Slice+Reshape 替代整数索引，用静态常量替代动态 shape。

```python
iou_token_out = hs[:, 0:1, :].reshape(1, 256)           # → Slice + Reshape
token_i = mask_tokens_out[:, i:i+1, :].reshape(1, 256)  # → Slice + Reshape x4
masks = (hyper_in @ upscaled.reshape(1, 32, 65536)).reshape(1, -1, 256, 256)
#                              ^^^^^^^^^^^^^^^^ 静态常量，无 Shape 算子
```

**关键**：不用 `squeeze()` 去掉维度（会导出为 `Squeeze` 算子，不在白名单中），用 `reshape()` 代替（Part 2 中有 29 个 Reshape 实例已确认通过）。

**精度影响**：零损失。数值计算完全等价。

---

### Fix E — 消除 Greater/Cast(bool)

**位置**：`edge_sam/utils/amg.py` → `calculate_stability_score()`

**原始代码**（bool 比较 + int 求和）：
```python
intersections = (
    (masks > (mask_threshold + threshold_offset))     # → Greater
    .sum(-1, dtype=torch.int16)                       # → Cast(bool→int16) + ReduceSum
    .sum(-1, dtype=torch.int32)
)
unions = (
    (masks > (mask_threshold - threshold_offset))
    .sum(-1, dtype=torch.int16)
    .sum(-1, dtype=torch.int32)
)
return intersections / unions
```

**解决方法**：用 sigmoid 阶跃函数近似 bool 比较，全程 float 运算。

```python
def _stability_score_npu(masks, mask_threshold, threshold_offset):
    k = 50.0  # 大斜率使 sigmoid 近似 step function
    high = torch.sigmoid(k * (masks - (mask_threshold + threshold_offset)))
    low  = torch.sigmoid(k * (masks - (mask_threshold - threshold_offset)))
    intersections = high.sum(-1).sum(-1)   # float ReduceSum
    unions = low.sum(-1).sum(-1)
    return intersections / unions
```

ONNX 算子：仅 `Mul`, `Sub`, `Sigmoid`, `ReduceSum(float)`, `Div`。

**精度影响**：score 最大偏差 < 0.004。二值化 mask 结果完全一致（IoU = 1.0）。

---

## 确认不需要的 Fix

### Fix D — GELU(Erf) → tanh 近似：不必要

消融实验确认 `Erf` 算子可通过 NPU 编译：

- Encoder 中有 27 个 Erf，编译通过
- `part3_fixCE`（保留 Erf）编译通过
- `part3_fixCDE`（去掉 Erf）也编译通过

移除 Fix D 可消除 tanh 近似带来的精度损失（mask 值最大偏差 0.014）。

---

## 附录：算子白名单

以下算子在 Encoder 或 Part 2 中存在且通过 NPU 编译，可安全使用：

```
Add, Concat, Constant, ConstantOfShape, Conv, Div, Equal, Erf, Expand,
MatMul, Mul, Pow, ReduceMean, Relu, Reshape, Resize, Shape, Sigmoid,
Slice, Softmax, Sqrt, Sub, Transpose, Where
```
