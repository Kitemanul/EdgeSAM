# NPU Fix 消融测试指南

## 目的

EdgeSAM decoder 的 NPU 导出当前包含 5 个 Fix，但不确定每个 Fix 是否真正必要。
特别是 Encoder（RepViT）中有 27 个 Erf 算子能通过编译，暗示 Fix D（GELU→tanh）可能多余。

本指南通过**逐个开关每个 Fix**，用 NPU 编译器验证，找出真正必要的最小 Fix 集合。

---

## 5 个 Fix 一览

| Fix | 位置 | 消除的算子 | 替代方案 | 可疑程度 |
|-----|------|-----------|---------|---------|
| **A** | Part 1 | `Sin`, `Cos` | PE 移到 CPU 预计算 | 低（之前已确认失败） |
| **B** | Part 1 | `Equal`, `Abs` | `relu(1 - diff²)` 浮点比较 | 低（之前已确认失败） |
| **C** | Part 3 | `Gather` | `[:, i:i+1, :].reshape()` | 低（之前已确认失败） |
| **D** | Part 3 | `Erf` (GELU) | tanh 近似 | **高 — Encoder 有 27 个 Erf 能过** |
| **E** | Part 3 | bool `ReduceSum` | sigmoid 阶跃近似 | 低（之前已确认失败） |

---

## 步骤 1：导出所有变体

```bash
python scripts/ablate_npu_fixes.py weights/edge_sam_3x.pth --output-dir ./npu_ablation
```

生成 13 个 ONNX 文件：

```
npu_ablation/
├── part1_vanilla.onnx      # 无 Fix
├── part1_fixA.onnx         # 仅 Fix A (去 Sin/Cos)
├── part1_fixB.onnx         # 仅 Fix B (去 Equal/Abs)
├── part1_fixAB.onnx        # Fix A+B (全部 Part 1 Fix)
├── part2_transformer.onnx  # 始终通过
├── part3_vanilla.onnx      # 无 Fix
├── part3_fixC.onnx         # 仅 Fix C (去 Gather)
├── part3_fixD.onnx         # 仅 Fix D (去 Erf)
├── part3_fixE.onnx         # 仅 Fix E (去 bool ReduceSum)
├── part3_fixCD.onnx        # Fix C+D
├── part3_fixCE.onnx        # Fix C+E
├── part3_fixDE.onnx        # Fix D+E
└── part3_fixCDE.onnx       # Fix C+D+E (全部 Part 3 Fix)
```

---

## 步骤 2：用 NPU 编译器逐个编译

对每个 ONNX 文件执行 NPU 编译，记录 PASS 或 FAIL（附错误信息）。

建议按以下顺序测试（先测 baseline，再测单个 Fix）：

### Part 2（确认 baseline）

| # | 文件 | 预期 | 实际结果 | 错误信息 |
|---|------|------|---------|---------|
| 1 | `part2_transformer.onnx` | PASS | | |

### Part 1（2 个 Fix，4 个变体）

| # | 文件 | 启用的 Fix | 预期 | 实际结果 | 错误信息 |
|---|------|-----------|------|---------|---------|
| 2 | `part1_vanilla.onnx` | 无 | FAIL | | |
| 3 | `part1_fixA.onnx` | A | ? | | |
| 4 | `part1_fixB.onnx` | B | ? | | |
| 5 | `part1_fixAB.onnx` | A+B | PASS | | |

### Part 3（3 个 Fix，8 个变体）

| # | 文件 | 启用的 Fix | 预期 | 实际结果 | 错误信息 |
|---|------|-----------|------|---------|---------|
| 6 | `part3_vanilla.onnx` | 无 | FAIL | | |
| 7 | `part3_fixC.onnx` | C | ? | | |
| 8 | `part3_fixD.onnx` | D | ? | | |
| 9 | `part3_fixE.onnx` | E | ? | | |
| 10 | `part3_fixCD.onnx` | C+D | ? | | |
| 11 | `part3_fixCE.onnx` | C+E | ? | | |
| 12 | `part3_fixDE.onnx` | D+E | ? | | |
| 13 | `part3_fixCDE.onnx` | C+D+E | PASS | | |

---

## 步骤 3：分析结果

### Part 1 判定逻辑

```
part1_vanilla = FAIL  (确认有问题)
                │
                ├── part1_fixA = PASS ?
                │     → Fix A 单独就够，Fix B 不需要
                │
                ├── part1_fixB = PASS ?
                │     → Fix B 单独就够，Fix A 不需要
                │
                ├── 都 FAIL, part1_fixAB = PASS ?
                │     → A 和 B 都需要
                │
                └── 都 PASS ?
                      → A 或 B 任选其一即可
```

**重点关注**：`part1_fixA.onnx` 的结果。Fix A 去掉了 Sin/Cos 但保留了 Equal/Cast 模式。
如果它能过，说明 NPU 实际上支持 Equal，Fix B 是多余的。

### Part 3 判定逻辑

```
part3_vanilla = FAIL  (确认有问题)
                │
                ├── 哪些单 Fix 能 PASS？
                │     fixC=PASS → Gather 是唯一问题
                │     fixD=PASS → Erf 是唯一问题
                │     fixE=PASS → bool ReduceSum 是唯一问题
                │     全 FAIL   → 至少需要组合
                │
                └── 看组合结果，找最小通过集合
```

**重点关注**：`part3_fixD.onnx`（仅去 Erf）。
Encoder 有 27 个 Erf 能过，所以预期 Fix D 单独不影响结果。
如果 `part3_fixCE.onnx`（不含 Fix D）能通过，就证实 Fix D 不需要。

### 快速判定表

| 场景 | 结论 |
|------|------|
| `part3_fixCE.onnx` = PASS | **Fix D 不需要**，最小集 = C+E |
| `part3_fixCE.onnx` = FAIL, `part3_fixCDE.onnx` = PASS | Fix D 也需要，最小集 = C+D+E |
| `part3_fixC.onnx` = PASS | Fix C 就够了，D 和 E 都不需要 |
| `part3_fixE.onnx` = PASS | Fix E 就够了，C 和 D 都不需要 |

---

## 步骤 4：精简 `export_onnx_model_npu.py`

根据测试结果，从 `NpuSafeDecoder` 中删除不需要的 Fix。

**如果确认 Fix D 不需要**（最可能的结果）：

1. 删除 `_GELUTanh` 类和 `_replace_gelu()` 函数
2. 删除 `NpuSafeDecoder.__init__` 中的 GELU 替换调用
3. 更新文档，将 Fix 数从 5 降为 4

**如果发现 Fix B 不需要**（Part 1 不需要 float 算术）：

1. `NpuSafeDecoder.forward` 中 Stage 1 可恢复为原始 `Equal`/索引赋值
2. 删除 `_float_eq()` 方法

---

## 预期结果

根据已有信息，最可能的结论：

| Fix | 预测 | 理由 |
|-----|------|------|
| A (Sin/Cos → CPU) | **需要** | Sin/Cos 不在任何 PASS 模型中 |
| B (Equal/Abs → float) | **需要** | Abs 不在任何 PASS 模型中；但 Equal 在 Part 2 中存在（1个） |
| C (Gather → Slice) | **需要** | Gather 不在任何 PASS 模型中 |
| D (Erf → tanh) | **不需要** | Encoder 有 27 个 Erf 且编译通过 |
| E (bool ReduceSum → sigmoid) | **需要** | bool ReduceSum 不在任何 PASS 模型中 |

关于 Fix B 的一个微妙点：Part 2 包含 `Equal(1)` 且通过编译。
但 Part 1 vanilla 中的 label 比较模式是 `Equal → Where → Cast`，与 Part 2 中 `Equal` 的用法
可能不同。需要实测确认。

---

## 附录 A：各变体的关键算子差异

下表列出每个变体中与 NPU 兼容性相关的关键算子（有 = 含该算子，无 = 不含）：

### Part 1 变体

| 变体 | Sin/Cos | Equal | Gather/GatherND/ScatterND | Cast |
|------|---------|-------|--------------------------|------|
| vanilla | 有 | 有 | 有 | 有 |
| fixA | 无 | 有 | 有 | 有 |
| fixB | 有 | 无 | 无 | 无 |
| fixAB | 无 | 无 | 无 | 无 |

注意：vanilla 和 fixA 导出了 `GatherND`/`ScatterND`/`NonZero`（来自 `label == val` 索引赋值模式），
这些在 Fix B 中被 float 算术完全消除。

### Part 3 变体

| 变体 | Gather | Erf | Greater/Cast(bool) | 关键新算子 |
|------|--------|-----|-------------------|-----------|
| vanilla | 有 | 有 | 有 | — |
| fixC | **无** | 有 | 有 | — |
| fixD | 有 | **无** | 有 | Tanh |
| fixE | 有 | 有 | **无** | Sigmoid |
| fixCD | **无** | **无** | 有 | Tanh |
| fixCE | **无** | 有 | **无** | Sigmoid |
| fixDE | 有 | **无** | **无** | Tanh, Sigmoid |
| fixCDE | **无** | **无** | **无** | Tanh, Sigmoid |

---

## 附录 B：算子到 Fix 的映射

| 算子 | 哪个 Fix 消除它 | 是否在 PASS 模型中出现过 |
|------|---------------|----------------------|
| `Sin` | Fix A | 否 |
| `Cos` | Fix A | 否 |
| `Equal` | Fix B | 是（Part 2 有 1 个） |
| `GatherND`/`ScatterND`/`NonZero` | Fix B | 否 |
| `Gather` | Fix C | 否 |
| `Erf` | Fix D | **是（Encoder 有 27 个）** |
| `Greater` | Fix E | 否 |
| `Cast(→int16)` | Fix E | 否 |
