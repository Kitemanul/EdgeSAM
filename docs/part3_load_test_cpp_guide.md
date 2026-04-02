# Part3 C++ Load 测试指南（给另一个 AI）

目的：让另一个 AI 直接写出 `tests/test_part3_load_only.cpp`，批量测试 Part3 变体的 `loadModel`。

---

## 1) 最重要结论

- 常规变体使用 4-mask 输出。
- 新增一个诊断变体：`p3_single_mask`（只输出 1 个 mask），用于优先排查共享子图复杂度问题。

### 变体列表
- `p3_base`
- `p3_no_score`
- `p3_no_deconv`
- `p3_no_gemm`
- `p3_no_tanh`
- `p3_no_unsqueeze`
- `p3_single_mask`（诊断优先）
- `p3_minimal`

---

## 2) 输入输出规格（必须写死）

统一输入（全部变体相同）：
- `hs`: `float32[1, 10, 256]`（2560）
- `src`: `float32[1, 4096, 256]`（1048576）

### 2.1 常规变体输出（4-mask）
适用：`p3_base/p3_no_score/p3_no_deconv/p3_no_gemm/p3_no_tanh/p3_no_unsqueeze/p3_minimal`
- `scores`: `float32[1, 4]`（4）
- `masks`: `float32[1, 4, 256, 256]`（262144）

### 2.2 单 mask 变体输出
适用：`p3_single_mask`
- `scores`: `float32[1, 1]`（1）
- `masks`: `float32[1, 1, 256, 256]`（65536）

---

## 3) C++ 程序必须实现的行为

目标文件：`tests/test_part3_load_only.cpp`

1. 支持输入多个 `.tvn` 路径。
2. 对每个 tvn：独立创建模型对象 -> 调用 `loadModel` -> 释放对象。
3. 不调用 `inference()`（只测 load）。
4. 每个模型输出一行：
   - `MODEL=<name> LOAD=PASS`
   - `MODEL=<name> LOAD=FAIL ERR=<message>`
5. 最后输出：`SUMMARY PASS=<n> FAIL=<n> TOTAL=<n>`。
6. 返回码：有 FAIL 返回 `1`，否则返回 `0`。

---

## 4) CLI 约定

示例：

```bash
./test_part3_load_only --w 0 --h 0 p3_single_mask.tvn p3_base.tvn p3_no_gemm.tvn
```

参数：
- `--w`：传给 `loadModel` 的 width（默认 0）
- `--h`：传给 `loadModel` 的 height（默认 0）
- 位置参数：若干 `.tvn`

建议固定跑两轮：
- `--w 0 --h 0`
- `--w 64 --h 64`

---

## 5) 日志规范（便于脚本解析）

- 只使用固定 key：`MODEL`、`LOAD`、`ERR`、`SUMMARY`
- 不要加额外前缀文本
- `try/catch` 仅包裹 `loadModel` 调用，并输出完整错误信息
