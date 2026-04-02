# Part3 C++ Load 测试指南（给另一个 AI）

目的：让另一个 AI 直接写出 `tests/test_part3_load_only.cpp`，批量测试 Part3 变体的 `loadModel`。

---

## 1) 最重要结论（请在代码里明确写注释）

**所有 Part3 变体输入/输出完全一致。**

### 变体列表
- `p3_base`
- `p3_no_score`
- `p3_no_deconv`
- `p3_no_gemm`
- `p3_no_tanh`
- `p3_no_unsqueeze`
- `p3_minimal`

### 统一输入（所有变体相同）
- `hs`: `float32[1, 10, 256]`（2560）
- `src`: `float32[1, 4096, 256]`（1048576）

### 统一输出（所有变体相同）
- `scores`: `float32[1, 4]`（4）
- `masks`: `float32[1, 4, 256, 256]`（262144）

---

## 2) C++ 程序必须实现的行为

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

## 3) CLI 约定

示例：

```bash
./test_part3_load_only --w 0 --h 0 p3_base.tvn p3_no_gemm.tvn p3_minimal.tvn
```

参数：
- `--w`：传给 `loadModel` 的 width（默认 0）
- `--h`：传给 `loadModel` 的 height（默认 0）
- 位置参数：若干 `.tvn`

建议固定跑两轮：
- `--w 0 --h 0`
- `--w 64 --h 64`

---

## 4) 日志规范（便于脚本解析）

- 只使用固定 key：`MODEL`、`LOAD`、`ERR`、`SUMMARY`
- 不要加额外前缀文本
- `try/catch` 仅包裹 `loadModel` 调用，并输出完整错误信息
