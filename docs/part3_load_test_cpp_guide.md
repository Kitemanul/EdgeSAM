# C++ 设计指南：Part3 TV `loadModel` 批量测试

此文档只用于指导另一个 AI 编写 C++ 程序。
不包含模型导出策略，不讨论 Python 侧实现。

---

## 1) 目标程序

生成文件：`tests/test_part3_load_only.cpp`

目标：批量测试多个 `*.tvn` 的 `loadModel` 结果，定位 TV 端加载失败。

---

## 2) 模型输入输出尺寸（必须写死在实现里）

> 下面是当前拆分流水线下的标准尺寸。你写 C++ 时请按这些尺寸定义常量，便于后续扩展到 inference 测试。

### 2.1 Encoder
- 输入：`image` `float32[1, 3, 1024, 1024]`
- 输出：`image_embeddings` `float32[1, 256, 64, 64]`

### 2.2 Part1 (Prompt Encoding)
- 输入0：`point_embedding_pe` `float32[1, N, 256]`
- 输入1：`point_labels` `float32[1, N]`
- 输出：`sparse_embedding` `float32[1, N, 256]`
- 本项目常用 `N=5`。

### 2.3 Part2 (Transformer)
- 输入0：`image_embeddings` `float32[1, 256, 64, 64]`
- 输入1：`sparse_embedding` `float32[1, 5, 256]`
- 输出0：`hs` `float32[1, 10, 256]`
- 输出1：`src` `float32[1, 4096, 256]`

### 2.4 Part3 (Mask Head)
- 输入0：`hs` `float32[1, 10, 256]`
- 输入1：`src` `float32[1, 4096, 256]`
- 输出0：`scores` `float32[1, 4]`
- 输出1：`masks` `float32[1, 4, 256, 256]`

### 2.5 Part3 消融变体（重点）
- `p3_base / p3_no_score / p3_no_deconv / p3_no_gemm / p3_no_tanh / p3_no_unsqueeze / p3_minimal`
- **以上所有 Part3 变体 I/O 尺寸必须与 2.4 完全一致**。

### 2.6 常用元素个数（便于 C++ 分配/校验）
- `hs`: `1*10*256 = 2,560`
- `src`: `1*4096*256 = 1,048,576`
- `scores`: `1*4 = 4`
- `masks`: `1*4*256*256 = 262,144`

---

## 3) 程序行为（必须）

1. 支持命令行输入多个 tvn 路径。
2. 每个模型独立创建模型对象，调用 `loadModel` 后立刻释放。
3. 当前阶段只做 `loadModel`，不调用 `inference()`。
4. 每个模型打印一行固定格式日志：
   - 成功：`MODEL=<name> LOAD=PASS`
   - 失败：`MODEL=<name> LOAD=FAIL ERR=<message>`
5. 最后打印汇总：`SUMMARY PASS=<n> FAIL=<n> TOTAL=<n>`。
6. 返回码规则：
   - 全部 PASS 返回 `0`
   - 任一 FAIL 返回 `1`

---

## 4) CLI 设计（必须）

示例：

```bash
./test_part3_load_only --w 0 --h 0 p3_base.tvn p3_no_gemm.tvn p3_minimal.tvn
```

参数约定：
- `--w`：传给 `loadModel` 的 width（默认 0）
- `--h`：传给 `loadModel` 的 height（默认 0）
- 位置参数：一个或多个 `.tvn` 路径

建议固定跑两轮：
1. `--w 0 --h 0`
2. `--w 64 --h 64`

---

## 5) 错误处理与日志规范

- `try/catch` 仅包围 `loadModel` 调用，避免吞掉上下文。
- 打印尽可能完整的错误信息：返回码、异常字符串、SDK 错误文本。
- 日志 key 固定为：`MODEL`、`LOAD`、`ERR`、`SUMMARY`（便于 grep/脚本解析）。
- 日志禁止自由文本前缀（避免解析困难）。

---

## 6) 代码结构建议

- `struct CaseResult { name, pass, err };`
- `bool run_one(const std::string& tvn_path, int w, int h, CaseResult& out)`
- `int main(...)` 负责：参数解析、循环调用、汇总、返回码

建议把“模型创建/销毁”放在 `run_one` 内部，保证每个 case 互不污染。

---

## 7) 最小验收标准

- 单次可测试多个 tvn。
- 输出稳定且可机读。
- 目前仅验证加载，不依赖推理输入数据。
- 文档中的 I/O 尺寸常量已在代码中体现（即使当前仅 load，不做 inference）。
