# C++ 设计指南：Part3 TV `loadModel` 批量测试

此文档只用于指导另一个 AI 编写 C++ 程序。
不包含模型导出策略，不讨论 Python 侧实现。

---

## 1) 目标程序

生成文件：`tests/test_part3_load_only.cpp`

目标：批量测试多个 `*.tvn` 的 `loadModel` 结果，定位 TV 端加载失败。

---

## 2) 程序行为（必须）

1. 支持命令行输入多个 tvn 路径。
2. 每个模型独立创建模型对象，调用 `loadModel` 后立刻释放。
3. 不调用 `inference()`。
4. 每个模型打印一行固定格式日志：
   - 成功：`MODEL=<name> LOAD=PASS`
   - 失败：`MODEL=<name> LOAD=FAIL ERR=<message>`
5. 最后打印汇总：`SUMMARY PASS=<n> FAIL=<n> TOTAL=<n>`。
6. 返回码规则：
   - 全部 PASS 返回 `0`
   - 任一 FAIL 返回 `1`

---

## 3) CLI 设计（必须）

示例：

```bash
./test_part3_load_only --w 0 --h 0 a.tvn b.tvn c.tvn
```

参数约定：
- `--w`：传给 `loadModel` 的 width（默认 0）
- `--h`：传给 `loadModel` 的 height（默认 0）
- 位置参数：一个或多个 `.tvn` 路径

建议固定跑两轮：
1. `--w 0 --h 0`
2. `--w 64 --h 64`

---

## 4) 错误处理与日志规范

- `try/catch` 仅包围 `loadModel` 调用，避免吞掉上下文。
- 打印尽可能完整的错误信息：返回码、异常字符串、SDK 错误文本。
- 日志 key 固定为：`MODEL`、`LOAD`、`ERR`、`SUMMARY`（便于 grep/脚本解析）。
- 日志禁止自由文本前缀（避免解析困难）。

---

## 5) 代码结构建议

- `struct CaseResult { name, pass, err };`
- `bool run_one(const std::string& tvn_path, int w, int h, CaseResult& out)`
- `int main(...)` 负责：参数解析、循环调用、汇总、返回码

建议把“模型创建/销毁”放在 `run_one` 内部，保证每个 case 互不污染。

---

## 6) 最小验收标准

- 单次可测试多个 tvn。
- 输出稳定且可机读。
- 不依赖推理输入数据。
- 可直接用于比较不同 Part3 变体在 TV 端的 load 成功率。
