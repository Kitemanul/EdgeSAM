# Part3 TVN `loadModel` 最小 C++ 测试指南（给另一个 AI）

目标：让另一个 AI 快速写出一个 **仅测试加载** 的 C++ 程序，批量验证以下 5 个 Part3 变体在 TV 端是否可加载：

- `part3_fixC.onnx`
- `part3_fixCE.onnx`
- `part3_fixCDE.onnx`
- `part3_fixCD.onnx`
- `part3_vanilla.onnx`

> 你（另一个 AI）只需要生成 C++ 测试代码，不要改模型，不要改编译链。

---

## 1) 产物要求

生成一个文件：`tests/test_part3_load_only.cpp`

程序行为：

1. 从命令行读取多个 `*.tvn` 路径（与上面 5 个变体一一对应）。
2. 对每个路径依次调用 `loadModel(...)`。
3. **不调用 inference**（当前阶段只关心 load）。
4. 每个模型打印一行结构化结果：
   - `MODEL=<name> LOAD=PASS`
   - 或 `MODEL=<name> LOAD=FAIL ERR=<error_code_or_msg>`
5. 程序最终返回码：
   - 全部 PASS 返回 `0`
   - 只要有一个 FAIL 返回 `1`

---

## 2) 输入参数约定

建议命令行：

```bash
./test_part3_load_only --w 0 --h 0 \
  part3_fixC.tvn part3_fixCE.tvn part3_fixCDE.tvn part3_fixCD.tvn part3_vanilla.tvn
```

要求支持：

- `--w` / `--h`：传给 `loadModel` 的宽高参数（默认 0/0）
- 位置参数：任意数量的 tvn 路径

备注：后续可用同一程序重复测试 `(w,h)=(64,64)`。

---

## 3) 实现要点（必须遵守）

1. 每次测试前后都创建/释放独立模型对象，避免句柄污染下一个测试。
2. `try/catch` 只包围 `loadModel` 调用和错误信息输出（不要吞掉错误）。
3. 输出必须稳定、可 grep（固定 key：`MODEL=`, `LOAD=`, `ERR=`）。
4. 失败时输出尽可能完整：错误码、异常字符串、SDK 返回值。
5. 不要做随机输入，不要分配大 buffer，不要推理。

---

## 4) 推荐输出示例

```text
MODEL=part3_fixC.tvn LOAD=PASS
MODEL=part3_fixCE.tvn LOAD=FAIL ERR=unsupported op: ConvTranspose
MODEL=part3_fixCDE.tvn LOAD=FAIL ERR=unsupported op: Gemm
SUMMARY PASS=1 FAIL=2 TOTAL=3
```

---

## 5) 可选增强（有时间再做）

- 增加 `--repeat N`，重复 load N 次，排查偶现加载失败。
- 增加 `--json out.json`，把结果写成 JSON，方便汇总。
- 记录每个模型加载耗时（ms）。

---

## 6) 验收标准

- 可以一次性批量测 5 个 tvn。
- 日志一眼可看出每个模型 PASS/FAIL。
- 不依赖 inference，执行时间应很短。
- 可直接用于后续“fix 组合 vs load 成功率”统计。
