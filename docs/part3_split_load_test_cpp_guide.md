# Part3 Split(A) C++ Load 验证指南（给另一个 AI）

目的：让另一个 AI 直接写出 `tests/test_part3_split_load_only.cpp`，用于批量验证 `part3a/part3b/part3c` 三个 TV 模型在设备侧 `loadModel` 的可加载性（优先诊断 load，不做推理性能评估）。

---

## 1) 背景与目标

你要验证的是 Part3 的三段拆分模型（Split-A）：

- `part3a`：上采样分支（`src -> upscaled`）
- `part3b`：hyper 分支（`hs -> hyper_in, iou_token`）
- `part3c`：融合+打分分支（`hyper_in + upscaled + iou_token -> scores, masks`）

测试目标：

1. 每个模型可独立 `loadModel`。
2. 模型输入输出名称/shape 与约定一致。
3. 仅做 load 验证时，不调用 `inference()`。
4. 可选增加一轮最小前向（全 0 输入）作为补充验证。

---

## 2) 输入输出规格（必须写死）

> 注意：以下 shape 均为静态 shape，测试代码中请写死并严格校验。

### 2.1 part3a（`p3a_upscale`）

- 输入：
  - `src`: `float32[1, 4096, 256]`
- 输出：
  - `upscaled`: `float32[1, 32, 256, 256]`

### 2.2 part3b（`p3b_hyper`）

- 输入：
  - `hs`: `float32[1, 10, 256]`
- 输出：
  - `hyper_in`: `float32[1, 4, 32]`
  - `iou_token`: `float32[1, 256]`

### 2.3 part3c（`p3c_fuse_score`）

- 输入：
  - `hyper_in`: `float32[1, 4, 32]`
  - `upscaled`: `float32[1, 32, 256, 256]`
  - `iou_token`: `float32[1, 256]`
- 输出：
  - `scores`: `float32[1, 4]`
  - `masks`: `float32[1, 4, 256, 256]`

---

## 3) C++ 程序必须实现的行为

目标文件：`tests/test_part3_split_load_only.cpp`

程序要求：

1. 支持通过 CLI 传入 3 个模型路径：`--p3a <tvn> --p3b <tvn> --p3c <tvn>`。
2. 对每个模型执行：
   - 创建模型对象
   - 调用 `loadModel(path, w, h)`
   - 校验 I/O 数量、名称、shape、dtype
   - 释放模型对象
3. 默认模式 **只做 load 验证**，不调用 `inference()`。
4. 增加 `--run-minimal-infer`（可选）：
   - 用全 0 输入跑 1 次前向
   - 只检查是否执行成功与输出 shape，不比较数值
5. 汇总输出 PASS/FAIL，并返回非零失败码。

---

## 4) CLI 约定

示例：

```bash
./test_part3_split_load_only \
  --w 0 --h 0 \
  --p3a p3a_upscale.tvn \
  --p3b p3b_hyper.tvn \
  --p3c p3c_fuse_score.tvn
```

可选前向验证：

```bash
./test_part3_split_load_only \
  --w 0 --h 0 \
  --p3a p3a_upscale.tvn \
  --p3b p3b_hyper.tvn \
  --p3c p3c_fuse_score.tvn \
  --run-minimal-infer
```

参数：

- `--w`：传给 `loadModel` 的 width（默认 0）
- `--h`：传给 `loadModel` 的 height（默认 0）
- `--p3a`/`--p3b`/`--p3c`：三个 tvn 模型路径（必填）
- `--run-minimal-infer`：是否跑最小前向（默认关闭）

建议固定跑两轮：

- `--w 0 --h 0`
- `--w 64 --h 64`

---

## 5) 日志规范（便于脚本解析）

只允许以下 key，避免自由文本：

- `MODEL`：`P3A` / `P3B` / `P3C`
- `STEP`：`LOAD` / `CHECK_IO` / `INFER` / `SUMMARY`
- `STATUS`：`PASS` / `FAIL`
- `ERR`：失败原因（完整错误文本）

单模型建议日志：

- `MODEL=P3A STEP=LOAD STATUS=PASS`
- `MODEL=P3A STEP=CHECK_IO STATUS=PASS`
- `MODEL=P3A STEP=INFER STATUS=PASS`
- `MODEL=P3A STEP=LOAD STATUS=FAIL ERR=<message>`

最终汇总：

- `STEP=SUMMARY STATUS=PASS PASS=<n> FAIL=<n> TOTAL=<n>`
- `STEP=SUMMARY STATUS=FAIL PASS=<n> FAIL=<n> TOTAL=<n>`

返回码规则：

- 任何模型任一步骤失败，返回 `1`
- 全部通过，返回 `0`

---

## 6) 实现细节建议（给另一个 AI）

1. **异常处理粒度**：
   - 每个模型的 `loadModel` 独立 try/catch
   - I/O 校验独立 try/catch
   - 可选 infer 独立 try/catch
   - 避免一个模型失败导致后续模型不执行

2. **I/O 校验建议**：
   - 先校验输入输出 tensor 数量
   - 再按名称匹配（若 runtime 名称可读）
   - 再校验 shape（逐维完全一致）
   - 最后校验 dtype（必须 float32）

3. **最小前向输入构造**：
   - 所有输入用 `0.0f` 填充
   - 仅校验推理调用成功与输出 shape
   - 不做数值阈值比较（避免不同后端差异）

4. **执行顺序建议**：
   - 固定 `P3A -> P3B -> P3C`
   - 这样日志更稳定，便于定位“首次失败点”

---

## 7) 预期产物

另一个 AI 完成后，应至少交付：

1. `tests/test_part3_split_load_only.cpp`
2. 使用说明（编译命令 + 运行命令）
3. 一段示例输出日志（含 PASS 与 FAIL 样例）

如果时间允许，再补一个脚本：

- `scripts/run_part3_split_load_test.sh`：一键跑两轮（`0x0` 与 `64x64`）并归档日志。
