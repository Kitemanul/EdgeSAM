# Decoder Part1/Part2/Part3 C++ 测试指南（给另一个 AI，简版）

目的：让另一个 AI 直接实现 `tests/test_decoder_part123.cpp`，用于验证 `part1/part2/part3` 三个模型在设备侧的：

1. `loadModel` 是否成功  
2. `minimal infer` 是否成功  
3. 单独测试 + 顺序测试（`P1->P2->P3`）

---

## 1) 固定 I/O 规格（先按 N=5 写死）

### Part1 (`part1_prompt_encoding`)
- 输入1 `point_embedding_pe`: `float32[1,5,256]`
- 输入2 `point_labels`: `float32[1,5]`
- 输出 `sparse_embedding`: `float32[1,5,256]`

### Part2 (`part2_transformer`)
- 输入1 `image_embeddings`: `float32[1,256,64,64]`
- 输入2 `sparse_embedding`: `float32[1,5,256]`
- 输出1 `hs`: `float32[1,10,256]`
- 输出2 `src`: `float32[1,4096,256]`

### Part3 (`part3_mask_head`)
- 输入1 `hs`: `float32[1,10,256]`
- 输入2 `src`: `float32[1,4096,256]`
- 输出1 `scores`: `float32[1,4]`
- 输出2 `masks`: `float32[1,4,256,256]`

---

## 2) C++ 程序要求

目标文件：`tests/test_decoder_part123.cpp`

必须支持两种模式：

1. `isolated`：分别单独测 `P1`、`P2`、`P3`（每个模型独立 load+infer）
2. `sequential`：按 `P1->P2->P3` 在同一进程串行测

每个模型步骤：

- `LOAD`：创建对象并 `loadModel`
- `CHECK_IO`：校验输入输出数量/shape/dtype
- `INFER`：用**程序内部构造的模拟输入数据**跑 1 次最小前向
- `DESTROY`：释放对象（以及必要同步）

> 强制要求：`INFER` 测试不能依赖真实业务输入文件，必须在 C++ 里自行构造 mock 数据（例如全 0、常数、随机但固定 seed）。

---

## 3) CLI 约定（最小）

```bash
./test_decoder_part123 \
  --mode isolated|sequential \
  --w 0 --h 0 \
  --p1 part1_prompt_encoding.tvn \
  --p2 part2_transformer.tvn \
  --p3 part3_mask_head.tvn \
  --timeout-ms 30000
```

可选：

- `--repeat 3`（重复跑，抓偶发卡住）
- `--run-infer`（默认开，若只测 load 可关闭）

---

## 4) 日志规范（固定 key）

- `MODE`：`ISOLATED` / `SEQUENTIAL`
- `MODEL`：`P1` / `P2` / `P3`
- `STEP`：`LOAD` / `CHECK_IO` / `INFER` / `DESTROY` / `SUMMARY`
- `STATUS`：`PASS` / `FAIL`
- `TIME_MS`
- `ERR`

示例：

- `MODE=ISOLATED MODEL=P1 STEP=LOAD STATUS=PASS TIME_MS=12`
- `MODE=SEQUENTIAL MODEL=P2 STEP=INFER STATUS=FAIL ERR=TIMEOUT TIME_MS=30000`

返回码：

- 任一步失败返回 `1`
- 全部通过返回 `0`

---

## 5) 执行建议（必须覆盖）

至少跑以下 4 组：

1. `isolated` + `--run-infer`
2. `isolated` + `--no-run-infer`（只 load）
3. `sequential` + `--run-infer`
4. `sequential` + `--no-run-infer`

并各跑两轮：

- `--w 0 --h 0`
- `--w 64 --h 64`

建议 mock 输入策略（固定即可）：

- `point_embedding_pe`：全 0 或固定常数
- `point_labels`：`[1, 0, -1, -1, -1]`
- `image_embeddings`：全 0 或固定 seed 的随机数
- Part3 独立测时：`hs/src` 也用程序内 mock 数据构造
