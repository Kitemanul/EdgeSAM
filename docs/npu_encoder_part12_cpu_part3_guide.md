# NPU 跑 Encoder+Part1+Part2，CPU 跑 Part3 的落地指南（给另一个 AI）

目的：指导另一个 AI 写出一个 C++ 推理程序，实现以下混合流水线：

- `Encoder`：NPU
- `Part1+Part2`（合并 ONNX）：NPU
- `Part3`（Mask Head）：CPU（ONNX Runtime）

---

## 1) 现状确认

按你当前实验结论，`Part1` / `Part2` 已可在 NPU 上 `load + infer`。本方案只把 `Part3` 放到 CPU，规避 Part3 在 NPU 侧的加载卡住问题。

---

## 2) 模型导出

### 2.1 导出“合并后的 Part1+Part2”（NPU 用）

使用新增脚本：

```bash
python3 scripts/export_part12_npu.py weights/edge_sam_3x.pth --output-dir ./part12_npu --num-points 5
```

产物：

- `part12_prompt_transformer.onnx`（单一 ONNX，内部已包含 Part1+Part2）

### 2.2 导出 Part3（CPU 用）

建议直接复用已验证的 3-part 产物中的 `part3_mask_head.onnx`（输入是 `hs/src`，可直接接 Part12 输出）：

```bash
python3 scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./part123_ref --num-points 5
```

产物：

- `part3_mask_head.onnx`（给 ONNX Runtime CPU）

> 约束：`--num-points` 必须一致（例如都用 5），否则 Part12 输出 token 长度与 Part3 输入不匹配。

---

## 3) I/O 契约（必须写死校验）

假设 `N=5`。

### Encoder（NPU）
- 输入：`image` `float32[1,3,1024,1024]`
- 输出：`image_embeddings` `float32[1,256,64,64]`

### Part12（NPU，合并）
- 输入1：`image_embeddings` `float32[1,256,64,64]`
- 输入2：`point_embedding_pe` `float32[1,5,256]`（CPU 先算好）
- 输入3：`point_labels` `float32[1,5]`
- 输出1：`hs` `float32[1,10,256]`（`10 = 1 + 4 + N`）
- 输出2：`src` `float32[1,4096,256]`

### Part3（CPU / ONNX Runtime）
- 输入1：`hs` `float32[1,10,256]`
- 输入2：`src` `float32[1,4096,256]`
- 输出1：`scores` `float32[1,4]`
- 输出2：`masks` `float32[1,4,256,256]`

---

## 4) 程序结构建议（另一个 AI 要实现）

目标文件：`tests/test_hybrid_npu_part12_cpu_part3.cpp`

### 4.1 CLI 约定

```bash
./test_hybrid_npu_part12_cpu_part3 \
  --encoder encoder.tvn \
  --part12 part12_prompt_transformer.tvn \
  --part3 part3_mask_head.onnx \
  --image test.jpg \
  --points "500,375,1;100,100,0" \
  --num-points 5
```

参数建议：

- `--encoder` / `--part12`：NPU 模型路径
- `--part3`：ONNX 路径
- `--num-points`：静态点数（与导出一致）
- `--points`：原图坐标 + label 列表（`x,y,l;...`）

### 4.2 执行步骤

1. 图像预处理（与现有 pipeline 一致，CPU）
2. Encoder NPU 推理，得到 `image_embeddings`
3. 坐标变换到模型坐标系（CPU）
4. CPU 计算 `point_embedding_pe`（必须和训练代码一致）
5. Part12 NPU 推理，得到 `hs/src`
6. Part3 用 ONNX Runtime CPU 推理，得到 `scores/masks`
7. 后处理上采样回原图并二值化

### 4.3 关键实现约束

1. **Part12 只接受 float32**（包括 `point_labels`）
2. **点提示需要 pad 到固定 N**，pad label 为 `-1`
3. **不要在 NPU 模型间复用未释放 buffer**
4. NPU 每步后打印耗时，遇错立即打出模型名/步骤名

---

## 5) 日志与返回码

建议日志 key（机读友好）：

- `STAGE`：`ENCODER` / `PART12` / `PART3_CPU` / `POST`
- `STEP`：`LOAD` / `INFER` / `CHECK_IO` / `SUMMARY`
- `STATUS`：`PASS` / `FAIL`
- `TIME_MS`
- `ERR`

返回码：

- 任一步失败：返回 `1`
- 全部通过：返回 `0`

---

## 6) 最小验证命令

```bash
# 1) 导出
python3 scripts/export_part12_npu.py weights/edge_sam_3x.pth --output-dir ./part12_npu --num-points 5
python3 scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./part123_ref --num-points 5

# 2) NPU 编译（示例，按你们工具链替换）
npu_compiler ./part12_npu/part12_prompt_transformer.onnx -o part12_prompt_transformer.tvn
npu_compiler ./encoder.onnx                         -o encoder.tvn

# 3) 运行混合推理
./test_hybrid_npu_part12_cpu_part3 \
  --encoder encoder.tvn \
  --part12 part12_prompt_transformer.tvn \
  --part3 ./part123_ref/part3_mask_head.onnx \
  --image test.jpg \
  --points "500,375,1;100,100,0" \
  --num-points 5
```
