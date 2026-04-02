# 方案 B：Decoder TVN 加载失败排查

合并 decoder TVN 在 TV 上无法加载。将 decoder 拆为 3 段 TVN，逐段测试定位问题。

## 三段模型规格

导出命令：
```bash
python scripts/diagnose_npu_ops.py weights/edge_sam_3x.pth --output-dir ./npu_diag --num-points 5
# 输出 3 个 ONNX，分别编译为 TVN
```

### Part 1: Prompt Encoding

| | 名称 | 类型 | 形状 | float 数 |
|--|------|------|------|---------|
| 输入 0 | `point_embedding_pe` | FLOAT32 | [1, 5, 256] | 1,280 |
| 输入 1 | `point_labels` | FLOAT32 | [1, 5] | 5 |
| 输出 0 | `sparse_embedding` | FLOAT32 | [1, 5, 256] | 1,280 |

最小的模型（约 8 种算子），如果这个都加载失败，说明问题出在 TVN 格式或库本身。

### Part 2: Transformer

| | 名称 | 类型 | 形状 | float 数 |
|--|------|------|------|---------|
| 输入 0 | `image_embeddings` | FLOAT32 | [1, 256, 64, 64] | 1,048,576 |
| 输入 1 | `sparse_embedding` | FLOAT32 | [1, 5, 256] | 1,280 |
| 输出 0 | `hs` | FLOAT32 | [1, 10, 256] | 2,560 |
| 输出 1 | `src` | FLOAT32 | [1, 4096, 256] | 1,048,576 |

计算量最大的部分（46 个 MatMul、7 个 Softmax）。

### Part 3: Mask Head

| | 名称 | 类型 | 形状 | float 数 |
|--|------|------|------|---------|
| 输入 0 | `hs` | FLOAT32 | [1, 10, 256] | 2,560 |
| 输入 1 | `src` | FLOAT32 | [1, 4096, 256] | 1,048,576 |
| 输出 0 | `scores` | FLOAT32 | [1, 4] | 4 |
| 输出 1 | `masks` | FLOAT32 | [1, 4, 256, 256] | 262,144 |

包含 ConvTranspose（上采样）、Gemm（MLP）、Erf（GELU）。

## 测试任务

写 3 个独立的 C++ 测试程序（`test_part1`、`test_part2`、`test_part3`），每个做同样的事：

1. `loadModel(tvn_path, ...)` — 验证能否加载
2. 构造 dummy 输入（随机 float 数据，按上表的形状和大小）
3. `inference()` — 验证能否推理
4. 检查输出：size 是否正确、是否全零、是否含 NaN/Inf
5. 打印 PASS 或 FAIL

库函数只支持单个 `const unsigned char* data` 输入和 `vector<float>` 输出。多个输入 tensor 需要按顺序拼接为一个连续 buffer（input 0 在前，input 1 紧随其后）。多个输出同理——库返回一个 `vector<float>`，前面是 output 0，后面是 output 1，按已知 size 拆分。

`loadModel` 的 `width`/`height` 参数含义不确定。如果库从 TVN 自动读取 shape，传 0 即可。如果需要指定输入空间尺寸：Part 1 无空间维度传 0，Part 2 传 64x64，Part 3 无空间维度传 0。需要实测调整。

## 结果判读

| Part 1 | Part 2 | Part 3 | 结论 |
|--------|--------|--------|------|
| PASS | PASS | PASS | 三段均可用，问题在合并模型的 TVN 大小或内存限制 |
| FAIL | FAIL | FAIL | 库函数或 TVN 格式本身有问题 |
| PASS | FAIL | PASS | Part 2 太大，可能超出 NPU 内存 |
| PASS | PASS | FAIL | Part 3 的 ConvTranspose/Gemm 可能运行时不支持 |
| LOAD FAIL | — | — | 检查 TVN 文件路径、loadModel 参数 |

如果三段都 PASS，可以考虑在 TV 上用三段 TVN 串联运行（替代合并版），参见 `npu_tv_deployment_guide.md` 中的完整推理流水线。
