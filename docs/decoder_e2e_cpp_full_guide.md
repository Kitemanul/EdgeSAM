# EdgeSAM 端到端 C++ 实现完整指南（给另一个 AI）

> 目标：指导另一个 AI 直接实现一个可运行的 C++ 程序，完成  
> **输入图片 + point prompt -> 输出 scores + masks（以及最终二值 mask）**。  
> 本指南覆盖：图像预处理、prompt 预计算（含 PE）、NPU 推理、后处理、日志与验收。

---

## 0) 你要实现的程序

建议目标文件：`tests/test_edgesam_e2e.cpp`

建议 CLI：

```bash
./test_edgesam_e2e \
  --encoder encoder.tvn \
  --decoder decoder.tvn \
  --gaussian pe_gaussian_matrix.bin \
  --image test.jpg \
  --points "500,375,1;100,100,0" \
  --num-points 5 \
  --mask-threshold 0.0 \
  --out-dir ./out
```

程序输出（至少）：
- `scores`：`float32[1,4]`
- `masks`：`float32[1,4,256,256]`
- `best_mask.png`（原图尺寸）
- `overlay.png`（原图叠加可视化）

---

## 1) 模型与张量契约（必须严格一致）

### 1.1 Encoder（NPU）
- 输入 `image`：`float32[1,3,1024,1024]`（NCHW）
- 输出 `image_embeddings`：`float32[1,256,64,64]`

### 1.2 Decoder（NPU）
- 输入1 `image_embeddings`：`float32[1,256,64,64]`
- 输入2 `point_embedding_pe`：`float32[1,N,256]`
- 输入3 `point_labels`：`float32[1,N]`（float32，不是 int）
- 输出1 `scores`：`float32[1,4]`
- 输出2 `masks`：`float32[1,4,256,256]`

默认 `N=5`（静态 shape）。点数不足必须 padding，超出需截断或报错（建议报错）。

---

## 2) 端到端流程（按顺序）

1. 图像预处理（CPU）  
2. Encoder 推理（NPU）  
3. 点坐标变换到模型空间（CPU）  
4. Prompt 预计算：PE 位置编码（CPU）  
5. Decoder 推理（NPU）  
6. 后处理：选 best mask、插值回原图、阈值化（CPU）

---

## 3) 图像预处理（CPU）

### 3.1 Resize（长边到 1024）

```text
scale = 1024.0 / max(orig_h, orig_w)
new_h = round(orig_h * scale)
new_w = round(orig_w * scale)
```

使用双线性插值 resize 到 `(new_h, new_w)`，保留 `new_h/new_w` 供后处理使用。

### 3.2 Normalize

按 RGB 通道：

```text
mean = [123.675, 116.28, 103.53]
std  = [58.395, 57.12, 57.375]
out = (pixel - mean) / std
```

若输入是 BGR，先转 RGB（推荐），再做 normalize。

### 3.3 Pad + 布局

- 将 resize+normalize 后图像放在左上角
- 右侧/底部补零到 `1024x1024`
- 组织成 `float32[1,3,1024,1024]` 连续内存（NCHW）

---

## 4) Prompt 预处理（CPU）

输入格式（原图坐标）：

```text
points: (x, y, label), label in {-1,0,1,2,3}
```

- `1` 前景点
- `0` 背景点
- `2`/`3` box 两角
- `-1` padding

### 4.1 坐标变换到模型空间

```text
x' = x * (new_w / orig_w)
y' = y * (new_h / orig_h)
```

### 4.2 点数对齐（固定 N）

若实际点数 `< N`：
- 坐标补 `(0,0)`
- 标签补 `-1`

若 `> N`：建议直接报错并退出（最可控）。

最终：
- `point_coords_model`：`float32[1,N,2]`
- `point_labels`：`float32[1,N]`

### 4.3 PE（位置编码）计算（必须 CPU）

从 `pe_gaussian_matrix.bin` 读取 `float32[2,128]` 常量矩阵。

对每个点 `(x', y')`：

1. `norm_x = (x' + 0.5) / 1024.0`，`norm_y = (y' + 0.5) / 1024.0`
2. 映射到 `[-1,1]`：`norm = 2*norm - 1`
3. 投影：`proj[j] = norm_x * G[0][j] + norm_y * G[1][j]`（j=0..127）
4. 缩放：`proj[j] *= 2π`
5. 拼接：
   - `pe[j] = sin(proj[j])`
   - `pe[j+128] = cos(proj[j])`

得到 `point_embedding_pe`：`float32[1,N,256]`。

---

## 5) NPU 推理调用顺序

### 5.1 Encoder

输入：`image[1,3,1024,1024]`  
输出：`image_embeddings[1,256,64,64]`

### 5.2 Decoder

输入按顺序绑定：
1. `image_embeddings`
2. `point_embedding_pe`
3. `point_labels`

输出读取：
- `scores[1,4]`
- `masks[1,4,256,256]`

> 强制校验：每次 load 后都检查 I/O 数量、shape、dtype，防止错绑。

---

## 6) 后处理（CPU）

1. `best_idx = argmax(scores[0])`
2. 取 `masks[0][best_idx]`（`256x256`）
3. 双线性插值到 `1024x1024`
4. 裁掉 pad 区域到 `new_h x new_w`
5. 再插值到原图 `orig_h x orig_w`
6. 二值化：`mask > mask_threshold`（默认 0.0）

输出：
- `best_mask.png`（0/255）
- `overlay.png`（彩色叠加）

---

## 7) 代码结构建议

建议拆成以下函数：

1. `preprocess_image(...) -> image_tensor, orig_h, orig_w, new_h, new_w`
2. `transform_points(...) -> point_coords_model`
3. `pad_points(...) -> point_coords_N, point_labels_N`
4. `compute_point_pe(...) -> point_embedding_pe`
5. `run_encoder_npu(...) -> image_embeddings`
6. `run_decoder_npu(...) -> scores, masks`
7. `postprocess_mask(...) -> final_mask`
8. `save_outputs(...)`

---

## 8) 日志规范（机读）

每步统一打印：

```text
STEP=<name> STATUS=PASS TIME_MS=<ms>
STEP=<name> STATUS=FAIL ERR=<msg>
```

推荐 step 名：
- `PREPROCESS`
- `ENCODER_LOAD`
- `ENCODER_INFER`
- `PROMPT_PRECOMP`
- `DECODER_LOAD`
- `DECODER_INFER`
- `POSTPROCESS`
- `SAVE`
- `SUMMARY`

返回码：
- 任一步失败：`return 1`
- 全部通过：`return 0`

---

## 9) 最小验收标准（给另一个 AI）

实现完成后，至少满足：

1. 能跑通单张图单点提示（前景点）  
2. 能跑通前景+背景双点提示  
3. 能输出 `scores` 数组和 `best_mask.png`  
4. 所有 step 都有日志和耗时  
5. I/O 契约错误时能明确报错并退出

---

## 10) 易错点清单（必须避免）

1. 把 `point_labels` 当成 int 传入（错误，必须 float32）
2. 忘记做点 padding 到固定 `N`
3. 坐标未按 resize 后尺度变换
4. PE 没按 `(coord+0.5)/1024` 处理
5. 忘记去 pad 区域就直接回原图
6. Decoder 输入顺序绑错

