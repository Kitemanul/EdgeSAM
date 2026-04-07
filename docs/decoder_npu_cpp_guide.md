# EdgeSAM Decoder NPU C++ 实现指南（给另一个 AI）

目标：让另一个 AI 直接写出一个 C++ 程序，在 TV/NPU 上跑 EdgeSAM Decoder（含前处理），输入图像+点提示，输出 mask。

---

## 1) 目标程序与输入输出

建议目标文件：`tests/test_decoder_npu_e2e.cpp`

程序输入：
- 图像（原始 `uint8`，HWC）
- 点坐标 `point_coords`（原图坐标系，`N` 个点）
- 点标签 `point_labels`（`N` 个）
- `encoder.tvn` + `decoder.tvn` 路径

程序输出：
- `scores`: `float32[1,4]`
- `masks`: `float32[1,4,256,256]`
- 最终二值 mask（原图尺寸）

---

## 2) 端到端流程（必须按顺序）

1. **图像前处理（CPU）**  
   Resize 长边到 1024、Normalize、Pad 到 `1x3x1024x1024`（NCHW）
2. **Encoder 推理（NPU）**  
   输入 `image[1,3,1024,1024]`，输出 `image_embeddings[1,256,64,64]`
3. **点坐标变换（CPU）**  
   原图坐标映射到模型坐标（resize 后坐标）
4. **PE 位置编码计算（CPU）**  
   生成 `point_embedding_pe[1,N,256]`（sin/cos）
5. **Decoder 推理（NPU）**  
   输入 `image_embeddings + point_embedding_pe + point_labels`
6. **后处理（CPU）**  
   选 `argmax(scores)` 的 mask，插值回原图并阈值化

---

## 3) 前处理细节（必须一致）

### 3.1 Resize
- `target = 1024`
- `scale = 1024.0 / max(orig_h, orig_w)`
- `new_h = round(orig_h * scale)`
- `new_w = round(orig_w * scale)`

### 3.2 Normalize（RGB）
- `mean = [123.675, 116.28, 103.53]`
- `std  = [58.395, 57.12, 57.375]`
- `out = (pixel - mean) / std`

> 若输入是 BGR，先转 RGB，或改用 BGR 顺序的 mean/std。

### 3.3 Pad
- 把 resize+normalize 后图像放在左上角
- 右侧与底部补 0 到 `1024x1024`

---

## 4) Decoder I/O 合同（按导出静态 N）

常用 `N=5`（导出时固定）。

Decoder 输入：
- `image_embeddings`: `float32[1,256,64,64]`
- `point_embedding_pe`: `float32[1,N,256]`
- `point_labels`: `float32[1,N]`（注意是 float）

Decoder 输出：
- `scores`: `float32[1,4]`
- `masks`: `float32[1,4,256,256]`

point_labels 编码：
- `1` 前景点
- `0` 背景点
- `2/3` box 两角
- `-1` padding

当点数 `< N` 时，必须 pad：
- 坐标补 `(0,0)`
- label 补 `-1`

---

## 5) PE（位置编码）CPU 计算要求

必须在 CPU 做，NPU 模型不包含这部分。

需要常量矩阵：`gaussian_matrix[2,128]`（从 checkpoint 提取一次，落盘加载）。

对每个点：
1. `(x,y)` 加 `0.5`，再除以 `1024`
2. 映射到 `[-1,1]`
3. 与 `gaussian_matrix` 做投影得到 128 维
4. 乘 `2π`
5. 拼接 `sin` 和 `cos` 成 256 维

输出 `point_embedding_pe[1,N,256]` 连续内存。

---

## 6) 另一个 AI 的实现要求（强制）

1. 提供 CLI：
   - `--encoder encoder.tvn`
   - `--decoder decoder.tvn`
   - `--image xxx.jpg`
   - `--points "x,y,l;..."`
   - `--num-points 5`
2. 每步打印日志（`STEP/STATUS/TIME_MS/ERR`）。
3. 所有输入输出都做 shape+dtype 校验。
4. `point_labels` 必须按 float32 传入。
5. 输出至少保存：
   - 最佳 mask png
   - 原图叠加可视化 png

---

## 7) 最小运行示例

```bash
./test_decoder_npu_e2e \
  --encoder encoder.tvn \
  --decoder decoder.tvn \
  --image test.jpg \
  --points "500,375,1;100,100,0" \
  --num-points 5
```

