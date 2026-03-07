# EdgeSAM ONNX 评估指南

本文档详细描述评估 EdgeSAM ONNX 模型时，输入前和输出后所需的全部处理步骤。

---

## 模型结构

EdgeSAM ONNX 分为两个独立模型：

```
encoder.onnx   输入: 图像        输出: image_embeddings
decoder.onnx   输入: embeddings + point_coords + point_labels   输出: scores + low_res_masks
```

---

## 一、Encoder 输入预处理

### 1. 读取图像

```python
import cv2
import numpy as np

image = cv2.imread("image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为 RGB，shape: (H, W, 3)
original_size = image.shape[:2]                 # 保存原始尺寸 (H, W)，后处理要用
```

### 2. Resize（最长边缩放到 1024）

```python
def get_preprocess_shape(oldh, oldw, long_side=1024):
    scale = long_side / max(oldh, oldw)
    newh = int(oldh * scale + 0.5)
    neww = int(oldw * scale + 0.5)
    return newh, neww

from PIL import Image
target_size = get_preprocess_shape(original_size[0], original_size[1])  # (newH, newW)
input_image = np.array(Image.fromarray(image).resize(target_size[::-1]))  # PIL resize 用 (W, H)
input_size = input_image.shape[:2]  # 保存 resize 后尺寸 (newH, newW)，后处理要用
```

### 3. HWC → BCHW

```python
input_image = input_image.transpose(2, 0, 1)[None, :, :, :]  # shape: (1, 3, newH, newW)
```

### 4. ImageNet 标准化

```python
pixel_mean = np.array([123.675, 116.28, 103.53])[None, :, None, None]
pixel_std  = np.array([58.395,  57.12,  57.375])[None, :, None, None]

input_image = (input_image - pixel_mean) / pixel_std  # 输出范围约 [-2, 2]
```

> **注意**：不是除以 255，不是归一化到 [0,1]。

### 5. Padding 到 1024×1024

```python
h, w = input_image.shape[-2:]
padh = 1024 - h
padw = 1024 - w
input_image = np.pad(input_image, ((0,0),(0,0),(0,padh),(0,padw)), mode='constant', constant_values=0)
# 最终 shape: (1, 3, 1024, 1024)
```

### 6. 转 float32 并送入 encoder

```python
input_image = input_image.astype(np.float32)
image_embeddings = encoder.run(None, {'image': input_image})[0]
# 输出 shape: (1, 256, 64, 64)
```

---

## 二、Decoder 输入预处理

### point_coords 坐标缩放

用户输入的坐标是基于**原始图像**的像素坐标，需要按 resize 比例缩放到 resize 后的坐标空间。

```python
def apply_coords(coords, original_size, target_size):
    """
    coords: np.ndarray, shape (N, 2)，格式为 (X, Y)，原始图像像素坐标
    original_size: (H, W) 原始图像尺寸
    target_size:   (newH, newW) resize 后尺寸
    """
    old_h, old_w = original_size
    new_h, new_w = target_size
    coords = coords.copy().astype(float)
    coords[:, 0] = coords[:, 0] * (new_w / old_w)  # X 方向
    coords[:, 1] = coords[:, 1] * (new_h / old_h)  # Y 方向
    return coords
```

> **注意**：
> - 坐标格式为 **(X, Y)**，即 `[width方向, height方向]`
> - 归一化到 [0,1] 和位置编码在 ONNX 模型内部完成，脚本里不需要处理

### point_labels

```
0 = 负样本点（背景）
1 = 正样本点（前景）
2 = box 左上角
3 = box 右下角
```

### 送入 decoder

```python
point_coords = apply_coords(point_coords, original_size, input_size)

outputs = decoder.run(None, {
    'image_embeddings': image_embeddings,                      # (1, 256, 64, 64)
    'point_coords':     point_coords[None].astype(np.float32), # (1, N, 2)
    'point_labels':     point_labels[None].astype(np.float32), # (1, N)
})

scores, low_res_masks = outputs[0], outputs[1]
# scores shape:         (1, num_masks)
# low_res_masks shape:  (1, num_masks, 256, 256)
```

---

## 三、Decoder 输出后处理

decoder 输出的 mask 是低分辨率的（256×256），需要还原到原始图像尺寸。

```python
def postprocess_masks(low_res_masks, input_size, original_size):
    """
    low_res_masks: np.ndarray, shape (1, num_masks, 256, 256)
    input_size:    (newH, newW) resize 后尺寸（含 padding 前）
    original_size: (H, W) 原始图像尺寸
    """
    # (1, num_masks, 256, 256) → (num_masks, 256, 256, 1) 方便 cv2 处理
    mask = low_res_masks.squeeze(0).transpose(1, 2, 0)  # (256, 256, num_masks)

    # Step 1: resize 到 1024×1024
    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    # Step 2: 裁掉 padding，还原到 resize 后的有效区域
    mask = mask[:input_size[0], :input_size[1], :]

    # Step 3: resize 回原始图像尺寸
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

    # 还原 shape: (1, num_masks, H, W)
    mask = mask.transpose(2, 0, 1)[None, :, :, :]
    return mask

masks = postprocess_masks(low_res_masks, input_size, original_size)

# 二值化
mask_threshold = 0.0
binary_masks = masks > mask_threshold  # shape: (1, num_masks, H, W)，dtype: bool
```

---

## 四、完整流程示意

```
原始图像 (H, W, 3) uint8
    ↓ BGR → RGB
    ↓ resize 最长边到 1024             → (newH, newW, 3)    保存 original_size, input_size
    ↓ HWC → BCHW                      → (1, 3, newH, newW)
    ↓ (x - mean) / std                → 约 [-2, 2]
    ↓ padding 到 1024×1024            → (1, 3, 1024, 1024)
    ↓ encoder.onnx
    ↓ image_embeddings                → (1, 256, 64, 64)

用户点击坐标 (X, Y) 基于原始图像
    ↓ apply_coords 缩放               → 基于 resize 后图像的坐标
    ↓ decoder.onnx（内部做归一化+位置编码）
    ↓ low_res_masks                   → (1, num_masks, 256, 256)

    ↓ resize 到 1024×1024
    ↓ 裁掉 padding
    ↓ resize 回原始尺寸               → (1, num_masks, H, W)
    ↓ > threshold 二值化              → bool mask
```

---

## 五、常见错误

| 错误 | 原因 |
|------|------|
| mask 位置偏移 | `apply_coords` 缩放比例计算错误，或忘记缩放 |
| mask 尺寸不对 | 后处理时忘记裁掉 padding，直接从 256 resize 到原图 |
| 分割结果很差 | 图像预处理用了 `/255` 而不是 ImageNet 标准化 |
| decoder 报错 | `point_coords` 没有加 batch 维度，shape 应为 `(1, N, 2)` |
