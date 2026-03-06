# ONNX Model Evaluation Guide

This guide explains the complete pre-processing and post-processing pipeline for evaluating EdgeSAM ONNX models with point prompts, and how to use the `scripts/eval_onnx_mIoU.py` evaluation script.

## Pipeline Overview

```
Image (H, W, 3, uint8, RGB)
  → Resize (longest side → 1024)
  → Normalize (ImageNet mean/std)
  → Pad (right & bottom to 1024×1024)
  → Encoder ONNX → image embeddings (1, 256, 64, 64)

Point prompts (x, y) in original image coordinates
  → Scale coordinates (same ratio as image resize)
  → Decoder ONNX → scores (1, M) + low-res masks (1, M, 256, 256)

Low-res masks
  → Resize to 1024×1024
  → Crop padding region
  → Resize to original (H, W)
  → Select best mask (by score or area)
  → Threshold > 0.0 → binary mask
  → Compare with GT → IoU
```

## Step 1: Image Pre-processing

### 1.1 Resize (keep aspect ratio, longest side = 1024)

```python
scale = 1024.0 / max(H, W)
new_h = int(H * scale + 0.5)
new_w = int(W * scale + 0.5)
```

Use **PIL bilinear** interpolation (not cv2.resize):

```python
from torchvision.transforms.functional import resize, to_pil_image
resized = np.array(resize(to_pil_image(image), (new_h, new_w)))
```

**Important**: The image is resized proportionally, NOT stretched to 1024×1024.

Example: a 1920×1080 image becomes 1024×576.

### 1.2 Convert layout and dtype

```python
# HWC → CHW, add batch dimension, convert to float32
x = resized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)
# Result shape: (1, 3, new_h, new_w)
```

### 1.3 Normalize (ImageNet statistics, 0-255 scale)

```python
pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 3, 1, 1)
pixel_std  = np.array([58.395, 57.12, 57.375],   dtype=np.float32).reshape(1, 3, 1, 1)
x = (x - pixel_mean) / pixel_std
```

**Common mistake**: If you first divide pixels by 255.0, then you must also divide mean/std by 255.0. The default pipeline operates on **raw 0-255 pixel values**.

### 1.4 Pad to 1024×1024 (right and bottom only)

```python
padh = 1024 - new_h
padw = 1024 - new_w
x = np.pad(x, ((0,0), (0,0), (0,padh), (0,padw)), mode='constant', constant_values=0)
# Result shape: (1, 3, 1024, 1024), dtype=float32
```

**Important**: Padding is applied to the **right and bottom** edges only, not evenly on all sides.

### Summary

```
Input:  (H, W, 3) uint8 RGB
Output: (1, 3, 1024, 1024) float32 normalized + padded
Side outputs:
  - original_size = (H, W)          # needed for post-processing
  - input_size    = (new_h, new_w)   # needed for post-processing (crop padding)
```

## Step 2: Encoder Inference

```python
features = encoder_session.run(None, {"image": input_image})[0]
# features shape: (1, 256, 64, 64), dtype=float32
```

The encoder only needs to run **once per image**. The features can be reused for all point prompts on the same image.

## Step 3: Point Coordinate Transformation

Point prompts must be transformed from original image coordinates to the resized image coordinate space.

```python
# coords: (N, 2) array in (x, y) format, in original image space
coords_transformed = coords.copy().astype(np.float32)
coords_transformed[:, 0] *= (new_w / W)  # scale x
coords_transformed[:, 1] *= (new_h / H)  # scale y
```

**Key details**:
- Coordinates are in **(x, y)** order, i.e., **(width, height)**, NOT (row, col)
- No padding offset is needed — the model handles padding internally
- Point labels: `1` = positive (foreground), `0` = negative (background)

Decoder input format:

```python
point_coords = coords_transformed[None, :, :]   # (1, N, 2), float32
point_labels = labels[None, :]                    # (1, N),    float32
```

## Step 4: Decoder Inference

```python
scores, low_res_masks = decoder_session.run(None, {
    "image_embeddings": features,        # (1, 256, 64, 64)
    "point_coords":     point_coords,    # (1, N, 2)
    "point_labels":     point_labels,    # (1, N)
})
# scores:        (1, M) — M mask candidates (typically 4)
# low_res_masks: (1, M, 256, 256) — logits, not probabilities
```

## Step 5: Mask Post-processing

The decoder outputs low-resolution (256×256) mask logits. To get the final binary mask at original image resolution:

### 5.1 Resize to 1024×1024

```python
# For each candidate mask m of shape (256, 256):
m = cv2.resize(m, (1024, 1024), interpolation=cv2.INTER_LINEAR)
```

### 5.2 Crop to remove padding

```python
m = m[:input_size[0], :input_size[1]]   # input_size = (new_h, new_w)
```

### 5.3 Resize to original image resolution

```python
m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
```

**Important**: The order matters: **1024 → crop → original size**. Do NOT skip the intermediate 1024 step or directly resize from 256 to original.

### 5.4 Select the best mask

Multiple mask candidates are returned. Selection strategies:

- **`score`**: Pick the mask with highest predicted score (`scores.argmax()`)
- **`area`**: Pick the mask with largest foreground area (`(m > 0).sum()` for each)

**Recommendation**: Use `area` strategy. EdgeSAM's IoU score token was not fully distilled, so score predictions are unreliable. In our tests, `area` consistently outperforms `score` (e.g., mIoU 0.87 vs 0.30).

### 5.5 Binarize

```python
binary_mask = (selected_mask > 0.0)    # threshold = 0.0 for predictions
gt_binary   = (gt_mask > 0.5)          # threshold = 0.5 for ground truth
```

**Note**: The prediction threshold is **0.0** (the logit boundary), not 0.5.

## Step 6: IoU Calculation

```python
intersection = (pred_binary & gt_binary).sum()
union        = (pred_binary | gt_binary).sum()
iou = intersection / union
```

## Common Pitfalls

| Issue | Wrong | Correct |
|-------|-------|---------|
| Resize method | Stretch to 1024×1024 | Scale longest side to 1024, keep aspect ratio |
| Normalization | Divide by 255 first, then use ImageNet mean ~0.485 | Use raw 0-255 values with mean=[123.675, 116.28, 103.53] |
| Padding | Center padding or all-sides padding | Right and bottom padding only |
| Coordinate order | (row, col) or (h, w) | (x, y) i.e. (width, height) |
| Coordinate transform | Add padding offset | Only scale, no padding offset |
| Post-process order | 256 → original directly | 256 → 1024 → crop padding → original |
| Pred threshold | 0.5 | 0.0 |
| Data type | float64 (numpy default) | float32 (ONNX requires float32) |
| Mask selection | Trust score predictions | Use area — score is unreliable in EdgeSAM |

## Evaluation Script Usage

### Basic Usage

```bash
python scripts/eval_onnx_mIoU.py \
    --encoder weights/edge_sam_3x_encoder.onnx \
    --decoder weights/edge_sam_3x_decoder.onnx \
    --data-root /path/to/coco \
    --split val
```

### Custom Dataset

```bash
python scripts/eval_onnx_mIoU.py \
    --encoder weights/edge_sam_3x_encoder.onnx \
    --decoder weights/edge_sam_3x_decoder.onnx \
    --data-root /path/to/dataset \
    --anno-file /path/to/annotations.json \
    --img-dir images
```

### All Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder` | (required) | Path to encoder ONNX model |
| `--decoder` | (required) | Path to decoder ONNX model |
| `--data-root` | (required) | Root directory of dataset |
| `--split` | `val` | Dataset split |
| `--annotation` | `coco` | Annotation format: `coco`, `lvis`, `cocofied_lvis` |
| `--anno-file` | None | Direct path to annotation JSON (overrides `--annotation` and `--split`) |
| `--img-dir` | auto | Image directory relative to `--data-root` |
| `--num-samples` | -1 | Number of images to evaluate (-1 = all) |
| `--num-points` | 1 | Number of point prompts per instance |
| `--point-strategy` | `random` | Point sampling: `random` (from GT mask) or `center` (distance transform) |
| `--mask-select` | `score` | Mask selection: `score` (highest score) or `area` (largest area) |
| `--seed` | 42 | Random seed |
| `--max-prompts-per-image` | -1 | Max instances per image (-1 = all) |

### Annotation Format

The script expects COCO-format JSON annotations:

```json
{
  "images": [{"id": 1, "width": 1920, "height": 1080, "file_name": "image.jpg"}],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "segmentation": [[x1,y1,x2,y2,...]] or {"counts": "...", "size": [h,w]},
      "bbox": [x, y, w, h],
      "area": 1234,
      "iscrowd": 0
    }
  ]
}
```

### Example Results

Evaluated on 7 images with 9 instances (1 random point per instance):

| Mask Selection | mIoU | Median IoU | Std |
|----------------|------|------------|-----|
| `score` | 0.30 | 0.18 | 0.31 |
| `area` | **0.87** | **0.90** | 0.09 |
