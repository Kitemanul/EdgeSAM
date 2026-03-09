# EdgeSAM Fine-tuning Guide: Point Prompt Segmentation

This guide walks you through fine-tuning EdgeSAM's mask decoder with point prompts on your own COCO-format dataset. A typical use case is training the model to segment **whole objects** (e.g., athletes) from a single point click, instead of only segmenting local parts (clothes, skin, etc.).

## Table of Contents

1. [Training Strategy](#training-strategy)
2. [Why Fine-tune?](#why-fine-tune)
3. [Prerequisites](#prerequisites)
4. [Data Preparation](#data-preparation)
5. [Training](#training)
6. [Parameter Tuning Guide](#parameter-tuning-guide)
7. [Evaluation](#evaluation)
8. [Inference](#inference)
9. [Troubleshooting](#troubleshooting)

---

## Training Strategy

### The Core Problem

SAM / EdgeSAM is trained on SA-1B, which has **multi-granularity** annotations: the same person might have separate masks for the whole body, upper body, shirt, pants, shoes, skin, and even logos. A single point prompt is inherently ambiguous — the model must guess which granularity you want.

When you click on an athlete, the original model often returns the most "visually coherent" local region (a piece of clothing, a patch of skin) rather than the whole person. This is by design in SAM, not a bug.

### Strategy Overview

Fine-tuning solves this by **redefining what a point means in your domain**: a point on a person always means the whole person.

```
Before fine-tuning:  point on shirt → shirt mask (part-level)
After fine-tuning:   point on shirt → whole athlete mask (object-level)
```

### Strategy 1: Single-Point Supervised Fine-tuning (Recommended Start)

The simplest and most effective approach. Each training sample is:
- **Input**: one image + one random point sampled from inside the GT mask
- **Target**: the whole-object GT mask
- **Loss**: BCE + Dice on predicted mask vs GT mask

```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --num-points 1 --epochs 10 --lr 1e-4
```

**Why it works**: Across many training iterations, the random point lands on different body parts (head, torso, legs, shoes). Every time, the target is the **same** whole-body mask. The decoder learns that regardless of where the point is, the output should be the full person.

**Limitation**: Each iteration only gives one "perspective" of the object. May need more epochs to converge.

### Strategy 2: Multi-Point Training (Stronger Whole-Object Signal)

Sample 2-3 positive points from different locations within the same GT mask. This gives the decoder a stronger geometric signal: "these scattered points all belong to one object."

```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --num-points 3 --epochs 10 --lr 1e-4
```

**When to use**: If Strategy 1 still produces part-level masks after sufficient training. Multi-point is especially helpful for:
- Tall/elongated objects (full-body athletes in action poses)
- Objects with visually distinct parts (multi-colored uniforms)
- Complex poses where body parts are separated (spread arms/legs)

**Trade-off**: The model learns with multi-point input, but at inference you might only provide 1 point. Training with `--num-points 3` and inferring with 1 point still works — the decoder generalizes — but the best results come when train and inference point counts are similar.

### Strategy 3: Progressive Training (Two-Stage)

If your dataset is large enough, train in two stages:

**Stage 1**: Coarse alignment with multi-point (fast convergence)
```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --num-points 3 --epochs 5 --lr 3e-4 \
    --output output/stage1
```

**Stage 2**: Refine with single-point (match inference scenario)
```bash
python training/finetune.py \
    --checkpoint output/stage1/finetune_best.pth \
    --ann-file train.json --img-dir images/train/ \
    --num-points 1 --epochs 10 --lr 5e-5 \
    --output output/stage2
```

This gives the decoder a strong whole-object prior first, then adapts it to single-point input.

### Strategy 4: Hard Example Mining with Focal Loss

If the model handles easy cases (torso clicks) well but fails on hard cases (edge clicks on hands, feet, head), add focal loss to focus on boundary regions:

```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --num-points 1 --epochs 15 --lr 1e-4 \
    --bce-weight 2.0 --dice-weight 5.0 --focal-weight 5.0
```

Focal loss down-weights easy pixels (clear foreground/background) and up-weights hard pixels (boundaries, ambiguous regions).

### Choosing a Strategy

```
Start here
    │
    ▼
Strategy 1 (single-point, default settings)
    │
    ├── mIoU > 0.85? ──▶ Done! Use this model.
    │
    ▼
Still segmenting parts?
    │
    ├── Yes ──▶ Strategy 2 (multi-point, --num-points 3)
    │               │
    │               ├── Better? ──▶ Strategy 3 (progressive, 2-stage)
    │               │
    │               └── Still bad? ──▶ Check annotation quality.
    │                                   Masks must be whole-body.
    │
    └── Boundaries are poor ──▶ Strategy 4 (focal loss)
```

### Key Principles

1. **Data quality > training tricks**: Whole-body GT masks are the single most important factor. If your masks only cover torsos, no training strategy will produce full-body segmentation.

2. **Freeze the encoder**: The RepViT encoder already extracts excellent features. Fine-tuning it risks catastrophic forgetting and requires much more data/compute. Only consider unfreezing the encoder if you have 10,000+ domain-specific images and the encoder features are clearly inadequate.

3. **Use `multimask_output=False` at inference**: The fine-tuned decoder is trained with single mask output (`num_multimask_outputs=1`). Always use `multimask_output=False` in `SamPredictor.predict()` for best results.

4. **Evaluation point sampling matters**: During validation, mIoU is computed with random points. If you care about specific click locations (e.g., always clicking on the torso), evaluate with those specific locations too.

5. **Domain narrowing helps**: "Segment athletes in sports photos" is easier to learn than "segment any person in any context." If possible, make your training data match your deployment scenario.

---

## Why Fine-tune?

EdgeSAM (distilled from SAM) was trained on SA-1B, which contains masks at **all granularities** — from whole objects to fine-grained parts (sleeves, shoes, skin patches). When you click a point on a person, the model doesn't know whether you want:
- The whole person
- Just the shirt
- Just the visible skin

Fine-tuning the mask decoder on your domain-specific data teaches the model that **a point on a person → the whole person mask**.

### What gets trained?

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| Image Encoder (RepViT) | ~9.6M | Frozen |
| Prompt Encoder | ~6K | Frozen |
| Mask Decoder | ~4.1M | **Yes** |

Only the mask decoder is trained, which means:
- Fast training (minutes to hours, not days)
- Low GPU memory requirement (single consumer GPU is enough)
- Preserves the encoder's strong feature extraction ability
- Low risk of catastrophic forgetting

---

## Prerequisites

```bash
# Install EdgeSAM
pip install -e .

# Download pretrained weights (pick one)
mkdir -p weights
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth
```

**Recommended starting checkpoint**: `edge_sam_3x.pth` (trained 3x longer, stronger baseline).

---

## Data Preparation

### Required Format

Your dataset must follow **standard COCO annotation format**:

```
your_dataset/
├── images/
│   ├── train/
│   │   ├── 001.jpg
│   │   ├── 002.jpg
│   │   └── ...
│   └── val/
│       ├── 100.jpg
│       └── ...
└── annotations/
    ├── train.json
    └── val.json
```

### Annotation JSON Structure

```json
{
  "images": [
    {"id": 1, "file_name": "001.jpg", "height": 720, "width": 1280}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "segmentation": {"size": [720, 1280], "counts": "..."},
      "bbox": [100, 200, 150, 300],
      "area": 45000,
      "category_id": 1,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "athlete"}
  ]
}
```

Supported segmentation formats:
- **Compressed RLE** (recommended): `{"size": [h, w], "counts": "encoded_string"}`
- Uncompressed RLE: `{"size": [h, w], "counts": [int_list]}`
- Polygon: `[[x1,y1,x2,y2,...], ...]`

### Critical: Annotation Quality

For "whole athlete" segmentation, your masks **must** cover the entire person body, not just parts. If your annotations are part-level (e.g., only upper body), the model will only learn to segment that part.

### Dataset Size Guidelines

| Dataset Size | Expected Result |
|-------------|----------------|
| 100-500 images | Reasonable results if your domain is narrow (e.g., one sport type) |
| 500-2000 images | Good results for most use cases |
| 2000-10000 images | Strong results with good generalization |
| 10000+ images | Diminishing returns, likely overkill for decoder-only fine-tuning |

---

## Training

### Quick Start (Single GPU)

```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file /path/to/annotations/train.json \
    --img-dir /path/to/images/train/ \
    --output output/finetune \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 4 \
    --num-points 1
```

### With Validation

```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file /path/to/annotations/train.json \
    --img-dir /path/to/images/train/ \
    --val-ann-file /path/to/annotations/val.json \
    --val-img-dir /path/to/images/val/ \
    --output output/finetune \
    --epochs 10 \
    --lr 1e-4
```

### Multi-GPU

```bash
torchrun --nproc_per_node NUM_GPUS training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file /path/to/annotations/train.json \
    --img-dir /path/to/images/train/ \
    --output output/finetune \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 4
```

Note: `--batch-size` is **per GPU**. With 4 GPUs and `--batch-size 4`, effective batch size is 16.

### Output Files

```
output/finetune/
├── finetune_epoch_0.pth    # Checkpoint after epoch 0
├── finetune_epoch_1.pth    # Checkpoint after epoch 1
├── finetune_step_500.pth   # Step checkpoint (if --save-step-freq is set)
├── ...
├── finetune_best.pth       # Best checkpoint (highest val mIoU)
└── tensorboard/            # TensorBoard event files
```

All checkpoints are full model state_dicts, directly loadable by `sam_model_registry['edge_sam']()`.

### Monitoring with TensorBoard

Training metrics are logged automatically to `{output}/tensorboard/`. Launch TensorBoard in a separate terminal:

```bash
tensorboard --logdir output/finetune/tensorboard
# Then open http://localhost:6006 in your browser
```

Logged metrics:

| Metric | X-axis | Description |
|--------|--------|-------------|
| `train/loss_step` | global step | Per-step loss |
| `train/mIoU_step` | global step | Per-step mIoU |
| `train/lr` | global step | Learning rate schedule |
| `train/loss_epoch` | epoch | Epoch-averaged training loss |
| `train/mIoU_epoch` | epoch | Epoch-averaged training mIoU |
| `val/loss_epoch` | epoch | Epoch-averaged validation loss (if val set provided) |
| `val/mIoU_epoch` | epoch | Epoch-averaged validation mIoU (if val set provided) |

### Saving Checkpoints by Step

Use `--save-step-freq` to save checkpoints at fixed step intervals, useful for long training runs or early stopping:

```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --output output/finetune \
    --epochs 10 \
    --save-step-freq 500    # Save every 500 steps
```

This produces `finetune_step_500.pth`, `finetune_step_1000.pth`, etc., in addition to per-epoch checkpoints. Set to `0` (default) to disable step-based saving.

---

## Parameter Tuning Guide

### All Parameters

```
Data:
  --ann-file            COCO annotation JSON path (train) [required]
  --img-dir             Image directory path (train) [required]
  --val-ann-file        COCO annotation JSON path (val) [optional]
  --val-img-dir         Image directory path (val) [optional]

Model:
  --checkpoint          Pretrained EdgeSAM weights path [required]

Training:
  --output              Output directory [default: output/finetune]
  --epochs              Number of training epochs [default: 10]
  --lr                  Learning rate [default: 1e-4]
  --weight-decay        AdamW weight decay [default: 0.01]
  --batch-size          Images per GPU per step [default: 4]
  --num-workers         DataLoader workers [default: 4]
  --max-prompts         Max mask prompts sampled per image [default: 16]
  --num-points          Positive points sampled per mask [default: 1]
  --bce-weight          BCE loss weight [default: 5.0]
  --dice-weight         Dice loss weight [default: 5.0]
  --focal-weight        Focal loss weight [default: 0.0]
  --save-freq           Save checkpoint every N epochs [default: 1]
  --save-step-freq      Save checkpoint every N steps, 0 = disabled [default: 0]
  --print-freq          Print log every N steps [default: 50]
  --seed                Random seed [default: 42]
```

### Learning Rate (`--lr`)

The most important hyperparameter. Since we only train the decoder with frozen encoder:

| LR | When to use |
|----|------------|
| `5e-5` | Very small dataset (<200 images), or fine-tuning an already fine-tuned model |
| **`1e-4`** | **Recommended starting point for most cases** |
| `3e-4` | Larger datasets (>2000 images), or if `1e-4` converges too slowly |
| `5e-4` | Large dataset + large batch size. Monitor for instability |
| `1e-3+` | Too high. Will likely cause divergence or oscillation |

The scheduler uses **cosine annealing**, decaying to `lr * 0.01` by the final epoch. No manual scheduling needed.

### Epochs (`--epochs`)

| Dataset Size | Recommended Epochs |
|-------------|-------------------|
| <500 images | 15-30 (more epochs to compensate for less data) |
| 500-2000 | 10-20 |
| 2000-5000 | 5-10 |
| 5000+ | 3-5 |

**How to decide**: Watch the training mIoU. If it's still improving at the end, increase epochs. If val mIoU peaks early and then drops, reduce epochs or use the best checkpoint.

### Batch Size (`--batch-size`)

| GPU Memory | Max Batch Size |
|-----------|---------------|
| 8 GB | 1-2 |
| 12 GB | 2-4 |
| 24 GB | 4-8 |
| 40-80 GB | 8-16 |

Larger batch sizes give more stable gradients. If you reduce batch size, also reduce the learning rate proportionally (e.g., batch_size=2 → lr=5e-5).

### Number of Points (`--num-points`)

Controls how many positive points are randomly sampled from each GT mask during training.

| Value | Effect |
|-------|--------|
| **1** | **Default. One random point per mask.** Across epochs, points are sampled from different locations, so the model sees diverse prompt positions |
| 2-3 | Multiple points from different body parts. Gives stronger "whole object" signal per iteration. Useful if 1-point results are still too part-focused |
| 5+ | Diminishing returns. May overfit to multi-point scenarios, weakening single-point inference |

**Recommendation**: Start with `--num-points 1`. If the model still segments parts instead of the whole object after training, try `--num-points 3`.

### Max Prompts (`--max-prompts`)

Limits how many masks are sampled from each image per iteration. This controls:
- Memory usage (more prompts = more decoder forward passes)
- Training balance (prevents images with many annotations from dominating)

| Value | When to use |
|-------|------------|
| 8 | Memory-constrained, or images have many (>20) annotations |
| **16** | **Default. Good balance** |
| 32+ | Large GPU memory, images typically have many objects |

### Loss Weights

Three loss functions are available:

| Loss | Default Weight | Role |
|------|---------------|------|
| BCE (`--bce-weight`) | 5.0 | Pixel-level binary classification. Good for overall mask shape |
| Dice (`--dice-weight`) | 5.0 | Region-level overlap. Handles class imbalance (small vs large masks) |
| Focal (`--focal-weight`) | 0.0 | Hard example mining. Helps with boundary precision |

**Recommended configurations**:

```bash
# Default (works well for most cases)
--bce-weight 5.0 --dice-weight 5.0 --focal-weight 0.0

# If masks have fine boundary details (e.g., hair, fingers)
--bce-weight 5.0 --dice-weight 5.0 --focal-weight 2.0

# If the model struggles with small objects
--bce-weight 2.0 --dice-weight 5.0 --focal-weight 0.0
```

### Recommended Configurations by Scenario

**Scenario A: Small dataset (<500 images), simple domain**
```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --epochs 20 --lr 5e-5 --batch-size 2 --num-points 1
```

**Scenario B: Medium dataset (500-2000 images), typical use case**
```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --val-ann-file val.json --val-img-dir images/val/ \
    --epochs 10 --lr 1e-4 --batch-size 4 --num-points 1
```

**Scenario C: Large dataset (2000+ images), multi-point training**
```bash
python training/finetune.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file train.json --img-dir images/train/ \
    --val-ann-file val.json --val-img-dir images/val/ \
    --epochs 5 --lr 3e-4 --batch-size 8 --num-points 3
```

---

## Evaluation

Use `scripts/eval_iou.py` to measure mIoU on your validation set. This is essential for comparing models before/after fine-tuning.

### Basic Evaluation

```bash
python scripts/eval_iou.py \
    --checkpoint output/finetune/finetune_best.pth \
    --ann-file /path/to/annotations/val.json \
    --img-dir /path/to/images/val/
```

### Compare Before vs After Fine-tuning

```bash
# Before (original EdgeSAM)
python scripts/eval_iou.py \
    --checkpoint weights/edge_sam_3x.pth \
    --ann-file val.json --img-dir images/val/ \
    --output results/baseline.json

# After (fine-tuned)
python scripts/eval_iou.py \
    --checkpoint output/finetune/finetune_best.pth \
    --ann-file val.json --img-dir images/val/ \
    --output results/finetuned.json
```

### Point Sampling Strategies

The evaluation supports three point strategies:

| Strategy | Flag | Description |
|----------|------|-------------|
| **random** | `--point-strategy random` | Random point from inside GT mask (default). Reflects real-world usage where the user clicks anywhere on the object |
| **center** | `--point-strategy center` | Centroid of the GT mask. Best-case scenario: clicking dead center |
| **bbox_center** | `--point-strategy bbox_center` | Center of the bounding box. Approximates a "rough click" |

Evaluate with all three to understand model robustness:

```bash
for strategy in random center bbox_center; do
    echo "=== $strategy ==="
    python scripts/eval_iou.py \
        --checkpoint output/finetune/finetune_best.pth \
        --ann-file val.json --img-dir images/val/ \
        --point-strategy $strategy
done
```

### Multi-Point Evaluation

If you trained with `--num-points 3`, evaluate with the same setting:

```bash
python scripts/eval_iou.py \
    --checkpoint output/finetune/finetune_best.pth \
    --ann-file val.json --img-dir images/val/ \
    --num-points 3
```

Also test with 1 point to verify single-point generalization:

```bash
python scripts/eval_iou.py \
    --checkpoint output/finetune/finetune_best.pth \
    --ann-file val.json --img-dir images/val/ \
    --num-points 1
```

### Output

The script prints:

```
==================================================
  Checkpoint : output/finetune/finetune_best.pth
  Dataset    : val.json
  Strategy   : random, 1 point(s)
  Images     : 500
  Masks      : 1823
==================================================
  mIoU       : 0.8634
  Median IoU : 0.8921
  Std        : 0.1245
  Min IoU    : 0.1023
  Max IoU    : 0.9876

  IoU distribution:
    >= 0.50 :  95.2%  ###############################################
    >= 0.75 :  82.1%  #########################################
    >= 0.90 :  45.3%  ######################
    >= 0.95 :  12.7%  ######

  Worst 10 images:
    hard_case_001.jpg                         mIoU=0.3214  (3 masks)
    ...
```

Use `--output results.json` to save full per-image results for further analysis.

### Evaluation Parameters

```
--checkpoint          Model weights to evaluate [required]
--ann-file            COCO annotation JSON [required]
--img-dir             Image directory [required]
--num-points          Points per mask [default: 1]
--point-strategy      random / center / bbox_center [default: random]
--max-masks-per-image Limit masks per image (-1 = all) [default: -1]
--max-images          Limit total images (-1 = all) [default: -1]
--seed                Random seed for reproducibility [default: 42]
--output              Save results JSON to this path [optional]
```

### Interpreting Results

| mIoU Range | Interpretation |
|-----------|---------------|
| < 0.60 | Poor. Check data, annotations, or training config |
| 0.60 - 0.75 | Moderate. May need more training or better data |
| 0.75 - 0.85 | Good. Usable for most applications |
| 0.85 - 0.92 | Very good. Fine boundary details may still be off |
| > 0.92 | Excellent |

**Tip**: Look at the "Worst 10 images" to identify failure patterns (occlusion, unusual poses, small objects, etc.).

---

## Inference

### Using SamPredictor

```python
import numpy as np
from PIL import Image
from edge_sam import sam_model_registry, SamPredictor

# Load fine-tuned model
model = sam_model_registry['edge_sam'](
    checkpoint='output/finetune/finetune_best.pth'
)
model.cuda()

predictor = SamPredictor(model)

# Set image
image = np.array(Image.open('test.jpg'))
predictor.set_image(image)

# Predict with point prompt
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),   # pixel coordinates
    point_labels=np.array([1]),         # 1 = positive point
    multimask_output=False,             # single mask output
)

# masks[0] is the binary mask (H, W), bool
# scores[0] is the confidence score
```

### Multi-point Inference

```python
# Multiple positive points on the athlete
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x1, y1], [x2, y2], [x3, y3]]),
    point_labels=np.array([1, 1, 1]),
    multimask_output=False,
)
```

### Batch Inference on Multiple Images

```python
import glob

predictor = SamPredictor(model)

for img_path in glob.glob('test_images/*.jpg'):
    image = np.array(Image.open(img_path))
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=np.array([[x, y]]),
        point_labels=np.array([1]),
        multimask_output=False,
    )

    # Process masks...
```

### Export Fine-tuned Model

The fine-tuned model can be exported to ONNX/CoreML using the same scripts:

```bash
# Export to ONNX
python scripts/export_onnx_model.py output/finetune/finetune_best.pth
python scripts/export_onnx_model.py output/finetune/finetune_best.pth --decoder --use-stability-score

# Export to CoreML
python scripts/export_coreml_model.py output/finetune/finetune_best.pth
python scripts/export_coreml_model.py output/finetune/finetune_best.pth --decoder --use-stability-score
```

---

## Troubleshooting

### Model still segments parts instead of whole objects

1. **Check your annotations**: Make sure GT masks cover the **entire** object, not just parts
2. **Increase `--num-points` to 3**: Multi-point training gives stronger whole-object signal
3. **Train longer**: Increase `--epochs`, the model may need more iterations
4. **Check learning rate**: If loss isn't decreasing, try increasing LR. If loss oscillates, decrease it

### Training loss is not decreasing

- Learning rate too low → try `3e-4` or `5e-4`
- Data loading issue → check that `--img-dir` and `--ann-file` are correct
- Bad annotations → check that masks are non-empty and correctly aligned with images

### Out of memory (OOM)

- Reduce `--batch-size` (try 1 or 2)
- Reduce `--max-prompts` (try 4 or 8)
- Make sure the encoder is frozen (it should be by default)

### mIoU is high in training but low in validation

- Overfitting → reduce `--epochs`, increase `--weight-decay`
- Too few validation samples → add more diverse val data
- Data distribution mismatch between train and val

### Multi-GPU training is slow

- The decoder is very small, so multi-GPU overhead may dominate for small datasets
- Recommendation: use single GPU for datasets under 5000 images

### Checkpoint compatibility

Fine-tuned checkpoints are fully compatible with the original EdgeSAM inference pipeline, including `SamPredictor`, `SamAutomaticMaskGenerator`, ONNX export, and CoreML export. No code changes needed.
