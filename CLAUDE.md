# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EdgeSAM is an accelerated variant of the Segment Anything Model (SAM) optimized for edge devices. It uses knowledge distillation to compress the original ViT-based SAM image encoder into a CNN-based architecture (RepViT), achieving 40x speedup over SAM while maintaining competitive performance. The project supports deployment to PyTorch, CoreML (iOS), and ONNX formats.

## Installation and Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install EdgeSAM package in editable mode
pip install -e .
```

For ONNX runtime (web demo acceleration):
```bash
# CPU backend
pip install onnxruntime

# GPU backend (do not install both)
pip install onnxruntime-gpu
```

## Key Commands

### Download Checkpoints

```bash
mkdir weights
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam.pth
wget -P weights/ https://huggingface.co/spaces/chongzhou/EdgeSAM/resolve/main/weights/edge_sam_3x.pth
```

### Inference

Run the web demo:
```bash
python web_demo/gradio_app.py
# With ONNX acceleration:
python web_demo/gradio_app.py --enable-onnx --checkpoint [CHECKPOINT] --server-name [SERVER_NAME] --port [PORT]
```

### Training

**Phase 1: Encoder-only knowledge distillation**
```bash
# Download RepViT pretrained weights
wget https://github.com/THU-MIG/RepViT/releases/download/v1.0/repvit_m0_9_distill_300e.pth
mv repvit_m0_9_distill_300e.pth weights/repvit_m1_distill_300.pth

# Train encoder
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
    training/train.py --cfg training/configs/rep_vit_m1_fuse_sa_distill.yaml \
    --output ./output/ \
    --batch-size 8 \
    --use-sync-bn

# Convert encoder-only checkpoint
python scripts/convert_weights.py output/rep_vit_m1_fuse_sa_distill/default/ckpt_epoch_9.pth --encoder-only
```

**Phase 2: Prompt-in-the-loop knowledge distillation**
```bash
# Extract teacher model weights first
python scripts/extract_weights.py

# Train full model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
    training/train.py --cfg training/configs/rep_vit_m1_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml \
    --output ./output/ \
    --batch-size 2
```

**Prepare teacher embeddings** (required before training):
```bash
# Download SAM ViT-H weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P weights/

# Save teacher embeddings
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port 29501 --nproc_per_node 8 \
    training/save_embedding.py --cfg training/configs/teacher/sam_vit_huge_sa1b.yaml \
    --batch-size 8 \
    --eval \
    --resume weights/sam_vit_h_4b8939.pth
```

### Evaluation

Run evaluation suite:
```bash
bash scripts/eval_mIoU.sh [CONFIG_FILE] [CHECKPOINT_PATH]
```

Direct evaluation:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 \
    evaluation/eval_mIoU.py --launcher pytorch \
    [CONFIG_FILE] \
    --checkpoint [CHECKPOINT_PATH] \
    --num-samples 1000 \
    --dataset coco  # or 'lvis'
```

### Model Export

**Export to CoreML** (for iOS deployment):
```bash
# Encoder
python scripts/export_coreml_model.py [CHECKPOINT]

# Decoder
python scripts/export_coreml_model.py [CHECKPOINT] --decoder --use-stability-score
```

**Export to ONNX**:
```bash
# Encoder
python scripts/export_onnx_model.py [CHECKPOINT]

# Decoder
python scripts/export_onnx_model.py [CHECKPOINT] --decoder --use-stability-score
```

## Architecture Overview

### Core Components

**Model Architecture** (`edge_sam/modeling/`):
- `sam.py` / `sam_batch.py`: Main SAM model class, combining encoder, prompt encoder, and mask decoder
- `image_encoder.py`: ViT-based encoder for original SAM
- `rep_vit.py`: RepViT CNN encoder for EdgeSAM (lightweight replacement)
- `prompt_encoder.py`: Encodes point/box prompts and mask inputs
- `mask_decoder.py`: Predicts masks from image embeddings and prompts
- `transformer.py`: Two-way transformer for mask decoder

**Model Registry** (`edge_sam/build_sam.py`):
- `sam_model_registry`: Dictionary mapping model names to builder functions
- Supported models: `"edge_sam"`, `"vit_h"`, `"vit_l"`, `"vit_b"`
- Use `build_sam_from_config()` for training (supports distillation config)
- Use `sam_model_registry["edge_sam"](checkpoint=...)` for inference

**Inference** (`edge_sam/predictor.py`):
- `SamPredictor`: Standard predictor interface compatible with original SAM
- `SamOnnxPredictor` (`edge_sam/onnx/predictor_onnx.py`): ONNX runtime predictor

**Utilities** (`edge_sam/utils/`):
- `amg.py`: Automatic mask generation utilities
- `transforms.py`: Image preprocessing (resize, normalize, pad)
- `coreml.py`: CoreML conversion helpers

### Training Pipeline

**Data** (`training/data/`):
- `sa1b_dataset.py`: SA-1B dataset loader (primary training dataset)
- `coco_dataset.py`: COCO dataset for validation
- `augmentation/`: Data augmentation managers and wrappers
- Dataset expects structure: `datasets/SA-1B/{images,annotations}/{train,val}/`

**Training Strategy**:
1. **Encoder-only distillation**: Distill image encoder from SAM teacher to RepViT student using pixel-wise embedding loss
2. **Prompt-in-the-loop distillation**: Full model distillation including prompt encoder and mask decoder, with prompts (points/boxes) in the training loop

**Key Training Files**:
- `training/train.py`: Main training script with distributed support
- `training/optimizer.py`: Optimizer configuration
- `training/lr_scheduler.py`: Learning rate scheduling
- Config files in `training/configs/`: YAML configs for different training phases

### Knowledge Distillation Architecture

EdgeSAM uses a two-phase distillation approach:

**Phase 1**: Task-agnostic encoder distillation from SAM ViT-H to RepViT
- Teacher embeddings pre-computed and saved to `teacher_embed/sa-1b/`
- Student encoder (RepViT) trained to match teacher embeddings
- Config: `rep_vit_m1_fuse_sa_distill.yaml`

**Phase 2**: Prompt-in-the-loop full model distillation
- Teacher prompt encoder and mask decoder weights loaded from extracted SAM weights
- Student trained with prompts (points/boxes) to match full prediction pipeline
- Config: `rep_vit_m1_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml`
- Requires: `weights/sam_vit_h_prompt_encoder.pth` and `weights/sam_vit_h_mask_decoder.pth`

## Dataset Requirements

Training requires SA-1B dataset organized as:
```
datasets/SA-1B/
├── images/
│   ├── train/
│   └── val/
└── annotations/
    ├── train/
    └── val/
```

Subsets defined in:
- `training/sa_train_subset.txt` (1% training subset)
- `training/sa_val_subset.txt` (validation subset)

## Model Differences

**EdgeSAM vs SAM**:
- EdgeSAM uses RepViT (CNN) encoder instead of ViT, reducing params from 641M to 9.6M
- EdgeSAM can output 1, 3, or 4 mask candidates (SAM outputs 1 or 3)
- EdgeSAM uses stability score for mask selection instead of IoU predictions in CoreML/ONNX exports (IoU token not distilled)

## Important Notes

- EdgeSAM maintains the same encoder-decoder architecture as SAM, so usage is similar
- For ONNX/CoreML models, pre-processing (resize-norm-pad) and post-processing are not included due to framework limitations
- CoreML point coordinates follow `(height, width)` format with `(0, 0)` at top-left
- Point labels: `0=negative`, `1=positive`, `2=box top-left`, `3=box bottom-right`
- Training uses distributed data parallel with PyTorch's `torch.distributed.launch`
- Adjust GPU count and batch size based on available resources
