# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EdgeSAM is an accelerated variant of the Segment Anything Model (SAM) optimized for edge devices. It uses knowledge distillation to compress the original ViT-based SAM image encoder into a CNN-based architecture (RepViT), achieving 40x speedup over SAM while maintaining competitive performance. The project supports deployment to PyTorch, CoreML (iOS), ONNX, and NPU-optimized ONNX formats.

License: NTU S-Lab License 1.0

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

Dev tools (linting/formatting):
```bash
pip install -e ".[dev]"
# Provides: flake8, isort, black, mypy
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

Evaluation supports SA-1B, COCO, LVIS, and cocofied-LVIS datasets with point prompts, box prompts, and iterative refinement. Metrics include mIoU, AP, AP_s/m/l.

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

**Export to NPU-optimized ONNX** (for Qualcomm/Mediatek NPUs):
```bash
# Decoder with NPU optimizations
python scripts/export_onnx_model_npu.py [CHECKPOINT] --decoder --use-stability-score

# Inspect operator support
python scripts/export_onnx_model_npu.py [CHECKPOINT] --decoder --check-ops-only
```

See `docs/npu_onnx_export.md` for detailed technical guide on NPU optimizations (LayerNorm decomposition, GELU approximation, int64→int32 conversion, graph simplification).

## Repository Structure

```
EdgeSAM/
├── edge_sam/                          # Main Python package
│   ├── __init__.py                    # Public API exports
│   ├── build_sam.py                   # Model registry & builder functions
│   ├── predictor.py                   # SamPredictor inference interface
│   ├── automatic_mask_generator.py    # SamAutomaticMaskGenerator
│   ├── config.py                      # YACS-based configuration system
│   ├── modeling/                      # Neural network architectures
│   │   ├── sam.py                     # Main Sam model class
│   │   ├── sam_batch.py               # Batch-enabled variant for training
│   │   ├── rep_vit.py                 # RepViT CNN encoder (EdgeSAM)
│   │   ├── image_encoder.py           # ViT encoder (original SAM)
│   │   ├── prompt_encoder.py          # Point/box/mask prompt encoding
│   │   ├── mask_decoder.py            # Transformer mask decoder
│   │   ├── transformer.py             # Two-way transformer blocks
│   │   └── common.py                  # Shared layers (LayerNorm2d, MLPBlock)
│   ├── onnx/                          # ONNX runtime inference
│   │   └── predictor_onnx.py          # SamOnnxPredictor (drop-in replacement)
│   └── utils/                         # Utilities
│       ├── transforms.py              # ResizeLongestSide image preprocessing
│       ├── amg.py                     # Mask generation utilities, NMS, RLE
│       ├── common.py                  # Point/mask/coordinate utilities
│       └── coreml.py                  # SamCoreMLModel wrapper
│
├── training/                          # Training & distillation pipeline
│   ├── train.py                       # Main distributed training script
│   ├── save_embedding.py              # Teacher embedding extraction
│   ├── optimizer.py                   # AdamW/SGD builder
│   ├── lr_scheduler.py                # Cosine/step/linear schedulers
│   ├── utils.py                       # Checkpoint loading, prompt sampling
│   ├── my_meter.py                    # AverageMeter for metrics
│   ├── logger.py                      # Training logger
│   ├── configs/                       # YAML training configurations
│   │   ├── rep_vit_m1_fuse_sa_distill.yaml                            # Phase 1: encoder distillation
│   │   ├── rep_vit_m1_fuse_enc_dec_4m_ft_bp_iter2b_sa_distill.yaml   # Phase 2: full model
│   │   └── teacher/sam_vit_huge_sa1b.yaml                             # Teacher config
│   ├── data/                          # Dataset implementations
│   │   ├── build.py                   # Dataset/dataloader builder
│   │   ├── sa1b_dataset.py            # SA-1B dataset loader
│   │   ├── coco_dataset.py            # COCO dataset loader
│   │   ├── sampler.py                 # Distributed sampler
│   │   └── augmentation/              # Data augmentation
│   │       ├── manager.py             # Augmentation manager
│   │       ├── aug_random.py          # Random augmentations
│   │       └── dataset_wrapper.py     # Teacher embedding wrapper
│   ├── sa_train_subset.txt            # 1% SA-1B training subset (11 files)
│   └── sa_val_subset.txt              # Validation subset (2 files)
│
├── evaluation/                        # Evaluation code
│   └── eval_mIoU.py                   # mIoU/AP evaluation on SA-1B/COCO/LVIS
│
├── scripts/                           # Export & utility scripts
│   ├── export_coreml_model.py         # PyTorch → CoreML
│   ├── export_onnx_model.py           # PyTorch → ONNX
│   ├── export_onnx_model_npu.py       # PyTorch → NPU-optimized ONNX
│   ├── convert_weights.py             # Encoder-only → full SAM checkpoint
│   ├── extract_weights.py             # Extract SAM prompt encoder/decoder
│   ├── eval_mIoU.sh                   # Evaluation shell wrapper
│   ├── distill_embedding.sh           # Distillation shell script
│   └── save_embedding.sh              # Embedding extraction shell script
│
├── web_demo/                          # Interactive web interface
│   ├── gradio_app.py                  # Gradio app (point/box prompts)
│   ├── utils/
│   │   ├── tools.py                   # Annotation utilities
│   │   └── tools_gradio.py            # Gradio-specific utilities
│   └── assets/                        # Demo assets
│
├── docs/                              # Additional documentation
│   └── npu_onnx_export.md             # NPU ONNX export technical guide
│
├── notebooks/                         # Jupyter notebooks
│   └── predictor_example.ipynb        # Interactive usage example
│
├── setup.py                           # Package setup (name: edge_sam, version: 1.0)
├── setup.cfg                          # isort configuration
├── requirements.txt                   # 16 pinned dependencies
├── README.md                          # Project documentation
└── README_TRAIN.md                    # Training guide
```

## Architecture Overview

### Core Components

**Model Architecture** (`edge_sam/modeling/`):
- `sam.py`: Main `Sam` model class combining encoder, prompt encoder, and mask decoder. Supports optional RPN head integration.
- `sam_batch.py`: `SamBatch`, `PromptEncoderBatch`, `MaskDecoderBatch` - batch-enabled variants used during training
- `rep_vit.py`: `RepViT` CNN encoder for EdgeSAM (9.6M params). Supports m0, m1, m2, m3 variants.
- `image_encoder.py`: `ImageEncoderViT` - ViT-based encoder for original SAM (641M params)
- `prompt_encoder.py`: `PromptEncoder` - encodes point/box prompts and mask inputs into embeddings
- `mask_decoder.py`: `MaskDecoder` - predicts masks from image embeddings and prompt tokens
- `transformer.py`: `TwoWayTransformer` - self-attention + cross-attention blocks for mask decoder
- `common.py`: Shared components (`LayerNorm2d`, `MLPBlock`)

**Model Registry** (`edge_sam/build_sam.py`):
- `sam_model_registry`: Dictionary mapping model names to builder functions
- Supported keys: `"default"`, `"edge_sam"`, `"vit_h"`, `"vit_l"`, `"vit_b"`
- `build_sam_from_config(cfg_file, checkpoint, enable_distill, enable_batch)` for training
- `sam_model_registry["edge_sam"](checkpoint=...)` for inference

**Inference** (`edge_sam/predictor.py`):
- `SamPredictor`: Standard predictor interface compatible with original SAM
  - `set_image()` → pre-compute image embeddings
  - `predict()` → generate masks from points/boxes
  - Multi-mask output (1, 3, or 4 candidates)
  - Stability score calculation and mask refinement

**Automatic Mask Generation** (`edge_sam/automatic_mask_generator.py`):
- `SamAutomaticMaskGenerator`: Generates masks for entire images without prompts
  - Point grid generation, multi-scale cropping, NMS, quality filtering

**ONNX Inference** (`edge_sam/onnx/predictor_onnx.py`):
- `SamOnnxPredictor`: ONNX Runtime-based inference, drop-in replacement for `SamPredictor`
- Supports CUDA GPU acceleration

**Utilities** (`edge_sam/utils/`):
- `transforms.py`: `ResizeLongestSide` - image preprocessing (resize, normalize, pad)
- `amg.py`: Mask generation utilities, stability scores, RLE encoding, NMS
- `common.py`: Point/mask utilities, coordinate transforms
- `coreml.py`: `SamCoreMLModel` - CoreML inference wrapper

### Configuration System

`edge_sam/config.py` uses YACS (`CfgNode`) with these key sections:

| Section | Key Parameters |
|---------|---------------|
| `DATA` | `BATCH_SIZE`, `IMG_SIZE` (1024), `DATASET`, `NUM_WORKERS`, `PIN_MEMORY` |
| `MODEL` | `TYPE` (model variant), `PRETRAINED`, `RESUME` |
| `DISTILL` | `ENCODER_ONLY`, `PIXEL_WISE`, `CHANNEL_WISE`, `DECODER_BCE/DICE/FOCAL`, `DECODER_IOU`, `PROMPT_TYPE`, `DECODE_ITERS`, `MULTIMASK_OUTPUT`, `SAVE_TEACHER_EMBED`, `TEACHER_EMBED_PATH` |
| `TRAIN` | `EPOCHS`, `BASE_LR`, `WARMUP_LR/EPOCHS`, `OPTIMIZER`, `LR_SCHEDULER` |
| `AMP_ENABLE` | Automatic mixed precision toggle |

Command-line overrides: `python train.py --cfg config.yaml --batch-size 4 --opt TRAIN.EPOCHS 10`

### Training Pipeline

**Data** (`training/data/`):
- `sa1b_dataset.py`: SA-1B dataset loader with prompt sampling and mask filtering
- `coco_dataset.py`: COCO dataset for evaluation
- `build.py`: Dataset/dataloader construction
- `sampler.py`: Custom distributed sampler
- `augmentation/`: Random augmentations, augmentation manager, teacher embedding dataset wrapper

**Training Strategy**:
1. **Encoder-only distillation**: Distill image encoder from SAM ViT-H to RepViT using pixel-wise embedding loss
2. **Prompt-in-the-loop distillation**: Full model distillation with prompts (points/boxes) in the training loop

**Loss Functions** (configured via `DISTILL` config section):
- Pixel-wise L2 embedding matching
- Channel-wise distillation
- Decoder BCE, Dice, Focal losses
- IoU prediction loss
- Attention map distillation

**Key Training Files**:
- `training/train.py`: Main training script with distributed support, AMP, gradient clipping, W&B/TensorBoard logging
- `training/save_embedding.py`: Pre-compute and save teacher (SAM ViT-H) embeddings
- `training/optimizer.py`: AdamW/SGD optimizer builder with weight decay
- `training/lr_scheduler.py`: Cosine/step/linear schedulers (timm-based)
- `training/utils.py`: Checkpoint loading, uncertainty-based prompt sampling, point generation
- `training/my_meter.py`: `AverageMeter` for tracking training metrics

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
- `training/sa_train_subset.txt` (1% training subset, 11 TAR files)
- `training/sa_val_subset.txt` (validation subset, 2 TAR files)

## Dependencies

Core dependencies (pinned in `requirements.txt`):
- `torch==2.0.0`, `torchvision==0.15.1` - deep learning framework
- `opencv-python==4.8.0.74` - image processing
- `pycocotools==2.0.6` - COCO dataset tools
- `gradio==4.7.1` - web demo UI
- `timm==0.4.12` - vision model utilities (LR schedulers)
- `coremltools==7.1` - CoreML conversion
- `yacs==0.1.8` - configuration management
- `loralib==0.1.2` - LoRA fine-tuning support
- `kornia==0.7.1` - differentiable computer vision ops
- `wandb==0.16.3` - experiment tracking
- `mmengine==0.10.3`, `mmcv==2.0.0rc4` - OpenMMLab utilities (evaluation)
- `tensorboard==2.14.0` - training visualization

## Code Style and Conventions

- **Formatting**: isort configured in `setup.cfg` (line length 100, multi-line mode 3, trailing commas)
- **Dev tools available**: `flake8`, `isort`, `black`, `mypy` (install via `pip install -e ".[dev]"`)
- **Import order** (isort sections): FUTURE → STDLIB → THIRDPARTY → MYSELF (edge_sam) → FIRSTPARTY → LOCALFOLDER
- **No test suite**: There is no formal test directory. Validation is done through the evaluation pipeline (`evaluation/eval_mIoU.py`) and manual testing via notebooks/web demo.

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
- Key model constants: `prompt_embed_dim=256`, `image_size=1024`, `vit_patch_size=16`, `image_embedding_size=64`
- NPU export applies 5 optimizations: LayerNorm decomposition, GELU→tanh approximation, int16→int32, ONNX graph simplification, int64→int32 conversion
