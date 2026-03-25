"""Fine-tune EdgeSAM mask decoder with point prompts on a COCO-format dataset.

Strategy — 2-Round Iterative Decode
=====================================
Within every training step, each (image, GT-mask) pair is decoded **twice**:

  Round 1: 1 positive point sampled inside the GT mask
           → predict mask  →  BCE + Dice + IoU loss₁
  Round 2: Round-1 point  +  1 correction point sampled from the error region
           (false-positive or false-negative pixels wrt round-1 prediction)
           → predict mask  →  BCE + Dice + IoU loss₂

  total_loss = (loss₁ + loss₂) / 2

The image encoder and prompt encoder are kept **frozen**; only the mask decoder
is trained.  The image embedding is computed once per step and reused across
both rounds.

Checkpoints
===========
Every saved `.pth` file contains a ``training_strategy`` block — a plain dict
recording every hyperparameter needed to reproduce the run:

  {
    "architecture": "EdgeSAM",
    "pretrained_checkpoint": "...",
    "decode_iters": 2,
    "loss_bce": 5.0,
    "loss_dice": 5.0,
    ...
  }

Evaluation
==========
After each epoch, **both** the training set and the validation set (if
provided) are evaluated under identical conditions: ``model.eval()`` mode,
single-round decode (no correction point), full-resolution 1024×1024 IoU.
This makes the two mIoU numbers directly comparable for diagnosing
overfitting.  Use ``--train-eval-frac`` (default 1.0) to evaluate only a
random subset of the training set to save time.

At the end of each validation run, samples whose mean IoU falls below
``--val-iou-thresh`` are written to::

    <output>/bad_preds/epoch_<N>/

For each bad (image, prompt) pair three files are saved:
  ``<name>_mask.png``  — predicted binary mask (white = foreground)
  ``<name>_gt.png``    — GT binary mask (reference)
  ``<name>_conf.png``  — confidence / certainty map
      white pixel (255)  model is certain  (sigmoid ≈ 0 or ≈ 1)
      black pixel (0)    model is uncertain (sigmoid ≈ 0.5)

Usage
=====
  # Single GPU
  python training/finetune.py \\
      --checkpoint weights/edge_sam_3x.pth \\
      --ann-file /path/to/annotations/train.json \\
      --img-dir /path/to/images/train/ \\
      --output output/finetune

  # Multi-GPU (torchrun)
  torchrun --nproc_per_node 4 training/finetune.py \\
      --checkpoint weights/edge_sam_3x.pth \\
      --ann-file /path/to/annotations/train.json \\
      --img-dir /path/to/images/train/ \\
      --output output/finetune

  # With validation
  python training/finetune.py \\
      --checkpoint weights/edge_sam_3x.pth \\
      --ann-file /path/to/annotations/train.json \\
      --img-dir /path/to/images/train/ \\
      --val-ann-file /path/to/annotations/val.json \\
      --val-img-dir /path/to/images/val/ \\
      --output output/finetune
"""

import os
import sys
import time
import argparse
import datetime
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_sam import sam_model_registry
from edge_sam.utils.common import sample_point_in_mask
from training.data.finetune_dataset import COCOFinetuneDataset


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

IMG_SIZE = 1024       # padded image resolution fed to the model
MASK_SIZE = 256       # low-res mask resolution produced by the mask decoder


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

def bce_loss(pred, target, valid=None):
    """Binary cross-entropy.  pred: logits, target: 0/1 float."""
    pred_sig = pred.sigmoid()
    loss = F.binary_cross_entropy(pred_sig, target, reduction='none')
    if valid is not None:
        loss = loss.flatten(1)
        valid = valid.flatten(1)
        loss = (loss * valid).sum(-1) / valid.sum(-1).clamp(min=1)
    return loss.mean()


def dice_loss(pred, target, valid=None):
    """Dice loss.  pred: logits, target: 0/1 float."""
    pred_sig = pred.sigmoid().flatten(1)
    target = target.flatten(1)
    if valid is not None:
        valid = valid.flatten(1)
        pred_sig = pred_sig * valid
        target = target * valid
    num = 2 * (pred_sig * target).sum(-1)
    den = pred_sig.sum(-1) + target.sum(-1)
    return (1 - (num + 1) / (den + 1)).mean()


def focal_loss(pred, target, valid=None, alpha=0.25, gamma=2.0):
    """Focal loss.  pred: logits, target: 0/1 float."""
    pred_sig = pred.sigmoid()
    ce = F.binary_cross_entropy(pred_sig, target, reduction='none')
    p_t = pred_sig * target + (1 - pred_sig) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    if valid is not None:
        loss = loss.flatten(1)
        valid = valid.flatten(1)
        loss = (loss * valid).sum(-1) / valid.sum(-1).clamp(min=1)
    return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_iou_per_mask(pred_logits, gt_mask):
    """Per-mask IoU at low resolution (256x256).

    Used during training for quick monitoring.  For accurate evaluation,
    use ``compute_iou_full_res`` which operates at 1024x1024.

    Returns tensor of shape [N] (N = number of prompts).
    """
    pred = (pred_logits > 0).float()          # threshold at 0 (= mask_threshold)
    gt = gt_mask.float()
    inter = (pred * gt).sum(dim=(-2, -1))     # [N, 1]
    union = (pred + gt - pred * gt).sum(dim=(-2, -1)).clamp(min=1)
    return (inter / union).squeeze(1)         # [N]


@torch.no_grad()
def compute_iou_full_res(low_res_masks, gt_masks, img_size_before_pad):
    """Per-mask IoU computed at 1024x1024 (cropped to valid region).

    This matches the resolution used by ``eval_mIoU.py`` and avoids the
    inflated scores that come from evaluating at 256x256 where boundary
    errors are hidden.

    Args:
        low_res_masks:      [N, 1, 256, 256]  logits from the mask decoder
        gt_masks:           [N, 1024, 1024]    binary GT masks (before padding)
        img_size_before_pad: (H, W)            image size after resize, before padding

    Returns:
        iou: tensor of shape [N]
    """
    # Upsample prediction to 1024x1024
    pred_up = F.interpolate(
        low_res_masks, size=(IMG_SIZE, IMG_SIZE),
        mode='bilinear', align_corners=False)       # [N, 1, 1024, 1024]
    pred_up = (pred_up[:, 0] > 0).float()           # [N, 1024, 1024]

    # Crop to valid (un-padded) region
    h, w = img_size_before_pad
    pred_crop = pred_up[:, :h, :w]
    gt_crop = gt_masks[:, :h, :w].float()

    inter = (pred_crop * gt_crop).sum(dim=(-2, -1))
    union = (pred_crop + gt_crop - pred_crop * gt_crop).sum(dim=(-2, -1)).clamp(min=1)
    return inter / union                            # [N]


# ─────────────────────────────────────────────────────────────────────────────
# Single-image inference helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_image(model, image):
    """Encode one image (frozen encoder).  Returns [1, C, 64, 64]."""
    return model.image_encoder(image.unsqueeze(0))


@torch.no_grad()
def encode_prompts(model, point_coords, point_labels):
    """Encode point prompts (frozen prompt encoder).
    point_coords: [N, P, 2], point_labels: [N, P]
    Returns sparse_emb [N, tokens, C], dense_emb [N, C, 64, 64].
    """
    return model.prompt_encoder(
        points=(point_coords, point_labels),
        boxes=None,
        masks=None,
    )


def decode_masks(model, image_embeddings, point_coords, point_labels, num_multimask=1):
    """Run prompt encoder + mask decoder.  Image encoder must have been called first.

    The mask decoder is inside autograd — gradients flow through it during training.
    The prompt encoder call is wrapped in no_grad (it is always frozen).

    Returns:
        low_res_masks: [N, num_multimask, 256, 256]  (logits)
        iou_pred:      [N, num_multimask]
    """
    sparse_emb, dense_emb = encode_prompts(model, point_coords, point_labels)
    image_pe = model.prompt_encoder.get_dense_pe()  # [1, C, 64, 64]

    low_res_masks, iou_pred = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        num_multimask_outputs=num_multimask,
    )
    return low_res_masks, iou_pred


def select_best_mask(low_res_masks, iou_pred, gt_low=None):
    """Pick the best mask from multi-mask output.

    During **training** pass *gt_low* so that the mask with the highest
    **actual** IoU against the ground truth is selected.  This prevents the
    not-yet-calibrated IoU head from steering the selection toward a poor
    mask and creating a vicious cycle.

    During **inference / evaluation** leave *gt_low=None*; the mask with the
    highest **predicted** IoU is selected (standard SAM behaviour).

    Returns:
        selected_masks: [N, 1, 256, 256]
        selected_iou:   [N]  — the predicted IoU score for the selected mask
    """
    if low_res_masks.shape[1] == 1:
        return low_res_masks, iou_pred.squeeze(1)

    rng = torch.arange(low_res_masks.shape[0], device=low_res_masks.device)

    if gt_low is not None:
        # Training: select by ground-truth IoU (no grad needed)
        with torch.no_grad():
            pred_bin = (low_res_masks > 0).float()                   # [N, M, 256, 256]
            gt_exp   = gt_low.expand_as(pred_bin)                    # [N, M, 256, 256]
            inter    = (pred_bin * gt_exp).sum(dim=(-2, -1))         # [N, M]
            union    = (pred_bin + gt_exp - pred_bin * gt_exp).sum(dim=(-2, -1)).clamp(min=1)
            gt_iou   = inter / union                                 # [N, M]
        best = gt_iou.argmax(dim=1)                                  # [N]
    else:
        # Inference / evaluation: select by predicted IoU
        best = iou_pred.argmax(dim=1)                                # [N]

    return low_res_masks[rng, best].unsqueeze(1), iou_pred[rng, best]


# ─────────────────────────────────────────────────────────────────────────────
# Shared loss computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_round_loss(low_res_masks, gt_low, valid_low, args,
                       pred_iou=None):
    """BCE + Dice (+ optional Focal) + IoU prediction loss at 256×256.

    low_res_masks: [N, 1, 256, 256] logits
    gt_low:        [N, 1, 256, 256] binary float (0/1)
    valid_low:     [N, 1, 256, 256] float mask (0 = padding, 1 = valid)
    pred_iou:      [N] predicted IoU scores (None to skip IoU loss)

    Returns:
        total:      scalar tensor  (weighted sum, used for backward)
        components: dict with float values {'bce', 'dice', 'focal', 'iou'}
    """
    device = low_res_masks.device
    l_bce   = torch.tensor(0.0, device=device)
    l_dice  = torch.tensor(0.0, device=device)
    l_focal = torch.tensor(0.0, device=device)
    l_iou   = torch.tensor(0.0, device=device)
    if args.bce_weight > 0:
        l_bce   = bce_loss(low_res_masks, gt_low, valid_low)   * args.bce_weight
    if args.dice_weight > 0:
        l_dice  = dice_loss(low_res_masks, gt_low, valid_low)  * args.dice_weight
    if args.focal_weight > 0:
        l_focal = focal_loss(low_res_masks, gt_low, valid_low) * args.focal_weight
    if args.iou_weight > 0 and pred_iou is not None:
        # GT IoU: actual overlap between prediction and GT (detached)
        with torch.no_grad():
            gt_iou = compute_iou_per_mask(low_res_masks, gt_low)  # [N]
        l_iou = F.mse_loss(pred_iou, gt_iou) * args.iou_weight
    total = l_bce + l_dice + l_focal + l_iou
    return total, {'bce': l_bce.item(), 'dice': l_dice.item(),
                   'focal': l_focal.item(), 'iou': l_iou.item()}


def prepare_gt_and_valid(gt_masks, img_size_before_pad, device):
    """Downsample GT masks to 256×256 and build a valid-pixel mask.

    gt_masks: [N, H, W] binary float (still at 1024×1024)
    Returns:
        gt_low:    [N, 1, 256, 256]
        valid_low: [N, 1, 256, 256]  (1 inside original image, 0 in padding)
    """
    gt_low = F.interpolate(
        gt_masks.unsqueeze(1), size=(MASK_SIZE, MASK_SIZE), mode='nearest')

    # valid region in the padded 1024×1024 image
    img_h, img_w = img_size_before_pad
    valid = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    valid[:, :, :img_h, :img_w] = 1
    valid_low = F.interpolate(valid, size=(MASK_SIZE, MASK_SIZE),
                              mode='bilinear', align_corners=False)
    valid_low = (valid_low > 0.5).expand(gt_low.shape[0], 1, MASK_SIZE, MASK_SIZE).float()

    return gt_low, valid_low


# ─────────────────────────────────────────────────────────────────────────────
# Correction point sampling  (used in round 2+)
# ─────────────────────────────────────────────────────────────────────────────

def sample_correction_point(low_res_masks, gt_low):
    """Sample one correction point per mask from the error region.

    Operates in 256×256 space then scales to 1024×1024 for the prompt encoder.

    low_res_masks: [N, 1, 256, 256] logits (detached)
    gt_low:        [N, 1, 256, 256] binary float

    Returns:
        corr_xy:  [N, 1, 2]  — correction point coords in 1024×1024 space
        corr_lbl: [N, 1]     — 0 (FP correction) or 1 (FN correction) or -2 (ignore)
    """
    mask_bin = (low_res_masks > 0).float()   # threshold at 0
    gt_bin   = (gt_low > 0.5).float()

    # sample_point_in_mask works on [N, H, W] tensors
    corr_xy, corr_lbl = sample_point_in_mask(mask_bin, gt_bin, num_samples=1)
    # corr_xy: [N, 1, 2] in 256×256 pixel coords

    # Scale from MASK_SIZE space → IMG_SIZE space
    corr_xy = corr_xy.float() * (IMG_SIZE / MASK_SIZE)

    return corr_xy, corr_lbl.float()


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch  (2-round iterative decode)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, epoch, args,
                    distributed, world_size, rank, writer, global_step):
    """One training epoch using the 2-round iterative decode strategy.

    Round 1 — use the dataset's pre-sampled positive point prompt.
    Round 2 — accumulate round-1 point + 1 correction point sampled from
               the error region between the round-1 prediction and the GT.
    total_loss = (loss₁ + loss₂) / decode_iters

    Tracked and logged (step + epoch):
        train/loss_total   — weighted sum of all components
        train/loss_bce     — BCE component
        train/loss_dice    — Dice component
        train/loss_focal   — Focal component (0 if weight=0)
        train/loss_r{i+1}  — per-round total loss (r1, r2, ...)
        train/mIoU         — mean IoU over last round
        train/lr           — learning rate
    """
    model.mask_decoder.train()   if not args.freeze_mask_decoder   else model.mask_decoder.eval()
    model.image_encoder.train()  if not args.freeze_image_encoder  else model.image_encoder.eval()
    model.prompt_encoder.train() if not args.freeze_prompt_encoder else model.prompt_encoder.eval()

    # Running sums for epoch-level averages
    sum_total = 0.0
    sum_bce   = 0.0
    sum_dice  = 0.0
    sum_focal = 0.0
    sum_iou_loss = 0.0
    sum_iou   = 0.0
    sum_round = [0.0] * args.decode_iters  # per-round
    count     = 0
    start     = time.time()

    for step, batch in enumerate(dataloader):
        batch_loss  = torch.tensor(0.0, device='cuda')
        batch_iou   = 0.0
        # Accumulate per-component and per-round values across images in batch
        batch_bce   = 0.0
        batch_dice  = 0.0
        batch_focal = 0.0
        batch_iou_loss = 0.0
        batch_round = [0.0] * args.decode_iters
        n_valid     = 0

        for sample in batch:
            if sample['num_prompts'] == 0:
                continue

            image    = sample['image'].cuda(non_blocking=True)        # [3, 1024, 1024]
            gt_masks = sample['gt_masks'].cuda(non_blocking=True)     # [N, 1024, 1024]
            pts      = sample['point_coords'].cuda(non_blocking=True) # [N, 1, 2]
            lbls     = sample['point_labels'].cuda(non_blocking=True) # [N, 1]
            sz       = sample['img_size_before_pad']                  # (H, W)

            # Prepare GT and valid mask at 256×256 — done once per image
            gt_low, valid_low = prepare_gt_and_valid(gt_masks, sz, image.device)

            # ── Encode image (frozen encoder, no grad) ─────────────────────
            img_emb = encode_image(model, image)                      # [1, C, 64, 64]

            # ── 2-Round Iterative Decode ───────────────────────────────────
            cur_pts  = pts.clone()    # [N, num_pts, 2]  — accumulated across rounds
            cur_lbls = lbls.clone()   # [N, num_pts]
            total_round_loss = torch.tensor(0.0, device=image.device)
            # Per-component sums for this image (averaged over rounds)
            img_bce      = 0.0
            img_dice     = 0.0
            img_focal    = 0.0
            img_iou_loss = 0.0
            last_masks = None          # logits from the previous round

            for round_i in range(args.decode_iters):

                if round_i > 0 and last_masks is not None:
                    # ── Sample correction point from error region ──────────
                    with torch.no_grad():
                        corr_xy, corr_lbl = sample_correction_point(
                            last_masks.detach(), gt_low)
                        # Accumulate: append correction to existing prompt points
                        cur_pts  = torch.cat([cur_pts,  corr_xy],  dim=1)
                        cur_lbls = torch.cat([cur_lbls, corr_lbl], dim=1)

                # ── Decode (mask decoder has gradients) ───────────────────
                low_res_masks, iou_pred = decode_masks(
                    model, img_emb, cur_pts, cur_lbls,
                    num_multimask=args.num_multimask_outputs)

                low_res_masks, sel_iou = select_best_mask(
                    low_res_masks, iou_pred, gt_low=gt_low)
                last_masks = low_res_masks                 # keep for next round

                # ── Loss for this round ────────────────────────────────────
                loss_round, components = compute_round_loss(
                    low_res_masks, gt_low, valid_low, args,
                    pred_iou=sel_iou)
                total_round_loss = total_round_loss + loss_round / args.decode_iters

                img_bce      += components['bce']   / args.decode_iters
                img_dice     += components['dice']  / args.decode_iters
                img_focal    += components['focal'] / args.decode_iters
                img_iou_loss += components['iou']   / args.decode_iters
                batch_round[round_i] += loss_round.item()

            # ── Track IoU (last round prediction) ─────────────────────────
            iou = compute_iou_per_mask(low_res_masks, gt_low).mean()

            batch_loss  = batch_loss + total_round_loss
            batch_iou  += iou.item()
            batch_bce  += img_bce
            batch_dice += img_dice
            batch_focal += img_focal
            batch_iou_loss += img_iou_loss
            n_valid += 1

        if n_valid == 0:
            continue

        batch_loss = batch_loss / n_valid

        optimizer.zero_grad()
        batch_loss.backward()

        # Gradient sync for multi-GPU (manual, no DDP automatic sync)
        if distributed:
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.clip_grad)
        optimizer.step()

        # ── Step-level scalars ─────────────────────────────────────────────
        loss_val     = batch_loss.item()
        iou_val      = batch_iou     / n_valid
        bce_val      = batch_bce     / n_valid
        dice_val     = batch_dice    / n_valid
        focal_val    = batch_focal   / n_valid
        iou_loss_val = batch_iou_loss / n_valid
        round_vals = [v / n_valid for v in batch_round]

        # Running epoch sums
        sum_total    += loss_val
        sum_iou      += iou_val
        sum_bce      += bce_val
        sum_dice     += dice_val
        sum_focal    += focal_val
        sum_iou_loss += iou_loss_val
        for i, rv in enumerate(round_vals):
            sum_round[i] += rv
        count       += 1
        global_step += 1

        if rank == 0 and writer is not None:
            writer.add_scalar('train/loss_total', loss_val, global_step)
            writer.add_scalar('train/loss_bce',   bce_val,  global_step)
            writer.add_scalar('train/loss_dice',  dice_val, global_step)
            if args.focal_weight > 0:
                writer.add_scalar('train/loss_focal', focal_val, global_step)
            if args.iou_weight > 0:
                writer.add_scalar('train/loss_iou', iou_loss_val, global_step)
            for i, rv in enumerate(round_vals):
                writer.add_scalar(f'train/loss_r{i + 1}', rv, global_step)
            writer.add_scalar('train/mIoU', iou_val, global_step)
            writer.add_scalar('train/lr',   optimizer.param_groups[0]['lr'], global_step)

        if rank == 0 and args.save_step_freq > 0 and global_step % args.save_step_freq == 0:
            _save_ckpt(model, optimizer, epoch, global_step, 0.0, args.output,
                       f'finetune_step_{global_step}.pth', args)
            print(f'  [step {global_step}] checkpoint saved')

        if step % args.print_freq == 0 and rank == 0:
            elapsed = time.time() - start
            eta = elapsed / max(step + 1, 1) * (len(dataloader) - step - 1)
            round_str = '  '.join(
                f'r{i + 1}={rv:.4f}' for i, rv in enumerate(round_vals))
            print(f'  [{epoch}][{step}/{len(dataloader)}]  '
                  f'loss {loss_val:.4f} (avg {sum_total / count:.4f})  '
                  f'bce {bce_val:.4f}  dice {dice_val:.4f}'
                  + (f'  focal {focal_val:.4f}' if args.focal_weight > 0 else '')
                  + (f'  iou_l {iou_loss_val:.4f}' if args.iou_weight > 0 else '')
                  + f'  {round_str}'
                    f'  mIoU {iou_val:.4f} (avg {sum_iou / count:.4f})'
                    f'  eta {datetime.timedelta(seconds=int(eta))}')

    n = max(count, 1)
    epoch_components = {
        'total': sum_total / n,
        'bce':   sum_bce   / n,
        'dice':  sum_dice  / n,
        'focal': sum_focal / n,
        'iou':   sum_iou_loss / n,
        'round': [s / n for s in sum_round],
    }
    return epoch_components, sum_iou / n, global_step


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_confidence_map(logit_low, img_size_before_pad):
    """Convert a [1, 256, 256] logit tensor to a confidence PNG (uint8).

    Confidence = |sigmoid(logit) - 0.5| * 2 ∈ [0, 1]
      255  →  model is certain   (sigmoid close to 0 or 1)
        0  →  model is uncertain (sigmoid close to 0.5)

    The output is cropped to the un-padded region of the original image.
    """
    h, w = img_size_before_pad
    # upsample to 1024×1024 then crop
    logit_up = F.interpolate(logit_low[None], size=(IMG_SIZE, IMG_SIZE),
                             mode='bilinear', align_corners=False)[0, 0]  # [1024, 1024]
    prob = torch.sigmoid(logit_up)
    conf = ((prob - 0.5).abs() * 2 * 255).clamp(0, 255)
    return conf[:h, :w].cpu().numpy().astype(np.uint8)


def _make_mask_png(logit_low, img_size_before_pad):
    """Convert a [1, 256, 256] logit to a binary mask PNG (uint8)."""
    h, w = img_size_before_pad
    logit_up = F.interpolate(logit_low[None], size=(IMG_SIZE, IMG_SIZE),
                             mode='bilinear', align_corners=False)[0, 0]
    mask = (logit_up > 0).cpu().numpy().astype(np.uint8) * 255
    return mask[:h, :w]


def _make_gt_png(gt_mask_1024, img_size_before_pad):
    """Convert a [1024, 1024] GT mask to a reference PNG (uint8)."""
    h, w = img_size_before_pad
    gt = gt_mask_1024.cpu().numpy().astype(np.uint8) * 255
    return gt[:h, :w]


def _save_bad_prediction(save_dir, name, logit_low, gt_mask_1024, iou_val,
                         img_size_before_pad):
    """Save mask + GT + confidence map for one bad (image, prompt) pair."""
    os.makedirs(save_dir, exist_ok=True)

    conf = _make_confidence_map(logit_low, img_size_before_pad)
    mask = _make_mask_png(logit_low, img_size_before_pad)
    gt   = _make_gt_png(gt_mask_1024, img_size_before_pad)

    prefix = os.path.join(save_dir, f'{name}_iou{iou_val:.2f}')
    Image.fromarray(mask, mode='L').save(f'{prefix}_mask.png')
    Image.fromarray(gt,   mode='L').save(f'{prefix}_gt.png')
    Image.fromarray(conf, mode='L').save(f'{prefix}_conf.png')


# ─────────────────────────────────────────────────────────────────────────────
# Validation epoch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, dataloader, epoch, args, rank, split='val'):
    """Evaluate on a dataset using identical conditions regardless of split.

    When *split* is ``'val'``, bad predictions (mean IoU < --val-iou-thresh)
    are saved to::

        <output>/bad_preds/epoch_<N>/
          <img_id>_p<i>_iou<X.XX>_mask.png   — predicted binary mask
          <img_id>_p<i>_iou<X.XX>_gt.png     — GT binary mask (reference)
          <img_id>_p<i>_iou<X.XX>_conf.png   — certainty map

    Bad-prediction saving is skipped for ``split='train_eval'``.

    Returns:
        components: dict {'total', 'bce', 'dice', 'focal'} — epoch-averaged losses
        avg_iou:    float
    """
    model.eval()

    # Save and fix random state so that validation uses identical point
    # prompts every epoch, making the mIoU curve purely reflect model
    # improvement rather than sampling noise.
    rng_state_py    = random.getstate()
    rng_state_np    = np.random.get_state()
    rng_state_torch = torch.random.get_rng_state()
    rng_state_cuda  = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    save_bad = (split == 'val')
    save_dir    = os.path.join(args.output, 'bad_preds', f'epoch_{epoch}')
    sum_total   = 0.0
    sum_bce     = 0.0
    sum_dice    = 0.0
    sum_focal   = 0.0
    sum_iou_l   = 0.0
    all_ious    = []           # collect per-mask IoUs (matching eval_mIoU.py)
    count       = 0            # image count (for loss averaging)
    saved_count = 0

    for batch in dataloader:
        for sample in batch:
            if sample['num_prompts'] == 0:
                continue

            image    = sample['image'].cuda(non_blocking=True)
            gt_masks = sample['gt_masks'].cuda(non_blocking=True)  # [N, 1024, 1024]
            pts      = sample['point_coords'].cuda(non_blocking=True)
            lbls     = sample['point_labels'].cuda(non_blocking=True)
            sz       = sample['img_size_before_pad']
            img_id   = sample.get('img_id', count)

            gt_low, valid_low = prepare_gt_and_valid(gt_masks, sz, image.device)

            img_emb = encode_image(model, image)
            low_res_masks, iou_pred = decode_masks(
                model, img_emb, pts, lbls,
                num_multimask=args.num_multimask_outputs)
            low_res_masks, sel_iou = select_best_mask(low_res_masks, iou_pred)

            # loss (all components)
            loss, components = compute_round_loss(
                low_res_masks, gt_low, valid_low, args, pred_iou=sel_iou)

            # per-mask IoU at full resolution (matches eval_mIoU.py)
            iou_per  = compute_iou_full_res(low_res_masks, gt_masks, sz)  # [N]
            all_ious.append(iou_per.cpu())

            sum_total += loss.item()
            sum_bce   += components['bce']
            sum_dice  += components['dice']
            sum_focal += components['focal']
            sum_iou_l += components['iou']
            count     += 1

            # ── Save bad predictions (val split only) ─────────────────────
            img_mean_iou = iou_per.mean().item()
            if save_bad and img_mean_iou < args.val_iou_thresh and saved_count < args.val_max_save:
                N = gt_masks.shape[0]
                for pi in range(N):
                    if saved_count >= args.val_max_save:
                        break
                    iou_val = iou_per[pi].item()
                    name = f'{img_id}_p{pi}'
                    _save_bad_prediction(
                        save_dir=save_dir,
                        name=name,
                        logit_low=low_res_masks[pi],        # [1, 256, 256]
                        gt_mask_1024=gt_masks[pi],          # [1024, 1024]
                        iou_val=iou_val,
                        img_size_before_pad=sz,
                    )
                    saved_count += 1

    n = max(count, 1)
    epoch_components = {
        'total': sum_total / n,
        'bce':   sum_bce   / n,
        'dice':  sum_dice  / n,
        'focal': sum_focal / n,
        'iou':   sum_iou_l / n,
    }
    # Global per-mask mean (same as eval_mIoU.py: equal weight per mask)
    if all_ious:
        all_ious_cat = torch.cat(all_ious, dim=0)
        all_ious_cat = all_ious_cat[~all_ious_cat.isnan()]
        avg_iou = all_ious_cat.mean().item()
    else:
        avg_iou = 0.0

    if rank == 0 and save_bad and saved_count > 0:
        print(f'  => {split} bad_preds ({saved_count} saved) → {save_dir}')

    # Restore random state so training randomness is unaffected.
    random.setstate(rng_state_py)
    np.random.set_state(rng_state_np)
    torch.random.set_rng_state(rng_state_torch)
    if rng_state_cuda is not None:
        torch.cuda.set_rng_state(rng_state_cuda)

    # Restore train/eval modes according to freeze flags
    model.mask_decoder.train()   if not args.freeze_mask_decoder   else model.mask_decoder.eval()
    model.image_encoder.train()  if not args.freeze_image_encoder  else model.image_encoder.eval()
    model.prompt_encoder.train() if not args.freeze_prompt_encoder else model.prompt_encoder.eval()
    return epoch_components, avg_iou


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_training_strategy(args):
    """Return a self-documenting dict of every hyperparameter."""
    return {
        # architecture
        'architecture': 'EdgeSAM',
        'pretrained_checkpoint': args.checkpoint,
        # decode strategy
        'decode_iters': args.decode_iters,
        'num_multimask_outputs': args.num_multimask_outputs,
        # frozen components
        'frozen': [n for n, f in [('image_encoder', args.freeze_image_encoder),
                                   ('prompt_encoder', args.freeze_prompt_encoder),
                                   ('mask_decoder',   args.freeze_mask_decoder)] if f],
        'trained': [n for n, f in [('image_encoder', args.freeze_image_encoder),
                                    ('prompt_encoder', args.freeze_prompt_encoder),
                                    ('mask_decoder',   args.freeze_mask_decoder)] if not f],
        # optimisation
        'epochs': args.epochs,
        'batch_size_per_gpu': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'clip_grad': args.clip_grad,
        # loss
        'loss_bce': args.bce_weight,
        'loss_dice': args.dice_weight,
        'loss_focal': args.focal_weight,
        'loss_iou': args.iou_weight,
        # data
        'ann_file': args.ann_file,
        'img_dir': args.img_dir,
        'max_prompts_per_image': args.max_prompts,
        'num_points_per_mask': args.num_points,
    }


def _save_ckpt(model, optimizer, epoch, global_step, best_iou, output_dir,
               filename, args):
    """Save model state + training strategy to a .pth file."""
    save_path = os.path.join(output_dir, filename)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_iou': best_iou,
        # ← every .pth is self-documenting
        'training_strategy': _make_training_strategy(args),
    }, save_path)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('EdgeSAM fine-tuning — 2-round iterative decode')

    # Data
    p.add_argument('--ann-file',     required=True, help='COCO annotation JSON (train)')
    p.add_argument('--img-dir',      required=True, help='Image directory (train)')
    p.add_argument('--val-ann-file', default=None,  help='COCO annotation JSON (val)')
    p.add_argument('--val-img-dir',  default=None,  help='Image directory (val)')

    # Model
    p.add_argument('--checkpoint', required=True, help='EdgeSAM pretrained weights')
    p.add_argument('--freeze-image-encoder',  action='store_true', default=True,
                   help='Freeze image encoder (default: True).')
    p.add_argument('--no-freeze-image-encoder', dest='freeze_image_encoder', action='store_false',
                   help='Unfreeze image encoder so it is trained jointly.')
    p.add_argument('--freeze-prompt-encoder', action='store_true', default=True,
                   help='Freeze prompt encoder (default: True).')
    p.add_argument('--no-freeze-prompt-encoder', dest='freeze_prompt_encoder', action='store_false',
                   help='Unfreeze prompt encoder so it is trained jointly.')
    p.add_argument('--freeze-mask-decoder',  action='store_true', default=False,
                   help='Freeze mask decoder (default: False).')
    p.add_argument('--no-freeze-mask-decoder', dest='freeze_mask_decoder', action='store_false',
                   help='Unfreeze mask decoder (default behaviour).')

    # Training
    p.add_argument('--output',       default='output/finetune')
    p.add_argument('--epochs',       type=int,   default=10)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--clip-grad',    type=float, default=0.1,
                   help='Gradient norm clip for mask decoder.')
    p.add_argument('--batch-size',   type=int,   default=4,
                   help='Images per GPU.')
    p.add_argument('--num-workers',  type=int,   default=4)
    p.add_argument('--max-prompts',  type=int,   default=16,
                   help='Max GT masks kept per image.')
    p.add_argument('--num-points',   type=int,   default=1,
                   help='Positive points sampled per mask for the initial prompt.')

    # Iterative decode
    p.add_argument('--decode-iters',         type=int, default=2,
                   help='Decode rounds per step (1 = single-round, 2 = with correction).')
    p.add_argument('--num-multimask-outputs', type=int, default=3,
                   help='Number of mask candidates (1, 3, or 4). Best is kept.')

    # Loss weights
    p.add_argument('--bce-weight',   type=float, default=0.0)
    p.add_argument('--dice-weight',  type=float, default=1.0)
    p.add_argument('--focal-weight', type=float, default=20.0)
    p.add_argument('--iou-weight',   type=float, default=1.0,
                   help='Weight for IoU prediction loss (MSE between predicted '
                        'and actual IoU). Default 1.0.')

    # Checkpointing & logging
    p.add_argument('--save-freq',      type=int, default=1,
                   help='Save epoch checkpoint every N epochs.')
    p.add_argument('--save-step-freq', type=int, default=0,
                   help='Save step checkpoint every N steps (0 = disabled).')
    p.add_argument('--print-freq',     type=int, default=50)
    p.add_argument('--seed',           type=int, default=42)

    # Validation
    p.add_argument('--val-iou-thresh', type=float, default=0.5,
                   help='Predictions with mIoU below this are saved as bad_preds.')
    p.add_argument('--val-max-save',   type=int,   default=100,
                   help='Maximum number of bad predictions saved per epoch.')

    # Train-set evaluation (same conditions as val)
    p.add_argument('--train-eval-frac', type=float, default=1.0,
                   help='Fraction of training set to evaluate after each epoch '
                        '(0.0–1.0, default 1.0 = 100%%).')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return batch


def main():
    args = parse_args()

    # ── Distributed setup ────────────────────────────────────────────────────
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if distributed:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    else:
        rank, world_size, local_rank = 0, 1, 0

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output, exist_ok=True)

    if rank == 0:
        print('=' * 60)
        print('EdgeSAM Fine-tuning — 2-Round Iterative Decode')
        print('=' * 60)
        print(f'  Checkpoint    : {args.checkpoint}')
        print(f'  Train ann     : {args.ann_file}')
        print(f'  Train images  : {args.img_dir}')
        print(f'  Output        : {args.output}')
        print(f'  GPUs          : {world_size}')
        print(f'  Batch / GPU   : {args.batch_size}')
        print(f'  Epochs        : {args.epochs}')
        print(f'  LR            : {args.lr}')
        print(f'  Points / mask : {args.num_points}')
        print(f'  Decode iters  : {args.decode_iters}  '
              f'(round 2 adds a correction point from the error region)')
        print(f'  Loss weights  : BCE={args.bce_weight} '
              f'Dice={args.dice_weight} Focal={args.focal_weight}')
        print()

    # ── Model ────────────────────────────────────────────────────────────────
    model = sam_model_registry['edge_sam'](checkpoint=args.checkpoint)
    model.cuda()

    # Freeze / unfreeze components according to CLI flags
    for p in model.image_encoder.parameters():
        p.requires_grad = not args.freeze_image_encoder
    for p in model.prompt_encoder.parameters():
        p.requires_grad = not args.freeze_prompt_encoder
    for p in model.mask_decoder.parameters():
        p.requires_grad = not args.freeze_mask_decoder

    frozen = [n for n, flag in [('image_encoder', args.freeze_image_encoder),
                                 ('prompt_encoder', args.freeze_prompt_encoder),
                                 ('mask_decoder',   args.freeze_mask_decoder)] if flag]
    trained = [n for n, flag in [('image_encoder', args.freeze_image_encoder),
                                  ('prompt_encoder', args.freeze_prompt_encoder),
                                  ('mask_decoder',   args.freeze_mask_decoder)] if not flag]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f'  Frozen        : {", ".join(frozen) or "none"}')
        print(f'  Trainable     : {", ".join(trained) or "none"} '
              f'— {n_train:,} / {n_total:,} ({100 * n_train / n_total:.1f}%)')
        print()

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = COCOFinetuneDataset(
        ann_file=args.ann_file, img_dir=args.img_dir,
        max_prompts_per_image=args.max_prompts,
        num_points_per_mask=args.num_points)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader  = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn)

    val_loader = None
    if args.val_ann_file and args.val_img_dir:
        val_ds = COCOFinetuneDataset(
            ann_file=args.val_ann_file, img_dir=args.val_img_dir,
            max_prompts_per_image=args.max_prompts,
            num_points_per_mask=args.num_points)
        val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
        val_loader  = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn)

    # ── Train-eval loader (same conditions as val) ────────────────────────
    frac = max(0.0, min(1.0, args.train_eval_frac))
    if frac > 0:
        n_total = len(train_ds)
        n_eval  = max(1, int(n_total * frac))
        if frac < 1.0:
            rng_sub = np.random.RandomState(args.seed)
            indices = rng_sub.choice(n_total, size=n_eval, replace=False).tolist()
            train_eval_ds = Subset(train_ds, indices)
        else:
            train_eval_ds = train_ds
        train_eval_sampler = (DistributedSampler(train_eval_ds, shuffle=False)
                              if distributed else None)
        train_eval_loader = DataLoader(
            train_eval_ds, batch_size=args.batch_size,
            shuffle=False, sampler=train_eval_sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn)
    else:
        train_eval_loader = None
        n_eval = 0

    if rank == 0:
        parts = [f'  Train images  : {len(train_ds)}']
        if frac > 0:
            parts.append(f'  Train-eval    : {n_eval} ({frac * 100:.0f}%)')
        if val_loader is not None:
            parts.append(f'  Val images    : {len(val_ds)}')
        print('\n'.join(parts))

    if rank == 0:
        print()

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = None
    if rank == 0:
        tb_dir = os.path.join(args.output, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_dir)
        print(f'  TensorBoard   : tensorboard --logdir {tb_dir}')
        print()

    # ── Training loop ─────────────────────────────────────────────────────
    best_iou    = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f'Epoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]["lr"]:.6f}')

        train_comps, train_iou, global_step = train_one_epoch(
            model, train_loader, optimizer, epoch, args,
            distributed, world_size, rank, writer, global_step)
        scheduler.step()

        if rank == 0:
            round_str = '  '.join(
                f'r{i + 1}={v:.4f}'
                for i, v in enumerate(train_comps['round']))
            print(f'  => Train  loss={train_comps["total"]:.4f}'
                  f'  bce={train_comps["bce"]:.4f}'
                  f'  dice={train_comps["dice"]:.4f}'
                  + (f'  focal={train_comps["focal"]:.4f}' if args.focal_weight > 0 else '')
                  + (f'  iou_l={train_comps["iou"]:.4f}' if args.iou_weight > 0 else '')
                  + f'  {round_str}'
                  + f'  mIoU={train_iou:.4f}  (in-training, 256px)')
            if writer is not None:
                writer.add_scalar('train/loss_total_epoch', train_comps['total'], epoch)
                writer.add_scalar('train/loss_bce_epoch',   train_comps['bce'],   epoch)
                writer.add_scalar('train/loss_dice_epoch',  train_comps['dice'],  epoch)
                if args.focal_weight > 0:
                    writer.add_scalar('train/loss_focal_epoch', train_comps['focal'], epoch)
                if args.iou_weight > 0:
                    writer.add_scalar('train/loss_iou_epoch', train_comps['iou'], epoch)
                for i, v in enumerate(train_comps['round']):
                    writer.add_scalar(f'train/loss_r{i + 1}_epoch', v, epoch)
                writer.add_scalar('train/mIoU_epoch', train_iou, epoch)

        # ── Train-eval: evaluate train set with same conditions as val ────
        train_eval_iou = train_iou  # fallback if train eval is disabled
        if train_eval_loader is not None:
            te_comps, train_eval_iou = validate(
                model, train_eval_loader, epoch, args, rank, split='train_eval')
            if rank == 0:
                print(f'  => TrainE loss={te_comps["total"]:.4f}'
                      f'  bce={te_comps["bce"]:.4f}'
                      f'  dice={te_comps["dice"]:.4f}'
                      + (f'  focal={te_comps["focal"]:.4f}' if args.focal_weight > 0 else '')
                      + (f'  iou_l={te_comps["iou"]:.4f}' if args.iou_weight > 0 else '')
                      + f'  mIoU={train_eval_iou:.4f}  (eval mode, 1024px)')
                if writer is not None:
                    writer.add_scalar('train_eval/loss_total_epoch', te_comps['total'], epoch)
                    writer.add_scalar('train_eval/loss_bce_epoch',   te_comps['bce'],   epoch)
                    writer.add_scalar('train_eval/loss_dice_epoch',  te_comps['dice'],  epoch)
                    if args.focal_weight > 0:
                        writer.add_scalar('train_eval/loss_focal_epoch', te_comps['focal'], epoch)
                    if args.iou_weight > 0:
                        writer.add_scalar('train_eval/loss_iou_epoch', te_comps['iou'], epoch)
                    writer.add_scalar('train_eval/mIoU_epoch', train_eval_iou, epoch)

        # ── Val eval ──────────────────────────────────────────────────────
        val_iou = train_eval_iou
        if val_loader is not None:
            val_comps, val_iou = validate(
                model, val_loader, epoch, args, rank, split='val')
            if rank == 0:
                print(f'  => Val    loss={val_comps["total"]:.4f}'
                      f'  bce={val_comps["bce"]:.4f}'
                      f'  dice={val_comps["dice"]:.4f}'
                      + (f'  focal={val_comps["focal"]:.4f}' if args.focal_weight > 0 else '')
                      + (f'  iou_l={val_comps["iou"]:.4f}' if args.iou_weight > 0 else '')
                      + f'  mIoU={val_iou:.4f}  (eval mode, 1024px)')
                if writer is not None:
                    writer.add_scalar('val/loss_total_epoch', val_comps['total'], epoch)
                    writer.add_scalar('val/loss_bce_epoch',   val_comps['bce'],   epoch)
                    writer.add_scalar('val/loss_dice_epoch',  val_comps['dice'],  epoch)
                    if args.focal_weight > 0:
                        writer.add_scalar('val/loss_focal_epoch', val_comps['focal'], epoch)
                    if args.iou_weight > 0:
                        writer.add_scalar('val/loss_iou_epoch', val_comps['iou'], epoch)
                    writer.add_scalar('val/mIoU_epoch', val_iou, epoch)

        if rank == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs - 1):
            path = _save_ckpt(model, optimizer, epoch, global_step, best_iou,
                              args.output, f'finetune_epoch_{epoch}.pth', args)
            print(f'  Saved: {path}')

            if val_iou > best_iou:
                best_iou  = val_iou
                best_path = _save_ckpt(model, optimizer, epoch, global_step, best_iou,
                                       args.output, 'finetune_best.pth', args)
                print(f'  ** New best mIoU={best_iou:.4f} → {best_path}')

    if rank == 0:
        if writer is not None:
            writer.close()
        print(f'\nDone.  Best mIoU: {best_iou:.4f}')

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
