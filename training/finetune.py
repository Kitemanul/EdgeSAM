"""Fine-tune EdgeSAM mask decoder with point prompts on a COCO-format dataset.

Freezes image_encoder and prompt_encoder, only trains mask_decoder.
Supports single-GPU and multi-GPU (via torchrun).

Usage:
  # Single GPU
  python training/finetune.py \
      --checkpoint weights/edge_sam_3x.pth \
      --ann-file /path/to/annotations/train.json \
      --img-dir /path/to/images/train/ \
      --output output/finetune

  # Multi-GPU
  torchrun --nproc_per_node 4 training/finetune.py \
      --checkpoint weights/edge_sam_3x.pth \
      --ann-file /path/to/annotations/train.json \
      --img-dir /path/to/images/train/ \
      --output output/finetune

  # With validation
  python training/finetune.py \
      --checkpoint weights/edge_sam_3x.pth \
      --ann-file /path/to/annotations/train.json \
      --img-dir /path/to/images/train/ \
      --val-ann-file /path/to/annotations/val.json \
      --val-img-dir /path/to/images/val/ \
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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_sam import sam_model_registry
from training.data.finetune_dataset import COCOFinetuneDataset


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def bce_loss(pred, target, valid=None):
    """Binary cross-entropy loss. pred: logits, target: 0/1."""
    pred_sig = pred.sigmoid()
    loss = F.binary_cross_entropy(pred_sig, target, reduction='none')
    if valid is not None:
        loss = loss.flatten(1)
        valid = valid.flatten(1)
        loss = (loss * valid).sum(-1) / valid.sum(-1).clamp(min=1)
    return loss.mean()


def dice_loss(pred, target, valid=None):
    """Dice loss. pred: logits, target: 0/1."""
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
    """Focal loss. pred: logits, target: 0/1."""
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_iou(pred_logits, gt_mask):
    """Mean IoU between predicted (logits) and GT binary masks."""
    pred = (pred_logits > 0).float()
    gt = gt_mask.float()
    inter = (pred * gt).sum(dim=(-2, -1))
    union = pred.sum(dim=(-2, -1)) + gt.sum(dim=(-2, -1)) - inter
    return (inter / union.clamp(min=1)).mean()


# ---------------------------------------------------------------------------
# Forward pass for one image
# ---------------------------------------------------------------------------

def forward_one_image(model, image, point_coords, point_labels, img_size_before_pad):
    """Run encoder → prompt_encoder → mask_decoder for a single image.

    Args:
        model: Sam model (unwrapped, on GPU).
        image: [3, 1024, 1024] normalized & padded.
        point_coords: [N, P, 2] transformed point coordinates.
        point_labels: [N, P] point labels (1=positive).
        img_size_before_pad: (H, W) before padding.

    Returns:
        low_res_masks: [N, 1, 256, 256] predicted mask logits.
        iou_pred: [N, 1] predicted IoU scores.
    """
    with torch.no_grad():
        image_embeddings = model.image_encoder(image.unsqueeze(0))
        sparse_emb, dense_emb = model.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        image_pe = model.prompt_encoder.get_dense_pe()

    low_res_masks, iou_pred = model.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        num_multimask_outputs=1,
    )
    return low_res_masks, iou_pred


def compute_loss(low_res_masks, gt_masks, img_size_before_pad, device, args):
    """Compute BCE + Dice + Focal loss at 256×256 resolution."""
    gt_low = F.interpolate(
        gt_masks.unsqueeze(1), size=(256, 256), mode='nearest')

    valid = torch.zeros(1, 1, 1024, 1024, device=device)
    img_h, img_w = img_size_before_pad
    valid[:, :, :img_h, :img_w] = 1
    valid_low = F.interpolate(valid, size=(256, 256), mode='bilinear', align_corners=False)
    valid_low = (valid_low > 0.5).expand_as(low_res_masks).float()

    loss = torch.tensor(0.0, device=device)
    if args.bce_weight > 0:
        loss = loss + bce_loss(low_res_masks, gt_low, valid_low) * args.bce_weight
    if args.dice_weight > 0:
        loss = loss + dice_loss(low_res_masks, gt_low, valid_low) * args.dice_weight
    if args.focal_weight > 0:
        loss = loss + focal_loss(low_res_masks, gt_low, valid_low) * args.focal_weight

    iou = compute_iou(low_res_masks, gt_low)
    return loss, iou


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, epoch, args, distributed, world_size, rank,
                    writer, global_step):
    model.mask_decoder.train()
    model.image_encoder.eval()
    model.prompt_encoder.eval()

    loss_sum, iou_sum, count = 0.0, 0.0, 0
    start = time.time()

    for step, batch in enumerate(dataloader):
        batch_loss = torch.tensor(0.0, device='cuda')
        batch_iou = 0.0
        n_valid = 0

        for sample in batch:
            if sample['num_prompts'] == 0:
                continue

            image = sample['image'].cuda(non_blocking=True)
            gt_masks = sample['gt_masks'].cuda(non_blocking=True)
            pts = sample['point_coords'].cuda(non_blocking=True)
            lbls = sample['point_labels'].cuda(non_blocking=True)
            sz = sample['img_size_before_pad']

            low_res_masks, _ = forward_one_image(model, image, pts, lbls, sz)
            loss, iou = compute_loss(low_res_masks, gt_masks, sz, image.device, args)

            batch_loss = batch_loss + loss
            batch_iou += iou.item()
            n_valid += 1

        if n_valid == 0:
            continue

        batch_loss = batch_loss / n_valid

        optimizer.zero_grad()
        batch_loss.backward()

        # Manual gradient sync for multi-GPU
        if distributed:
            for p in model.mask_decoder.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad /= world_size

        torch.nn.utils.clip_grad_norm_(model.mask_decoder.parameters(), 0.1)
        optimizer.step()

        loss_val = batch_loss.item()
        iou_val = batch_iou / n_valid
        loss_sum += loss_val
        iou_sum += iou_val
        count += 1
        global_step += 1

        if rank == 0:
            writer.add_scalar('train/loss_step', loss_val, global_step)
            writer.add_scalar('train/mIoU_step', iou_val, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        if rank == 0 and args.save_step_freq > 0 and global_step % args.save_step_freq == 0:
            ckpt_path = os.path.join(args.output, f'finetune_step_{global_step}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  [step {global_step}] Saved: {ckpt_path}')

        if step % args.print_freq == 0 and rank == 0:
            elapsed = time.time() - start
            eta = elapsed / max(step + 1, 1) * (len(dataloader) - step - 1)
            print(f'  [{epoch}][{step}/{len(dataloader)}]  '
                  f'loss {loss_val:.4f} (avg {loss_sum / count:.4f})  '
                  f'mIoU {iou_val:.4f} (avg {iou_sum / count:.4f})  '
                  f'eta {datetime.timedelta(seconds=int(eta))}')

    return loss_sum / max(count, 1), iou_sum / max(count, 1), global_step


@torch.no_grad()
def validate(model, dataloader, args, rank):
    model.eval()

    loss_sum, iou_sum, count = 0.0, 0.0, 0
    for batch in dataloader:
        for sample in batch:
            if sample['num_prompts'] == 0:
                continue

            image = sample['image'].cuda(non_blocking=True)
            gt_masks = sample['gt_masks'].cuda(non_blocking=True)
            pts = sample['point_coords'].cuda(non_blocking=True)
            lbls = sample['point_labels'].cuda(non_blocking=True)
            sz = sample['img_size_before_pad']

            low_res_masks, _ = forward_one_image(model, image, pts, lbls, sz)
            loss, iou = compute_loss(low_res_masks, gt_masks, sz, image.device, args)

            loss_sum += loss.item()
            iou_sum += iou.item()
            count += 1

    avg_loss = loss_sum / max(count, 1)
    avg_iou = iou_sum / max(count, 1)
    return avg_loss, avg_iou


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return batch


def parse_args():
    p = argparse.ArgumentParser('EdgeSAM fine-tuning with point prompts')
    # Data
    p.add_argument('--ann-file', required=True, help='COCO annotation JSON (train)')
    p.add_argument('--img-dir', required=True, help='Image directory (train)')
    p.add_argument('--val-ann-file', default=None, help='COCO annotation JSON (val)')
    p.add_argument('--val-img-dir', default=None, help='Image directory (val)')
    # Model
    p.add_argument('--checkpoint', required=True, help='EdgeSAM pretrained weights')
    # Training
    p.add_argument('--output', default='output/finetune')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=4, help='Images per GPU')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--max-prompts', type=int, default=16, help='Max masks per image')
    p.add_argument('--num-points', type=int, default=1, help='Positive points per mask')
    p.add_argument('--bce-weight', type=float, default=5.0)
    p.add_argument('--dice-weight', type=float, default=5.0)
    p.add_argument('--focal-weight', type=float, default=0.0)
    p.add_argument('--save-freq', type=int, default=1, help='Save checkpoint every N epochs')
    p.add_argument('--save-step-freq', type=int, default=0,
                   help='Save checkpoint every N steps (0 = disabled)')
    p.add_argument('--print-freq', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # Distributed setup
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if distributed:
        rank = int(os.environ['RANK'])
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
        print('EdgeSAM Fine-tuning (mask decoder only, point prompts)')
        print('=' * 60)
        print(f'  Checkpoint   : {args.checkpoint}')
        print(f'  Train ann    : {args.ann_file}')
        print(f'  Train images : {args.img_dir}')
        print(f'  Output       : {args.output}')
        print(f'  GPUs         : {world_size}')
        print(f'  Batch / GPU  : {args.batch_size}')
        print(f'  Epochs       : {args.epochs}')
        print(f'  LR           : {args.lr}')
        print(f'  Points / mask: {args.num_points}')
        print(f'  Loss weights : BCE={args.bce_weight} Dice={args.dice_weight} Focal={args.focal_weight}')
        print(f'  Save step freq: {args.save_step_freq if args.save_step_freq > 0 else "disabled"}')
        print()

    # ---- Model ----
    model = sam_model_registry['edge_sam'](checkpoint=args.checkpoint)
    model.cuda()

    # Freeze encoder & prompt encoder
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False

    n_train = sum(p.numel() for p in model.mask_decoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f'  Trainable params : {n_train:,} / {n_total:,} '
              f'({100 * n_train / n_total:.1f}%)')
        print()

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.mask_decoder.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ---- Datasets ----
    train_ds = COCOFinetuneDataset(
        ann_file=args.ann_file, img_dir=args.img_dir,
        max_prompts_per_image=args.max_prompts,
        num_points_per_mask=args.num_points)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
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
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            shuffle=False, sampler=val_sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn)
        if rank == 0:
            print(f'  Train images: {len(train_ds)},  Val images: {len(val_ds)}')
    else:
        if rank == 0:
            print(f'  Train images: {len(train_ds)}')
    if rank == 0:
        print()

    # ---- TensorBoard ----
    writer = None
    if rank == 0:
        tb_dir = os.path.join(args.output, 'tensorboard')
        writer = SummaryWriter(log_dir=tb_dir)
        print(f'  TensorBoard  : tensorboard --logdir {tb_dir}')
        print()

    # ---- Training loop ----
    best_iou = 0.0
    global_step = 0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f'Epoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]["lr"]:.6f}')

        train_loss, train_iou, global_step = train_one_epoch(
            model, train_loader, optimizer, epoch, args,
            distributed, world_size, rank, writer, global_step)
        scheduler.step()

        if rank == 0:
            print(f'  => Train  loss={train_loss:.4f}  mIoU={train_iou:.4f}')
            writer.add_scalar('train/loss_epoch', train_loss, epoch)
            writer.add_scalar('train/mIoU_epoch', train_iou, epoch)

        val_iou = train_iou
        if val_loader is not None:
            val_loss, val_iou = validate(model, val_loader, args, rank)
            if rank == 0:
                print(f'  => Val    loss={val_loss:.4f}  mIoU={val_iou:.4f}')
                writer.add_scalar('val/loss_epoch', val_loss, epoch)
                writer.add_scalar('val/mIoU_epoch', val_iou, epoch)

        if rank == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs - 1):
            ckpt_path = os.path.join(args.output, f'finetune_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f'  Saved: {ckpt_path}')

            if val_iou > best_iou:
                best_iou = val_iou
                best_path = os.path.join(args.output, 'finetune_best.pth')
                torch.save(model.state_dict(), best_path)
                print(f'  ** New best mIoU={best_iou:.4f} -> {best_path}')

    if rank == 0:
        writer.close()
        print(f'\nDone. Best mIoU: {best_iou:.4f}')

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
