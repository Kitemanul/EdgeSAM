"""Evaluate EdgeSAM mIoU with point prompts on a COCO-format dataset.

For each annotation, samples random point(s) from the GT mask, runs
inference, and computes per-mask IoU. Reports overall mIoU and per-image
statistics.

Supports multiple point sampling strategies:
  - random:  N random points from inside the GT mask (default)
  - center:  point inside the GT mask farthest from the mask boundary
  - bbox_center: center of the bounding box

Usage:
  python scripts/eval_iou.py \
      --checkpoint weights/edge_sam_3x.pth \
      --ann-file /path/to/annotations/val.json \
      --img-dir /path/to/images/val/ \
      --num-points 1 \
      --point-strategy random

  # Evaluate with 3 random points per mask
  python scripts/eval_iou.py \
      --checkpoint output/finetune/finetune_best.pth \
      --ann-file /path/to/annotations/val.json \
      --img-dir /path/to/images/val/ \
      --num-points 3

  # Compare before/after fine-tuning
  python scripts/eval_iou.py --checkpoint weights/edge_sam_3x.pth    --ann-file val.json --img-dir images/val/
  python scripts/eval_iou.py --checkpoint output/finetune/finetune_best.pth --ann-file val.json --img-dir images/val/
"""

import os
import sys
import json
import argparse
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_sam import sam_model_registry, SamPredictor


def parse_args():
    p = argparse.ArgumentParser('Evaluate EdgeSAM mIoU with point prompts')
    p.add_argument('--checkpoint', required=True, help='Model checkpoint path')
    p.add_argument('--ann-file', required=True, help='COCO annotation JSON path')
    p.add_argument('--img-dir', required=True, help='Image directory path')
    p.add_argument('--num-points', type=int, default=1,
                   help='Number of point prompts per mask')
    p.add_argument('--point-strategy', default='random',
                   choices=['random', 'center', 'bbox_center'],
                   help='Point sampling strategy')
    p.add_argument('--max-masks-per-image', type=int, default=-1,
                   help='Max masks to evaluate per image (-1 = all)')
    p.add_argument('--max-images', type=int, default=-1,
                   help='Max images to evaluate (-1 = all)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default=None,
                   help='Save per-image results to JSON file')
    return p.parse_args()


def decode_mask(segm, h, w):
    """Decode COCO segmentation to binary numpy mask."""
    if isinstance(segm, list):
        rles = mask_utils.frPyObjects(segm, h, w)
        rle = mask_utils.merge(rles)
    elif isinstance(segm, dict):
        if isinstance(segm.get('counts'), list):
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            rle = segm
    else:
        return None
    return mask_utils.decode(rle)


def sample_points(binary_mask, num_points, strategy, bbox=None, ann=None):
    """Sample point prompts from a GT mask.

    Args:
        binary_mask: [H, W] uint8 numpy array.
        num_points: Number of points to sample.
        strategy: 'random', 'center', or 'bbox_center'.
            'center' reads ann['center_point'] (pre-computed largest-inscribed-
            circle centre).  Run scripts/add_center_points.py to add this field.
        bbox: [x, y, w, h] COCO format bounding box (for bbox_center).
        ann: annotation dict (required when strategy='center').

    Returns:
        point_coords: [N, 2] array in (x, y) pixel format.
        point_labels: [N] array, all 1 (positive).
    """
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        point_coords = np.zeros((num_points, 2), dtype=np.float32)
    elif strategy == 'center':
        if ann is None or 'center_point' not in ann:
            raise ValueError(
                "strategy='center' requires ann['center_point']. "
                "Run scripts/add_center_points.py to pre-compute centres.")
        cx, cy = ann['center_point']
        center = np.array([[float(cx), float(cy)]], dtype=np.float32)
        if num_points == 1:
            point_coords = center
        else:
            extra = num_points - 1
            indices = [random.randint(0, len(xs) - 1) for _ in range(extra)]
            extra_coords = np.array(
                [[float(xs[i]), float(ys[i])] for i in indices], dtype=np.float32)
            point_coords = np.concatenate([center, extra_coords], axis=0)
    elif strategy == 'bbox_center' and bbox is not None:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        point_coords = np.array([[cx, cy]] * num_points, dtype=np.float32)
    else:  # random
        indices = [random.randint(0, len(xs) - 1) for _ in range(num_points)]
        point_coords = np.array(
            [[float(xs[i]), float(ys[i])] for i in indices], dtype=np.float32)

    point_labels = np.ones(num_points, dtype=np.int32)
    return point_coords, point_labels


def compute_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load annotations
    with open(args.ann_file, 'r') as f:
        coco = json.load(f)

    images = {img['id']: img for img in coco['images']}
    img_anns = defaultdict(list)
    for ann in coco['annotations']:
        if ann.get('iscrowd', 0):
            continue
        img_anns[ann['image_id']].append(ann)

    img_ids = [iid for iid in img_anns if iid in images]
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]

    # Load model
    print(f'Loading model: {args.checkpoint}')
    model = sam_model_registry['edge_sam'](checkpoint=args.checkpoint)
    model.cuda()
    model.eval()
    predictor = SamPredictor(model)

    print(f'Evaluating on {len(img_ids)} images, '
          f'strategy={args.point_strategy}, num_points={args.num_points}')
    print()

    # Evaluate
    all_ious = []
    per_image_results = []

    for img_id in tqdm(img_ids, desc='Evaluating'):
        img_info = images[img_id]
        anns = img_anns[img_id]
        h, w = img_info['height'], img_info['width']

        img_path = os.path.join(args.img_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            continue

        image = np.array(Image.open(img_path).convert('RGB'))
        predictor.set_image(image)

        if args.max_masks_per_image > 0 and len(anns) > args.max_masks_per_image:
            anns = random.sample(anns, args.max_masks_per_image)

        img_ious = []
        for ann in anns:
            gt_mask = decode_mask(ann['segmentation'], h, w)
            if gt_mask is None or gt_mask.sum() == 0:
                continue

            point_coords, point_labels = sample_points(
                gt_mask, args.num_points, args.point_strategy,
                bbox=ann.get('bbox'), ann=ann)

            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                num_multimask_outputs=1,
            )

            pred_mask = masks[0]  # [H, W] bool
            iou = compute_iou(pred_mask, gt_mask)
            img_ious.append(iou)
            all_ious.append(iou)

        if img_ious:
            per_image_results.append({
                'image_id': img_id,
                'file_name': img_info['file_name'],
                'num_masks': len(img_ious),
                'mean_iou': float(np.mean(img_ious)),
                'min_iou': float(np.min(img_ious)),
                'max_iou': float(np.max(img_ious)),
            })

    # Summary
    all_ious = np.array(all_ious)
    print()
    print('=' * 50)
    print(f'  Checkpoint : {args.checkpoint}')
    print(f'  Dataset    : {args.ann_file}')
    print(f'  Strategy   : {args.point_strategy}, {args.num_points} point(s)')
    print(f'  Images     : {len(per_image_results)}')
    print(f'  Masks      : {len(all_ious)}')
    print('=' * 50)
    print(f'  mIoU       : {all_ious.mean():.4f}')
    print(f'  Median IoU : {np.median(all_ious):.4f}')
    print(f'  Std        : {all_ious.std():.4f}')
    print(f'  Min IoU    : {all_ious.min():.4f}')
    print(f'  Max IoU    : {all_ious.max():.4f}')
    print()

    # IoU distribution
    thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    print('  IoU distribution:')
    for t in thresholds:
        pct = (all_ious >= t).mean() * 100
        bar = '#' * int(pct / 2)
        print(f'    >= {t:.2f} : {pct:5.1f}%  {bar}')
    print()

    # Worst images
    if per_image_results:
        per_image_results.sort(key=lambda x: x['mean_iou'])
        print('  Worst 10 images:')
        for r in per_image_results[:10]:
            print(f"    {r['file_name']:40s}  mIoU={r['mean_iou']:.4f}  "
                  f"({r['num_masks']} masks)")
        print()

    # Save results
    if args.output:
        results = {
            'config': {
                'checkpoint': args.checkpoint,
                'ann_file': args.ann_file,
                'point_strategy': args.point_strategy,
                'num_points': args.num_points,
                'seed': args.seed,
            },
            'summary': {
                'mIoU': float(all_ious.mean()),
                'median_iou': float(np.median(all_ious)),
                'std': float(all_ious.std()),
                'num_images': len(per_image_results),
                'num_masks': len(all_ious),
            },
            'per_image': per_image_results,
        }
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Results saved to: {args.output}')


if __name__ == '__main__':
    main()
