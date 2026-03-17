"""
Model comparison script for EdgeSAM.

Runs multiple models on the same dataset with per-model inference strategies,
computes per-image IoU, generates a comparison report, and saves prediction
masks for images where one model improves most over another.

Each model can have its own mask selection strategy via a JSON config string.

Usage:
    # Basic: compare two checkpoints with default settings (box prompt, 1 mask)
    python scripts/compare_models.py \
        --models edge_sam:weights/edge_sam.pth edge_sam:weights/edge_sam_3x.pth \
        --dataset coco --num-samples 200 --output-dir output/compare

    # Per-model strategy override via --strategies (JSON per model, same order as --models)
    python scripts/compare_models.py \
        --models edge_sam:weights/edge_sam.pth:EdgeSAM_1x \
                 edge_sam:weights/edge_sam_3x.pth:EdgeSAM_3x \
        --strategies \
            '{"prompt_type":"point","point_from":"mask-center","num_multimask":4,"multimask_select":"oracle","refine_iter":2}' \
            '{"prompt_type":"box","num_multimask":1,"refine_iter":1}' \
        --dataset coco --num-samples 200 --output-dir output/compare

    # Shorthand: use global defaults, override only what differs
    python scripts/compare_models.py \
        --models edge_sam:weights/edge_sam.pth vit_b:weights/sam_vit_b.pth \
        --prompt-type point --point-from mask-rand \
        --num-multimask 4 --multimask-select score --refine-iter 2 \
        --dataset sa --num-samples 100 --output-dir output/compare --top-k 10

Strategy parameters (global flags or per-model JSON):
    prompt_type       : "box" | "point" | "both"  (default: "box")
    point_from        : "point" | "random" | "mask-rand" | "mask-center" | "box-center"
                        (default: "point")
                        - "point"       : use dataset-provided point annotations
                        - "random"      : uniformly sample N random points inside GT mask
                        - "mask-rand"   : 1 random point inside GT mask (legacy)
                        - "mask-center" : mask centroid via distance transform
                        - "box-center"  : center of bounding box
    num_points        : int >= 1  (default: 1, number of point prompts per mask,
                        mainly for point_from="random")
    num_multimask     : 1 | 3 | 4  (default: 1)
    multimask_select  : "score" | "area" | "oracle"  (default: "score")
    refine_iter       : int >= 1  (default: 1, iterative mask refinement rounds)
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edge_sam import sam_model_registry
from edge_sam.utils.common import cal_iou, sample_point_in_mask, get_centroid_from_mask

# Lazy imports for datasets
_SA1BDataset = None
_COCODataset = None


def _get_sa1b_dataset():
    global _SA1BDataset
    if _SA1BDataset is None:
        from training.data import SA1BDataset
        _SA1BDataset = SA1BDataset
    return _SA1BDataset


def _get_coco_dataset():
    global _COCODataset
    if _COCODataset is None:
        from training.data import COCODataset
        _COCODataset = COCODataset
    return _COCODataset


# ---------------------------------------------------------------------------
# Strategy config: controls how each model does inference
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY = {
    "prompt_type": "box",         # "box" | "point" | "both"
    "point_from": "point",        # "point" | "random" | "mask-rand" | "mask-center" | "box-center"
    "num_points": 1,              # number of point prompts per mask
    "num_multimask": 1,           # 1, 3, or 4
    "multimask_select": "score",  # "score" | "area" | "oracle"
    "refine_iter": 1,             # >= 1
}


def build_strategy(global_args, per_model_json=None):
    """Merge global CLI defaults with optional per-model JSON overrides."""
    strategy = dict(DEFAULT_STRATEGY)
    # Apply global CLI flags
    strategy["prompt_type"] = global_args.prompt_type
    strategy["point_from"] = global_args.point_from
    strategy["num_points"] = global_args.num_points
    strategy["num_multimask"] = global_args.num_multimask
    strategy["multimask_select"] = global_args.multimask_select
    strategy["refine_iter"] = global_args.refine_iter
    # Apply per-model overrides
    if per_model_json is not None:
        overrides = json.loads(per_model_json) if isinstance(per_model_json, str) else per_model_json
        for k, v in overrides.items():
            if k not in DEFAULT_STRATEGY:
                raise ValueError(f"Unknown strategy key '{k}'. Valid: {list(DEFAULT_STRATEGY.keys())}")
            strategy[k] = v
    return strategy


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def parse_model_spec(spec):
    """Parse 'model_type:checkpoint_path' or 'model_type:checkpoint_path:label'."""
    parts = spec.split(":")
    if len(parts) == 2:
        model_type, ckpt = parts
        label = f"{model_type}_{os.path.basename(ckpt).replace('.pth', '')}"
    elif len(parts) >= 3:
        model_type = parts[0]
        ckpt = parts[1]
        label = ":".join(parts[2:])
    else:
        raise ValueError(f"Invalid model spec '{spec}'. Expected 'type:checkpoint' or 'type:checkpoint:label'.")
    return model_type, ckpt, label


def load_model(model_type, checkpoint, device):
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(sam_model_registry.keys())}")
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(args):
    if args.dataset == "sa":
        SA1BDataset = _get_sa1b_dataset()
        dataset = SA1BDataset(
            data_root=args.data_root, split="val",
            num_samples=args.num_samples,
            filter_by_area=None, sort_by_area=False,
            load_gt_mask=True,
            max_allowed_prompts=args.max_prompts,
            fix_seed=True,
        )
    elif args.dataset in ["coco", "cocofied_lvis", "lvis"]:
        COCODataset = _get_coco_dataset()
        dataset = COCODataset(
            data_root=args.data_root, split="val",
            num_samples=args.num_samples,
            filter_by_area=None, sort_by_area=False,
            load_gt_mask=True,
            max_allowed_prompts=args.max_prompts,
            fix_seed=True, annotation=args.dataset,
        )
    else:
        raise ValueError(f"Unsupported dataset '{args.dataset}'")

    from mmengine.dataset import pseudo_collate
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=pseudo_collate, num_workers=args.num_workers,
    )
    return dataset, dataloader


# ---------------------------------------------------------------------------
# Prompt preparation
# ---------------------------------------------------------------------------

def _sample_random_points_in_mask(gt_mask, num_points, device):
    """Sample num_points random points uniformly from within each GT mask.

    Args:
        gt_mask: (N, 1, H, W) binary-ish tensor
        num_points: int, number of points to sample per mask
        device: torch device

    Returns:
        points: (N, num_points, 2) in (x, y) format
        labels: (N, num_points) all ones (foreground)
    """
    N = gt_mask.shape[0]
    all_points = []
    for i in range(N):
        mask_i = gt_mask[i, 0]  # (H, W)
        candidate_indices = mask_i.nonzero()  # (K, 2) in (row, col) = (y, x)
        if len(candidate_indices) >= num_points:
            perm = torch.randperm(len(candidate_indices), device=device)[:num_points]
            selected = candidate_indices[perm]
        elif len(candidate_indices) > 0:
            # Fewer mask pixels than requested: sample with replacement
            indices = torch.randint(0, len(candidate_indices), (num_points,), device=device)
            selected = candidate_indices[indices]
        else:
            selected = torch.zeros(num_points, 2, device=device, dtype=torch.long)
        # Convert (y, x) -> (x, y)
        pts = selected.flip(1).float()
        all_points.append(pts)
    points = torch.stack(all_points, dim=0)  # (N, num_points, 2)
    labels = torch.ones(N, num_points, device=device)
    return points, labels


def prepare_prompts(annos, boxes_cat, gt_mask, strategy, device, img_size_pad):
    """Prepare point and box prompts according to strategy.

    Returns:
        points: (coords, labels) or None
        boxes: tensor or None
    """
    prompt_type = strategy["prompt_type"]
    point_from = strategy["point_from"]
    num_points = strategy["num_points"]

    points = None
    boxes = boxes_cat

    # Build point prompts based on point_from
    if prompt_type in ("point", "both"):
        if point_from == "random":
            # Uniformly random N points inside GT mask, controlled by seed
            pts, labels = _sample_random_points_in_mask(gt_mask, num_points, device)
            points = (pts, labels)
        elif point_from == "point" and "prompt_point" in annos:
            pts = torch.cat(annos["prompt_point"], dim=0).to(device)
            # pts: (N, K, 2) where K is original point count
            if num_points > 1 and pts.shape[1] < num_points:
                # Supplement with random mask points
                extra_pts, extra_labels = _sample_random_points_in_mask(
                    gt_mask, num_points - pts.shape[1], device)
                pts = torch.cat([pts, extra_pts], dim=1)
            elif num_points < pts.shape[1]:
                pts = pts[:, :num_points]
            labels = torch.ones(pts.shape[:2], device=device)
            points = (pts, labels)
        elif point_from == "mask-rand":
            # Legacy: 1 random point per mask (ignores num_points for backward compat)
            point_list = []
            for g in gt_mask.squeeze(1):
                candidate_indices = g.nonzero()
                if len(candidate_indices) > 0:
                    selected_index = random.randint(0, len(candidate_indices) - 1)
                    p = candidate_indices[selected_index].flip(0)
                else:
                    p = torch.zeros(2, device=device)
                point_list.append(p)
            pts = torch.stack(point_list, dim=0)[:, None]
            # If num_points > 1, add more random points
            if num_points > 1:
                extra_pts, _ = _sample_random_points_in_mask(gt_mask, num_points - 1, device)
                pts = torch.cat([pts, extra_pts], dim=1)
            labels = torch.ones(pts.shape[:2], device=device)
            points = (pts, labels)
        elif point_from == "mask-center":
            pts = get_centroid_from_mask(gt_mask > 0.5)
            if num_points > 1:
                extra_pts, _ = _sample_random_points_in_mask(gt_mask, num_points - 1, device)
                pts = torch.cat([pts, extra_pts], dim=1)
            labels = torch.ones(pts.shape[:2], device=device)
            points = (pts, labels)
        elif point_from == "box-center":
            cx = (boxes_cat[:, 0] + boxes_cat[:, 2]) / 2
            cy = (boxes_cat[:, 1] + boxes_cat[:, 3]) / 2
            pts = torch.stack([cx, cy], dim=1)[:, None]
            if num_points > 1:
                extra_pts, _ = _sample_random_points_in_mask(gt_mask, num_points - 1, device)
                pts = torch.cat([pts, extra_pts], dim=1)
            labels = torch.ones(pts.shape[:2], device=device)
            points = (pts, labels)
        else:
            # Fallback: random points
            pts, labels = _sample_random_points_in_mask(gt_mask, num_points, device)
            points = (pts, labels)

    # Suppress prompts based on prompt_type
    if prompt_type == "point":
        boxes = None
    elif prompt_type == "box":
        points = None

    return points, boxes


# ---------------------------------------------------------------------------
# Mask selection after multi-mask output
# ---------------------------------------------------------------------------

def select_mask(mask_pred, iou_pred, gt_mask_down, strategy, mask_threshold):
    """Select one mask per prompt from multi-mask output.

    Args:
        mask_pred: (N, C, H, W)  C = num_multimask
        iou_pred:  (N, C)
        gt_mask_down: (N, 1, H, W) GT at mask_pred resolution (only for oracle)
        strategy: dict
        mask_threshold: float

    Returns:
        mask_pred: (N, 1, H, W) selected mask
        iou_pred:  (N, 1)
    """
    num_masks = mask_pred.shape[1]
    if num_masks <= 1:
        return mask_pred, iou_pred

    method = strategy["multimask_select"]
    n, c, h, w = mask_pred.shape

    if method == "score":
        idx = iou_pred.argmax(dim=1)[:, None, None, None].expand(n, 1, h, w)
        mask_pred = torch.gather(mask_pred, 1, idx)
        iou_idx = iou_pred.argmax(dim=1)[:, None]
        iou_pred = torch.gather(iou_pred, 1, iou_idx)
    elif method == "area":
        area = (mask_pred > mask_threshold).sum(dim=(2, 3))
        idx = area.argmax(dim=1)[:, None, None, None].expand(n, 1, h, w)
        mask_pred = torch.gather(mask_pred, 1, idx)
        iou_idx = area.argmax(dim=1)[:, None]
        iou_pred = torch.gather(iou_pred, 1, iou_idx)
    elif method == "oracle":
        # Use GT to pick best mask
        gt_expanded = gt_mask_down.expand_as(mask_pred)
        iou_per_mask = cal_iou(mask_pred, gt_expanded)  # (N, C)
        idx = iou_per_mask.argmax(dim=1)[:, None, None, None].expand(n, 1, h, w)
        mask_pred = torch.gather(mask_pred, 1, idx)
        iou_idx = iou_per_mask.argmax(dim=1)[:, None]
        iou_pred = torch.gather(iou_pred, 1, iou_idx)

    return mask_pred, iou_pred


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, dataloader, strategy, device):
    """Run model on the dataloader with the given strategy.

    Returns:
        ious: list of float, per-image mean IoU (from the last refine iteration)
        all_masks: dict[img_idx] -> { pred, gt, img, file_name, iou_per_prompt }
        ious_per_iter: list of list of float, per-image mIoU for each refine iteration
    """
    img_size = model.image_encoder.img_size
    mask_threshold = model.mask_threshold
    num_multimask = strategy["num_multimask"]
    refine_iter = strategy["refine_iter"]

    ious_per_iter = [[] for _ in range(refine_iter)]
    all_masks = {}

    for img_idx, (imgs, annos) in enumerate(dataloader):
        imgs = torch.stack(imgs, dim=0).to(device)
        img_size_before_pad = annos["img_size_before_pad"]
        img_size_pad = (img_size, img_size)

        image_embeddings = model.image_encoder(model.preprocess(imgs))
        dense_pe = model.prompt_encoder.get_dense_pe()

        # Prompts
        boxes_raw = annos["prompt_box"]
        num_prompts = [b.size(0) for b in boxes_raw]
        boxes_cat = torch.cat(boxes_raw, dim=0).to(device)

        gt_mask = torch.cat(annos["gt_mask"], dim=0).float().to(device)[:, None]

        points, boxes = prepare_prompts(annos, boxes_cat, gt_mask, strategy, device, img_size_pad)

        # Padding valid mask
        img_bs = imgs.size(0)
        valid = torch.zeros(img_bs, 1, *img_size_pad, device=device)
        valid_list = []
        for i in range(img_bs):
            h, w = img_size_before_pad[i][1:]
            valid[i, :, :h, :w] = 1
            valid_list.append(valid[i:i + 1].expand(num_prompts[i], *valid.shape[1:]))
        valid = torch.cat(valid_list, dim=0)

        # Iterative decode
        prev_point = points
        mask_pred = None
        for iter_i in range(refine_iter):
            if iter_i > 0 and mask_pred is not None:
                # Sample refinement points from prediction errors
                valid_down = F.interpolate(valid, mask_pred.shape[2:], mode="bilinear", align_corners=False)
                gt_mask_down = F.interpolate(gt_mask, mask_pred.shape[2:], mode="bilinear", align_corners=False)

                mask_pred_valid = (mask_pred > mask_threshold) * valid_down
                gt_mask_valid = (gt_mask_down > mask_threshold) * valid_down

                point, label = sample_point_in_mask(mask_pred_valid, gt_mask_valid, num_samples=1)
                point[..., 0] = point[..., 0] / mask_pred.shape[3] * img_size_pad[1]
                point[..., 1] = point[..., 1] / mask_pred.shape[2] * img_size_pad[0]

                if prev_point is not None:
                    point = torch.cat([prev_point[0], point], dim=1)
                    label = torch.cat([prev_point[1], label], dim=1)
                points = (point, label)
                prev_point = points

            sparse_emb, dense_emb = model.prompt_encoder(
                points=points, boxes=boxes, masks=None,
            )
            mask_pred, iou_pred = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=dense_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                num_multimask_outputs=num_multimask,
                num_prompts=num_prompts,
            )

            # Multi-mask selection
            if num_multimask > 1:
                gt_mask_for_select = F.interpolate(gt_mask, mask_pred.shape[2:],
                                                   mode="bilinear", align_corners=False)
                mask_pred, iou_pred = select_mask(
                    mask_pred, iou_pred, gt_mask_for_select, strategy, mask_threshold)

            # Compute IoU at this iteration
            mask_pred_up = F.interpolate(mask_pred, img_size_pad, mode="bilinear", align_corners=False)
            iter_ious = []
            for i in range(img_bs):
                if i == 0:
                    cur = slice(0, num_prompts[i])
                else:
                    cur = slice(sum(num_prompts[:i]), sum(num_prompts[:i + 1]))

                cur_pred = mask_pred_up[cur]
                cur_gt = gt_mask[cur]

                h, w = img_size_before_pad[i][1:]
                ori_h, ori_w = annos["info"]["height"][i], annos["info"]["width"][i]

                cur_pred = cur_pred[..., :h, :w]
                cur_gt = cur_gt[..., :h, :w]
                cur_pred = F.interpolate(cur_pred, (ori_h, ori_w), mode="bilinear", align_corners=False)
                cur_gt = F.interpolate(cur_gt, (ori_h, ori_w), mode="bilinear", align_corners=False)

                iou = cal_iou(cur_pred > mask_threshold, cur_gt > mask_threshold)
                if iou.dim() > 1:
                    iou, _ = iou.max(dim=1)
                iter_ious.append(iou.cpu())

            iter_iou_tensor = torch.cat(iter_ious)
            mean_iou = iter_iou_tensor[~iter_iou_tensor.isnan()].mean().item()
            ious_per_iter[iter_i].append(mean_iou)

        # Save masks from last iteration for visualization
        prompt_offset = 0
        pred_masks_list = []
        gt_masks_list = []
        final_ious = []
        for i in range(img_bs):
            n = num_prompts[i]
            cur_pred = mask_pred_up[prompt_offset:prompt_offset + n]
            cur_gt = gt_mask[prompt_offset:prompt_offset + n]

            h, w = img_size_before_pad[i][1:]
            ori_h, ori_w = annos["info"]["height"][i], annos["info"]["width"][i]

            cur_pred = cur_pred[..., :h, :w]
            cur_gt = cur_gt[..., :h, :w]
            cur_pred = F.interpolate(cur_pred, (ori_h, ori_w), mode="bilinear", align_corners=False)
            cur_gt = F.interpolate(cur_gt, (ori_h, ori_w), mode="bilinear", align_corners=False)

            cur_pred_bin = cur_pred > mask_threshold
            cur_gt_bin = cur_gt > mask_threshold

            iou = cal_iou(cur_pred_bin, cur_gt_bin)
            if iou.dim() > 1:
                iou, _ = iou.max(dim=1)
            final_ious.append(iou.cpu())

            pred_masks_list.append(cur_pred_bin[:, 0].cpu().numpy().astype(np.uint8) * 255)
            gt_masks_list.append(cur_gt_bin[:, 0].cpu().numpy().astype(np.uint8) * 255)
            prompt_offset += n

        final_iou_tensor = torch.cat(final_ious)

        # Recover original image for visualization
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        raw_img = (imgs * pixel_std + pixel_mean).clamp(0, 255)
        raw_img = raw_img[0]
        h, w = img_size_before_pad[0][1:]
        ori_h, ori_w = annos["info"]["height"][0], annos["info"]["width"][0]
        raw_img = raw_img[:, :h, :w]
        raw_img = F.interpolate(raw_img[None], (ori_h, ori_w), mode="bilinear", align_corners=False)[0]
        raw_img = raw_img.permute(1, 2, 0).byte().cpu().numpy()

        file_name = annos["info"]["file_name"][0] if "file_name" in annos["info"] else f"img_{img_idx}"

        all_masks[img_idx] = {
            "pred": np.concatenate(pred_masks_list, axis=0),
            "gt": np.concatenate(gt_masks_list, axis=0),
            "img": raw_img,
            "file_name": file_name,
            "iou_per_prompt": final_iou_tensor.numpy(),
        }

    # Use last iteration as the final IoU
    ious = ious_per_iter[-1]
    return ious, all_masks, ious_per_iter


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_comparison_masks(output_dir, top_entries, model_masks, model_labels):
    """Save prediction masks for top improved images."""
    base_label, cmp_label = model_labels
    vis_dir = os.path.join(output_dir, "top_improvements")
    os.makedirs(vis_dir, exist_ok=True)

    for rank, (img_idx, delta, iou_base, iou_cmp) in enumerate(top_entries):
        base_data = model_masks[base_label][img_idx]
        cmp_data = model_masks[cmp_label][img_idx]
        img = base_data["img"]
        file_name = base_data["file_name"]
        safe_name = os.path.splitext(os.path.basename(str(file_name)))[0]

        sub_dir = os.path.join(vis_dir, f"rank{rank:03d}_{safe_name}_delta{delta:.4f}")
        os.makedirs(sub_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sub_dir, "image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        n_prompts = base_data["pred"].shape[0]
        for p in range(min(n_prompts, 8)):
            gt_mask = base_data["gt"][p]
            base_mask = base_data["pred"][p]
            cmp_mask = cmp_data["pred"][p]

            cv2.imwrite(os.path.join(sub_dir, f"p{p}_gt.png"), gt_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{base_label}.png"), base_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{cmp_label}.png"), cmp_mask)

            overlay = _make_overlay(img, gt_mask, cmp_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{cmp_label}_overlay.png"), overlay)

            overlay_base = _make_overlay(img, gt_mask, base_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{base_label}_overlay.png"), overlay_base)

    print(f"Saved {len(top_entries)} comparison visualizations to {vis_dir}")


def _make_overlay(img, gt_mask, pred_mask, alpha=0.5):
    h, w = gt_mask.shape
    img_resized = cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img.copy()
    overlay = img_resized.copy()

    gt_bin = gt_mask > 127
    pred_bin = pred_mask > 127
    tp = gt_bin & pred_bin
    fn = gt_bin & ~pred_bin
    fp = ~gt_bin & pred_bin

    overlay[tp] = [255, 255, 255]
    overlay[fp] = [0, 0, 255]
    overlay[fn] = [0, 255, 0]

    result = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 1 - alpha, overlay, alpha, 0)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(output_dir, model_labels, all_ious, strategies, ious_per_iter_all):
    """Generate a JSON report with per-image comparison stats and strategy info."""
    report = {
        "models": model_labels,
        "num_images": len(all_ious[model_labels[0]]),
        "strategies": {label: strategies[label] for label in model_labels},
    }

    model_stats = {}
    for label in model_labels:
        ious = all_ious[label]
        arr = np.array(ious)
        valid = arr[~np.isnan(arr)]
        stat = {
            "mean_iou": float(valid.mean()) if len(valid) > 0 else 0.0,
            "median_iou": float(np.median(valid)) if len(valid) > 0 else 0.0,
            "std_iou": float(valid.std()) if len(valid) > 0 else 0.0,
            "num_images": len(ious),
        }
        # Per-iteration IoU if refine_iter > 1
        per_iter = ious_per_iter_all[label]
        if len(per_iter) > 1:
            stat["iou_per_iter"] = [
                float(np.nanmean(per_iter[i])) for i in range(len(per_iter))
            ]
        model_stats[label] = stat
    report["model_stats"] = model_stats

    if len(model_labels) >= 2:
        base = model_labels[0]
        comparisons = {}
        for cmp in model_labels[1:]:
            base_ious = np.array(all_ious[base])
            cmp_ious = np.array(all_ious[cmp])
            delta = cmp_ious - base_ious
            valid = ~(np.isnan(base_ious) | np.isnan(cmp_ious))
            delta_valid = delta[valid]
            comparisons[f"{cmp}_vs_{base}"] = {
                "mean_delta_iou": float(delta_valid.mean()) if len(delta_valid) > 0 else 0.0,
                "median_delta_iou": float(np.median(delta_valid)) if len(delta_valid) > 0 else 0.0,
                "num_improved": int((delta_valid > 0).sum()),
                "num_degraded": int((delta_valid < 0).sum()),
                "num_tied": int((delta_valid == 0).sum()),
                "max_improvement": float(delta_valid.max()) if len(delta_valid) > 0 else 0.0,
                "max_degradation": float(delta_valid.min()) if len(delta_valid) > 0 else 0.0,
            }
        report["comparisons"] = comparisons

    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")
    return report


def print_summary(report):
    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)
    print(f"Images evaluated: {report['num_images']}")
    print()

    # Strategy summary
    for label in report["models"]:
        s = report["strategies"][label]
        parts = [f"prompt={s['prompt_type']}"]
        if s["prompt_type"] in ("point", "both"):
            parts.append(f"point_from={s['point_from']}")
            parts.append(f"num_points={s['num_points']}")
        parts.append(f"multimask={s['num_multimask']}")
        if s["num_multimask"] > 1:
            parts.append(f"select={s['multimask_select']}")
        if s["refine_iter"] > 1:
            parts.append(f"refine={s['refine_iter']}")
        print(f"  [{label}] {', '.join(parts)}")
    print()

    print(f"{'Model':<30} {'mIoU':>8} {'Median':>8} {'Std':>8}")
    print("-" * 56)
    for label, stats in report["model_stats"].items():
        line = f"{label:<30} {stats['mean_iou']*100:>7.2f}% {stats['median_iou']*100:>7.2f}% {stats['std_iou']*100:>7.2f}%"
        if "iou_per_iter" in stats:
            iter_str = " | ".join(f"iter{i+1}={v*100:.2f}%" for i, v in enumerate(stats["iou_per_iter"]))
            line += f"  [{iter_str}]"
        print(line)

    if "comparisons" in report:
        print()
        for key, cmp in report["comparisons"].items():
            print(f"--- {key} ---")
            print(f"  Mean delta IoU:    {cmp['mean_delta_iou']*100:+.2f}%")
            print(f"  Median delta IoU:  {cmp['median_delta_iou']*100:+.2f}%")
            print(f"  Improved / Degraded / Tied: {cmp['num_improved']} / {cmp['num_degraded']} / {cmp['num_tied']}")
            print(f"  Max improvement:   {cmp['max_improvement']*100:+.2f}%")
            print(f"  Max degradation:   {cmp['max_degradation']*100:+.2f}%")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple SAM models on the same dataset with per-model strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model specs: 'model_type:checkpoint[:label]'",
    )
    parser.add_argument(
        "--strategies", nargs="*", default=None,
        help="Per-model strategy JSON strings (same order as --models). "
             "Omit to use global defaults for all models.",
    )
    # Global strategy defaults
    parser.add_argument("--prompt-type", type=str, default="box", choices=["box", "point", "both"])
    parser.add_argument("--point-from", type=str, default="point",
                        choices=["point", "random", "mask-rand", "mask-center", "box-center"],
                        help="How to generate point prompts. "
                             "'random' samples N uniform random points inside GT mask.")
    parser.add_argument("--num-points", type=int, default=1,
                        help="Number of point prompts per mask (works with all point_from modes)")
    parser.add_argument("--num-multimask", type=int, default=1, choices=[1, 3, 4],
                        help="Number of mask candidates per prompt")
    parser.add_argument("--multimask-select", type=str, default="score",
                        choices=["score", "area", "oracle"],
                        help="Mask selection strategy when num_multimask > 1")
    parser.add_argument("--refine-iter", type=int, default=1,
                        help="Iterative refinement rounds (sample error points and re-decode)")

    # Dataset & output
    parser.add_argument("--dataset", type=str, default="coco", choices=["sa", "coco", "cocofied_lvis", "lvis"])
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="output/compare")
    parser.add_argument("--top-k", type=int, default=20, help="Save masks for top-K most improved images")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = "datasets/SA-1B" if args.dataset == "sa" else "datasets/coco"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse model specs & strategies
    model_specs = [parse_model_spec(s) for s in args.models]
    model_labels = [label for _, _, label in model_specs]

    strategy_jsons = args.strategies or [None] * len(model_specs)
    if len(strategy_jsons) != len(model_specs):
        raise ValueError(f"--strategies count ({len(strategy_jsons)}) must match --models count ({len(model_specs)})")

    strategies = {}
    for (_, _, label), sj in zip(model_specs, strategy_jsons):
        strategies[label] = build_strategy(args, sj)

    print(f"Models to compare: {model_labels}")
    for label in model_labels:
        print(f"  [{label}] strategy: {strategies[label]}")

    # Build dataset once
    print(f"\nLoading dataset: {args.dataset} ({args.num_samples} samples)...")
    dataset, dataloader = build_dataset(args)
    print(f"Dataset size: {len(dataset)}")

    # Evaluate each model
    all_ious = {}
    all_masks = {}
    ious_per_iter_all = {}
    for model_type, ckpt, label in model_specs:
        strategy = strategies[label]
        print(f"\nEvaluating [{label}] ({model_type}: {ckpt})...")
        model = load_model(model_type, ckpt, args.device)
        ious, masks, ious_per_iter = evaluate_model(model, dataloader, strategy, args.device)
        all_ious[label] = ious
        all_masks[label] = masks
        ious_per_iter_all[label] = ious_per_iter
        print(f"  mIoU: {np.nanmean(ious)*100:.2f}%")
        if len(ious_per_iter) > 1:
            for it_i, it_ious in enumerate(ious_per_iter):
                print(f"    iter{it_i+1}: {np.nanmean(it_ious)*100:.2f}%")
        del model
        torch.cuda.empty_cache()

    # Report
    report = generate_report(args.output_dir, model_labels, all_ious, strategies, ious_per_iter_all)
    print_summary(report)

    # Save top-K improved images
    if len(model_labels) >= 2:
        base_label = model_labels[0]
        cmp_label = model_labels[-1]
        base_arr = np.array(all_ious[base_label])
        cmp_arr = np.array(all_ious[cmp_label])
        delta = cmp_arr - base_arr
        valid_mask = ~(np.isnan(base_arr) | np.isnan(cmp_arr))

        indices = np.where(valid_mask)[0]
        sorted_idx = indices[np.argsort(-delta[indices])]
        top_k = min(args.top_k, len(sorted_idx))
        top_entries = [
            (int(i), float(delta[i]), float(base_arr[i]), float(cmp_arr[i]))
            for i in sorted_idx[:top_k]
        ]
        save_comparison_masks(args.output_dir, top_entries, all_masks, (base_label, cmp_label))

        csv_path = os.path.join(args.output_dir, "per_image_iou.csv")
        with open(csv_path, "w") as f:
            headers = ["img_idx", "file_name"] + [f"iou_{l}" for l in model_labels] + ["delta_iou"]
            f.write(",".join(headers) + "\n")
            for i in range(len(base_arr)):
                fname = all_masks[base_label].get(i, {}).get("file_name", f"img_{i}")
                row = [str(i), str(fname)]
                for label in model_labels:
                    row.append(f"{all_ious[label][i]:.6f}")
                row.append(f"{delta[i]:.6f}")
                f.write(",".join(row) + "\n")
        print(f"Per-image IoU saved to {csv_path}")


if __name__ == "__main__":
    main()
