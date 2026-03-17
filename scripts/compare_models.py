"""
Model comparison script for EdgeSAM.

Runs multiple models on the same dataset, computes per-image IoU,
generates a comparison report, and saves prediction masks for images
where one model improves most over another.

Usage:
    # Compare two models using COCO val with box prompts
    python scripts/compare_models.py \
        --models edge_sam:weights/edge_sam.pth edge_sam:weights/edge_sam_3x.pth \
        --dataset coco --num-samples 200 --prompt-type box \
        --output-dir output/compare --top-k 20

    # Compare EdgeSAM vs SAM ViT-B with point prompts
    python scripts/compare_models.py \
        --models edge_sam:weights/edge_sam_3x.pth vit_b:weights/sam_vit_b.pth \
        --dataset sa --num-samples 100 --prompt-type point \
        --output-dir output/compare --top-k 10
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

from edge_sam import sam_model_registry, SamPredictor
from edge_sam.utils.common import cal_iou

# Lazy imports for datasets — only needed when using SA-1B or COCO
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
    """Load a SAM-family model from the registry."""
    if model_type not in sam_model_registry:
        raise ValueError(f"Unknown model type '{model_type}'. Available: {list(sam_model_registry.keys())}")
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device)
    model.eval()
    return model


def build_dataset(args):
    """Build dataset and dataloader based on args."""
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

    # Use simple collate that returns a list
    from mmengine.dataset import pseudo_collate
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=pseudo_collate, num_workers=args.num_workers,
    )
    return dataset, dataloader


@torch.no_grad()
def evaluate_model(model, dataloader, prompt_type, device):
    """Run model on the entire dataloader. Returns per-image mean IoU list and raw masks dict.

    Returns:
        ious: list of float, per-image mean IoU
        all_masks: dict mapping image_index -> (pred_masks_np, gt_masks_np, img_np, meta)
            where pred/gt masks are uint8 H×W arrays (per prompt, best mask selected).
    """
    predictor = SamPredictor(model)
    img_size = model.image_encoder.img_size
    mask_threshold = model.mask_threshold

    ious = []
    all_masks = {}

    for img_idx, (imgs, annos) in enumerate(dataloader):
        imgs = torch.stack(imgs, dim=0).to(device)
        img_size_before_pad = annos["img_size_before_pad"]
        img_size_pad = (img_size, img_size)

        image_embeddings = model.image_encoder(model.preprocess(imgs))
        dense_pe = model.prompt_encoder.get_dense_pe()

        # Prompts
        boxes = annos["prompt_box"]
        num_prompts = [b.size(0) for b in boxes]
        boxes_cat = torch.cat(boxes, dim=0).to(device)

        points = None
        if "prompt_point" in annos and prompt_type == "point":
            pts = torch.cat(annos["prompt_point"], dim=0).to(device)
            labels = torch.ones(pts.shape[:2], device=device)
            points = (pts, labels)

        if prompt_type == "point" and points is None:
            # Use box center as point prompt fallback
            cx = (boxes_cat[:, 0] + boxes_cat[:, 2]) / 2
            cy = (boxes_cat[:, 1] + boxes_cat[:, 3]) / 2
            pts = torch.stack([cx, cy], dim=1)[:, None]
            labels = torch.ones(pts.shape[:2], device=device)
            points = (pts, labels)
            boxes_cat = None
        elif prompt_type == "box":
            points = None
        # else: both

        sparse_emb, dense_emb = model.prompt_encoder(
            points=points, boxes=boxes_cat, masks=None,
        )
        mask_pred, iou_pred = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            num_multimask_outputs=1,
        )

        # GT mask
        gt_mask = torch.cat(annos["gt_mask"], dim=0).float().to(device)[:, None]

        # Upsample to original resolution and crop padding
        mask_pred_up = F.interpolate(mask_pred, img_size_pad, mode="bilinear", align_corners=False)
        gt_mask_up = gt_mask  # already at img_size_pad

        img_bs = imgs.size(0)
        prompt_offset = 0
        img_ious = []
        pred_masks_list = []
        gt_masks_list = []

        for i in range(img_bs):
            n = num_prompts[i]
            cur_pred = mask_pred_up[prompt_offset:prompt_offset + n]
            cur_gt = gt_mask_up[prompt_offset:prompt_offset + n]

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
            img_ious.append(iou.cpu())

            pred_masks_list.append((cur_pred_bin[:, 0].cpu().numpy().astype(np.uint8) * 255))
            gt_masks_list.append((cur_gt_bin[:, 0].cpu().numpy().astype(np.uint8) * 255))
            prompt_offset += n

        img_iou_tensor = torch.cat(img_ious)
        mean_iou = img_iou_tensor[~img_iou_tensor.isnan()].mean().item()
        ious.append(mean_iou)

        # Recover original image for visualization
        pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
        raw_img = (imgs * pixel_std + pixel_mean).clamp(0, 255)
        raw_img = raw_img[0]  # take first image in batch
        h, w = img_size_before_pad[0][1:]
        ori_h, ori_w = annos["info"]["height"][0], annos["info"]["width"][0]
        raw_img = raw_img[:, :h, :w]
        raw_img = F.interpolate(raw_img[None], (ori_h, ori_w), mode="bilinear", align_corners=False)[0]
        raw_img = raw_img.permute(1, 2, 0).byte().cpu().numpy()

        file_name = annos["info"]["file_name"][0] if "file_name" in annos["info"] else f"img_{img_idx}"

        all_masks[img_idx] = {
            "pred": np.concatenate(pred_masks_list, axis=0),  # (N, H, W)
            "gt": np.concatenate(gt_masks_list, axis=0),
            "img": raw_img,
            "file_name": file_name,
            "iou_per_prompt": img_iou_tensor.numpy(),
        }

    return ious, all_masks


def save_comparison_masks(output_dir, top_entries, model_masks, model_labels):
    """Save prediction masks for top improved images.

    top_entries: list of (img_idx, delta_iou, iou_base, iou_compare)
    model_masks: dict[label] -> all_masks from evaluate_model
    model_labels: (base_label, compare_label)
    """
    base_label, cmp_label = model_labels
    vis_dir = os.path.join(output_dir, "top_improvements")
    os.makedirs(vis_dir, exist_ok=True)

    for rank, (img_idx, delta, iou_base, iou_cmp) in enumerate(top_entries):
        base_data = model_masks[base_label][img_idx]
        cmp_data = model_masks[cmp_label][img_idx]
        img = base_data["img"]  # RGB uint8
        file_name = base_data["file_name"]
        safe_name = os.path.splitext(os.path.basename(str(file_name)))[0]

        sub_dir = os.path.join(vis_dir, f"rank{rank:03d}_{safe_name}_delta{delta:.4f}")
        os.makedirs(sub_dir, exist_ok=True)

        # Save original image
        cv2.imwrite(os.path.join(sub_dir, "image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # For each prompt, save GT, base pred, compare pred, and overlay
        n_prompts = base_data["pred"].shape[0]
        for p in range(min(n_prompts, 8)):  # limit to 8 prompts per image
            gt_mask = base_data["gt"][p]
            base_mask = base_data["pred"][p]
            cmp_mask = cmp_data["pred"][p]

            cv2.imwrite(os.path.join(sub_dir, f"p{p}_gt.png"), gt_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{base_label}.png"), base_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{cmp_label}.png"), cmp_mask)

            # Overlay: green=GT, red=pred, white=overlap
            overlay = _make_overlay(img, gt_mask, cmp_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{cmp_label}_overlay.png"), overlay)

            overlay_base = _make_overlay(img, gt_mask, base_mask)
            cv2.imwrite(os.path.join(sub_dir, f"p{p}_{base_label}_overlay.png"), overlay_base)

    print(f"Saved {len(top_entries)} comparison visualizations to {vis_dir}")


def _make_overlay(img, gt_mask, pred_mask, alpha=0.5):
    """Create an overlay image: green = GT only, red = pred only, white = overlap."""
    h, w = gt_mask.shape
    img_resized = cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img.copy()
    overlay = img_resized.copy()

    gt_bin = gt_mask > 127
    pred_bin = pred_mask > 127
    tp = gt_bin & pred_bin
    fn = gt_bin & ~pred_bin
    fp = ~gt_bin & pred_bin

    overlay[tp] = [255, 255, 255]  # white = correct
    overlay[fp] = [0, 0, 255]      # red = false positive (BGR)
    overlay[fn] = [0, 255, 0]      # green = false negative (BGR)

    result = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 1 - alpha, overlay, alpha, 0)
    return result


def generate_report(output_dir, model_labels, all_ious, all_masks):
    """Generate a JSON report with per-image comparison stats."""
    report = {"models": model_labels, "num_images": len(all_ious[model_labels[0]])}

    # Per-model summary
    model_stats = {}
    for label in model_labels:
        ious = all_ious[label]
        arr = np.array(ious)
        valid = arr[~np.isnan(arr)]
        model_stats[label] = {
            "mean_iou": float(valid.mean()) if len(valid) > 0 else 0.0,
            "median_iou": float(np.median(valid)) if len(valid) > 0 else 0.0,
            "std_iou": float(valid.std()) if len(valid) > 0 else 0.0,
            "num_images": len(ious),
        }
    report["model_stats"] = model_stats

    # Pairwise comparison (first model as baseline)
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
    """Print a readable summary table."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)
    print(f"Images evaluated: {report['num_images']}")
    print()

    # Model stats
    print(f"{'Model':<30} {'mIoU':>8} {'Median':>8} {'Std':>8}")
    print("-" * 56)
    for label, stats in report["model_stats"].items():
        print(f"{label:<30} {stats['mean_iou']*100:>7.2f}% {stats['median_iou']*100:>7.2f}% {stats['std_iou']*100:>7.2f}%")

    # Pairwise
    if "comparisons" in report:
        print()
        for key, cmp in report["comparisons"].items():
            print(f"--- {key} ---")
            print(f"  Mean delta IoU:    {cmp['mean_delta_iou']*100:+.2f}%")
            print(f"  Median delta IoU:  {cmp['median_delta_iou']*100:+.2f}%")
            print(f"  Improved / Degraded / Tied: {cmp['num_improved']} / {cmp['num_degraded']} / {cmp['num_tied']}")
            print(f"  Max improvement:   {cmp['max_improvement']*100:+.2f}%")
            print(f"  Max degradation:   {cmp['max_degradation']*100:+.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare multiple SAM models on the same dataset.")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Model specs in format 'model_type:checkpoint[:label]'. "
             "E.g. 'edge_sam:weights/edge_sam.pth:EdgeSAM_1x'",
    )
    parser.add_argument("--dataset", type=str, default="coco", choices=["sa", "coco", "cocofied_lvis", "lvis"])
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset root. Default: datasets/SA-1B for sa, datasets/coco for coco/lvis")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-prompts", type=int, default=64)
    parser.add_argument("--prompt-type", type=str, default="box", choices=["box", "point", "both"])
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

    # Parse model specs
    model_specs = [parse_model_spec(s) for s in args.models]
    model_labels = [label for _, _, label in model_specs]
    print(f"Models to compare: {model_labels}")

    # Build dataset once
    print(f"Loading dataset: {args.dataset} ({args.num_samples} samples)...")
    dataset, dataloader = build_dataset(args)
    print(f"Dataset size: {len(dataset)}")

    # Evaluate each model
    all_ious = {}
    all_masks = {}
    for model_type, ckpt, label in model_specs:
        print(f"\nEvaluating [{label}] ({model_type}: {ckpt})...")
        model = load_model(model_type, ckpt, args.device)
        ious, masks = evaluate_model(model, dataloader, args.prompt_type, args.device)
        all_ious[label] = ious
        all_masks[label] = masks
        print(f"  mIoU: {np.nanmean(ious)*100:.2f}%")
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Generate report
    report = generate_report(args.output_dir, model_labels, all_ious, all_masks)
    print_summary(report)

    # Save top-K improved images (compare last model vs first model)
    if len(model_labels) >= 2:
        base_label = model_labels[0]
        cmp_label = model_labels[-1]
        base_arr = np.array(all_ious[base_label])
        cmp_arr = np.array(all_ious[cmp_label])
        delta = cmp_arr - base_arr
        valid_mask = ~(np.isnan(base_arr) | np.isnan(cmp_arr))

        # Get top-K most improved
        indices = np.where(valid_mask)[0]
        sorted_idx = indices[np.argsort(-delta[indices])]
        top_k = min(args.top_k, len(sorted_idx))
        top_entries = [
            (int(i), float(delta[i]), float(base_arr[i]), float(cmp_arr[i]))
            for i in sorted_idx[:top_k]
        ]
        save_comparison_masks(args.output_dir, top_entries, all_masks, (base_label, cmp_label))

        # Also save per-image IoU CSV
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
