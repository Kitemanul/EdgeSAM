"""
Compare multiple SAM models on the same dataset.

Computes per-mask IoU, generates a comparison report, and saves prediction
masks for images where one model improves most over another.

Usage:
    python scripts/compare_models.py \
        --models edge_sam:weights/edge_sam.pth edge_sam:weights/edge_sam_3x.pth \
        --ann-file /path/to/instances_val2017.json \
        --img-dir /path/to/val2017/ \
        --num-points 1 --point-strategy random \
        --output-dir output/compare --top-k 20

    # Per-model strategy override
    python scripts/compare_models.py \
        --models edge_sam:weights/edge_sam.pth:baseline \
                 edge_sam:weights/edge_sam_3x.pth:improved \
        --ann-file val.json --img-dir images/ \
        --strategies \
            '{"point_strategy":"random","num_points":1,"num_multimask":1}' \
            '{"point_strategy":"center","num_points":3,"num_multimask":4,"multimask_select":"oracle"}'
"""

import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from edge_sam import sam_model_registry, SamPredictor


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY = {
    "point_strategy": "random",   # random | center | bbox_center
    "num_points": 1,
    "num_multimask": 1,           # 1, 3, or 4
    "multimask_select": "score",  # score | area | oracle
}


def build_strategy(args, per_model_json=None):
    s = dict(DEFAULT_STRATEGY)
    s["point_strategy"] = args.point_strategy
    s["num_points"] = args.num_points
    s["num_multimask"] = args.num_multimask
    s["multimask_select"] = args.multimask_select
    if per_model_json:
        overrides = json.loads(per_model_json) if isinstance(per_model_json, str) else per_model_json
        for k, v in overrides.items():
            if k not in DEFAULT_STRATEGY:
                raise ValueError(f"Unknown strategy key '{k}'. Valid: {list(DEFAULT_STRATEGY.keys())}")
            s[k] = v
    return s


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def parse_model_spec(spec):
    parts = spec.split(":")
    if len(parts) == 2:
        model_type, ckpt = parts
        label = f"{model_type}_{os.path.basename(ckpt).replace('.pth', '')}"
    elif len(parts) >= 3:
        model_type, ckpt = parts[0], parts[1]
        label = ":".join(parts[2:])
    else:
        raise ValueError(f"Invalid model spec '{spec}'")
    return model_type, ckpt, label


# ---------------------------------------------------------------------------
# Annotation helpers (same as scripts/eval_iou.py)
# ---------------------------------------------------------------------------

def decode_mask(segm, h, w):
    if isinstance(segm, list):
        rle = mask_utils.merge(mask_utils.frPyObjects(segm, h, w))
    elif isinstance(segm, dict):
        if isinstance(segm.get("counts"), list):
            rle = mask_utils.frPyObjects(segm, h, w)
        else:
            rle = segm
    else:
        return None
    return mask_utils.decode(rle)


def sample_points(binary_mask, num_points, strategy, bbox=None):
    if strategy == "center":
        ys, xs = np.where(binary_mask > 0)
        cx, cy = float(xs.mean()), float(ys.mean())
        coords = np.array([[cx, cy]] * num_points, dtype=np.float32)
    elif strategy == "bbox_center" and bbox is not None:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        coords = np.array([[cx, cy]] * num_points, dtype=np.float32)
    else:  # random
        ys, xs = np.where(binary_mask > 0)
        indices = [random.randint(0, len(xs) - 1) for _ in range(num_points)]
        coords = np.array([[float(xs[i]), float(ys[i])] for i in indices], dtype=np.float32)
    labels = np.ones(num_points, dtype=np.int32)
    return coords, labels


def compute_iou(pred_mask, gt_mask):
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def select_best_mask(masks, scores, gt_mask, strategy):
    """From multi-mask output, select one mask.

    Args:
        masks: (C, H, W) bool ndarray
        scores: (C,) float ndarray
        gt_mask: (H, W) binary ndarray
        strategy: "score" | "area" | "oracle"
    Returns:
        selected mask (H, W) bool
    """
    if masks.shape[0] == 1:
        return masks[0]
    if strategy == "score":
        return masks[scores.argmax()]
    elif strategy == "area":
        return masks[masks.sum(axis=(1, 2)).argmax()]
    elif strategy == "oracle":
        ious = [compute_iou(m, gt_mask) for m in masks]
        return masks[np.argmax(ious)]
    return masks[scores.argmax()]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model_type, checkpoint, images, img_anns, img_ids,
                   img_dir, strategy, max_masks):
    """Run one model on the dataset. Returns per-mask IoU (no heavy data kept)."""
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.cuda().eval()
    predictor = SamPredictor(model)

    per_image = {}  # img_id -> { file_name, ious, mean_iou }

    for img_id in tqdm(img_ids, desc=f"  {model_type}"):
        img_info = images[img_id]
        anns = img_anns[img_id]
        h, w = img_info["height"], img_info["width"]

        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(image)

        if max_masks > 0 and len(anns) > max_masks:
            anns = random.sample(anns, max_masks)

        ious = []
        for ann in anns:
            gt = decode_mask(ann["segmentation"], h, w)
            if gt is None or gt.sum() == 0:
                continue

            coords, labels = sample_points(
                gt, strategy["num_points"], strategy["point_strategy"],
                bbox=ann.get("bbox"))

            masks, scores, _ = predictor.predict(
                point_coords=coords, point_labels=labels,
                num_multimask_outputs=strategy["num_multimask"])

            pred = select_best_mask(masks, scores, gt, strategy["multimask_select"])
            iou = compute_iou(pred, gt)
            ious.append(iou)

        if ious:
            per_image[img_id] = dict(
                file_name=img_info["file_name"],
                ious=ious, mean_iou=float(np.mean(ious)))

    del model, predictor
    torch.cuda.empty_cache()
    return per_image


@torch.no_grad()
def generate_visualizations(model_type, checkpoint, images, img_anns,
                            vis_img_ids, img_dir, strategy, max_masks):
    """Re-run a model on selected images to produce masks for visualization."""
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.cuda().eval()
    predictor = SamPredictor(model)

    vis_data = {}  # img_id -> { pred_masks, gt_masks, image }

    for img_id in tqdm(vis_img_ids, desc=f"  vis {model_type}"):
        img_info = images[img_id]
        anns = img_anns[img_id]
        h, w = img_info["height"], img_info["width"]

        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        predictor.set_image(image)

        if max_masks > 0 and len(anns) > max_masks:
            anns = random.sample(anns, max_masks)

        pred_masks, gt_masks = [], []
        for ann in anns:
            gt = decode_mask(ann["segmentation"], h, w)
            if gt is None or gt.sum() == 0:
                continue

            coords, labels = sample_points(
                gt, strategy["num_points"], strategy["point_strategy"],
                bbox=ann.get("bbox"))

            masks, scores, _ = predictor.predict(
                point_coords=coords, point_labels=labels,
                num_multimask_outputs=strategy["num_multimask"])

            pred = select_best_mask(masks, scores, gt, strategy["multimask_select"])
            pred_masks.append(pred.astype(np.uint8) * 255)
            gt_masks.append(gt.astype(np.uint8) * 255)

        if pred_masks:
            vis_data[img_id] = dict(
                file_name=img_info["file_name"],
                pred_masks=pred_masks, gt_masks=gt_masks,
                image=image)

    del model, predictor
    torch.cuda.empty_cache()
    return vis_data


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _make_overlay(img, gt_mask, pred_mask, alpha=0.5):
    h, w = gt_mask.shape[:2]
    base = cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img.copy()
    ov = base.copy()
    gt_b, pr_b = gt_mask > 127, pred_mask > 127
    ov[gt_b & pr_b] = [255, 255, 255]
    ov[~gt_b & pr_b] = [0, 0, 255]
    ov[gt_b & ~pr_b] = [0, 255, 0]
    return cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 1 - alpha, ov, alpha, 0)


def save_top_masks(output_dir, top_entries, vis_data, model_labels, top_k):
    base_label, cmp_label = model_labels
    vis_dir = os.path.join(output_dir, "top_improvements")
    os.makedirs(vis_dir, exist_ok=True)

    saved = 0
    for rank, (img_id, delta, iou_base, iou_cmp) in enumerate(top_entries[:top_k]):
        if img_id not in vis_data[base_label] or img_id not in vis_data[cmp_label]:
            continue
        bd = vis_data[base_label][img_id]
        cd = vis_data[cmp_label][img_id]
        safe = os.path.splitext(bd["file_name"])[0].replace("/", "_")
        sub = os.path.join(vis_dir, f"rank{rank:03d}_{safe}_delta{delta:.4f}")
        os.makedirs(sub, exist_ok=True)

        cv2.imwrite(os.path.join(sub, "image.png"),
                    cv2.cvtColor(bd["image"], cv2.COLOR_RGB2BGR))

        n = min(len(bd["pred_masks"]), len(cd["pred_masks"]), 8)
        for p in range(n):
            gt = bd["gt_masks"][p]
            cv2.imwrite(os.path.join(sub, f"p{p}_gt.png"), gt)
            cv2.imwrite(os.path.join(sub, f"p{p}_{base_label}.png"), bd["pred_masks"][p])
            cv2.imwrite(os.path.join(sub, f"p{p}_{cmp_label}.png"), cd["pred_masks"][p])
            cv2.imwrite(os.path.join(sub, f"p{p}_{base_label}_overlay.png"),
                        _make_overlay(bd["image"], gt, bd["pred_masks"][p]))
            cv2.imwrite(os.path.join(sub, f"p{p}_{cmp_label}_overlay.png"),
                        _make_overlay(cd["image"], gt, cd["pred_masks"][p]))
        saved += 1

    print(f"Saved {saved} visualizations to {vis_dir}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(model_labels, all_results, strategies):
    report = dict(models=model_labels,
                  strategies={l: strategies[l] for l in model_labels})

    stats = {}
    for label in model_labels:
        ious = []
        for d in all_results[label].values():
            ious.extend(d["ious"])
        arr = np.array(ious) if ious else np.array([0.0])
        stats[label] = dict(
            mean_iou=float(arr.mean()), median_iou=float(np.median(arr)),
            std_iou=float(arr.std()), num_masks=len(ious),
            num_images=len(all_results[label]))
    report["model_stats"] = stats

    if len(model_labels) >= 2:
        base, cmp = model_labels[0], model_labels[-1]
        common = set(all_results[base]) & set(all_results[cmp])
        deltas = []
        for img_id in common:
            deltas.append(all_results[cmp][img_id]["mean_iou"] - all_results[base][img_id]["mean_iou"])
        d = np.array(deltas) if deltas else np.array([0.0])
        report["comparison"] = dict(
            pair=f"{cmp}_vs_{base}",
            mean_delta=float(d.mean()), median_delta=float(np.median(d)),
            num_improved=int((d > 0).sum()), num_degraded=int((d < 0).sum()),
            num_tied=int((d == 0).sum()),
            max_improve=float(d.max()), max_degrade=float(d.min()))

    return report


def print_summary(report):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON REPORT")
    print("=" * 60)

    for label in report["models"]:
        s = report["strategies"][label]
        print(f"  [{label}] {s['point_strategy']}, {s['num_points']}pt, "
              f"multimask={s['num_multimask']}, select={s['multimask_select']}")
    print()

    print(f"{'Model':<30} {'mIoU':>8} {'Median':>8} {'Masks':>8}")
    print("-" * 56)
    for label, st in report["model_stats"].items():
        print(f"{label:<30} {st['mean_iou']*100:>7.2f}% {st['median_iou']*100:>7.2f}% {st['num_masks']:>8}")

    if "comparison" in report:
        c = report["comparison"]
        print(f"\n--- {c['pair']} ---")
        print(f"  Mean delta:  {c['mean_delta']*100:+.2f}%")
        print(f"  Median delta: {c['median_delta']*100:+.2f}%")
        print(f"  Improved/Degraded/Tied: {c['num_improved']}/{c['num_degraded']}/{c['num_tied']}")
        print(f"  Max improve: {c['max_improve']*100:+.2f}%  Max degrade: {c['max_degrade']*100:+.2f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Compare SAM models on the same dataset.")

    # Models
    p.add_argument("--models", nargs="+", required=True,
                   help="'model_type:checkpoint[:label]'")
    p.add_argument("--strategies", nargs="*", default=None,
                   help="Per-model strategy JSON (same order as --models)")

    # Dataset paths (like scripts/eval_iou.py)
    p.add_argument("--ann-file", required=True, help="COCO annotation JSON path")
    p.add_argument("--img-dir", required=True, help="Image directory path")

    # Strategy defaults
    p.add_argument("--point-strategy", default="random",
                   choices=["random", "center", "bbox_center"])
    p.add_argument("--num-points", type=int, default=1)
    p.add_argument("--num-multimask", type=int, default=1, choices=[1, 3, 4])
    p.add_argument("--multimask-select", default="score",
                   choices=["score", "area", "oracle"])

    # Limits
    p.add_argument("--max-masks-per-image", type=int, default=-1)
    p.add_argument("--max-images", type=int, default=-1)

    # Output
    p.add_argument("--output-dir", default="output/compare")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse models & strategies
    model_specs = [parse_model_spec(s) for s in args.models]
    model_labels = [label for _, _, label in model_specs]
    strategy_jsons = args.strategies or [None] * len(model_specs)
    if len(strategy_jsons) != len(model_specs):
        raise ValueError("--strategies count must match --models count")
    strategies = {}
    for (_, _, label), sj in zip(model_specs, strategy_jsons):
        strategies[label] = build_strategy(args, sj)

    # Load annotations once
    print(f"Loading annotations: {args.ann_file}")
    with open(args.ann_file, "r") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_anns[ann["image_id"]].append(ann)
    img_ids = [iid for iid in img_anns if iid in images]
    if args.max_images > 0:
        img_ids = img_ids[:args.max_images]
    print(f"Images: {len(img_ids)}")

    # Evaluate each model
    all_results = {}
    for model_type, ckpt, label in model_specs:
        strategy = strategies[label]
        print(f"\nEvaluating [{label}] ({model_type}: {ckpt})")
        print(f"  strategy: {strategy}")

        # Reset seed so each model sees the same random points
        random.seed(args.seed)
        np.random.seed(args.seed)

        per_image = evaluate_model(
            model_type, ckpt, images, img_anns, img_ids,
            args.img_dir, strategy, args.max_masks_per_image)
        all_results[label] = per_image

        ious = [iou for d in per_image.values() for iou in d["ious"]]
        print(f"  mIoU: {np.mean(ious)*100:.2f}% ({len(ious)} masks)")

    # Report
    report = build_report(model_labels, all_results, strategies)
    print_summary(report)

    with open(os.path.join(args.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # Top-K visualizations and CSV
    if len(model_labels) >= 2:
        base_label, cmp_label = model_labels[0], model_labels[-1]
        common = sorted(set(all_results[base_label]) & set(all_results[cmp_label]))

        entries = []
        for img_id in common:
            ib = all_results[base_label][img_id]["mean_iou"]
            ic = all_results[cmp_label][img_id]["mean_iou"]
            entries.append((img_id, ic - ib, ib, ic))
        entries.sort(key=lambda x: -x[1])

        # Per-image CSV (lightweight, no re-inference needed)
        csv_path = os.path.join(args.output_dir, "per_image_iou.csv")
        with open(csv_path, "w") as f:
            f.write("image_id,file_name," +
                    ",".join(f"iou_{l}" for l in model_labels) + ",delta_iou\n")
            for img_id, delta, ib, ic in entries:
                fn = all_results[base_label][img_id]["file_name"]
                vals = [str(img_id), fn]
                for label in model_labels:
                    r = all_results[label].get(img_id)
                    vals.append(f"{r['mean_iou']:.6f}" if r else "nan")
                vals.append(f"{delta:.6f}")
                f.write(",".join(vals) + "\n")
        print(f"Per-image IoU saved to {csv_path}")

        # Re-run only on top-K images for visualization (avoids storing
        # all masks/images in memory during the full evaluation pass)
        if args.top_k > 0:
            vis_img_ids = set(img_id for img_id, _, _, _ in entries[:args.top_k])
            print(f"\nGenerating visualizations for top-{args.top_k} images...")

            vis_data = {}
            for model_type, ckpt, label in model_specs:
                strategy = strategies[label]
                random.seed(args.seed)
                np.random.seed(args.seed)
                vis_data[label] = generate_visualizations(
                    model_type, ckpt, images, img_anns, vis_img_ids,
                    args.img_dir, strategy, args.max_masks_per_image)

            save_top_masks(args.output_dir, entries, vis_data,
                           (base_label, cmp_label), args.top_k)


if __name__ == "__main__":
    main()
