import os
import json
import random

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from pycocotools import mask as mask_utils

from edge_sam.utils.transforms import ResizeLongestSide


class COCOFinetuneDataset(torch.utils.data.Dataset):
    """COCO-format dataset for fine-tuning EdgeSAM with point prompts.

    Supports compressed RLE, uncompressed RLE, and polygon segmentations.

    Args:
        ann_file: Path to COCO annotation JSON file.
        img_dir: Path to image directory.
        img_size: Model input size (default: 1024).
        max_prompts_per_image: Max masks to sample per image per iteration.
        num_points_per_mask: Number of positive points sampled per mask.
    """

    def __init__(self, ann_file, img_dir, img_size=1024,
                 max_prompts_per_image=16, num_points_per_mask=1):
        super().__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        self.max_prompts_per_image = max_prompts_per_image
        self.num_points_per_mask = num_points_per_mask
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        self.transform = ResizeLongestSide(img_size)

        with open(ann_file, 'r') as f:
            coco = json.load(f)

        self.images = {img['id']: img for img in coco['images']}

        self.img_anns = {}
        for ann in coco['annotations']:
            if ann.get('iscrowd', 0):
                continue
            img_id = ann['image_id']
            if img_id not in self.img_anns:
                self.img_anns[img_id] = []
            self.img_anns[img_id].append(ann)

        self.img_ids = [iid for iid in self.img_anns if iid in self.images]

    def __len__(self):
        return len(self.img_ids)

    def _decode_mask(self, segm, h, w):
        """Decode segmentation to binary mask, handling all COCO formats."""
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

    def _sample_points(self, binary_mask):
        """Sample random positive points from inside the mask."""
        ys, xs = np.where(binary_mask > 0)
        pts, lbls = [], []
        for _ in range(self.num_points_per_mask):
            idx = random.randint(0, len(xs) - 1)
            pts.append([float(xs[idx]), float(ys[idx])])
            lbls.append(1)
        return pts, lbls

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        anns = self.img_anns[img_id]
        h, w = img_info['height'], img_info['width']

        img = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')
        img = pil_to_tensor(img)
        original_size = img.shape[1:]

        masks, points, labels = [], [], []
        for ann in anns:
            binary_mask = self._decode_mask(ann['segmentation'], h, w)
            if binary_mask is None or binary_mask.sum() == 0:
                continue
            masks.append(binary_mask)
            pts, lbls = self._sample_points(binary_mask)
            points.append(pts)
            labels.append(lbls)

        if len(masks) == 0:
            return self._dummy_sample()

        if len(masks) > self.max_prompts_per_image:
            sel = random.sample(range(len(masks)), self.max_prompts_per_image)
            masks = [masks[i] for i in sel]
            points = [points[i] for i in sel]
            labels = [labels[i] for i in sel]

        masks = torch.from_numpy(np.stack(masks, axis=0)).float()
        points = torch.tensor(points, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)

        # Resize image & masks
        img = self.transform.apply_image_torch(img[None].float()).squeeze(0)
        masks = self.transform.apply_masks_torch(masks, original_size)

        # Resize point coordinates
        pts_np = points.numpy()
        for i in range(pts_np.shape[0]):
            pts_np[i] = self.transform.apply_coords(pts_np[i], original_size)
        points = torch.from_numpy(pts_np)

        img_size_before_pad = img.shape[1:]

        img = self._pad(self._norm(img))
        masks = self._pad(masks)

        return {
            'image': img,
            'gt_masks': masks,
            'point_coords': points,
            'point_labels': labels,
            'img_size_before_pad': img_size_before_pad,
            'num_prompts': len(masks),
        }

    def _dummy_sample(self):
        return {
            'image': torch.zeros(3, self.img_size, self.img_size),
            'gt_masks': torch.zeros(1, self.img_size, self.img_size),
            'point_coords': torch.zeros(1, 1, 2),
            'point_labels': torch.ones(1, 1),
            'img_size_before_pad': (self.img_size, self.img_size),
            'num_prompts': 0,
        }

    def _norm(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def _pad(self, x):
        h, w = x.shape[-2:]
        return F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
