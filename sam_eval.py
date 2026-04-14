import argparse
import csv
import math
import os
import random
import time
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.metrics import HausdorffDistanceMetric
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class JointTransform:
    def __init__(self, image_size: int = 256):
        self.img_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        self.mask_resize = transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = self.img_resize(image)
        mask = self.mask_resize(mask)
        image_tensor = self.to_tensor(image)
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image_tensor, mask_tensor


class VOCSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, year: str, image_set: str, joint_transform: JointTransform):
        self.base_dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=False,
        )
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, mask = self.base_dataset[idx]
        return self.joint_transform(image, mask)


@torch.no_grad()
def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    valid = targets != ignore_index
    preds = preds[valid]
    targets = targets[valid]

    k = (targets >= 0) & (targets < num_classes)
    inds = num_classes * targets[k].to(torch.int64) + preds[k]
    confmat = torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return confmat


def compute_metrics_from_confmat(confmat: torch.Tensor) -> Dict[str, Any]:
    confmat = confmat.float()

    tp = torch.diag(confmat)
    gt_per_class = confmat.sum(dim=1)
    pred_per_class = confmat.sum(dim=0)
    union = gt_per_class + pred_per_class - tp

    iou = tp / torch.clamp(union, min=1.0)
    acc_per_class = tp / torch.clamp(gt_per_class, min=1.0)
    dice_per_class = 2 * tp / torch.clamp(gt_per_class + pred_per_class, min=1.0)

    pixel_acc = tp.sum() / torch.clamp(confmat.sum(), min=1.0)

    return {
        "pixel_acc": pixel_acc.item(),
        "miou": iou.mean().item(),
        "mean_dice": dice_per_class.mean().item(),
        "mean_class_acc": acc_per_class.mean().item(),
        "per_class_iou": iou.cpu().numpy(),
        "per_class_acc": acc_per_class.cpu().numpy(),
    }


def prepare_one_hot_for_hd95(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_mask = targets != ignore_index

    safe_targets = targets.clone()
    safe_targets[~valid_mask] = 0

    safe_preds = preds.clone()
    safe_preds[~valid_mask] = 0

    pred_one_hot = F.one_hot(safe_preds, num_classes=num_classes).permute(0, 3, 1, 2).float()
    tgt_one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1).float()
    pred_one_hot = pred_one_hot * valid_mask
    tgt_one_hot = tgt_one_hot * valid_mask
    return pred_one_hot, tgt_one_hot


def clean_mask(mask: np.ndarray, num_classes: int = 21, ignore_index: int = 255) -> np.ndarray:
    out = mask.copy()
    out[(out < 0) | (out >= num_classes)] = 0
    out[out == ignore_index] = 0
    return out


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def build_sam2_predictor(model_cfg: str, checkpoint: str, device: str):
    model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def binary_mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def binary_mask_to_center_point(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    cx = float(xs.mean())
    cy = float(ys.mean())
    point_coords = np.array([[cx, cy]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    return point_coords, point_labels


def instance_masks_from_semantic_mask(mask: np.ndarray, ignore_index: int = 255) -> List[Tuple[int, np.ndarray]]:
    """
    Simplified prompt extraction:
    for each semantic class > 0, treat all pixels of that class as one region.
    """
    results = []
    class_ids = np.unique(mask)
    for cls_id in class_ids:
        if cls_id in [0, ignore_index]:
            continue
        region = (mask == cls_id).astype(np.uint8)
        if region.sum() == 0:
            continue
        results.append((int(cls_id), region))
    return results


def merge_object_mask_into_semantic(pred_semantic: np.ndarray, obj_mask: np.ndarray, cls_id: int):
    pred_semantic[obj_mask > 0] = cls_id


@torch.no_grad()
def run_sam_on_one_image(
    predictor: SAM2ImagePredictor,
    image_tensor: torch.Tensor,
    gt_mask_tensor: torch.Tensor,
    prompt_type: str,
    num_classes: int,
    ignore_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    image_np = (denormalize_image(image_tensor) * 255.0).astype(np.uint8)
    gt_mask = gt_mask_tensor.cpu().numpy()

    predictor.set_image(image_np)
    pred_semantic = np.zeros_like(gt_mask, dtype=np.int64)

    objects = instance_masks_from_semantic_mask(gt_mask, ignore_index=ignore_index)

    for cls_id, obj_region in objects:
        if obj_region.sum() == 0:
            continue

        try:
            if prompt_type == "box":
                box = binary_mask_to_box(obj_region)
                masks, scores, _ = predictor.predict(
                    box=box[None, :],
                    multimask_output=False,
                )
            elif prompt_type == "point":
                point_coords, point_labels = binary_mask_to_center_point(obj_region)
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False,
                )
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")

            best_mask = masks[0].astype(np.uint8)
            merge_object_mask_into_semantic(pred_semantic, best_mask, cls_id)
        except Exception:
            continue

    pred_semantic[gt_mask == ignore_index] = 0
    return pred_semantic, gt_mask


def save_visualization(
    image_tensor: torch.Tensor,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    num_classes: int,
    ignore_index: int,
):
    gt_vis = clean_mask(gt_mask, num_classes, ignore_index)
    pred_vis = clean_mask(pred_mask, num_classes, ignore_index)
    img = denormalize_image(image_tensor)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(gt_vis, cmap="tab20", vmin=0, vmax=num_classes - 1)
    ax2.set_title("Ground Truth")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(pred_vis, cmap="tab20", vmin=0, vmax=num_classes - 1)
    ax3.set_title("SAM Prediction")
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_per_class_metrics(metrics: Dict[str, Any], save_path: str):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "iou", "accuracy"])
        for i, cls_name in enumerate(VOC_CLASSES):
            writer.writerow([
                cls_name,
                f"{metrics['per_class_iou'][i]:.6f}",
                f"{metrics['per_class_acc'][i]:.6f}",
            ])


def save_metrics_csv_header(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "pixel_acc",
                "miou",
                "mean_dice",
                "mean_class_acc",
                "hd95",
                "epoch_time_sec",
            ])


def append_metrics_csv(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["epoch"],
            "nan" if math.isnan(row["train_loss"]) else f"{row['train_loss']:.6f}",
            "nan" if math.isnan(row["val_loss"]) else f"{row['val_loss']:.6f}",
            f"{row['pixel_acc']:.6f}",
            f"{row['miou']:.6f}",
            f"{row['mean_dice']:.6f}",
            f"{row['mean_class_acc']:.6f}",
            "nan" if math.isnan(row["hd95"]) else f"{row['hd95']:.6f}",
            f"{row['epoch_time_sec']:.4f}",
        ])


def save_config(cfg: Dict[str, Any], exp_dir: str) -> None:
    with open(os.path.join(exp_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


@torch.no_grad()
def evaluate_sam(
    predictor: SAM2ImagePredictor,
    loader: DataLoader,
    prompt_type: str,
    num_classes: int,
    ignore_index: int,
    save_dir: str,
    max_vis: int = 6,
) -> Dict[str, Any]:
    total_confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
        directed=False,
        reduction="mean",
        get_not_nans=False,
    )

    vis_count = 0
    total_time_start = time.time()

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"SAM Eval ({prompt_type})",
        leave=True,
    )

    for images, masks in progress_bar:
        image_tensor = images[0]
        gt_mask_tensor = masks[0]

        pred_mask_np, gt_mask_np = run_sam_on_one_image(
            predictor=predictor,
            image_tensor=image_tensor,
            gt_mask_tensor=gt_mask_tensor,
            prompt_type=prompt_type,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        preds = torch.from_numpy(pred_mask_np).unsqueeze(0)
        targets = torch.from_numpy(gt_mask_np).unsqueeze(0)

        confmat = compute_confusion_matrix(
            preds=preds,
            targets=targets,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        total_confmat += confmat

        pred_one_hot, tgt_one_hot = prepare_one_hot_for_hd95(
            preds=preds,
            targets=targets,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        hd95_metric(
            y_pred=pred_one_hot.cpu().float(),
            y=tgt_one_hot.cpu().float(),
        )

        if vis_count < max_vis:
            save_visualization(
                image_tensor=image_tensor,
                gt_mask=gt_mask_np,
                pred_mask=pred_mask_np,
                save_path=os.path.join(save_dir, "visualizations", f"sample_{vis_count:02d}.png"),
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            vis_count += 1

        progress_bar.set_postfix(vis=vis_count)

    total_time = time.time() - total_time_start

    metrics = compute_metrics_from_confmat(total_confmat)
    try:
        metrics["hd95"] = hd95_metric.aggregate().item()
    except Exception:
        metrics["hd95"] = float("nan")
    hd95_metric.reset()

    metrics["loss"] = float("nan")
    metrics["epoch_time_sec"] = total_time
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 evaluation for Pascal VOC semantic segmentation")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--year", type=str, default="2007")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sam_cfg", type=str, required=True)
    parser.add_argument("--sam_ckpt", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, choices=["box", "point"], default="box")

    parser.add_argument("--save_root", type=str, default="./outputs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--vis_num_samples", type=int, default=6)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    exp_name = args.exp_name if args.exp_name else f"sam_{args.prompt_type}"
    exp_dir = os.path.join(args.save_root, exp_name)
    ensure_dir(exp_dir)
    ensure_dir(os.path.join(exp_dir, "visualizations"))

    cfg = {
        "data_root": args.data_root,
        "year": args.year,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "seed": args.seed,
        "save_root": args.save_root,
        "exp_name": exp_name,
        "vis_num_samples": args.vis_num_samples,
        "model": "sam2",
        "loss": "prompted",
        "augmentation": False,
        "pretrained": True,
        "prompt_type": args.prompt_type,
        "sam_cfg": args.sam_cfg,
        "sam_ckpt": args.sam_ckpt,
        "device": get_device(),
    }
    save_config(cfg, exp_dir)

    print("Resolved config:")
    for k, v in cfg.items():
        print(f"{k}: {v}")

    print(f"\nUsing device: {cfg['device']}")
    print(f"Experiment dir: {exp_dir}")

    transform = JointTransform(image_size=args.image_size)
    val_dataset = VOCSegmentationDataset(
        root=args.data_root,
        year=args.year,
        image_set="val",
        joint_transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Val/Test samples: {len(val_dataset)}")

    predictor = build_sam2_predictor(
        model_cfg=args.sam_cfg,
        checkpoint=args.sam_ckpt,
        device=cfg["device"],
    )

    metrics = evaluate_sam(
        predictor=predictor,
        loader=val_loader,
        prompt_type=args.prompt_type,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        save_dir=exp_dir,
        max_vis=args.vis_num_samples,
    )

    print("\nSAM Evaluation Results")
    print(f"Prompt Type     : {args.prompt_type}")
    print(f"Pixel Accuracy  : {metrics['pixel_acc']:.4f}")
    print(f"mIoU            : {metrics['miou']:.4f}")
    print(f"Mean Dice       : {metrics['mean_dice']:.4f}")
    print(f"Mean Class Acc  : {metrics['mean_class_acc']:.4f}")

    hd95_str = "nan" if math.isnan(metrics["hd95"]) else f"{metrics['hd95']:.4f}"
    print(f"HD95            : {hd95_str}")

    save_per_class_metrics(metrics, os.path.join(exp_dir, "per_class_metrics_best.csv"))

    metrics_csv_path = os.path.join(exp_dir, "metrics.csv")
    save_metrics_csv_header(metrics_csv_path)
    row = {
        "epoch": 1,
        "train_loss": float("nan"),
        "val_loss": float("nan"),
        "pixel_acc": metrics["pixel_acc"],
        "miou": metrics["miou"],
        "mean_dice": metrics["mean_dice"],
        "mean_class_acc": metrics["mean_class_acc"],
        "hd95": metrics["hd95"],
        "epoch_time_sec": metrics["epoch_time_sec"],
    }
    append_metrics_csv(metrics_csv_path, row)

    summary = {
        "best_epoch": 1,
        "best_miou": float(metrics["miou"]),
        "best_hd95": None if math.isnan(metrics["hd95"]) else float(metrics["hd95"]),
        "model": "sam2",
        "loss": "prompted",
        "augmentation": False,
        "pretrained": True,
        "device": cfg["device"],
        "prompt_type": args.prompt_type,
    }
    with open(os.path.join(exp_dir, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"\nSaved results to: {exp_dir}")


if __name__ == "__main__":
    main()