import argparse
import csv
import math
import os
import random
import time
from typing import Any, Dict, Tuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from monai.metrics import HausdorffDistanceMetric
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation

from models.unet import UNet
from models.deeplab import DeepLabV3Wrapper
import segmentation_models_pytorch as smp

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified CLI for VOC segmentation experiments")

    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet", "unet_pretrained", "deeplab"],
        default=None,
    )
    parser.add_argument("--loss", type=str, choices=["ce", "dice", "combined"], default=None)
    parser.add_argument("--augmentation", type=int, choices=[0, 1], default=None)
    parser.add_argument("--pretrained", type=int, choices=[0, 1], default=None)
    parser.add_argument("--exp_name", type=str, default=None)

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--vis_num_samples", type=int, default=None)

    return parser.parse_args()


def merge_config(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value

    if isinstance(cfg.get("augmentation"), int):
        cfg["augmentation"] = bool(cfg["augmentation"])
    if isinstance(cfg.get("pretrained"), int):
        cfg["pretrained"] = bool(cfg["pretrained"])

    cfg["device"] = get_device()
    return cfg


class JointTransform:
    def __init__(self, image_size: int, use_augmentation: bool):
        self.use_augmentation = use_augmentation
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

        if self.use_augmentation and random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        image = self.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return image, mask


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


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid_mask = target != self.ignore_index

        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        one_hot = F.one_hot(safe_target, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dims)
        denominator = torch.sum(probs + one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        dice_loss = self.dice(logits, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    if cfg["model"] == "unet":
        return UNet(in_channels=3, num_classes=cfg["num_classes"])

    if cfg["model"] == "unet_pretrained":
        encoder_weights = "imagenet" if cfg["pretrained"] else None
        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=cfg["num_classes"],
        )

    if cfg["model"] == "deeplab":
        return DeepLabV3Wrapper(
            num_classes=cfg["num_classes"],
            pretrained_backbone=cfg["pretrained"],
        )

    raise ValueError(f"Unsupported model: {cfg['model']}")


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    if cfg["loss"] == "ce":
        return nn.CrossEntropyLoss(ignore_index=cfg["ignore_index"])
    if cfg["loss"] == "dice":
        return DiceLoss(num_classes=cfg["num_classes"], ignore_index=cfg["ignore_index"])
    if cfg["loss"] == "combined":
        return CombinedLoss(num_classes=cfg["num_classes"], ignore_index=cfg["ignore_index"])
    raise ValueError(f"Unsupported loss: {cfg['loss']}")


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
        "confmat": confmat.cpu().numpy(),
    }


def prepare_one_hot_for_hd95(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    preds:   [B, H, W]
    targets: [B, H, W]

    Returns:
      pred_one_hot: [B, C, H, W]
      tgt_one_hot : [B, C, H, W]

    Ignore pixels are masked out in both prediction and target.
    """
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


def clean_mask(mask: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    out = mask.copy()
    out[(out < 0) | (out >= num_classes)] = 0
    out[out == ignore_index] = 0
    return out


def denormalize_image(img_tensor: torch.Tensor) -> np.ndarray:
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def make_exp_dir(cfg: Dict[str, Any]) -> str:
    if cfg.get("exp_name"):
        exp_name = cfg["exp_name"]
    else:
        aug_tag = "aug" if cfg["augmentation"] else "noaug"
        if cfg["model"] == "deeplab":
            pre_tag = "pretrain" if cfg["pretrained"] else "scratch"
            exp_name = f"{cfg['model']}_{cfg['loss']}_{aug_tag}_{pre_tag}"
        else:
            exp_name = f"{cfg['model']}_{cfg['loss']}_{aug_tag}"

    exp_dir = os.path.join(cfg["save_root"], exp_name)
    ensure_dir(exp_dir)
    ensure_dir(os.path.join(exp_dir, "visualizations"))
    return exp_dir


def save_config(cfg: Dict[str, Any], exp_dir: str) -> None:
    with open(os.path.join(exp_dir, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


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
            f"{row['train_loss']:.6f}",
            f"{row['val_loss']:.6f}",
            f"{row['pixel_acc']:.6f}",
            f"{row['miou']:.6f}",
            f"{row['mean_dice']:.6f}",
            f"{row['mean_class_acc']:.6f}",
            f"{row['hd95']:.6f}" if not math.isnan(row["hd95"]) else "nan",
            f"{row['epoch_time_sec']:.4f}",
        ])


def save_per_class_metrics(metrics: Dict[str, Any], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "iou", "accuracy"])
        for i, cls_name in enumerate(VOC_CLASSES):
            writer.writerow([
                cls_name,
                f"{metrics['per_class_iou'][i]:.6f}",
                f"{metrics['per_class_acc'][i]:.6f}",
            ])


def plot_curves(metrics_csv_path: str, save_path: str) -> None:
    import pandas as pd

    df = pd.read_csv(metrics_csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["miou"], label="mIoU")
    plt.plot(df["epoch"], df["mean_dice"], label="Mean Dice")
    plt.plot(df["epoch"], df["pixel_acc"], label="Pixel Acc")
    if "hd95" in df.columns:
        plt.plot(df["epoch"], df["hd95"], label="HD95")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path.replace(".png", "_metrics.png"), dpi=150)
    plt.close()


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None):
    model.train()
    running_loss = 0.0
    total_batches = 0

    desc = f"Train Epoch {epoch}" if epoch is not None else "Training"
    train_bar = tqdm(loader, desc=desc, leave=False)

    for images, masks in train_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes, ignore_index, epoch=None):
    model.eval()
    running_loss = 0.0
    total_batches = 0
    total_confmat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        distance_metric="euclidean",
        percentile=95,
        directed=False,
        reduction="mean",
    )

    desc = f"Val Epoch {epoch}" if epoch is not None else "Validation"
    val_bar = tqdm(loader, desc=desc, leave=True)

    for images, masks in val_bar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        preds = torch.argmax(logits, dim=1)

        confmat = compute_confusion_matrix(
            preds=preds.cpu(),
            targets=masks.cpu(),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        total_confmat += confmat

        pred_one_hot, tgt_one_hot = prepare_one_hot_for_hd95(
            preds=preds,
            targets=masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        pred_one_hot = pred_one_hot.cpu().float()
        tgt_one_hot = tgt_one_hot.cpu().float()

        hd95_metric(y_pred=pred_one_hot, y=tgt_one_hot)

        running_loss += loss.item()
        total_batches += 1

        val_bar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics_from_confmat(total_confmat)
    metrics["loss"] = running_loss / max(total_batches, 1)

    try:
        hd95_value = hd95_metric.aggregate().item()
    except Exception:
        hd95_value = float("nan")

    hd95_metric.reset()
    metrics["hd95"] = hd95_value
    return metrics


@torch.no_grad()
def save_visualizations(model, loader, device, save_dir, num_samples, num_classes, ignore_index):
    model.eval()
    saved = 0

    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()

        for i in range(images.size(0)):
            if saved >= num_samples:
                return

            img = denormalize_image(images[i].cpu())
            gt = clean_mask(masks[i].cpu().numpy(), num_classes, ignore_index)
            pred = clean_mask(preds[i].numpy(), num_classes, ignore_index)

            fig = plt.figure(figsize=(12, 4))

            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(img)
            ax1.set_title("Image")
            ax1.axis("off")

            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(gt, cmap="tab20", vmin=0, vmax=num_classes - 1)
            ax2.set_title("Ground Truth")
            ax2.axis("off")

            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(pred, cmap="tab20", vmin=0, vmax=num_classes - 1)
            ax3.set_title("Prediction")
            ax3.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{saved:02d}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved += 1


def main():
    args = parse_args()
    cfg = merge_config(load_yaml(args.config), args)

    set_seed(cfg["seed"])
    exp_dir = make_exp_dir(cfg)
    save_config(cfg, exp_dir)

    print("Resolved config:")
    for k, v in cfg.items():
        print(f"{k}: {v}")

    print(f"\nUsing device: {cfg['device']}")
    print(f"Experiment dir: {exp_dir}")

    train_tf = JointTransform(image_size=cfg["image_size"], use_augmentation=cfg["augmentation"])
    val_tf = JointTransform(image_size=cfg["image_size"], use_augmentation=False)

    train_dataset = VOCSegmentationDataset(
        root=cfg["data_root"],
        year=cfg["year"],
        image_set="train",
        joint_transform=train_tf,
    )
    val_dataset = VOCSegmentationDataset(
        root=cfg["data_root"],
        year=cfg["year"],
        image_set="val",
        joint_transform=val_tf,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val/Test samples: {len(val_dataset)}")

    pin_memory = cfg["device"] == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=pin_memory,
    )

    model = build_model(cfg).to(cfg["device"])
    criterion = build_loss(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["scheduler_factor"],
        patience=cfg["scheduler_patience"],
    )

    metrics_csv_path = os.path.join(exp_dir, "metrics.csv")
    save_metrics_csv_header(metrics_csv_path)

    best_miou = -1.0
    best_epoch = -1
    best_hd95 = float("nan")

    for epoch in range(1, cfg["num_epochs"] + 1):
        start = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=cfg["device"],
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=cfg["device"],
            num_classes=cfg["num_classes"],
            ignore_index=cfg["ignore_index"],
            epoch=epoch,
        )

        epoch_time = time.time() - start
        scheduler.step(val_metrics["miou"])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "pixel_acc": val_metrics["pixel_acc"],
            "miou": val_metrics["miou"],
            "mean_dice": val_metrics["mean_dice"],
            "mean_class_acc": val_metrics["mean_class_acc"],
            "hd95": val_metrics["hd95"],
            "epoch_time_sec": epoch_time,
        }
        append_metrics_csv(metrics_csv_path, row)

        # print(f"\nEpoch [{epoch}/{cfg['num_epochs']}] - {epoch_time:.1f}s")
        # print(f"Train Loss      : {train_loss:.4f}")
        # print(f"Val Loss        : {val_metrics['loss']:.4f}")
        # print(f"Pixel Accuracy  : {val_metrics['pixel_acc']:.4f}")
        # print(f"mIoU            : {val_metrics['miou']:.4f}")
        # print(f"Mean Dice       : {val_metrics['mean_dice']:.4f}")
        # print(f"Mean Class Acc  : {val_metrics['mean_class_acc']:.4f}")
        # print(f"HD95            : {val_metrics['hd95']:.4f}" if not math.isnan(val_metrics["hd95"]) else "HD95            : nan")

        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_epoch = epoch
            best_hd95 = val_metrics["hd95"]

            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            save_per_class_metrics(val_metrics, os.path.join(exp_dir, "per_class_metrics_best.csv"))
            save_visualizations(
                model=model,
                loader=val_loader,
                device=cfg["device"],
                save_dir=os.path.join(exp_dir, "visualizations"),
                num_samples=cfg["vis_num_samples"],
                num_classes=cfg["num_classes"],
                ignore_index=cfg["ignore_index"],
            )
            print("Saved new best model.")

    plot_curves(metrics_csv_path, os.path.join(exp_dir, "curves.png"))

    summary = {
        "best_epoch": best_epoch,
        "best_miou": float(best_miou),
        "best_hd95": None if math.isnan(best_hd95) else float(best_hd95),
        "model": cfg["model"],
        "loss": cfg["loss"],
        "augmentation": cfg["augmentation"],
        "pretrained": cfg["pretrained"],
        "device": cfg["device"],
    }
    with open(os.path.join(exp_dir, "summary.yaml"), "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best mIoU : {best_miou:.4f}")
    print(f"Best HD95 : {'nan' if math.isnan(best_hd95) else f'{best_hd95:.4f}'}")
    print(f"Saved to  : {exp_dir}")


if __name__ == "__main__":
    main()