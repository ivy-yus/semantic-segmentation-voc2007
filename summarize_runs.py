import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt


def safe_read_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def collect_runs(outputs_root="./outputs"):
    rows = []

    for name in sorted(os.listdir(outputs_root)):
        run_dir = os.path.join(outputs_root, name)
        if not os.path.isdir(run_dir):
            continue

        summary_path = os.path.join(run_dir, "summary.yaml")
        config_path = os.path.join(run_dir, "resolved_config.yaml")
        metrics_path = os.path.join(run_dir, "metrics.csv")

        if not os.path.exists(metrics_path):
            continue

        summary = safe_read_yaml(summary_path)
        config = safe_read_yaml(config_path)

        try:
            df = pd.read_csv(metrics_path)
        except Exception:
            continue

        if len(df) == 0:
            continue

        # Best row by mIoU
        df = df[df["miou"].notna()].copy()
        if len(df) == 0:
            continue
        best_idx = df["miou"].idxmax()
        best_row = df.loc[best_idx]

        row = {
            "run_name": name,
            "model": config.get("model", summary.get("model")),
            "loss": config.get("loss", summary.get("loss")),
            "augmentation": config.get("augmentation", summary.get("augmentation")),
            "pretrained": config.get("pretrained", summary.get("pretrained")),
            "device": config.get("device", summary.get("device")),
            "batch_size": config.get("batch_size"),
            "image_size": config.get("image_size"),
            "num_epochs": config.get("num_epochs"),
            "best_epoch": int(best_row["epoch"]),
            "best_train_loss": float(best_row["train_loss"]),
            "best_val_loss": float(best_row["val_loss"]),
            "best_pixel_acc": float(best_row["pixel_acc"]),
            "best_miou": float(best_row["miou"]),
            "best_mean_dice": float(best_row["mean_dice"]),
            "best_mean_class_acc": float(best_row["mean_class_acc"]),
            "best_hd95": float(best_row["hd95"]) if "hd95" in best_row and pd.notna(best_row["hd95"]) else float("nan"),
            "last_epoch_time_sec": float(df.iloc[-1]["epoch_time_sec"]),
            "mean_epoch_time_sec": float(df["epoch_time_sec"].mean()),
            "run_dir": run_dir,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_summary_table(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "all_experiments.csv")
    df = df.sort_values(by="best_miou", ascending=False)
    df.to_csv(out_csv, index=False)
    print(f"Saved summary table to: {out_csv}")


def plot_bar(df, x_col, y_col, title, save_path, ascending=False):
    plot_df = df.sort_values(by=y_col, ascending=ascending).copy()

    plt.figure(figsize=(10, max(5, 0.5 * len(plot_df))))
    plt.barh(plot_df[x_col], plot_df[y_col])
    plt.xlabel(y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot to: {save_path}")


def main():
    outputs_root = "./outputs"
    summary_dir = os.path.join(outputs_root, "summary")

    df = collect_runs(outputs_root=outputs_root)
    if len(df) == 0:
        print("No experiment runs found.")
        return

    save_summary_table(df, summary_dir)

    plot_bar(
        df=df,
        x_col="run_name",
        y_col="best_miou",
        title="Best mIoU by Experiment",
        save_path=os.path.join(summary_dir, "best_miou_bar.png"),
        ascending=False,
    )

    # HD95: lower is better, so ascending=True looks nicer
    if "best_hd95" in df.columns and df["best_hd95"].notna().any():
        plot_bar(
            df=df[df["best_hd95"].notna()].copy(),
            x_col="run_name",
            y_col="best_hd95",
            title="Best HD95 by Experiment (Lower is Better)",
            save_path=os.path.join(summary_dir, "best_hd95_bar.png"),
            ascending=True,
        )

    # Optional: grouped quick view
    compact_cols = [
        "run_name", "model", "loss", "augmentation", "pretrained",
        "best_miou", "best_mean_dice", "best_pixel_acc", "best_hd95",
        "mean_epoch_time_sec"
    ]
    print("\nExperiment summary:")
    print(df[compact_cols].sort_values(by="best_miou", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()