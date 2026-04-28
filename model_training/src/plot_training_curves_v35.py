"""Parse v35-run.log and produce a 6-panel training-curve summary.

Panels (left-to-right, top-to-bottom):
    1. Train loss (per-batch + per-epoch avg) vs val loss per epoch
    2. Loss components L_att / L_rep / L_var per batch (smoothed)
    3. Val metrics per epoch: purity, efficiency, match rate, noise suppression
    4. LR schedule (left axis) + L_var weight ramp (right axis) per batch
    5. Wall-clock time per epoch (bar chart)
    6. Cumulative training wall time

Usage:
    cd model_training
    PYTHONPATH=. python src/plot_training_curves_v35.py \
        --log /home/marko.cechovic/cgatr/v35-run.log \
        --output_dir eval_results/v35_training_curves
"""

import os
import re
import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Each batch line, e.g.:
#   Epoch 5 | Batch 350 | Loss 0.7405 | Avg 0.7321 | L_att 0.025 L_rep 0.108 L_var 0.041 (w=0.30) | LR 5.42e-05 | Time 1024.7s
BATCH_RE = re.compile(
    r"Epoch (?P<ep>\d+) \| Batch (?P<batch>\d+) \| Loss (?P<loss>[\d.]+) \| "
    r"Avg (?P<avg>[\d.]+) \| L_att (?P<latt>[\d.]+) L_rep (?P<lrep>[\d.]+) "
    r"L_var (?P<lvar>[\d.]+) \(w=(?P<w>[-\d.]+)\) \| "
    r"LR (?P<lr>[\d.eE+\-]+) \| Time (?P<time>[\d.]+)s"
)
# Val metric line:
#   Epoch 5 | Val Loss: 0.6242 | Purity: 0.869 | Efficiency: 0.745 | Match Rate: 0.688 | Noise Supp: 0.846 (...)
VAL_RE = re.compile(
    r"Epoch (?P<ep>\d+) \| Val Loss: (?P<vl>[\d.]+) \| "
    r"Purity: (?P<pur>[\d.]+) \| Efficiency: (?P<eff>[\d.]+) \| "
    r"Match Rate: (?P<match>[\d.]+) \| Noise Supp: (?P<ns>[\d.]+)"
)
# Per-epoch summary line:
#   Epoch 5 Summary: Train Loss=0.7405, Val Loss=0.6242
SUM_RE = re.compile(
    r"Epoch (?P<ep>\d+) Summary: Train Loss=(?P<tl>[\d.]+), Val Loss=(?P<vl>[\d.]+)"
)


def parse_log(log_path):
    batches = []
    vals = []
    epoch_summaries = []
    with open(log_path, "r") as f:
        for line in f:
            m = BATCH_RE.search(line)
            if m:
                d = m.groupdict()
                batches.append({
                    "epoch": int(d["ep"]),
                    "batch": int(d["batch"]),
                    "loss": float(d["loss"]),
                    "avg":  float(d["avg"]),
                    "L_att": float(d["latt"]),
                    "L_rep": float(d["lrep"]),
                    "L_var": float(d["lvar"]),
                    "w":     float(d["w"]),
                    "lr":    float(d["lr"]),
                    "time":  float(d["time"]),
                })
                continue
            m = VAL_RE.search(line)
            if m:
                d = m.groupdict()
                vals.append({
                    "epoch": int(d["ep"]),
                    "val_loss": float(d["vl"]),
                    "purity": float(d["pur"]),
                    "efficiency": float(d["eff"]),
                    "match_rate": float(d["match"]),
                    "noise_supp": float(d["ns"]),
                })
                continue
            m = SUM_RE.search(line)
            if m:
                d = m.groupdict()
                epoch_summaries.append({
                    "epoch": int(d["ep"]),
                    "train_loss": float(d["tl"]),
                    "val_loss":   float(d["vl"]),
                })
    return batches, vals, epoch_summaries


def smooth(arr, window=51):
    """Centered moving average. Uses np.convolve in 'valid' mode then pads."""
    if len(arr) < window:
        return np.asarray(arr, dtype=float)
    kernel = np.ones(window) / window
    sm = np.convolve(arr, kernel, mode="valid")
    pad = (len(arr) - len(sm))
    left = pad // 2
    right = pad - left
    return np.concatenate([
        np.full(left, sm[0]),
        sm,
        np.full(right, sm[-1]),
    ])


def make_global_step(batches):
    """Convert (epoch, batch) pairs to a monotonically-increasing global step.
    Assumes batches per epoch is the same modulo the last batch in each epoch."""
    # Group by epoch and find the max batch index per epoch
    max_bat = {}
    for b in batches:
        max_bat[b["epoch"]] = max(max_bat.get(b["epoch"], 0), b["batch"])
    # Use (max+1) for each completed epoch to compute global step
    epoch_offset = {}
    cum = 0
    for ep in sorted(max_bat):
        epoch_offset[ep] = cum
        cum += max_bat[ep] + 1
    return np.array([epoch_offset[b["epoch"]] + b["batch"] for b in batches])


def make_plot(batches, vals, summaries, out_path, title="v35 training curves"):
    if not batches:
        print("No batch data parsed — empty plot.")
        return

    step = make_global_step(batches)
    arrs = {k: np.array([b[k] for b in batches], dtype=float)
            for k in ("loss", "avg", "L_att", "L_rep", "L_var", "w", "lr", "time")}
    epochs = np.array([b["epoch"] for b in batches])

    val_ep = np.array([v["epoch"] for v in vals])
    val_loss_arr = np.array([v["val_loss"] for v in vals])
    val_pur = np.array([v["purity"] for v in vals])
    val_eff = np.array([v["efficiency"] for v in vals])
    val_match = np.array([v["match_rate"] for v in vals])
    val_ns = np.array([v["noise_supp"] for v in vals])

    sum_ep = np.array([s["epoch"] for s in summaries])
    sum_train = np.array([s["train_loss"] for s in summaries])
    sum_val = np.array([s["val_loss"] for s in summaries])

    # Best-by-match-rate epoch
    if len(val_match):
        best_match_idx = int(np.argmax(val_match))
        best_epoch = int(val_ep[best_match_idx])
        best_match = float(val_match[best_match_idx])
    else:
        best_epoch, best_match = -1, 0.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # --- panel 1: train + val loss vs epoch -------------------------------
    ax = axes[0, 0]
    ax.plot(step, smooth(arrs["loss"], 31), color="#888", linewidth=0.8,
            label="train loss (per-batch, smoothed)")
    ax2 = ax  # plot per-epoch values vs global step at end-of-epoch
    # locate end-of-epoch step for each summary epoch
    end_of_epoch_step = {}
    for ep in sum_ep:
        mask = epochs == ep
        if mask.any():
            end_of_epoch_step[int(ep)] = int(step[mask].max())
    sum_x = np.array([end_of_epoch_step.get(int(e), e) for e in sum_ep])
    val_x = np.array([end_of_epoch_step.get(int(e), e) for e in val_ep])
    ax2.plot(sum_x, sum_train, "o-", color="C0", linewidth=2, markersize=4,
             label="train loss (epoch avg)")
    ax2.plot(val_x, val_loss_arr, "s-", color="C3", linewidth=2, markersize=4,
             label="val loss")
    if best_epoch > 0:
        ax2.axvline(end_of_epoch_step.get(best_epoch, best_epoch),
                    color="green", linestyle="--", alpha=0.5,
                    label=f"best.pt @ epoch {best_epoch}")
    ax.set_xlabel("global step (batches)")
    ax.set_ylabel("loss")
    ax.set_title("Loss curves")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # --- panel 2: loss components -----------------------------------------
    ax = axes[0, 1]
    for key, color, label in [
        ("L_att", "C0", "L_att (attractive)"),
        ("L_rep", "C1", "L_rep (repulsive)"),
        ("L_var", "C2", "L_var (within-cluster var)"),
    ]:
        ax.plot(step, smooth(arrs[key], 51), color=color, linewidth=1.0, label=label)
    ax.set_xlabel("global step")
    ax.set_ylabel("component value")
    ax.set_title("Loss components (smoothed, window=51 batches)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # --- panel 3: val metrics per epoch -----------------------------------
    ax = axes[0, 2]
    ax.plot(val_ep, val_pur, "o-", color="C0", label="purity", markersize=4)
    ax.plot(val_ep, val_eff, "o-", color="C1", label="efficiency", markersize=4)
    ax.plot(val_ep, val_match, "o-", color="C3", label="match rate", markersize=4)
    ax.plot(val_ep, val_ns, "o-", color="C2", label="noise supp", markersize=4)
    if best_epoch > 0:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
        ax.text(best_epoch + 0.3, best_match - 0.04,
                f"best match={best_match:.3f}\n@ epoch {best_epoch}",
                fontsize=8, color="green")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.set_title("Validation metrics per epoch")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.3, 1.0)
    ax.grid(alpha=0.3)

    # --- panel 4: LR schedule + L_var weight ramp -------------------------
    ax = axes[1, 0]
    ax.plot(step, arrs["lr"], color="C0", linewidth=1.0, label="LR")
    ax.set_xlabel("global step")
    ax.set_ylabel("learning rate", color="C0")
    ax.tick_params(axis="y", labelcolor="C0")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="major")
    axb = ax.twinx()
    axb.plot(step, arrs["w"], color="C2", linewidth=1.2,
             linestyle="--", label="L_var weight")
    axb.set_ylabel("L_var weight", color="C2")
    axb.tick_params(axis="y", labelcolor="C2")
    axb.set_ylim(-0.02, max(0.35, arrs["w"].max() * 1.1))
    ax.set_title("LR schedule + L_var weight ramp")

    # --- panel 5: wall-clock per epoch ------------------------------------
    ax = axes[1, 1]
    completed = {int(s["epoch"]) for s in summaries}  # only fully-trained epochs
    epoch_times = []
    epoch_idx = []
    for ep in sorted(completed):
        mask = epochs == ep
        if not mask.any():
            continue
        # The 'time' field is the cumulative seconds from epoch start. Final
        # batch's 'time' is approximately the epoch's wall-clock duration.
        epoch_idx.append(ep)
        epoch_times.append(arrs["time"][mask].max() / 60.0)  # to minutes
    ax.bar(epoch_idx, epoch_times, color="C0", alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("wall-clock time (minutes)")
    ax.set_title(f"Per-epoch wall time (mean {np.mean(epoch_times):.1f} min)")
    ax.grid(alpha=0.3, axis="y")

    # --- panel 6: cumulative wall time ------------------------------------
    ax = axes[1, 2]
    cum_min = np.cumsum(epoch_times)
    ax.plot(epoch_idx, cum_min / 60.0, "o-", color="C0", linewidth=2,
            markersize=5, label="cumulative training time")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cumulative wall time (hours)")
    total_h = cum_min[-1] / 60.0 if len(cum_min) else 0.0
    ax.set_title(f"Cumulative training time ({total_h:.1f} h total)")
    ax.grid(alpha=0.3)
    if best_epoch > 0:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
        ax.text(best_epoch + 0.5, cum_min[-1] / 60.0 * 0.5,
                f"best.pt @ epoch {best_epoch}",
                fontsize=9, color="green", rotation=90, va="center")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    return {
        "n_batches": len(batches),
        "n_epochs": len(summaries),
        "best_epoch": best_epoch,
        "best_match_rate": best_match,
        "final_match_rate": float(val_match[-1]) if len(val_match) else None,
        "final_train_loss": float(sum_train[-1]) if len(sum_train) else None,
        "final_val_loss": float(sum_val[-1]) if len(sum_val) else None,
        "total_training_hours": float(total_h),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str,
                        default="/home/marko.cechovic/cgatr/v35-run.log")
    parser.add_argument("--output_dir", type=str,
                        default="eval_results/v35_training_curves")
    parser.add_argument("--title", type=str,
                        default="v35 training curves (35 epochs, 4× V100 32 GB DDP)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Parsing {args.log}")
    batches, vals, summaries = parse_log(args.log)
    print(f"  {len(batches)} batch lines, {len(vals)} val lines, "
          f"{len(summaries)} epoch summaries")

    out_png = os.path.join(args.output_dir, "v35_training_curves.png")
    summary = make_plot(batches, vals, summaries, out_png, title=args.title)

    out_json = os.path.join(args.output_dir, "v35_training_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
