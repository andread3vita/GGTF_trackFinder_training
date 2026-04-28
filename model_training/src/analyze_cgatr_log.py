"""Preliminary analysis of a CGATr training run from its stdout log.

Scrapes per-epoch train/val numbers and per-batch component losses from
a log file produced by ``train_cgatr_parquet.py`` and writes plots plus a
short Markdown summary. CPU-only and read-only — safe to run while the
training is in progress.

Optionally takes a `--baseline_log` for head-to-head comparison plots
(useful when iterating on a new loss / architecture against a previous
run as a reference). The baseline log is parsed without component-loss
fields (which only the new run is expected to emit).

Outputs (under ``--out_dir``):
- ``training_curves.png``  — train and val loss curves
- ``val_metrics.png``      — purity / efficiency / match rate / noise suppression
- ``component_losses.png`` — OC loss components per epoch (current run)
- ``preview_report.md``    — Markdown summary with the latest epoch table

Usage:
    python src/analyze_cgatr_log.py \\
        --log run.log \\
        --total_epochs 40 \\
        --out_dir eval_results/cgatr_preview
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_SUMMARY_RE = re.compile(
    r"^Epoch (\d+) Summary: Train Loss=([\d.]+), Val Loss=([\d.]+)"
)
VAL_METRICS_RE = re.compile(
    r"Epoch (\d+) \| Val Loss: ([\d.]+) \| Purity: ([\d.]+) \| "
    r"Efficiency: ([\d.]+) \| Match Rate: ([\d.]+) \| Noise Supp: ([\d.]+)"
)
COMP_MEANS_RE = re.compile(
    r"Epoch (\d+) component means: L_att=([\d.]+) L_rep=([\d.]+) "
    r"L_beta_sig=([\d.]+) L_beta_noise=([\d.]+) L_beta_suppress=([\d.]+) "
    r"L_var=([\d.]+) \(weight=([\d.]+)\)"
)
BATCH_TIME_RE = re.compile(
    r"Epoch (\d+) \| Batch (\d+) \|.*Time ([\d.]+)s"
)


def parse_log(path, has_components=True):
    """Return dict with per-epoch arrays."""
    if not os.path.exists(path):
        return None
    epochs_summary = {}
    epochs_val = {}
    epochs_comp = {}
    last_batch_per_epoch = {}

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            m = EPOCH_SUMMARY_RE.match(line)
            if m:
                ep = int(m.group(1))
                epochs_summary[ep] = (float(m.group(2)), float(m.group(3)))
                continue
            m = VAL_METRICS_RE.search(line)
            if m:
                ep = int(m.group(1))
                epochs_val[ep] = {
                    "val_loss": float(m.group(2)),
                    "purity": float(m.group(3)),
                    "efficiency": float(m.group(4)),
                    "match_rate": float(m.group(5)),
                    "noise_supp": float(m.group(6)),
                }
                continue
            if has_components:
                m = COMP_MEANS_RE.search(line)
                if m:
                    ep = int(m.group(1))
                    epochs_comp[ep] = {
                        "L_att": float(m.group(2)),
                        "L_rep": float(m.group(3)),
                        "L_beta_sig": float(m.group(4)),
                        "L_beta_noise": float(m.group(5)),
                        "L_beta_suppress": float(m.group(6)),
                        "L_var": float(m.group(7)),
                        "var_weight": float(m.group(8)),
                    }
                    continue
            m = BATCH_TIME_RE.search(line)
            if m:
                ep = int(m.group(1))
                bt = int(m.group(2))
                t = float(m.group(3))
                if ep not in last_batch_per_epoch or bt > last_batch_per_epoch[ep][0]:
                    last_batch_per_epoch[ep] = (bt, t)

    return {
        "summary": epochs_summary,
        "val": epochs_val,
        "comp": epochs_comp,
        "last_batch": last_batch_per_epoch,
    }


def make_arrays(d, key, sub=None):
    eps, vals = [], []
    for ep in sorted(d.keys()):
        if sub is None:
            v = d[ep][key] if isinstance(d[ep], dict) else d[ep]
        else:
            v = d[ep].get(sub, None)
            if v is None:
                continue
        eps.append(ep)
        vals.append(v)
    return np.array(eps), np.array(vals)


def plot_training_curves(baseline, current, out_path,
                          baseline_label="baseline", current_label="current"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(axes, [0, 1], ["Train Loss", "Val Loss"]):
        if baseline:
            eps = sorted(baseline["summary"].keys())
            vs = [baseline["summary"][e][key] for e in eps]
            ax.plot(eps, vs, "o-", label=baseline_label, color="#888")
        if current:
            eps = sorted(current["summary"].keys())
            vs = [current["summary"][e][key] for e in eps]
            ax.plot(eps, vs, "o-", label=current_label, color="#c0392b")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Training and validation loss", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_val_metrics(baseline, current, out_path,
                      baseline_label="baseline", current_label="current"):
    metrics = [("purity", "Purity"), ("efficiency", "Efficiency"),
               ("match_rate", "Match rate (training-time greedy eval)"),
               ("noise_supp", "Noise suppression (β<0.1)")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (k, title) in zip(axes.flat, metrics):
        if baseline:
            eps, vs = make_arrays(baseline["val"], None, sub=k)
            ax.plot(eps, vs, "o-", label=baseline_label, color="#888")
        if current:
            eps, vs = make_arrays(current["val"], None, sub=k)
            ax.plot(eps, vs, "o-", label=current_label, color="#c0392b")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(
        "Validation metrics per epoch (training-time greedy eval, td=0.2, tbeta=0.1)",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_components(current, out_path):
    if not current or not current["comp"]:
        return
    eps = sorted(current["comp"].keys())
    keys = ["L_att", "L_rep", "L_beta_sig", "L_beta_noise",
            "L_beta_suppress", "L_var"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#c0392b"]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    for k, c in zip(keys, colors):
        vs = [current["comp"][e][k] for e in eps]
        ax1.plot(eps, vs, "o-", label=k, color=c, lw=1.6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss component (train, mean over batches)")
    ax1.set_yscale("log")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(loc="upper right")

    ax2 = ax1.twinx()
    vw = [current["comp"][e]["var_weight"] for e in eps]
    ax2.plot(eps, vw, "k--", lw=1.4, label="var_weight")
    ax2.set_ylabel("var_weight (warmup)")
    ax2.set_ylim(-0.02, max(vw) * 1.1 + 0.02 if vw else 1.0)
    ax2.legend(loc="upper left")

    fig.suptitle("OC loss components per epoch (training set)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def estimate_eta(current, total_epochs, batches_per_epoch=5000):
    """Estimate per-epoch wall clock + ETA from last_batch timestamps."""
    if not current:
        return None
    lb = current["last_batch"]
    if not lb:
        return None
    eps_sorted = sorted(lb.keys())
    if len(eps_sorted) < 1:
        return None
    fully_done = [e for e in eps_sorted if e in current["summary"]]
    per_epoch = lb[fully_done[-1]][1] if fully_done else float(batches_per_epoch * 2.0)
    last_ep = eps_sorted[-1]
    last_b, last_t = lb[last_ep]
    if last_b > 100:
        s_per_batch = last_t / max(last_b, 1)
        per_epoch = batches_per_epoch * s_per_batch
    remaining_in_current = max(0, batches_per_epoch - last_b) * (per_epoch / batches_per_epoch)
    remaining_full_epochs = max(0, total_epochs - last_ep) * per_epoch
    eta_s = remaining_in_current + remaining_full_epochs
    return {
        "current_epoch": last_ep,
        "current_batch": last_b,
        "per_epoch_s": per_epoch,
        "eta_remaining_s": eta_s,
        "eta_remaining_h": eta_s / 3600.0,
    }


def write_preview_report(baseline, current, eta, out_md, total_epochs,
                          baseline_label="baseline", current_label="current"):
    lines = []
    L = lines.append
    L("# CGATr training preliminary analysis\n")
    L("_Auto-generated. Reflects the log up to the latest epoch parsed._\n")

    if eta:
        L("## Status\n")
        L(f"- Current epoch: **{eta['current_epoch']}/{total_epochs}** "
          f"(batch {eta['current_batch']})")
        L(f"- Per-epoch wall clock: **{eta['per_epoch_s']/3600:.2f} h**")
        L(f"- ETA to finish: **{eta['eta_remaining_h']:.1f} h** remaining")
        L("")

    if current and current["val"] and baseline and baseline["val"]:
        L(f"## {baseline_label} vs {current_label} — validation per epoch\n")
        L(f"| Epoch | {baseline_label} Val | {baseline_label} Pur | {baseline_label} Eff | "
          f"{baseline_label} Match | {baseline_label} Noise | "
          f"{current_label} Val | {current_label} Pur | {current_label} Eff | "
          f"{current_label} Match | {current_label} Noise |")
        L("|-------|---------|---------|---------|-----------|-----------|---------|---------|---------|-----------|-----------|")
        common = sorted(set(baseline["val"]) & set(current["val"]))
        if common:
            step = max(1, len(common) // 25)
            picks = list(common[::step])
            if common[-1] not in picks:
                picks.append(common[-1])
        else:
            picks = []
        for ep in picks:
            a = baseline["val"][ep]
            b = current["val"][ep]
            L(f"| {ep} | {a['val_loss']:.4f} | {a['purity']:.3f} | {a['efficiency']:.3f} | "
              f"{a['match_rate']:.3f} | {a['noise_supp']:.3f} | "
              f"{b['val_loss']:.4f} | {b['purity']:.3f} | {b['efficiency']:.3f} | "
              f"{b['match_rate']:.3f} | {b['noise_supp']:.3f} |")
        L("")

        if common:
            ep = common[-1]
            a = baseline["val"][ep]
            b = current["val"][ep]
            L(f"### Latest common epoch ({ep})\n")
            L(f"- Val loss: {baseline_label} {a['val_loss']:.4f} -> "
              f"{current_label} {b['val_loss']:.4f} "
              f"({b['val_loss']-a['val_loss']:+.4f})")
            L(f"- Purity:   {baseline_label} {a['purity']:.3f} -> "
              f"{current_label} {b['purity']:.3f} "
              f"({b['purity']-a['purity']:+.3f})")
            L(f"- Efficiency: {baseline_label} {a['efficiency']:.3f} -> "
              f"{current_label} {b['efficiency']:.3f} "
              f"({b['efficiency']-a['efficiency']:+.3f})")
            L(f"- Match rate: {baseline_label} {a['match_rate']:.3f} -> "
              f"{current_label} {b['match_rate']:.3f} "
              f"({b['match_rate']-a['match_rate']:+.3f})")
            L(f"- Noise supp.: {baseline_label} {a['noise_supp']:.3f} -> "
              f"{current_label} {b['noise_supp']:.3f} "
              f"({b['noise_supp']-a['noise_supp']:+.3f})")
            L("")

    if current and current["comp"]:
        L(f"## {current_label} — OC loss components\n")
        L("| Epoch | var_weight | L_att | L_rep | L_beta_sig | L_beta_noise | L_beta_suppress | L_var |")
        L("|-------|-----------:|------:|------:|-----------:|-------------:|----------------:|------:|")
        eps = sorted(current["comp"].keys())
        step = max(1, len(eps) // 12)
        epicks = list(eps[::step])
        if eps and eps[-1] not in epicks:
            epicks.append(eps[-1])
        for ep in epicks:
            c = current["comp"][ep]
            L(f"| {ep} | {c['var_weight']:.2f} | {c['L_att']:.3f} | {c['L_rep']:.3f} | "
              f"{c['L_beta_sig']:.3f} | {c['L_beta_noise']:.3f} | "
              f"{c['L_beta_suppress']:.3f} | {c['L_var']:.3f} |")
        L("")
        first = current["comp"][eps[0]]
        last = current["comp"][eps[-1]]
        L(f"From epoch {eps[0]} -> {eps[-1]}: "
          f"L_var dropped {first['L_var']:.3f} -> {last['L_var']:.3f} "
          f"({last['L_var']/max(first['L_var'],1e-6):.2f}x of starting value), "
          f"L_att {first['L_att']:.3f} -> {last['L_att']:.3f}, "
          f"L_beta_sig {first['L_beta_sig']:.3f} -> {last['L_beta_sig']:.3f}.\n")

    L("## Plots\n")
    L("- `training_curves.png` — train and val loss curves.")
    L("- `val_metrics.png` — purity, efficiency, match rate, noise suppression.")
    L("- `component_losses.png` — OC-loss components with var_weight overlay.")

    with open(out_md, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="run.log",
                        help="Stdout log of the current training run.")
    parser.add_argument("--baseline_log", default=None,
                        help="Optional stdout log of a previous/baseline run "
                             "for head-to-head comparison plots.")
    parser.add_argument("--baseline_label", default="baseline",
                        help="Label for the baseline log in plots/tables.")
    parser.add_argument("--current_label", default="current",
                        help="Label for the current log in plots/tables.")
    parser.add_argument("--total_epochs", type=int, default=40)
    parser.add_argument("--batches_per_epoch", type=int, default=5000,
                        help="Approximate number of batches per epoch (used for ETA only).")
    parser.add_argument("--out_dir", default="eval_results/cgatr_preview")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Parsing {args.log} ...")
    current = parse_log(args.log, has_components=True)
    print(f"  summary epochs: {len(current['summary']) if current else 0}")
    print(f"  val epochs:     {len(current['val']) if current else 0}")
    print(f"  comp epochs:    {len(current['comp']) if current else 0}")

    baseline = None
    if args.baseline_log:
        print(f"Parsing {args.baseline_log} ...")
        baseline = parse_log(args.baseline_log, has_components=False)
        print(f"  summary epochs: {len(baseline['summary']) if baseline else 0}")
        print(f"  val epochs:     {len(baseline['val']) if baseline else 0}")

    eta = estimate_eta(current, args.total_epochs, args.batches_per_epoch)
    if eta:
        print(f"\nETA: epoch {eta['current_epoch']} batch {eta['current_batch']}, "
              f"~{eta['per_epoch_s']/3600:.2f} h/epoch, "
              f"~{eta['eta_remaining_h']:.1f} h remaining.")

    print("\nWriting plots ...")
    plot_training_curves(baseline, current,
                          os.path.join(args.out_dir, "training_curves.png"),
                          args.baseline_label, args.current_label)
    plot_val_metrics(baseline, current,
                      os.path.join(args.out_dir, "val_metrics.png"),
                      args.baseline_label, args.current_label)
    plot_components(current, os.path.join(args.out_dir, "component_losses.png"))

    print("Writing markdown report ...")
    write_preview_report(baseline, current, eta,
                          os.path.join(args.out_dir, "preview_report.md"),
                          args.total_epochs,
                          args.baseline_label, args.current_label)

    summary = {
        "current": {
            "n_epoch_summary": len(current["summary"]) if current else 0,
            "n_val_epoch": len(current["val"]) if current else 0,
            "latest_val": current["val"][max(current["val"])] if current and current["val"] else None,
            "latest_comp": current["comp"][max(current["comp"])] if current and current["comp"] else None,
        },
        "baseline_latest_val": (baseline["val"][max(baseline["val"])]
                                if baseline and baseline["val"] else None),
        "eta": eta,
    }
    with open(os.path.join(args.out_dir, "preview_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Done. Output: {args.out_dir}/")


if __name__ == "__main__":
    main()
