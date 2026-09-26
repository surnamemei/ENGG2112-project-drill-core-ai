"""
Generate final-report figures from saved result CSVs.

No model training or re-evaluation happens here. Inputs:
  results/training_history.csv
  results/robustness_analysis/clean_per_class_metrics.csv
  results/robustness_analysis/noisy_per_class_metrics.csv
  results/robustness_analysis/validation_threshold_analysis.csv
  results/robustness_analysis/confidence_threshold_analysis.csv
  results/robustness_analysis/noisy_test_predictions.csv
      (from export_noisy_predictions.py)

Outputs are written to results/report_figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from PIL import Image


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = Path("results")
ROBUST_DIR = RESULTS_DIR / "robustness_analysis"
OUTPUT_DIR = RESULTS_DIR / "report_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCKED_THRESHOLD = 0.97
NOISY_BASE_ACCURACY = 0.9792857142857143


# ============================================================
# Style
# ============================================================

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9892"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def style_axes(ax: Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def short_class(name: str) -> str:
    # "5.Granite" -> "Granite"
    return name.split(".", 1)[1] if "." in name else name


def save(fig: plt.Figure, name: str) -> None:
    path = OUTPUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ============================================================
# Figure 1: training curves
# ============================================================

history = pd.read_csv(RESULTS_DIR / "training_history.csv")

# Best checkpoint = first epoch reaching the maximum validation
# accuracy (train_resnet18.py uses a strict ">" comparison).
best_epoch = int(
    history.loc[history["val_accuracy"].idxmax(), "epoch"]
)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

panels = [
    ("loss", "Cross-entropy loss", "(a) Loss"),
    ("accuracy", "Accuracy (%)", "(b) Accuracy"),
]

for ax, (key, ylabel, title) in zip(axes, panels):

    scale = 100.0 if key == "accuracy" else 1.0

    ax.plot(
        history["epoch"], history[f"train_{key}"] * scale,
        color=BLUE, linewidth=2, marker="o", markersize=4,
        label="Training",
    )
    ax.plot(
        history["epoch"], history[f"val_{key}"] * scale,
        color=ORANGE, linewidth=2, marker="o", markersize=4,
        label="Validation",
    )
    ax.axvline(best_epoch, color=NEUTRAL, linestyle="--", linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.set_xticks(history["epoch"])
    style_axes(ax)

axes[0].set_yscale("log")
axes[1].set_ylim(85, 100.5)
axes[1].annotate(
    f"Selected checkpoint\n(epoch {best_epoch})",
    xy=(best_epoch, 99.9), xytext=(best_epoch - 3.6, 92.5),
    color=INK_SECONDARY, fontsize=9,
    arrowprops={"arrowstyle": "-", "color": NEUTRAL, "linewidth": 0.8},
)
axes[1].legend(loc="lower right")

save(fig, "fig_training_curves.png")


# ============================================================
# Figure 2: per-class noisy precision / recall + clean F1
# ============================================================

def load_per_class(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df[~df.index.isin(["accuracy", "macro avg", "weighted avg"])]


clean_pc = load_per_class(ROBUST_DIR / "clean_per_class_metrics.csv")
noisy_pc = load_per_class(ROBUST_DIR / "noisy_per_class_metrics.csv")

classes = list(noisy_pc.index)
labels = [short_class(c) for c in classes]
y = np.arange(len(classes))[::-1]

fig, ax = plt.subplots(figsize=(7.2, 3.8))

for yi, cls in zip(y, classes):
    p = noisy_pc.loc[cls, "precision"] * 100
    r = noisy_pc.loc[cls, "recall"] * 100
    ax.plot([p, r], [yi, yi], color=GRID, linewidth=3, zorder=1)

ax.scatter(
    clean_pc.loc[classes, "f1-score"] * 100, y,
    marker="|", s=160, color=NEUTRAL, linewidths=2,
    label="Clean F1", zorder=2,
)
ax.scatter(
    noisy_pc.loc[classes, "precision"] * 100, y,
    s=55, color=BLUE, edgecolors="white", linewidths=1.2,
    label="Noisy precision", zorder=3,
)
ax.scatter(
    noisy_pc.loc[classes, "recall"] * 100, y,
    s=55, color=ORANGE, marker="D", edgecolors="white", linewidths=1.2,
    label="Noisy recall", zorder=3,
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlim(95, 100.4)
ax.set_xlabel("Score (%)  (axis starts at 95%)")
style_axes(ax)
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3)

save(fig, "fig_per_class_precision_recall.png")


# ============================================================
# Figure 3: noisy-test confidence distribution
# ============================================================

predictions = pd.read_csv(ROBUST_DIR / "noisy_test_predictions.csv")

correct_conf = predictions.loc[predictions["correct"], "confidence"]
wrong_conf = predictions.loc[~predictions["correct"], "confidence"]

bins = np.linspace(0.3, 1.0, 36)

fig, ax = plt.subplots(figsize=(7.2, 3.4))

ax.hist(
    correct_conf, bins=bins, color=BLUE, alpha=0.85,
    edgecolor="white", linewidth=0.8,
    label=f"Correct (n = {len(correct_conf)})",
)
ax.hist(
    wrong_conf, bins=bins, color=ORANGE, alpha=0.9,
    edgecolor="white", linewidth=0.8,
    label=f"Incorrect (n = {len(wrong_conf)})",
)
ax.axvline(LOCKED_THRESHOLD, color=INK, linestyle="--", linewidth=1)
ax.text(
    LOCKED_THRESHOLD - 0.01, 1500, "threshold 0.97\n< refer | accept >",
    ha="right", va="center", fontsize=9, color=INK_SECONDARY,
)

ax.set_yscale("log")
ax.set_xlabel("Softmax confidence (max class probability)")
ax.set_ylabel("Number of images (log scale)")
ax.legend(loc="upper left")
style_axes(ax)

save(fig, "fig_noisy_confidence_hist.png")


# ============================================================
# Figure 4: threshold trade-off (validation vs test)
# ============================================================

val_thr = pd.read_csv(ROBUST_DIR / "validation_threshold_analysis.csv")
test_thr = pd.read_csv(ROBUST_DIR / "confidence_threshold_analysis.csv")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

# (a) error capture vs referral rate
ax = axes[0]
for df, colour, marker, name in [
    (val_thr, BLUE, "o", "Noisy validation (selection)"),
    (test_thr, ORANGE, "D", "Noisy test (reporting)"),
]:
    ax.plot(
        df["referral_rate"] * 100, df["error_capture_rate"] * 100,
        color=colour, linewidth=2, marker=marker, markersize=5,
        markeredgecolor="white", label=name,
    )
    locked = df[np.isclose(df["threshold"], LOCKED_THRESHOLD)].iloc[0]
    ax.scatter(
        locked["referral_rate"] * 100, locked["error_capture_rate"] * 100,
        s=140, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4,
    )

ax.text(
    0.3, 95, "circled points: locked threshold t = 0.97",
    fontsize=8.5, color=INK_SECONDARY,
)
ax.set_xlabel("Referral rate (% of all images sent to review)")
ax.set_ylabel("Error capture (% of errors referred)")
ax.set_title("(a) Review workload vs errors caught", loc="left")
ax.set_ylim(0, 100)
ax.legend(loc="lower right", fontsize=8.5)
style_axes(ax)

# (b) accepted-set accuracy vs threshold
ax = axes[1]
ax.plot(
    val_thr["threshold"], val_thr["accepted_accuracy"] * 100,
    color=BLUE, linewidth=2, marker="o", markersize=5,
    markeredgecolor="white", label="Noisy validation",
)
ax.plot(
    test_thr["threshold"], test_thr["accepted_accuracy"] * 100,
    color=ORANGE, linewidth=2, marker="D", markersize=5,
    markeredgecolor="white", label="Noisy test",
)
ax.axhline(
    NOISY_BASE_ACCURACY * 100, color=NEUTRAL, linestyle=":", linewidth=1.2,
)
ax.text(
    0.61, NOISY_BASE_ACCURACY * 100 - 0.08,
    "test accuracy with no referral (97.93%)",
    fontsize=8.5, color=INK_SECONDARY, va="top",
)
ax.axvline(LOCKED_THRESHOLD, color=INK, linestyle="--", linewidth=1)
ax.set_xlabel("Confidence threshold t")
ax.set_ylabel("Accuracy of auto-accepted set (%)")
ax.set_title("(b) Accuracy of predictions that are kept", loc="left")
ax.set_ylim(97.4, 100)
ax.legend(loc="upper left", fontsize=8.5)
style_axes(ax)

fig.tight_layout()
save(fig, "fig_threshold_tradeoff.png")


# ============================================================
# Figure 5: high-confidence residual errors (accepted at t = 0.97)
# ============================================================

residual = (
    predictions[
        (~predictions["correct"])
        & (predictions["confidence"] >= LOCKED_THRESHOLD)
    ]
    .sort_values("confidence", ascending=False)
    .reset_index(drop=True)
)

n_cols = 5
n_rows = int(np.ceil(len(residual) / n_cols))

fig, axes_grid = plt.subplots(
    n_rows, n_cols, figsize=(2.0 * n_cols, 2.85 * n_rows)
)
flat_axes = np.atleast_1d(axes_grid).ravel()

for ax in flat_axes:
    ax.axis("off")

for ax, (_, row) in zip(flat_axes, residual.iterrows()):
    with Image.open(str(row["source_path"])) as img:
        ax.imshow(img.convert("RGB"))
    ax.set_title(
        f"True: {short_class(str(row['true_class']))}\n"
        f"Pred: {short_class(str(row['predicted_class']))}\n"
        f"conf = {float(row['confidence']):.3f}",
        fontsize=8.5, color=INK,
    )

fig.tight_layout()
save(fig, "fig_high_confidence_errors.png")

print(f"\nResidual high-confidence errors: {len(residual)}")
