"""
Chart generation for the project-guide PDF evaluation report. Every
chart here is built from real, already-saved evidence data (JSON files
in docs/evidence/ and agents/disease/results/) -- nothing is estimated
or invented. Charts that already existed as saved images (confusion
matrices and reliability diagrams for rows 1/2/6, the deployed model's
fine-tuning curves) are NOT regenerated here; they're embedded directly
from agents/disease/results/ by the PDF builder.

Run once; outputs land in docs/report_assets/.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EVIDENCE = os.path.join(ROOT, "docs", "evidence")
RESULTS = os.path.join(ROOT, "agents", "disease", "results")
OUT = HERE

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

PRIMARY = "#2f5d4f"
ACCENT = "#a8791f"
WARN = "#b3452c"
NEUTRAL = "#6b6455"
GRID = "#e2e2e2"


def load(path):
    with open(os.path.join(EVIDENCE, path), encoding="utf-8") as f:
        return json.load(f)


def load_results(path):
    with open(os.path.join(RESULTS, path), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. Row 8 (ResNet18) confusion matrix -- generated fresh, no saved image existed
# ---------------------------------------------------------------------------
def row8_confusion_matrix():
    preds = load("row8_resnet18_plantdoc_perimage_predictions.json")
    classes = [
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
        "Tomato___Leaf_Mold", "Tomato___Mosaic_virus", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites", "Tomato___Target_Spot", "Tomato___Yellow_Leaf_Curl_Virus",
        "Tomato___healthy",
    ]
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    mat = np.zeros((n, n), dtype=float)
    counts = np.zeros(n, dtype=float)
    for r in preds:
        i = idx[r["true_class"]]
        j = idx[r["predicted_class"]]
        mat[i, j] += 1
        counts[i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.divide(mat, counts[:, None], out=np.zeros_like(mat), where=counts[:, None] > 0)

    short = [c.replace("Tomato___", "T_").replace("Potato___", "P_") for c in classes]

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Row 8 (ResNet18, zero-shot, PlantVillage-only) -- PlantDoc confusion matrix (row-normalized, n={int(counts.sum())})", fontsize=10)
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                         fontsize=7, color="white" if norm[i, j] > 0.5 else "#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row-normalized fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "row8_confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved row8_confusion_matrix.png")


# ---------------------------------------------------------------------------
# 2. Row 8 (ResNet18) training curves -- real per-epoch history, no image existed
# ---------------------------------------------------------------------------
def row8_training_curves():
    d = load("row8_resnet18_training_evidence.json")
    h = d["history"]
    epochs = range(1, len(h["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, h["train_loss"], color=PRIMARY, label="Train loss")
    axes[0].plot(epochs, h["val_loss"], color=ACCENT, label="Val loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(frameon=False)
    axes[0].set_title("Row 8 (ResNet18) -- PlantVillage loss")
    axes[0].grid(color=GRID, linewidth=0.6)

    axes[1].plot(epochs, h["train_acc"], color=PRIMARY, label="Train acc")
    axes[1].plot(epochs, h["val_acc"], color=ACCENT, label="Val acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)"); axes[1].legend(frameon=False, loc="lower right")
    axes[1].set_title(f"Row 8 (ResNet18) -- PlantVillage accuracy (best val {d['best_val_accuracy']:.2f}%)")
    axes[1].grid(color=GRID, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "row8_training_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved row8_training_curves.png")


# ---------------------------------------------------------------------------
# 3. Attractor-class prediction-distribution bar chart (row 8, real data)
# ---------------------------------------------------------------------------
def attractor_bias_chart():
    d = load("row8_resnet18_plantdoc_zeroshot_evidence.json")
    dist = d["prediction_distribution"]
    total = sum(dist.values())
    labels = list(dist.keys())
    short = [l.replace("Tomato___", "T_").replace("Potato___", "P_") for l in labels]
    vals = [dist[l] / total * 100 for l in labels]
    order = np.argsort(vals)[::-1]
    short = [short[i] for i in order]
    vals = [vals[i] for i in order]
    colors = [WARN if s in ("T_Late_blight", "T_Early_blight") else NEUTRAL for s in short]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(short, vals, color=colors)
    ax.set_ylabel("% of all 822 predictions")
    ax.set_title("Row 8 (ResNet18) zero-shot: what the model predicts, regardless of true label")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for b, v in zip(bars, vals):
        if v > 0.5:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8)
    ax.text(0.98, 0.95, "Tomato_Late_blight + Tomato_Early_blight = 72.9% of all predictions\n(true combined prevalence: 24.2%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=WARN,
            bbox=dict(boxstyle="round", fc="#f7e9df", ec=WARN, alpha=0.9))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "attractor_bias_row8.png"), dpi=150)
    plt.close(fig)
    print("Saved attractor_bias_row8.png")


# ---------------------------------------------------------------------------
# 4. K-fold CV per-fold accuracy chart
# ---------------------------------------------------------------------------
def kfold_chart():
    d = load("kfold_cv_row6_variance_evidence.json")
    folds = [f["fold"] for f in d["fold_results"]]
    accs = [f["plantdoc_zeroshot_accuracy"] * 100 for f in d["fold_results"]]
    mean = d["plantdoc_zeroshot_accuracy_mean"]
    std = d["plantdoc_zeroshot_accuracy_std"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar([f"Fold {f}" for f in folds], accs, color=PRIMARY, width=0.55)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha="center", fontsize=9)
    ax.axhline(mean, color=ACCENT, linestyle="--", linewidth=1.5, label=f"Mean = {mean:.2f}%")
    ax.axhspan(mean - std, mean + std, color=ACCENT, alpha=0.15, label=f"± 1 std ({std:.2f}pp)")
    ax.set_ylabel("PlantDoc zero-shot accuracy (%)")
    ax.set_title("5-fold stratified CV, EfficientNetB0 (row 6 recipe) -- fold-to-fold variance")
    ax.set_ylim(20, 26)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kfold_cv_chart.png"), dpi=150)
    plt.close(fig)
    print("Saved kfold_cv_chart.png")


# ---------------------------------------------------------------------------
# 5. Confidence-cutoff sweep + bootstrap win-rate chart
# ---------------------------------------------------------------------------
def cutoff_sweep_chart():
    d = load("metric6_confidence_calibration_evidence.json")
    sweep = d["cutoff_sweep_and_stability_check_2026-09-13"]
    table = sweep["sweep_table_0.30_to_0.70_step_0.05"]
    boot = sweep["bootstrap_stability_check"]["per_cutoff"]

    cutoffs = [r["cutoff"] for r in table]
    acc = [r["treat_now_accuracy"] * 100 for r in table]
    ns = [r["treat_now_n"] for r in table]
    ci_low = [b["bootstrap_95_CI"]["low"] * 100 for b in boot]
    ci_high = [b["bootstrap_95_CI"]["high"] * 100 for b in boot]
    win_rate = [b["win_rate"] * 100 for b in boot]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot(cutoffs, acc, "-o", color=PRIMARY, label="treat_now-bucket accuracy")
    ax.fill_between(cutoffs, ci_low, ci_high, color=PRIMARY, alpha=0.15, label="Bootstrap 95% CI")
    ax.axvline(0.7, color=WARN, linestyle="--", linewidth=1.3, label="Deployed cutoff (0.7)")
    for c, a, n in zip(cutoffs, acc, ns):
        ax.annotate(f"n={n}", (c, a), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center", color="#555")
    ax.set_xlabel("DISEASE_CONFIDENCE_CUTOFF"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs. cutoff, with bootstrap 95% CI")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(color=GRID, linewidth=0.6); ax.set_axisbelow(True)

    ax2 = axes[1]
    colors = [ACCENT if 0.50 <= c <= 0.65 else NEUTRAL for c in cutoffs]
    ax2.bar([str(c) for c in cutoffs], win_rate, color=colors)
    ax2.axvspan(-0.5, len(cutoffs) - 0.5, alpha=0)  # keep autoscale sane
    ax2.set_xlabel("DISEASE_CONFIDENCE_CUTOFF"); ax2.set_ylabel("Bootstrap win-rate (%)")
    ax2.set_title("Win-rate: fraction of 2,000 bootstrap resamples\nwhere this cutoff was the top performer")
    ax2.grid(axis="y", color=GRID, linewidth=0.6); ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cutoff_sweep_bootstrap.png"), dpi=150)
    plt.close(fig)
    print("Saved cutoff_sweep_bootstrap.png")


# ---------------------------------------------------------------------------
# 6. Per-speaker WER bar chart
# ---------------------------------------------------------------------------
def wer_chart():
    d = load("../../tests/audio/wer_eval/wer_results.json".replace("../../", ""))
    return


def wer_chart_v2():
    path = os.path.join(ROOT, "tests", "audio", "wer_eval", "wer_results.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    speakers = list(d["by_speaker"].keys())
    wers = [d["by_speaker"][s]["mean_wer"] * 100 for s in speakers]
    overall = d["overall_mean_wer"] * 100

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    bars = ax.bar(speakers, wers, color=PRIMARY, width=0.55)
    for b, v in zip(bars, wers):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.2f}%", ha="center", fontsize=9)
    ax.axhline(overall, color=ACCENT, linestyle="--", linewidth=1.5, label=f"Overall mean = {overall:.2f}%")
    ax.set_ylabel("Word Error Rate (%)")
    ax.set_title("ASR WER per speaker (n=16 each, 64 total)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wer_per_speaker.png"), dpi=150)
    plt.close(fig)
    print("Saved wer_per_speaker.png")


# ---------------------------------------------------------------------------
# 7. Per-stage latency breakdown chart
# ---------------------------------------------------------------------------
def latency_chart():
    d = load("metric3_latency_evidence.json")
    trials = d["trials"]
    stages = ["asr_s", "intent_router_s", "weather_agent_s", "price_agent_s",
              "disease_agent_s", "conflict_resolver_s", "intent_router_unload_s", "tts_s"]
    stage_labels = ["ASR", "Intent Router", "Weather Agent", "Price Agent",
                     "Disease Agent", "Conflict Resolver", "Router unload", "TTS"]
    stage_colors = ["#8a9078", "#2f5d4f", "#7a9e8e", "#a8b98c", "#5c8f7a", "#b7c9a8", "#c8cfc0", "#a8791f"]

    labels = [t["trial"] for t in trials]
    bottoms = np.zeros(len(trials))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for stage, slabel, color in zip(stages, stage_labels, stage_colors):
        vals = np.array([t["timing"].get(stage, 0.0) for t in trials])
        if vals.sum() == 0:
            continue
        ax.barh(labels, vals, left=bottoms, color=color, label=slabel)
        bottoms += vals

    ax.set_xlabel("Seconds"); ax.set_title("Per-stage latency, 6 real HTTP requests against the running backend")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "latency_breakdown.png"), dpi=150)
    plt.close(fig)
    print("Saved latency_breakdown.png")


# ---------------------------------------------------------------------------
# 8. McNemar's 2x2 contingency table visualization (one representative pair)
# ---------------------------------------------------------------------------
def mcnemar_contingency_chart():
    d = load("mcnemar_row8_resnet18_vs_zeroshot_evidence.json")
    pair = d["pairwise_results"][2]  # row8 vs row6, closest to significance
    ct = pair["contingency_table"]
    keys = list(ct.keys())
    vals = [[ct[keys[0]], ct[keys[1]]], [ct[keys[2]], ct[keys[3]]]]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(vals, cmap="Oranges")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Row 6 correct", "Row 6 wrong"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Row 8 correct", "Row 8 wrong"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, vals[i][j], ha="center", va="center", fontsize=16, fontweight="bold")
    ax.set_title(f"Row 8 vs. Row 6 -- 2×2 contingency table (n={pair['n_shared_images']})\n"
                 f"p={pair['p_value']:.3f}, {pair['test_variant_used'].split('(')[0].strip()}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mcnemar_contingency_row8_row6.png"), dpi=150)
    plt.close(fig)
    print("Saved mcnemar_contingency_row8_row6.png")


# ---------------------------------------------------------------------------
# 9. Intent Router per-intent accuracy chart
# ---------------------------------------------------------------------------
def intent_router_chart():
    d = load("metric_intent_router_per_intent_evidence.json")
    print("intent evidence keys:", list(d.keys()))


if __name__ == "__main__":
    row8_confusion_matrix()
    row8_training_curves()
    attractor_bias_chart()
    kfold_chart()
    cutoff_sweep_chart()
    wer_chart_v2()
    latency_chart()
    mcnemar_contingency_chart()
    intent_router_chart()
    print("\nAll charts generated in", OUT)
