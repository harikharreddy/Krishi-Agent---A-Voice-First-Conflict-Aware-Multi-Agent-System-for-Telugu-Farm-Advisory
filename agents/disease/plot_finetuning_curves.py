"""
Plots real training curves (loss + accuracy per epoch) from the PlantDoc
fine-tuning run that produced the currently-deployed Disease Agent
checkpoints -- for the paper's evaluation section.

Deliberately NOT a PlantVillage training curve (easy, near-saturated
dataset, converges to ~99%+ trivially and proves little about real-world
performance) -- these are the PlantDoc fine-tuning curves: train loss vs.
a genuine held-out internal validation split, on real field photos. Lower
peak accuracy than a PlantVillage curve would show, and that's the honest,
correct number to be showing.

Reads agents/disease/results/plantdoc_finetuned_results.json's
"training_history" (written by finetune_plantdoc.py), one curve per stage
model (crop classifier, tomato disease classifier, potato disease
classifier).
"""

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results", "plantdoc_finetuned_results.json")
OUT_DIR = os.path.join(HERE, "results")

STAGE_TITLES = {
    "stage1_crop": "Stage 1: Crop Classifier (Potato vs Tomato)",
    "stage2_tomato": "Stage 2: Tomato Disease Classifier",
    "stage2_potato": "Stage 2: Potato Disease Classifier",
}


def plot_stage(stage_key, history, ax_loss, ax_acc):
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_acc"] * 100 for h in history]

    ax_loss.plot(epochs, train_loss, marker="o", label="Train Loss", color="#c1622d")
    ax_loss.plot(epochs, val_loss, marker="o", label="Val Loss (PlantDoc, held out)", color="#3f6b35")
    ax_loss.set_title(STAGE_TITLES[stage_key])
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, val_acc, marker="o", color="#3f6b35")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val Accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(alpha=0.3)
    ax_acc.set_title(f"Final: {val_acc[-1]:.1f}% (best: {max(val_acc):.1f}%)")


def main():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)
    history = results["training_history"]

    stages = ["stage1_crop", "stage2_tomato", "stage2_potato"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        "Disease Agent fine-tuning on PlantDoc (real-world field photos) -- "
        "NOT PlantVillage (studio photos); see docs/evaluation_and_validation.md",
        fontsize=11,
    )

    for i, stage in enumerate(stages):
        plot_stage(stage, history[stage], axes[0, i], axes[1, i])

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "plantdoc_finetuning_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
