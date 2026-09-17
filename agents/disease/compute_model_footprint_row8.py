"""
Row 8 (ResNet18 cross-architecture baseline) compute footprint --
extends model_footprint_row2_row6_evidence.json's approach (ptflops,
untrained architecture matching the checkpoint's class count) to the
newly-added row 8 checkpoint, for the same lineage-table column.
"""

import json
import os

import torch.nn as nn
from ptflops import get_model_complexity_info
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")

IMG_SIZE = 224
NUM_CLASSES = 13


def main():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.eval()
    macs, params = get_model_complexity_info(
        m, (3, IMG_SIZE, IMG_SIZE), as_strings=False, print_per_layer_stat=False, verbose=False
    )
    flops = macs * 2
    print(f"Row 8 (ResNet18, flat 13-class): params={params:,}  MACs={macs:,}  FLOPs={flops:,} ({flops/1e9:.3f} GFLOPs)")

    evidence = {
        "metric": "Compute footprint (parameter count, FLOPs) for model lineage row 8 (ResNet18 cross-architecture baseline)",
        "method": (
            "ptflops (get_model_complexity_info), same method as model_footprint_row2_row6_evidence.json. "
            "Computed on the untrained architecture (torchvision resnet18, weights=None, fc head replaced "
            "with a 13-class Linear layer matching the checkpoint), input shape (3, 224, 224)."
        ),
        "params": params,
        "macs": macs,
        "flops": flops,
        "gflops": flops / 1e9,
        "context_note": (
            "ResNet18 has fewer parameters than row 2/5's combined 3-model hierarchical footprint "
            "(11.18M vs 12.04M) but MORE FLOPs per inference than EfficientNetB0's single-model "
            "footprint (3.65 GFLOPs vs 0.82 GFLOPs, row 1/6) -- EfficientNet's compound scaling is "
            "specifically designed for FLOP-efficiency per parameter, so this is expected, not anomalous. "
            "No real-world latency was measured for row 8 (no local deployment -- external Colab "
            "training/eval only), so this is a FLOPs-only figure, same honest gap already noted for "
            "row 6 in model_footprint_row2_row6_evidence.json."
        ),
    }

    out_path = os.path.join(EVIDENCE_DIR, "model_footprint_row8_evidence.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
