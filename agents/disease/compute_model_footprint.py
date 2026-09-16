"""
Phase 8.1, model lineage compute-footprint extension (2026-09-16):
parameter count and FLOPs for the two zero-shot checkpoints with formal
confusion-matrix/ECE evidence (model_lineage.md rows 2 and 6), using
ptflops (the only one of torchinfo/ptflops available in this environment
-- torchinfo was not installed; ptflops was installed for this script).

Row 2 (hierarchical): 3 separate EfficientNetB0 models (stage1 crop-ID,
stage2 tomato-disease, stage2 potato-disease) -- reports each
individually and the combined total, since a real inference call for a
tomato photo runs stage1 + stage2_tomato (2 of 3 models), not all 3.

Row 6 (v2, flat): 1 EfficientNetB0 model, single 13-class head.

Architecture-sharing note, not re-measured redundantly: row 1 (flat
baseline) uses the identical architecture family as row 6 (single flat
13-class EfficientNetB0) -- same param count and FLOPs by construction,
since these depend on architecture and class count, not trained weight
values. Row 5 (deployed) uses the identical architecture family as row 2
(hierarchical, same 3-model stage1/stage2 split) -- same param count and
FLOPs, and its ALREADY-MEASURED real latency
(agents/disease/compute_full_metrics.py, 240.2ms mean, 4.16 img/s
throughput) is architecturally representative of row 2 too, restated
here rather than re-measured, since latency depends on the compute graph
executed, not which weights are loaded into it.
"""

import json
import os
import sys

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from agents.disease.disease_agent import STAGE1_CLASSES, STAGE2_TOMATO_CLASSES, STAGE2_POTATO_CLASSES, IMG_SIZE

EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")


def build_model(num_classes):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    m.eval()
    return m


def measure(model, label):
    macs, params = get_model_complexity_info(
        model, (3, IMG_SIZE, IMG_SIZE), as_strings=False, print_per_layer_stat=False, verbose=False
    )
    flops = macs * 2  # ptflops reports MACs; 1 MAC = 2 FLOPs (1 mult + 1 add), the standard convention
    print(f"{label}: params={params:,}  MACs={macs:,}  FLOPs={flops:,} ({flops/1e9:.3f} GFLOPs)")
    return {"params": params, "macs": macs, "flops": flops}


def main():
    # Row 6: v2 flat 13-class checkpoint (same architecture as row 1's flat_baseline_best.pt)
    row6_model = build_model(13)
    row6_stats = measure(row6_model, "Row 6 (v2, flat 13-class) / Row 1 (flat baseline, identical architecture)")

    # Row 2: hierarchical 3-model checkpoint (same architecture as row 5's deployed stage1/2_*_best.pt)
    stage1_model = build_model(len(STAGE1_CLASSES))
    stage2_tomato_model = build_model(len(STAGE2_TOMATO_CLASSES))
    stage2_potato_model = build_model(len(STAGE2_POTATO_CLASSES))

    stage1_stats = measure(stage1_model, "Row 2/5 stage1 (crop ID, 2-class)")
    stage2_tomato_stats = measure(stage2_tomato_model, "Row 2/5 stage2_tomato (10-class)")
    stage2_potato_stats = measure(stage2_potato_model, "Row 2/5 stage2_potato (3-class)")

    row2_combined_all3 = {
        "params": stage1_stats["params"] + stage2_tomato_stats["params"] + stage2_potato_stats["params"],
        "flops": stage1_stats["flops"] + stage2_tomato_stats["flops"] + stage2_potato_stats["flops"],
    }
    # A real inference call only ever runs stage1 + ONE of stage2_tomato/stage2_potato (2 of 3
    # models), never all 3 -- this is what compute_full_metrics.py's real, measured 240.2ms
    # latency actually reflects. Reporting both the "3 models loaded in memory" total (for
    # footprint/checkpoint-size purposes) and the "2 models per real inference call" total (for
    # a FLOPs-per-inference comparison against row 6's single-model number) rather than
    # conflating them.
    row2_per_inference_2of3 = {
        "params_loaded_in_memory": row2_combined_all3["params"],  # all 3 loaded regardless
        "flops_tomato_path": stage1_stats["flops"] + stage2_tomato_stats["flops"],
        "flops_potato_path": stage1_stats["flops"] + stage2_potato_stats["flops"],
    }

    print()
    print(f"Row 2/5 combined (all 3 models loaded): params={row2_combined_all3['params']:,}  "
          f"FLOPs if all 3 ran={row2_combined_all3['flops']:,}")
    print(f"Row 2/5 per real inference (stage1 + stage2_tomato): FLOPs={row2_per_inference_2of3['flops_tomato_path']:,} "
          f"({row2_per_inference_2of3['flops_tomato_path']/1e9:.3f} GFLOPs)")
    print(f"Row 2/5 per real inference (stage1 + stage2_potato): FLOPs={row2_per_inference_2of3['flops_potato_path']:,} "
          f"({row2_per_inference_2of3['flops_potato_path']/1e9:.3f} GFLOPs)")

    evidence = {
        "metric": "Compute footprint (parameter count, FLOPs) for model lineage rows 2 and 6, per external review request",
        "method": (
            "ptflops (get_model_complexity_info), the only one of torchinfo/ptflops available in this "
            "environment -- torchinfo was not installed; ptflops was installed for this script. Computed on the "
            "untrained architecture (weights=None, matching each checkpoint's actual class-count/head "
            "configuration) -- params and FLOPs depend on architecture and input shape, not trained weight "
            "values, so this is identical to measuring the actual loaded checkpoint. Input shape (3, 224, 224), "
            "matching IMG_SIZE used throughout agents/disease/disease_agent.py."
        ),
        "row6_v2_flat_13class": row6_stats,
        "row1_flat_baseline_NOTE": (
            "Not separately measured -- row 1 (flat_baseline_best.pt) uses the identical architecture "
            "(single EfficientNetB0, 13-class flat head) as row 6, so its params/FLOPs are IDENTICAL to "
            "row6_v2_flat_13class above by construction, not independently computed."
        ),
        "row2_hierarchical_per_stage": {
            "stage1_crop_id_2class": stage1_stats,
            "stage2_tomato_10class": stage2_tomato_stats,
            "stage2_potato_3class": stage2_potato_stats,
        },
        "row2_hierarchical_combined": row2_combined_all3,
        "row2_hierarchical_per_real_inference": row2_per_inference_2of3,
        "row5_deployed_NOTE": (
            "Not separately measured -- row 5 (the deployed stage1/2_*_best.pt checkpoints) uses the "
            "identical architecture as row 2 (same 3-model hierarchical split, same per-stage class "
            "counts), so its params/FLOPs are IDENTICAL to the row2_hierarchical_* figures above by "
            "construction. Row 5's REAL measured latency/throughput (already computed, not re-measured "
            "here) is pulled in directly: mean inference latency 240.2ms (std 20.7ms, n=85), throughput "
            "4.16 img/sec -- see agents/disease/compute_full_metrics.py and "
            "evaluation_and_validation.md's Precision/recall/F1 subsection. This latency figure is "
            "architecturally representative of row 2 as well (same compute graph), restated rather than "
            "re-measured, since row 2 was never deployed/latency-profiled independently."
        ),
        "honest_gaps": [
            "FLOPs computed on the architecture in isolation, not inside a live inference call -- does "
            "not include preprocessing (resize/normalize) or postprocessing (softmax) overhead, which "
            "the measured 240.2ms latency figure for row 5 DOES include end-to-end. FLOPs and latency "
            "are therefore complementary, not directly convertible into each other from this data alone.",

            "Row 6's real measured latency/throughput was never separately profiled (only row 5's was, "
            "via compute_full_metrics.py) -- reporting row 6's FLOPs/params without a matching real-world "
            "latency number for the same single-model architecture; a single-model call would be "
            "expected to be faster than row 5's 2-model-per-inference call by rough proportion of FLOPs, "
            "but this is an inference from the FLOPs ratio, not an independent measurement.",
        ],
    }

    out_path = os.path.join(EVIDENCE_DIR, "model_footprint_row2_row6_evidence.json")
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
