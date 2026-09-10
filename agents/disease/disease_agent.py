"""
Disease Agent — hierarchical crop disease classifier.
Stage 1: crop classifier (Tomato/Potato). Stage 2: per-crop disease classifier.
Extracted from notebooks/phase1_disease_hierarchical_plantdoc.ipynb (Phase 1.4).

NOTE: Telugu disease-name labels below were reviewed by fluent Telugu
speakers on the team (Sep 2026, Phase 8 prep) and confirmed correct as-is --
same review step Phase 3.2's phrasing templates already went through.

NOTE (Sep 2026): checkpoints were originally trained on PlantVillage only
(studio-condition photos; 99.7% val accuracy there, but only ~23-28%
zero-shot on PlantDoc's real-world field photos). Since then, fine-tuned
on PlantDoc's own `train` split (see finetune_plantdoc.py) to close some of
that gap -- honest end-to-end accuracy on PlantDoc's held-out `test` split
went 28.2% -> 37.65%, and the earlier systematic over-prediction of
Tomato_Late_blight (49% of test predictions vs a true rate of 12%) is
resolved (now ~11%, matching the true distribution). See
results/plantdoc_finetuned_results.json for the full numbers.

Tomato__Target_Spot had ZERO real-world images anywhere in PlantDoc. A
first attempt (5 UF/IFAS extension photos) failed outright -- 0/2 on a
held-out check, and it cost accuracy elsewhere -- and was not promoted
(see results/plantdoc_target_spot_experiment.json). A second attempt added
15 more real field photos from the CC BY 4.0 "Tomato Leaf Dataset"
(Bangladesh tomato gardens, Mendeley DOI 10.17632/bpfd9cns5g.2), for 20
training images total. Result: 1/5 correct on a held-out check (up from
0/2), at a small additional cost elsewhere. This WAS promoted -- net
honest judgment call: some real ability to recognize Target_Spot beats
none, but it is still wrong 4 times out of 5 on held-out real photos. Not
"fixed," don't represent it as reliable.

Potato___healthy also had ZERO real-world images anywhere. Sourced 100
real field photos from the CC BY 4.0 "Potato Leaf (Healthy and Late
Blight)" dataset (Holeta, Ethiopia potato farm, Mendeley DOI
10.17632/v4w72bsts5.1) -- cleanly labeled by the dataset authors, unlike
Target_Spot's source. Result: 14/15 correct (93%) on a held-out check,
with high confidence (86-100%) -- a genuinely reliable fix, unlike
Target_Spot's. Collateral cost was minimal: 1 fewer correct image (out of
85) on PlantDoc's test split. Promoted without reservation.

Checkpoint lineage kept for comparison/rollback: checkpoints/
*_plantvillage_only.pt (original, no PlantDoc fine-tuning at all) ->
*_plantdoc_no_target_spot.pt (PlantDoc fine-tuned, before either extra-data
attempt) -> *_pre_potato_healthy.pt (adds Target_Spot, before healthy
potato) -> *_best.pt (current, adds Potato___healthy too).
"""

import os
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# Exact class orderings, pulled from the Phase 1 training run's printed output
# (ImageFolder alphabetical ordering — must match training exactly).
STAGE1_CLASSES = ['Potato', 'Tomato']

STAGE2_TOMATO_CLASSES = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]

STAGE2_POTATO_CLASSES = [
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
]

# Telugu labels — reviewed and confirmed by fluent-Telugu teammates (Sep 2026).
TELUGU_DISEASE_NAMES = {
    'Tomato_Bacterial_spot': 'బాక్టీరియల్ స్పాట్',
    'Tomato_Early_blight': 'ఎర్లీ బ్లైట్',
    'Tomato_Late_blight': 'లేట్ బ్లైట్',
    'Tomato_Leaf_Mold': 'లీఫ్ మోల్డ్',
    'Tomato_Septoria_leaf_spot': 'సెప్టోరియా లీఫ్ స్పాట్',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'స్పైడర్ మైట్స్',
    'Tomato__Target_Spot': 'టార్గెట్ స్పాట్',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'ఎల్లో లీఫ్ కర్ల్ వైరస్',
    'Tomato__Tomato_mosaic_virus': 'మొజాయిక్ వైరస్',
    'Tomato_healthy': 'ఆరోగ్యంగా ఉంది',
    'Potato___Early_blight': 'ఎర్లీ బ్లైట్',
    'Potato___Late_blight': 'లేట్ బ్లైట్',
    'Potato___healthy': 'ఆరోగ్యంగా ఉంది',
}

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_models_cache = {}


def _load_stage_model(checkpoint_path: str, num_classes: int):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    m.load_state_dict(torch.load(checkpoint_path, map_location=device))
    m = m.to(device)
    m.eval()
    return m


def _get_models():
    """Lazy-load all 3 stage models once, cache in memory."""
    if not _models_cache:
        _models_cache["stage1"] = _load_stage_model(
            os.path.join(CHECKPOINT_DIR, "stage1_crop_best.pt"), len(STAGE1_CLASSES)
        )
        _models_cache["stage2_tomato"] = _load_stage_model(
            os.path.join(CHECKPOINT_DIR, "stage2_tomato_best.pt"), len(STAGE2_TOMATO_CLASSES)
        )
        _models_cache["stage2_potato"] = _load_stage_model(
            os.path.join(CHECKPOINT_DIR, "stage2_potato_best.pt"), len(STAGE2_POTATO_CLASSES)
        )
    return _models_cache


def predict_disease(image_path: str) -> dict:
    """Main entry point. Returns advice dict in the shared agent schema.
    Confidence is the raw 0-1 softmax score, per the project guide's schema
    note for the Disease Agent (other agents use High/Medium/Low)."""
    try:
        img = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as e:
        return {
            "answer": "ఫోటో చదవడంలో సమస్య వచ్చింది.",
            "confidence": 0.0,
            "reason_for_confidence": f"Image load failed: {e}",
            "source_freshness": "unavailable",
        }

    img_tensor = val_transform(img).unsqueeze(0).to(device)
    m = _get_models()

    with torch.no_grad():
        stage1_out = m["stage1"](img_tensor)
        stage1_probs = F.softmax(stage1_out, dim=1)
        stage1_idx = stage1_probs.argmax(1).item()
        pred_crop = STAGE1_CLASSES[stage1_idx]
        stage1_conf = stage1_probs[0, stage1_idx].item()

        if pred_crop == "Tomato":
            stage2_out = m["stage2_tomato"](img_tensor)
            stage2_classes = STAGE2_TOMATO_CLASSES
        else:
            stage2_out = m["stage2_potato"](img_tensor)
            stage2_classes = STAGE2_POTATO_CLASSES

        stage2_probs = F.softmax(stage2_out, dim=1)
        stage2_idx = stage2_probs.argmax(1).item()
        pred_class = stage2_classes[stage2_idx]
        stage2_conf = stage2_probs[0, stage2_idx].item()

    telugu_label = TELUGU_DISEASE_NAMES.get(pred_class, pred_class)
    overall_confidence = stage1_conf * stage2_conf

    answer = f"{pred_crop} ఆకులో '{telugu_label}' కనిపిస్తోంది."

    reason = (
        f"Stage-1 crop={pred_crop} (conf {stage1_conf:.2f}), "
        f"Stage-2 disease={pred_class} (conf {stage2_conf:.2f})."
    )

    retrieved_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return {
        "answer": answer,
        "confidence": overall_confidence,
        "reason_for_confidence": reason,
        "source_freshness": f"Hierarchical EfficientNetB0 (PlantVillage-trained, PlantDoc-finetuned), predicted {retrieved_at}",
        "predicted_class": pred_class,
        "predicted_crop": pred_crop,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python disease_agent.py <path_to_leaf_image>")
    else:
        result = predict_disease(sys.argv[1])
        for k, v in result.items():
            print(f"{k}: {v}")
