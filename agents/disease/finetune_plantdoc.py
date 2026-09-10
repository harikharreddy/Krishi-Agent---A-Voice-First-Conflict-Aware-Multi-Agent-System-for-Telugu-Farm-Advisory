"""
Fine-tune the 3 hierarchical Disease Agent checkpoints (stage1 crop,
stage2 tomato, stage2 potato) on PlantDoc's own `train` split -- real-world
field photos, never used for training before (see plantdoc_confidence_eval.py's
"Zero-shot evaluation" note). Goal: close some of the 99.7% (PlantVillage,
studio photos) vs ~22-23% (PlantDoc, real-world photos) accuracy gap.

Starts from the existing PlantVillage-trained checkpoints (not from scratch)
and only unfreezes the classifier head + the backbone's last block, at a low
learning rate, for a small number of epochs -- standard conservative
transfer-learning recipe to adapt to the new (real-world) domain without
throwing away what the model already learned.

HONEST LIMITATION, not hidden: PlantDoc has ZERO images (train or test) for
Potato___healthy and Tomato__Target_Spot, and only 2 train / 0 test images
for Tomato_Spider_mites_Two_spotted_spider_mite. Those 3 classes get no real
fine-tuning signal and can't be validated against real-world photos either
way -- this script excludes Spider_mites from stage2_tomato's fine-tuning
set (2 images is not a training signal) but can't protect Potato___healthy
or Target_Spot from indirect softmax pressure during fine-tuning, since
that risk exists for every unrepresented class in a shared-softmax head.
Conservative hyperparameters (frozen backbone, low LR, few epochs) are
specifically chosen to minimize -- not eliminate -- that risk.

PlantDoc's `train` split is used ONLY for fine-tuning; `test` is held out
completely and used only by the separate before/after evaluation this
script runs at the end, so the reported PlantDoc accuracy stays honest.

Writes NEW checkpoint files (stage*_finetuned.pt) -- does not overwrite the
original stage*_best.pt files, so this is fully reversible.
"""

import copy
import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from disease_agent import (
    STAGE1_CLASSES,
    STAGE2_TOMATO_CLASSES,
    STAGE2_POTATO_CLASSES,
    IMG_SIZE,
    CHECKPOINT_DIR,
)

PLANTDOC_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "plantdoc_raw")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SEED = 42

# Tomato__Target_Spot has ZERO images anywhere in PlantDoc (train or test).
# Two real-world sources used instead, as model-training input only (not
# redistributed/displayed), for this non-commercial capstone project:
#
# 1. 7 images cropped from UF/IFAS EDIS publication PP351 ("Target Spot of
#    Tomato in Florida", (c) University of Florida, https://ask.ifas.ufl.edu/pp351).
# 2. 20 images from the CC BY 4.0 "Tomato Leaf Dataset" (Imtiaz et al.,
#    American International University Bangladesh, Mendeley Data DOI
#    10.17632/bpfd9cns5g.2) -- real field photos from tomato gardens in
#    Bangladesh. That dataset ships YOLO bounding-box labels but no
#    class-name file; class index 0 was identified as Target Spot by
#    downloading its labeled images and visually confirming the
#    characteristic concentric-ring lesion against the UF/IFAS reference
#    photos. 18 of 20 class-0-labeled images showed a clearly visible
#    lesion on inspection (2, IMG_0303/IMG_0304, showed no visible symptom
#    in frame and were excluded rather than trusted blindly).
TARGET_SPOT_EXTRA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "target_spot_extra", "cropped")
TARGET_SPOT_MENDELEY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "target_spot_extra", "mendeley")
TARGET_SPOT_TRAIN_FILES = [
    os.path.join(TARGET_SPOT_EXTRA_DIR, fn) for fn in [
        "ts_leaves_topright.jpg", "ts_spotted_closeup.jpg",
        "ts_leaflet_1.jpg", "ts_leaflet_2.jpg", "ts_leaflet_3.jpg",
    ]
] + [
    os.path.join(TARGET_SPOT_MENDELEY_DIR, f"{fn}.jpg") for fn in [
        "IMG_0229", "IMG_0322", "IMG_0323", "IMG_0324", "IMG_0325", "IMG_0326",
        "IMG_0348", "IMG_0350", "IMG_0374", "IMG_0654", "IMG_0655", "IMG_0656",
        "IMG_1035", "IMG_1036", "IMG_1126",
    ]
]
TARGET_SPOT_HELDOUT_FILES = [
    os.path.join(TARGET_SPOT_EXTRA_DIR, fn) for fn in ["ts_leaves_topleft.jpg", "ts_ring_lesion.jpg"]
] + [
    os.path.join(TARGET_SPOT_MENDELEY_DIR, f"{fn}.jpg") for fn in ["IMG_0653", "IMG_1128", "IMG_0349"]
]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Same folder->label mapping plantdoc_confidence_eval.py already validated.
# Tomato_Spider_mites excluded on purpose: only 2 train images / 0 test
# images exist anywhere in PlantDoc for it -- not a usable training signal.
PLANTDOC_MAP = {
    "Tomato leaf": "Tomato_healthy",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
}

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def list_plantdoc_images(split):
    """Returns [(filepath, our_label), ...] for every mapped PlantDoc image in `split`."""
    items = []
    for plantdoc_class, our_label in PLANTDOC_MAP.items():
        src_dir = os.path.join(PLANTDOC_ROOT, split, plantdoc_class)
        if not os.path.isdir(src_dir):
            continue
        for fname in os.listdir(src_dir):
            fpath = os.path.join(src_dir, fname)
            if os.path.isfile(fpath):
                items.append((fpath, our_label))
    return items


class LeafDataset(Dataset):
    def __init__(self, items, label_to_idx, transform):
        self.items = items
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        fpath, label = self.items[i]
        img = Image.open(fpath).convert("RGB")
        return self.transform(img), self.label_to_idx[label]


def stratified_split(items, val_frac=0.15, seed=SEED):
    rng = random.Random(seed)
    by_label = {}
    for item in items:
        by_label.setdefault(item[1], []).append(item)
    train_items, val_items = [], []
    for label, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_frac)) if len(group) > 1 else 0
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])
    return train_items, val_items


def load_model(checkpoint_name, num_classes):
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    m.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, checkpoint_name), map_location="cpu"))
    return m.to(device)


def set_trainable(m, unfreeze_last_block=True):
    """Freeze everything except the classifier head (+ optionally the
    backbone's last block) -- conservative transfer-learning recipe for a
    small real-world fine-tuning set."""
    for p in m.parameters():
        p.requires_grad = False
    for p in m.classifier.parameters():
        p.requires_grad = True
    if unfreeze_last_block:
        for p in m.features[-1].parameters():
            p.requires_grad = True


def class_weights_for(items, classes):
    counts = {c: 0 for c in classes}
    for _, label in items:
        counts[label] += 1
    total = len(items)
    weights = [total / (len(classes) * max(counts[c], 1)) for c in classes]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def finetune_stage(name, checkpoint_name, classes, items, epochs=8, batch_size=16):
    print(f"\n=== Fine-tuning {name} ===")
    label_to_idx = {c: i for i, c in enumerate(classes)}
    train_items, val_items = stratified_split(items)
    print(f"{name}: {len(train_items)} train / {len(val_items)} internal-val images "
          f"(from PlantDoc train split; PlantDoc test split untouched)")

    train_ds = LeafDataset(train_items, label_to_idx, train_transform)
    val_ds = LeafDataset(val_items, label_to_idx, eval_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = load_model(checkpoint_name, len(classes))
    set_trainable(model)

    weights = class_weights_for(train_items, classes)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": 3e-4},
        {"params": model.features[-1].parameters(), "lr": 3e-5},
    ])

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / max(len(train_ds), 1)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0
        print(f"  epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  internal_val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    out_path = os.path.join(CHECKPOINT_DIR, f"{name}_finetuned.pt")
    torch.save(best_state, out_path)
    print(f"  saved best checkpoint (internal_val_acc={best_val_acc:.4f}) -> {out_path}")
    return out_path, best_val_acc


def evaluate_on_plantdoc_test(checkpoints):
    """Honest before/after check -- runs ONLY on PlantDoc's held-out `test`
    split, which none of the fine-tuning above ever saw."""
    print("\n=== Evaluating fine-tuned models on PlantDoc TEST split (held out) ===")
    stage1 = load_model(checkpoints["stage1"], len(STAGE1_CLASSES))
    stage2_tomato = load_model(checkpoints["stage2_tomato"], len(STAGE2_TOMATO_CLASSES))
    stage2_potato = load_model(checkpoints["stage2_potato"], len(STAGE2_POTATO_CLASSES))
    for m in (stage1, stage2_tomato, stage2_potato):
        m.eval()

    test_items = list_plantdoc_images("test")
    print(f"PlantDoc test images evaluated: {len(test_items)}")

    correct_end_to_end, correct_stage1, total = 0, 0, 0
    per_class = {}

    with torch.no_grad():
        for fpath, true_label in test_items:
            img = Image.open(fpath).convert("RGB")
            x = eval_transform(img).unsqueeze(0).to(device)

            true_crop = "Tomato" if true_label.startswith("Tomato") else "Potato"
            stage1_idx = stage1(x).argmax(1).item()
            pred_crop = STAGE1_CLASSES[stage1_idx]

            if pred_crop == "Tomato":
                stage2_out = stage2_tomato(x)
                stage2_classes = STAGE2_TOMATO_CLASSES
            else:
                stage2_out = stage2_potato(x)
                stage2_classes = STAGE2_POTATO_CLASSES
            pred_class = stage2_classes[stage2_out.argmax(1).item()]

            total += 1
            correct_stage1 += int(pred_crop == true_crop)
            is_correct = pred_class == true_label
            correct_end_to_end += int(is_correct)

            bucket = per_class.setdefault(true_label, {"correct": 0, "total": 0})
            bucket["total"] += 1
            bucket["correct"] += int(is_correct)

    result = {
        "plantdoc_test_images_evaluated": total,
        "stage1_crop_accuracy": correct_stage1 / total if total else 0.0,
        "end_to_end_accuracy": correct_end_to_end / total if total else 0.0,
        "per_class": {
            label: {"accuracy": b["correct"] / b["total"], "n": b["total"]}
            for label, b in sorted(per_class.items())
        },
        "note": "PlantDoc `test` split only -- never used during fine-tuning. "
                "Potato___healthy and Tomato__Target_Spot are absent from PlantDoc "
                "entirely and cannot appear in this table.",
    }
    print(f"\nBEFORE (PlantVillage-only, from hierarchical_results.json): 23.45% end-to-end")
    print(f"AFTER (fine-tuned on PlantDoc train):  {result['end_to_end_accuracy']*100:.2f}% end-to-end")
    print(f"Stage-1 crop accuracy (Potato vs Tomato): {result['stage1_crop_accuracy']*100:.2f}%")
    return result


def check_target_spot_heldout(stage2_tomato_checkpoint):
    """The 2 Target_Spot images NEVER used in training or the internal val
    split (see TARGET_SPOT_HELDOUT_FILES) -- PlantDoc's test split has zero
    Target_Spot images, so this is the only real check available for this
    class. n=2 is not a statistically meaningful accuracy number; this is a
    sanity check, not a benchmark."""
    print(f"\n=== Checking Tomato__Target_Spot on {len(TARGET_SPOT_HELDOUT_FILES)} held-out images (never trained on) ===")
    model = load_model(stage2_tomato_checkpoint, len(STAGE2_TOMATO_CLASSES))
    model.eval()
    results = []
    with torch.no_grad():
        for fpath in TARGET_SPOT_HELDOUT_FILES:
            fn = os.path.basename(fpath)
            if not os.path.isfile(fpath):
                continue
            img = Image.open(fpath).convert("RGB")
            x = eval_transform(img).unsqueeze(0).to(device)
            probs = F.softmax(model(x), dim=1)[0]
            pred_idx = probs.argmax().item()
            pred_class = STAGE2_TOMATO_CLASSES[pred_idx]
            confidence = probs[pred_idx].item()
            correct = pred_class == "Tomato__Target_Spot"
            print(f"  {fn}: predicted={pred_class} (conf {confidence:.2f})  {'OK' if correct else 'WRONG'}")
            results.append({"file": fn, "predicted": pred_class, "confidence": confidence, "correct": correct})
    n_correct = sum(r["correct"] for r in results)
    print(f"  {n_correct}/{len(results)} correct")
    return results


def main():
    torch.manual_seed(SEED)
    train_items = list_plantdoc_images("train")

    stage1_items = [(f, "Tomato" if lbl.startswith("Tomato") else "Potato") for f, lbl in train_items]
    tomato_items = [(f, lbl) for f, lbl in train_items if lbl.startswith("Tomato")]
    potato_items = [(f, lbl) for f, lbl in train_items if lbl.startswith("Potato")]

    target_spot_train = [
        (fpath, "Tomato__Target_Spot")
        for fpath in TARGET_SPOT_TRAIN_FILES
        if os.path.isfile(fpath)
    ]
    print(f"\nAdding {len(target_spot_train)} real-world Tomato__Target_Spot images "
          f"(UF/IFAS EDIS PP351 + Mendeley Tomato Leaf Dataset, CC BY 4.0) -- "
          f"PlantDoc has zero images for this class.")
    tomato_items += target_spot_train

    # Always fine-tune from the ORIGINAL PlantVillage-only checkpoints, never
    # from an already-fine-tuned one -- keeps this a single, reproducible
    # PlantVillage -> PlantDoc(+extras) step instead of stacking fine-tunes.
    # Falls back to *_best.pt if the plantvillage_only backup isn't present
    # (e.g. first-ever run, before any promotion has happened).
    def source_checkpoint(stem):
        backup = f"{stem}_plantvillage_only.pt"
        return backup if os.path.isfile(os.path.join(CHECKPOINT_DIR, backup)) else f"{stem}_best.pt"

    checkpoints = {}
    checkpoints["stage1"], stage1_val_acc = finetune_stage(
        "stage1_crop", source_checkpoint("stage1_crop"), STAGE1_CLASSES, stage1_items
    )
    checkpoints["stage2_tomato"], tomato_val_acc = finetune_stage(
        "stage2_tomato", source_checkpoint("stage2_tomato"), STAGE2_TOMATO_CLASSES, tomato_items
    )
    checkpoints["stage2_potato"], potato_val_acc = finetune_stage(
        "stage2_potato", source_checkpoint("stage2_potato"), STAGE2_POTATO_CLASSES, potato_items
    )
    checkpoints = {k: os.path.basename(v) for k, v in checkpoints.items()}

    eval_result = evaluate_on_plantdoc_test(checkpoints)
    eval_result["target_spot_heldout_check"] = check_target_spot_heldout(checkpoints["stage2_tomato"])
    eval_result["internal_val_accuracy"] = {
        "stage1_crop": stage1_val_acc,
        "stage2_tomato": tomato_val_acc,
        "stage2_potato": potato_val_acc,
    }
    eval_result["finetuned_checkpoints"] = checkpoints

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "plantdoc_finetuned_results.json")
    with open(out_path, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"\nSaved results -> {out_path}")


if __name__ == "__main__":
    main()
