"""
Export per-image predictions of the locked ResNet-18 checkpoint on the
noisy/RWDA DCID-7 test set.

The evaluation protocol is unchanged from evaluate_robustness.py:
same checkpoint, same 224 x 224 resize, same ImageNet normalisation,
softmax confidence = max class probability.

The per-image CSV is used by make_report_figures.py to draw the full
confidence distribution (the summary CSVs only store means).
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ============================================================
# Paths and settings
# ============================================================

NOISY_TEST_DIR = Path("DCID/noise-512-7/test")

MODEL_PATH = Path("results/models/resnet18_dcid7_best.pth")

OUTPUT_PATH = Path(
    "results/robustness_analysis/noisy_test_predictions.csv"
)

BATCH_SIZE = 128
NUM_WORKERS = 8

EXPECTED_ACCURACY = 0.9792857142857143

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Transform (identical to evaluation transform)
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

dataset = datasets.ImageFolder(
    NOISY_TEST_DIR,
    transform=evaluation_transform,
)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

class_names: list[str] = list(checkpoint["class_names"])

if dataset.classes != class_names:
    raise ValueError(
        "Class mismatch.\n"
        f"Expected: {class_names}\n"
        f"Found: {dataset.classes}"
    )

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


# ============================================================
# Inference
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

rows: list[dict[str, object]] = []
global_index = 0

with torch.inference_mode():

    for images, labels in tqdm(loader, desc="Predicting"):

        images = images.to(DEVICE, non_blocking=True)

        probabilities = torch.softmax(model(images), dim=1)

        confidences, predictions = torch.max(probabilities, dim=1)

        for i in range(labels.size(0)):

            true_index = int(labels[i].item())
            pred_index = int(predictions[i].item())

            rows.append({
                "dataset_index": global_index,
                "source_path": dataset.samples[global_index][0],
                "true_class": class_names[true_index],
                "predicted_class": class_names[pred_index],
                "confidence": float(confidences[i].item()),
                "correct": true_index == pred_index,
            })

            global_index += 1


# ============================================================
# Save and sanity check
# ============================================================

df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

accuracy = float(df["correct"].mean())

print(f"\nImages: {len(df)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Incorrect: {int((~df['correct']).sum())}")

if abs(accuracy - EXPECTED_ACCURACY) > 1e-9:
    print(
        "WARNING: accuracy differs from the recorded noisy-test result "
        f"({EXPECTED_ACCURACY * 100:.2f}%)."
    )

print("\nSaved:", OUTPUT_PATH)
