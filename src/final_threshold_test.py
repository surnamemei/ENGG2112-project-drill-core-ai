from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

NOISY_TEST_DIR = Path(
    "DCID/noise-512-7/test"
)

MODEL_PATH = Path(
    "results/models/resnet18_dcid7_best.pth"
)

OUTPUT_PATH = Path(
    "results/robustness_analysis/"
    "final_threshold_test.csv"
)


# ============================================================
# Locked operating threshold
# ============================================================

THRESHOLD = 0.97

BATCH_SIZE = 128
NUM_WORKERS = 8

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# Transform
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# Dataset
# ============================================================

dataset = datasets.ImageFolder(
    root=NOISY_TEST_DIR,
    transform=transform,
)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

class_names: list[str] = list(
    checkpoint["class_names"]
)

num_classes = len(
    class_names
)

if dataset.classes != class_names:
    raise ValueError(
        "Class mismatch.\n"
        f"Expected: {class_names}\n"
        f"Found: {dataset.classes}"
    )


# ============================================================
# Build model
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# DataLoader
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


# ============================================================
# Collect predictions
# ============================================================

labels_list: list[int] = []
predictions_list: list[int] = []
confidence_list: list[float] = []

with torch.inference_mode():

    for images, labels in tqdm(
        loader,
        desc="Final noisy test evaluation",
    ):

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1,
        )

        labels_list.extend(
            labels.tolist()
        )

        predictions_list.extend(
            predictions
            .cpu()
            .tolist()
        )

        confidence_list.extend(
            confidences
            .cpu()
            .tolist()
        )


# ============================================================
# Convert to arrays
# ============================================================

labels = np.array(
    labels_list,
    dtype=np.int64,
)

predictions = np.array(
    predictions_list,
    dtype=np.int64,
)

confidence = np.array(
    confidence_list,
    dtype=np.float64,
)

correct = (
    labels == predictions
)

incorrect = ~correct

referred = (
    confidence < THRESHOLD
)

accepted = ~referred


# ============================================================
# Counts
# ============================================================

total_images = len(
    labels
)

total_correct = int(
    correct.sum()
)

total_incorrect = int(
    incorrect.sum()
)

referred_count = int(
    referred.sum()
)

accepted_count = int(
    accepted.sum()
)

referred_incorrect = int(
    (
        referred
        & incorrect
    ).sum()
)

referred_correct = int(
    (
        referred
        & correct
    ).sum()
)

accepted_incorrect = int(
    (
        accepted
        & incorrect
    ).sum()
)

accepted_correct = int(
    (
        accepted
        & correct
    ).sum()
)


# ============================================================
# Metrics
# ============================================================

base_accuracy = (
    total_correct
    / total_images
)

referral_rate = (
    referred_count
    / total_images
)

error_capture_rate = (
    referred_incorrect
    / total_incorrect
    if total_incorrect > 0
    else 0.0
)

correct_referral_rate = (
    referred_correct
    / total_correct
    if total_correct > 0
    else 0.0
)

accepted_accuracy = (
    accepted_correct
    / accepted_count
    if accepted_count > 0
    else 0.0
)


# ============================================================
# Save result
# ============================================================

result_df = pd.DataFrame([
    {
        "threshold":
            THRESHOLD,

        "total_images":
            total_images,

        "base_accuracy":
            base_accuracy,

        "referred_count":
            referred_count,

        "referral_rate":
            referral_rate,

        "referred_incorrect":
            referred_incorrect,

        "referred_correct":
            referred_correct,

        "error_capture_rate":
            error_capture_rate,

        "correct_referral_rate":
            correct_referral_rate,

        "accepted_count":
            accepted_count,

        "accepted_incorrect":
            accepted_incorrect,

        "accepted_accuracy":
            accepted_accuracy,
    }
])

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

result_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# Print final result
# ============================================================

print("\n" + "=" * 72)
print("FINAL LOCKED-THRESHOLD TEST")
print("=" * 72)

print(
    f"Threshold: "
    f"{THRESHOLD:.2f}"
)

print(
    f"Total images: "
    f"{total_images}"
)

print(
    f"Base accuracy: "
    f"{base_accuracy * 100:.2f}%"
)

print(
    f"Referred for review: "
    f"{referred_count} "
    f"({referral_rate * 100:.2f}%)"
)

print(
    f"Errors captured: "
    f"{referred_incorrect} / "
    f"{total_incorrect} "
    f"({error_capture_rate * 100:.2f}%)"
)

print(
    f"Correct predictions referred: "
    f"{referred_correct} / "
    f"{total_correct} "
    f"({correct_referral_rate * 100:.2f}%)"
)

print(
    f"Automatically accepted: "
    f"{accepted_count}"
)

print(
    f"Incorrect accepted predictions: "
    f"{accepted_incorrect}"
)

print(
    f"Accepted accuracy: "
    f"{accepted_accuracy * 100:.2f}%"
)

print("\nSaved:")
print(
    OUTPUT_PATH
)