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
    "confidence_threshold_analysis.csv"
)


# ============================================================
# Settings
# ============================================================

BATCH_SIZE = 128
NUM_WORKERS = 8

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.97,
    0.99,
]

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
# Load noisy test set
# ============================================================

dataset = datasets.ImageFolder(
    root=NOISY_TEST_DIR,
    transform=transform,
)

if dataset.classes != class_names:
    raise ValueError(
        "Class mismatch.\n"
        f"Expected: {class_names}\n"
        f"Found: {dataset.classes}"
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


# ============================================================
# Collect predictions and confidence
# ============================================================

labels_list: list[int] = []
predictions_list: list[int] = []
confidence_list: list[float] = []

with torch.inference_mode():

    for images, labels in tqdm(
        loader,
        desc="Evaluating noisy test set",
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
            predictions.cpu().tolist()
        )

        confidence_list.extend(
            confidences.cpu().tolist()
        )


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


# ============================================================
# Basic totals
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

print("\n" + "=" * 70)
print("NOISY TEST SUMMARY")
print("=" * 70)

print("Total images:", total_images)
print("Correct:", total_correct)
print("Incorrect:", total_incorrect)


# ============================================================
# Threshold analysis
# ============================================================

rows: list[dict[str, float | int]] = []

for threshold in THRESHOLDS:

    referred = (
        confidence < threshold
    )

    accepted = ~referred

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

    if total_incorrect > 0:

        error_capture_rate = (
            referred_incorrect
            / total_incorrect
        )

    else:

        error_capture_rate = 0.0

    if total_correct > 0:

        correct_referral_rate = (
            referred_correct
            / total_correct
        )

    else:

        correct_referral_rate = 0.0

    if referred_count > 0:

        referral_error_rate = (
            referred_incorrect
            / referred_count
        )

    else:

        referral_error_rate = 0.0

    if accepted_count > 0:

        accepted_accuracy = (
            accepted_correct
            / accepted_count
        )

    else:

        accepted_accuracy = 0.0

    referral_rate = (
        referred_count
        / total_images
    )

    rows.append({
        "threshold":
            threshold,

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

        "referral_error_rate":
            referral_error_rate,

        "accepted_count":
            accepted_count,

        "accepted_incorrect":
            accepted_incorrect,

        "accepted_accuracy":
            accepted_accuracy,
    })


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    rows
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# Print readable summary
# ============================================================

print("\n" + "=" * 70)
print("CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 70)

display_df = results_df.copy()

display_df[
    "referral_rate"
] *= 100

display_df[
    "error_capture_rate"
] *= 100

display_df[
    "correct_referral_rate"
] *= 100

display_df[
    "referral_error_rate"
] *= 100

display_df[
    "accepted_accuracy"
] *= 100

print(
    display_df[
        [
            "threshold",
            "referred_count",
            "referral_rate",
            "error_capture_rate",
            "correct_referral_rate",
            "accepted_accuracy",
        ]
    ].to_string(
        index=False,
        formatters={
            "referral_rate":
                "{:.2f}%".format,

            "error_capture_rate":
                "{:.2f}%".format,

            "correct_referral_rate":
                "{:.2f}%".format,

            "accepted_accuracy":
                "{:.2f}%".format,
        },
    )
)

print("\nSaved:")
print(OUTPUT_PATH)