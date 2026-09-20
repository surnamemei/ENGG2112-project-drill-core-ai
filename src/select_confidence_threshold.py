from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

NOISY_TRAIN_DIR = Path(
    "DCID/noise-512-7/train"
)

MODEL_PATH = Path(
    "results/models/resnet18_dcid7_best.pth"
)

OUTPUT_PATH = Path(
    "results/robustness_analysis/"
    "validation_threshold_analysis.csv"
)


# ============================================================
# Settings
# ============================================================

RANDOM_SEED = 2112
VALIDATION_RATIO = 0.20

BATCH_SIZE = 128
NUM_WORKERS = 8

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
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
# Reproducibility
# ============================================================

random.seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)

torch.manual_seed(
    RANDOM_SEED
)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(
        RANDOM_SEED
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
# Load dataset
# ============================================================

full_dataset = datasets.ImageFolder(
    root=NOISY_TRAIN_DIR,
    transform=transform,
)

targets = np.array(
    full_dataset.targets,
    dtype=np.int64,
)

indices = np.arange(
    len(full_dataset)
)


# ============================================================
# Stratified validation split
# ============================================================

_, val_indices = train_test_split(
    indices,
    test_size=VALIDATION_RATIO,
    random_state=RANDOM_SEED,
    stratify=targets,
)

val_dataset = Subset(
    full_dataset,
    val_indices,
)

print(
    "\nNoisy training images:",
    len(full_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
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

if full_dataset.classes != class_names:
    raise ValueError(
        "Class mismatch.\n"
        f"Expected: {class_names}\n"
        f"Found: {full_dataset.classes}"
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
    checkpoint[
        "model_state_dict"
    ]
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# DataLoader
# ============================================================

val_loader = DataLoader(
    val_dataset,
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
        val_loader,
        desc="Evaluating noisy validation",
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
# Base statistics
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

base_accuracy = (
    total_correct
    / total_images
)

print("\n" + "=" * 72)
print("NOISY VALIDATION SUMMARY")
print("=" * 72)

print(
    "Total images:",
    total_images
)

print(
    "Correct:",
    total_correct
)

print(
    "Incorrect:",
    total_incorrect
)

print(
    f"Base accuracy: "
    f"{base_accuracy * 100:.2f}%"
)


# ============================================================
# Threshold analysis
# ============================================================

rows: list[
    dict[str, float | int]
] = []

for threshold in THRESHOLDS:

    referred = (
        confidence
        < threshold
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

    referral_rate = (
        referred_count
        / total_images
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

    if accepted_count > 0:

        accepted_accuracy = (
            accepted_correct
            / accepted_count
        )

    else:

        accepted_accuracy = 0.0

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

        "accepted_count":
            accepted_count,

        "accepted_incorrect":
            accepted_incorrect,

        "accepted_accuracy":
            accepted_accuracy,
    })


# ============================================================
# Save analysis
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
# Print readable table
# ============================================================

display_df = (
    results_df.copy()
)

for column in [
    "referral_rate",
    "error_capture_rate",
    "correct_referral_rate",
    "accepted_accuracy",
]:

    display_df[
        column
    ] *= 100


print("\n" + "=" * 72)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 72)

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


# ============================================================
# Suggested operating points
# ============================================================

print("\n" + "=" * 72)
print("CANDIDATE OPERATING POINTS")
print("=" * 72)

candidate_df = (
    display_df[
        display_df[
            "threshold"
        ].isin([
            0.90,
            0.95,
            0.97,
        ])
    ]
)

print(
    candidate_df[
        [
            "threshold",
            "referral_rate",
            "error_capture_rate",
            "accepted_accuracy",
        ]
    ].to_string(
        index=False,
        formatters={
            "referral_rate":
                "{:.2f}%".format,

            "error_capture_rate":
                "{:.2f}%".format,

            "accepted_accuracy":
                "{:.2f}%".format,
        },
    )
)

print("\nSaved:")
print(
    OUTPUT_PATH
)