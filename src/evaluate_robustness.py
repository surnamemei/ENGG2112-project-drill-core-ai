from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

CLEAN_TEST_DIR = Path("DCID/DCID-512-7/test")
NOISY_TEST_DIR = Path("DCID/noise-512-7/test")

MODEL_PATH = Path(
    "results/models/resnet18_dcid7_best.pth"
)

OUTPUT_PATH = Path(
    "results/robustness_metrics.csv"
)


# ============================================================
# Settings
# ============================================================

BATCH_SIZE = 128
NUM_WORKERS = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Transform
# ============================================================

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]
num_classes = len(class_names)


# ============================================================
# Build model
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# Evaluation function
# ============================================================

def evaluate_dataset(
    dataset_name: str,
    dataset_path: Path,
) -> dict:

    dataset = datasets.ImageFolder(
        dataset_path,
        transform=transform
    )

    if dataset.classes != class_names:
        raise ValueError(
            f"Class mismatch for {dataset_name}.\n"
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

    all_labels = []
    all_predictions = []

    with torch.inference_mode():

        for images, labels in tqdm(
            loader,
            desc=f"Evaluating {dataset_name}"
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "dataset": dataset_name,
        "num_images": len(dataset),
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# ============================================================
# Run evaluations
# ============================================================

results = []

results.append(
    evaluate_dataset(
        "clean",
        CLEAN_TEST_DIR
    )
)

if NOISY_TEST_DIR.exists():

    results.append(
        evaluate_dataset(
            "noisy",
            NOISY_TEST_DIR
        )
    )

else:

    print(
        "\nNoisy test directory not found:"
    )

    print(
        NOISY_TEST_DIR
    )

    print(
        "\nCheck the noise-512-7 directory structure first."
    )


# ============================================================
# Save and print
# ============================================================

results_df = pd.DataFrame(
    results
)

if len(results_df) >= 2:

    clean_acc = results_df.loc[
        results_df["dataset"] == "clean",
        "accuracy"
    ].iloc[0]

    noisy_acc = results_df.loc[
        results_df["dataset"] == "noisy",
        "accuracy"
    ].iloc[0]

    clean_f1 = results_df.loc[
        results_df["dataset"] == "clean",
        "macro_f1"
    ].iloc[0]

    noisy_f1 = results_df.loc[
        results_df["dataset"] == "noisy",
        "macro_f1"
    ].iloc[0]

    results_df["accuracy_drop_pp"] = 0.0
    results_df["macro_f1_drop"] = 0.0

    results_df.loc[
        results_df["dataset"] == "noisy",
        "accuracy_drop_pp"
    ] = (
        clean_acc - noisy_acc
    ) * 100

    results_df.loc[
        results_df["dataset"] == "noisy",
        "macro_f1_drop"
    ] = (
        clean_f1 - noisy_f1
    )


results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 60)
print("ROBUSTNESS RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT_PATH)