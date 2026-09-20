from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

CLEAN_TEST_DIR = Path(
    "DCID/DCID-512-7/test"
)

NOISY_TEST_DIR = Path(
    "DCID/noise-512-7/test"
)

MODEL_PATH = Path(
    "results/models/resnet18_dcid7_best.pth"
)

RESULTS_DIR = Path(
    "results/robustness_analysis"
)

FIGURE_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Settings
# ============================================================

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
# Evaluation transform
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

evaluation_transform = transforms.Compose([
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
# Load model checkpoint
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
# Evaluation function
# ============================================================

def evaluate_dataset(
    dataset_name: str,
    dataset_path: Path,
) -> dict[str, object]:

    print(
        f"\nEvaluating: "
        f"{dataset_name}"
    )

    dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=evaluation_transform,
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

    labels_list: list[int] = []
    predictions_list: list[int] = []
    confidence_list: list[float] = []

    with torch.inference_mode():

        for images, labels in tqdm(
            loader,
            desc=dataset_name,
        ):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(
                images
            )

            probabilities = (
                torch.softmax(
                    outputs,
                    dim=1,
                )
            )

            confidences, predictions = (
                torch.max(
                    probabilities,
                    dim=1,
                )
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

    labels_array = np.array(
        labels_list,
        dtype=np.int64,
    )

    predictions_array = np.array(
        predictions_list,
        dtype=np.int64,
    )

    confidence_array = np.array(
        confidence_list,
        dtype=np.float64,
    )

    correct_array = (
        labels_array
        == predictions_array
    )

    return {
        "dataset_name":
            dataset_name,

        "labels":
            labels_array,

        "predictions":
            predictions_array,

        "confidence":
            confidence_array,

        "correct":
            correct_array,

        "num_images":
            len(dataset),
    }


# ============================================================
# Save classification report
# ============================================================

def save_classification_report(
    result: dict[str, object],
) -> None:

    dataset_name = str(
        result["dataset_name"]
    )

    labels = np.asarray(
        result["labels"]
    )

    predictions = np.asarray(
        result["predictions"]
    )

    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    output_path = (
        RESULTS_DIR
        / f"{dataset_name}_per_class_metrics.csv"
    )

    report_df.to_csv(
        output_path
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Save confusion matrix
# ============================================================

def save_confusion_matrix(
    result: dict[str, object],
) -> np.ndarray:

    dataset_name = str(
        result["dataset_name"]
    )

    labels = np.asarray(
        result["labels"]
    )

    predictions = np.asarray(
        result["predictions"]
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(
            range(num_classes)
        ),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(
        f"{dataset_name.capitalize()} "
        "Confusion Matrix"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / f"{dataset_name}_confusion_matrix.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )

    return matrix


# ============================================================
# Confidence analysis
# ============================================================

def save_confidence_analysis(
    result: dict[str, object],
) -> None:

    dataset_name = str(
        result["dataset_name"]
    )

    confidence = np.asarray(
        result["confidence"],
        dtype=np.float64,
    )

    correct = np.asarray(
        result["correct"],
        dtype=bool,
    )

    correct_confidence = (
        confidence[
            correct
        ]
    )

    incorrect_confidence = (
        confidence[
            ~correct
        ]
    )

    correct_mean = float(
        correct_confidence.mean()
    )

    if len(
        incorrect_confidence
    ) > 0:

        incorrect_mean = float(
            incorrect_confidence.mean()
        )

    else:
        incorrect_mean = float(
            "nan"
        )

    summary_df = pd.DataFrame([
        {
            "dataset":
                dataset_name,

            "num_images":
                len(confidence),

            "num_correct":
                int(
                    correct.sum()
                ),

            "num_incorrect":
                int(
                    (~correct).sum()
                ),

            "mean_confidence_correct":
                correct_mean,

            "mean_confidence_incorrect":
                incorrect_mean,
        }
    ])

    output_path = (
        RESULTS_DIR
        / f"{dataset_name}_confidence_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Confidence distribution figure
# ============================================================

def save_confidence_plot(
    result: dict[str, object],
) -> None:

    dataset_name = str(
        result["dataset_name"]
    )

    confidence = np.asarray(
        result["confidence"],
        dtype=np.float64,
    )

    correct = np.asarray(
        result["correct"],
        dtype=bool,
    )

    correct_confidence = (
        confidence[
            correct
        ]
    )

    incorrect_confidence = (
        confidence[
            ~correct
        ]
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        correct_confidence,
        bins=30,
        alpha=0.6,
        label="Correct",
    )

    if len(
        incorrect_confidence
    ) > 0:

        plt.hist(
            incorrect_confidence,
            bins=30,
            alpha=0.6,
            label="Incorrect",
        )

    plt.xlabel(
        "Softmax Confidence"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        f"{dataset_name.capitalize()} "
        "Prediction Confidence"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / f"{dataset_name}_confidence_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(
        "Saved:",
        output_path
    )


# ============================================================
# Most common confusion pairs
# ============================================================

def save_confusion_pairs(
    dataset_name: str,
    matrix: np.ndarray,
) -> None:

    confusion_rows: list[
        dict[str, object]
    ] = []

    for true_index in range(
        num_classes
    ):

        for predicted_index in range(
            num_classes
        ):

            if (
                true_index
                == predicted_index
            ):
                continue

            count = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

            if count == 0:
                continue

            confusion_rows.append({
                "true_class":
                    class_names[
                        true_index
                    ],

                "predicted_class":
                    class_names[
                        predicted_index
                    ],

                "count":
                    count,
            })

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    if not confusion_df.empty:

        confusion_df = (
            confusion_df
            .sort_values(
                by="count",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    output_path = (
        RESULTS_DIR
        / f"{dataset_name}_confusion_pairs.csv"
    )

    confusion_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "Saved:",
        output_path
    )

    if not confusion_df.empty:

        print(
            f"\nTop confusion pairs "
            f"for {dataset_name}:"
        )

        print(
            confusion_df.head(10)
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    clean_result = evaluate_dataset(
        "clean",
        CLEAN_TEST_DIR,
    )

    noisy_result = evaluate_dataset(
        "noisy",
        NOISY_TEST_DIR,
    )

    results = [
        clean_result,
        noisy_result,
    ]

    for result in results:

        save_classification_report(
            result
        )

        matrix = save_confusion_matrix(
            result
        )

        save_confidence_analysis(
            result
        )

        save_confidence_plot(
            result
        )

        save_confusion_pairs(
            str(
                result[
                    "dataset_name"
                ]
            ),
            matrix,
        )

    print(
        "\nRobustness analysis complete."
    )


if __name__ == "__main__":
    main()