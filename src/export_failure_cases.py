from pathlib import Path
import shutil

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

OUTPUT_DIR = Path(
    "results/failure_cases"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CSV_PATH = OUTPUT_DIR / "failure_cases.csv"


# ============================================================
# Settings
# ============================================================

BATCH_SIZE = 128
NUM_WORKERS = 8

TOP_N_PER_PAIR = 5

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
# Model
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
# Collect prediction results
# ============================================================

rows: list[dict[str, object]] = []

global_index = 0

with torch.inference_mode():

    for images, labels in tqdm(
        loader,
        desc="Finding failure cases",
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

        batch_size = labels.size(0)

        for i in range(batch_size):

            true_index = int(
                labels[i].item()
            )

            predicted_index = int(
                predictions[i].item()
            )

            confidence = float(
                confidences[i].item()
            )

            source_path = Path(
                dataset.samples[
                    global_index
                ][0]
            )

            if true_index != predicted_index:

                rows.append({
                    "dataset_index":
                        global_index,

                    "source_path":
                        str(source_path),

                    "true_class":
                        class_names[
                            true_index
                        ],

                    "predicted_class":
                        class_names[
                            predicted_index
                        ],

                    "confidence":
                        confidence,

                    "confusion_pair":
                        (
                            f"{class_names[true_index]}"
                            " -> "
                            f"{class_names[predicted_index]}"
                        ),
                })

            global_index += 1


# ============================================================
# DataFrame
# ============================================================

failure_df = pd.DataFrame(
    rows
)

failure_df = failure_df.sort_values(
    by="confidence",
    ascending=False,
).reset_index(
    drop=True
)

failure_df.to_csv(
    CSV_PATH,
    index=False,
)

print(
    "\nTotal misclassified images:",
    len(failure_df)
)

print(
    "Saved:",
    CSV_PATH
)


# ============================================================
# Export representative images
# ============================================================

pair_counts = (
    failure_df[
        "confusion_pair"
    ]
    .value_counts()
)

print(
    "\nMost common confusion pairs:"
)

print(
    pair_counts.head(10)
)


for pair_name, pair_group in failure_df.groupby(
    "confusion_pair"
):

    pair_safe_name = (
        str(pair_name)
        .replace(" ", "_")
        .replace(".", "")
        .replace("->", "to")
        .replace("/", "_")
    )

    pair_dir = (
        OUTPUT_DIR
        / pair_safe_name
    )

    pair_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Highest-confidence wrong predictions are especially
    # useful because they show difficult failure cases.
    selected = (
        pair_group
        .sort_values(
            by="confidence",
            ascending=False,
        )
        .head(
            TOP_N_PER_PAIR
        )
    )

    for rank, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        source_path = Path(
            str(
                row[
                    "source_path"
                ]
            )
        )

        confidence = float(
            row[
                "confidence"
            ]
        )

        true_class = str(
            row[
                "true_class"
            ]
        )

        predicted_class = str(
            row[
                "predicted_class"
            ]
        )

        destination_name = (
            f"{rank:02d}_"
            f"true-{true_class}_"
            f"pred-{predicted_class}_"
            f"conf-{confidence:.3f}"
            f"{source_path.suffix}"
        )

        destination_name = (
            destination_name
            .replace(" ", "_")
        )

        destination_path = (
            pair_dir
            / destination_name
        )

        shutil.copy2(
            source_path,
            destination_path,
        )


print(
    "\nRepresentative failure images exported to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nFailure-case export complete."
)