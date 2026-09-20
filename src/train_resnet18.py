from pathlib import Path
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = Path("DCID/DCID-512-7")
TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"

RESULTS_DIR = Path("results")
MODEL_DIR = RESULTS_DIR / "models"
FIGURE_DIR = RESULTS_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 2112

BATCH_SIZE = 128
NUM_WORKERS = 8
NUM_EPOCHS = 10

LEARNING_RATE = 1e-3
VALIDATION_RATIO = 0.10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# Image transformations
# ============================================================

# ImageNet normalization because ResNet-18 pretrained weights
# were trained using ImageNet statistics.

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=imagenet_mean,
        std=imagenet_std
    ),
])


# ============================================================
# Dataset loading
# ============================================================

# Two copies of the same training dataset are used because
# training and validation require different transformations.

full_train_augmented = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

full_train_evaluation = datasets.ImageFolder(
    TRAIN_DIR,
    transform=evaluation_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=evaluation_transform
)


class_names = full_train_augmented.classes
num_classes = len(class_names)

print("\nClasses:")

for index, name in enumerate(class_names):
    print(index, name)


# ============================================================
# Stratified train / validation split
# ============================================================

all_indices = np.arange(
    len(full_train_augmented)
)

all_targets = np.array(
    full_train_augmented.targets
)

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=VALIDATION_RATIO,
    random_state=RANDOM_SEED,
    stratify=all_targets
)


train_dataset = Subset(
    full_train_augmented,
    train_indices
)

val_dataset = Subset(
    full_train_evaluation,
    val_indices
)


print("\nDataset split:")
print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


# ============================================================
# ResNet-18 model
# ============================================================

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

# Replace ImageNet 1000-class classifier
# with a 7-class DCID classifier.

input_features = model.fc.in_features

model.fc = nn.Linear(
    input_features,
    num_classes
)

model = model.to(DEVICE)


# ============================================================
# Loss and optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# AMP
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE.type == "cuda")
)


# ============================================================
# Train one epoch
# ============================================================

def train_one_epoch(model, loader):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return epoch_loss, epoch_accuracy


# ============================================================
# Validation
# ============================================================

def evaluate(model, loader):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in tqdm(
            loader,
            desc="Evaluating",
            leave=False
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    loss = running_loss / total
    accuracy = correct / total

    return (
        loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# Training loop
# ============================================================

history = []

best_val_accuracy = -1.0
best_model_state = copy.deepcopy(
    model.state_dict()
)

start_time = time.time()


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    print(
        f"\nEpoch {epoch}/{NUM_EPOCHS}"
    )

    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader
        )
    )

    (
        val_loss,
        val_accuracy,
        _,
        _
    ) = evaluate(
        model,
        val_loader
    )

    scheduler.step(
        val_loss
    )

    print(
        f"Train loss: {train_loss:.4f}"
    )

    print(
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        print(
            "New best model saved."
        )


# ============================================================
# Save training history
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    RESULTS_DIR / "training_history.csv",
    index=False
)


# ============================================================
# Save best model
# ============================================================

model.load_state_dict(
    best_model_state
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_names":
            class_names,

        "best_val_accuracy":
            best_val_accuracy,
    },

    MODEL_DIR / "resnet18_dcid7_best.pth"
)


# ============================================================
# Final test evaluation
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


(
    test_loss,
    test_accuracy,
    test_labels,
    test_predictions,
) = evaluate(
    model,
    test_loader
)


precision, recall, f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="macro",
        zero_division=0
    )
)


print(
    f"Test loss: "
    f"{test_loss:.4f}"
)

print(
    f"Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Macro precision: "
    f"{precision:.4f}"
)

print(
    f"Macro recall: "
    f"{recall:.4f}"
)

print(
    f"Macro F1: "
    f"{f1:.4f}"
)


# ============================================================
# Save final metrics
# ============================================================

metrics_df = pd.DataFrame([
    {
        "accuracy": test_accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }
])

metrics_df.to_csv(
    RESULTS_DIR / "test_metrics.csv",
    index=False
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(11, 9)
)

display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    colorbar=False
)

plt.title(
    "DCID-7 ResNet-18 Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "resnet18_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# Training curves
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    marker="o",
    label="Train"
)

plt.plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    marker="o",
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet-18 Training Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "training_accuracy.png",
    dpi=300
)

plt.close()


plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["train_loss"],
    marker="o",
    label="Train"
)

plt.plot(
    history_df["epoch"],
    history_df["val_loss"],
    marker="o",
    label="Validation"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet-18 Training Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR /
    "training_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# Finish
# ============================================================

elapsed_time = (
    time.time() - start_time
)

print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Training time: "
    f"{elapsed_time / 60:.1f} minutes"
)

print("\nSaved:")
print(
    MODEL_DIR /
    "resnet18_dcid7_best.pth"
)

print(
    RESULTS_DIR /
    "training_history.csv"
)

print(
    RESULTS_DIR /
    "test_metrics.csv"
)

print(
    FIGURE_DIR /
    "resnet18_confusion_matrix.png"
)

print(
    FIGURE_DIR /
    "training_accuracy.png"
)

print(
    FIGURE_DIR /
    "training_loss.png"
)