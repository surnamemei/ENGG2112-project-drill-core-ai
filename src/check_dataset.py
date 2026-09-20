from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_ROOT = Path("DCID/DCID-512-7")

TRAIN_DIR = DATA_ROOT / "train"
TEST_DIR = DATA_ROOT / "test"


# --------------------------------------------------
# Basic transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# --------------------------------------------------
# Load datasets
# --------------------------------------------------
train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=transform
)


# --------------------------------------------------
# Print dataset information
# --------------------------------------------------
print("=" * 60)
print("DCID-7 DATASET CHECK")
print("=" * 60)

print("\nDataset root:")
print(DATA_ROOT.resolve())

print("\nClasses:")
for index, class_name in enumerate(train_dataset.classes):
    print(f"{index}: {class_name}")

print("\nNumber of classes:")
print(len(train_dataset.classes))

print("\nTrain images:")
print(len(train_dataset))

print("\nTest images:")
print(len(test_dataset))


# --------------------------------------------------
# Check class distribution
# --------------------------------------------------
print("\nTrain class counts:")

train_targets = torch.tensor(train_dataset.targets)

for class_index, class_name in enumerate(train_dataset.classes):
    count = int((train_targets == class_index).sum())
    print(f"{class_name}: {count}")

print("\nTest class counts:")

test_targets = torch.tensor(test_dataset.targets)

for class_index, class_name in enumerate(test_dataset.classes):
    count = int((test_targets == class_index).sum())
    print(f"{class_name}: {count}")


# --------------------------------------------------
# DataLoader test
# --------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

images, labels = next(iter(train_loader))

print("\nBatch check:")
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Example labels:", labels[:10].tolist())

print("\nImage tensor range:")
print("Min:", images.min().item())
print("Max:", images.max().item())

print("\nDataset check complete.")