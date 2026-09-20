# ENGG2112 Drill-Core AI

AI-assisted drill-core image analysis for preliminary lithology classification and future material-layer segmentation.

## Project Overview

This project investigates whether computer vision can support preliminary drill-core analysis in civil/geotechnical site investigation. The current implementation focuses on **lithology classification** using a pretrained **ResNet-18** model. A later extension will investigate **pixel-level segmentation** of different lithological regions using a U-Net-based architecture.

The system is intended as a **decision-support tool**, not a replacement for professional geological or geotechnical judgement.

## Current Status

The DCID-7 classification pipeline is now operational:

- dataset loading and verification complete;
- stratified training/validation split implemented;
- ImageNet-pretrained ResNet-18 fine-tuned on DCID-7;
- clean test evaluation complete;
- noisy/RWDA robustness evaluation complete;
- training curves and confusion matrix generated.

### Current Classification Results

| Evaluation set | Images | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Clean DCID-7 test | 7,000 | 99.93% | 0.9993 | 0.9993 | 0.9993 |
| Noisy/RWDA test | 2,800 | 97.93% | 0.9797 | 0.9793 | 0.9793 |

The noisy/RWDA evaluation produced an accuracy decrease of approximately **2.0 percentage points** and a macro-F1 decrease of approximately **0.020** relative to the clean test set.

These are preliminary project results and will be followed by more detailed per-class, confidence, and error analysis.

## Dataset

The project uses the **Drill Core Image Dataset (DCID)** developed by Jia-Yu Li, Ji-Zhou Tang, et al.

DCID provides:

- **DCID-7:** 7 lithology classes, 5,000 images per class;
- **DCID-35:** 35 lithology classes, 1,000 images per class;
- 512 × 512 RGB drill-core images;
- official 80:20 train/test splits;
- noisy / real-world data augmentation (RWDA) variants.

The current project starts with **DCID-7**.

### DCID-7 Classes

1. Red sandstone
2. Light sandstone
3. Gray siltstone
4. Mudstone
5. Granite
6. Basalt
7. Marble

### Dataset Source

Official repository:  
https://github.com/JiayuLi1120/drill-core-image-dataset

Hugging Face dataset:  
https://huggingface.co/datasets/168sir/drill-core-image-dataset

Paper DOI:  
https://doi.org/10.1016/j.petsci.2025.04.013

License: **CC BY-NC 4.0**

The dataset itself is **not stored in this repository**. Download it separately and place it under:

```text
DCID/
├── DCID-512-7/
│   ├── train/
│   └── test/
├── noise-512-7/
│   ├── train/
│   └── test/
├── DCID-512-35/
└── noise-512-35/
```

## Current Method

### Classification

The classification pipeline uses:

- ResNet-18 with ImageNet pretrained weights;
- transfer learning for 7-class lithology classification;
- input resolution of 224 × 224;
- ImageNet normalization;
- random resized crop;
- horizontal and vertical flips;
- small rotations;
- controlled colour jitter;
- AdamW optimisation;
- mixed-precision CUDA training.

The official DCID-7 training set contains 28,000 images. It is further divided into:

- **25,200 training images**
- **2,800 validation images**

The official **7,000-image test set** is reserved for final evaluation.

### Evaluation Metrics

The current classification evaluation reports:

- accuracy;
- macro precision;
- macro recall;
- macro F1;
- confusion matrix;
- robustness degradation from clean to noisy/RWDA images.

## Repository Structure

```text
ENGG2112-project-drill-core-ai/
├── src/
│   ├── check_dataset.py
│   └── train_resnet18.py
├── results/
│   ├── figures/
│   ├── models/
│   ├── training_history.csv
│   └── test_metrics.csv
├── requirements.txt
├── .gitignore
└── README.md
```

The local `DCID/` directory and trained model checkpoints are excluded from Git where appropriate.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-capable NVIDIA GPU is recommended for training.

## Running the Current Pipeline

Check the dataset:

```bash
python src/check_dataset.py
```

Train and evaluate ResNet-18:

```bash
python src/train_resnet18.py
```

The training script saves:

- best validation checkpoint;
- training history;
- final test metrics;
- confusion matrix;
- training accuracy curve;
- training loss curve.

## Next Steps

1. Generate clean and noisy confusion matrices.
2. Calculate per-class precision, recall, and F1.
3. Analyse prediction confidence for correct and incorrect predictions.
4. Identify the lithology pairs most commonly confused under noisy conditions.
5. Evaluate whether additional RWDA-based training is useful.
6. Investigate a supplementary segmentation dataset with pixel-level lithology masks.
7. Develop a U-Net-based material-layer segmentation extension.
8. Integrate results into the ENGG2112 final report and demonstration.

## Intended Use and Limitations

The current model is a preliminary classification system trained on a benchmark image dataset. High benchmark performance does not by itself demonstrate reliable deployment across all geological sites, cameras, lighting conditions, weathering states, or rock-strength conditions.

Predictions should therefore be treated as **decision support only**. Uncertain or safety-relevant cases should remain subject to review by qualified engineers or geologists.

## Citation

If using DCID, cite the original dataset paper:

> J.-Y. Li, J.-Z. Tang, X.-Z. Zhao, B. Fan, W.-Y. Jiang, S.-Y. Song, J.-B. Li, K.-D. Chen, and Z.-G. Zhao, “A large-scale, high-quality dataset for lithology identification: Construction and applications,” *Petroleum Science*, 2025. DOI: 10.1016/j.petsci.2025.04.013.
