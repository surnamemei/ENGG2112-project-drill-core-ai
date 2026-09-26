# ENGG2112 Drill-Core AI

AI-assisted drill-core image analysis for preliminary lithology classification and future material-layer segmentation.

## Project Overview

This project investigates whether computer vision can support preliminary drill-core analysis in civil/geotechnical site investigation.

The current implementation focuses on **lithology classification** using a pretrained **ResNet-18** model. A later extension will investigate **pixel-level segmentation** of different lithological regions using a U-Net-based architecture.

The system is intended as a **decision-support tool**, not a replacement for professional geological or geotechnical judgement.

## Current Status

The DCID-7 classification and robustness pipeline is operational.

Completed work includes:

- DCID-7 dataset verification;
- stratified training/validation split;
- ImageNet-pretrained ResNet-18 transfer learning;
- clean test evaluation;
- noisy/RWDA robustness evaluation;
- per-class performance analysis;
- clean and noisy confusion matrices;
- confidence analysis;
- confusion-pair analysis;
- confidence-threshold selection using a held-out noisy validation subset;
- locked-threshold evaluation on the final noisy test set.

## Dataset

The project uses the **Drill Core Image Dataset (DCID)** developed by Jia-Yu Li, Ji-Zhou Tang, et al.

DCID provides:

- **DCID-7:** 7 lithology classes with 5,000 images per class;
- **DCID-35:** 35 lithology classes with 1,000 images per class;
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
- 224 × 224 model input;
- ImageNet normalization;
- random resized crop;
- horizontal and vertical flips;
- small rotations;
- controlled colour jitter;
- AdamW optimisation;
- mixed-precision CUDA training.

The official DCID-7 training set contains 28,000 images and is further divided into:

- **25,200 training images**
- **2,800 validation images**

The official **7,000-image clean test set** is reserved for final clean evaluation.

### Evaluation Metrics

The classification workflow reports:

- accuracy;
- macro precision;
- macro recall;
- macro F1;
- confusion matrix;
- per-class precision, recall, and F1;
- clean-to-noisy performance degradation;
- prediction-confidence statistics;
- confidence-threshold referral performance.

## Classification Results

### Clean DCID-7 Test Set

| Metric | Result |
|---|---:|
| Images | 7,000 |
| Accuracy | 99.93% |
| Macro Precision | 0.9993 |
| Macro Recall | 0.9993 |
| Macro F1 | 0.9993 |

The clean test set is almost perfectly classified, indicating that the ResNet-18 transfer-learning pipeline performs strongly under the standard DCID-7 benchmark conditions.

### Noisy/RWDA Test Set

| Metric | Result |
|---|---:|
| Images | 2,800 |
| Accuracy | 97.93% |
| Macro Precision | 0.9797 |
| Macro Recall | 0.9793 |
| Macro F1 | 0.9793 |

Relative to the clean test set:

- accuracy decreased by approximately **2.0 percentage points**;
- macro F1 decreased by approximately **0.020**.

This indicates a measurable robustness gap under degraded image conditions while overall classification performance remains high.

## Per-Class Robustness

On the noisy/RWDA test set, class-level F1 scores remained high across all seven lithologies.

Representative noisy-set F1 scores include:

- Red sandstone: **0.9766**
- Light sandstone: **0.9851**
- Gray siltstone: **0.9887**
- Mudstone: **0.9739**
- Granite: **0.9770**
- Basalt: **0.9809**
- Marble: **0.9729**

The lowest F1 was observed for Marble, followed by Mudstone, Red sandstone, and Granite.

Granite showed particularly high precision but lower recall, indicating that true Granite samples were more likely to be misclassified than non-Granite samples were to be incorrectly labelled as Granite.

## Confusion Analysis

The most common noisy/RWDA confusion pairs included:

1. Granite → Red sandstone
2. Basalt → Marble
3. Granite → Marble
4. Marble → Mudstone
5. Gray siltstone → Mudstone

These results show that classification errors under degraded conditions are concentrated in specific lithology pairs rather than being uniformly distributed.

The current analysis reports these observed confusion patterns only; geological or visual explanations require further inspection of representative misclassified images.

## Prediction Confidence

Prediction confidence differed substantially between correct and incorrect predictions on the noisy/RWDA test set.

| Prediction outcome | Mean confidence |
|---|---:|
| Correct predictions | 0.992 |
| Incorrect predictions | 0.736 |

This supports the use of confidence as one component of a human-in-the-loop review process.

However, confidence is not treated as a guarantee of correctness, because some incorrect predictions may still have relatively high confidence.

## Human-in-the-Loop Confidence Threshold

A confidence threshold was selected using a **held-out subset of the noisy training data**, rather than the final noisy test set.

The selected operating threshold was:

- **confidence threshold: 0.97**

This threshold was then locked before final noisy-test evaluation.

### Final Locked-Threshold Test

| Metric | Result |
|---|---:|
| Noisy test images | 2,800 |
| Base accuracy | 97.93% |
| Referred for human review | 175 |
| Referral rate | 6.25% |
| Classification errors captured | 49 / 58 |
| Error capture rate | 84.48% |
| Correct predictions referred | 126 / 2742 |
| Correct referral rate | 4.60% |
| Automatically accepted predictions | 2,625 |
| Incorrect accepted predictions | 9 |
| Accuracy of automatically accepted predictions | 99.66% |

This provides a preliminary operating strategy in which a small proportion of lower-confidence cases are referred for human review while the remaining automatically accepted predictions achieve higher accuracy.

The threshold is a project-specific experimental operating point and is not presented as a universally safe deployment threshold.

## Repository Structure

```text
ENGG2112-project-drill-core-ai/
├── src/
│   ├── check_dataset.py
│   ├── train_resnet18.py
│   ├── evaluate_robustness.py
│   ├── analyze_robustness.py
│   ├── confidence_threshold_analysis.py
│   ├── select_confidence_threshold.py
│   ├── final_threshold_test.py
│   ├── export_failure_cases.py
│   ├── export_noisy_predictions.py
│   └── make_report_figures.py
├── results/
│   ├── figures/
│   ├── models/
│   ├── robustness_analysis/
│   ├── training_history.csv
│   └── test_metrics.csv
├── requirements.txt
├── .gitignore
└── README.md
```

The local `DCID/` directory and trained model checkpoints should not be committed to Git unless explicitly required.

## Setup

Create and activate a Python virtual environment:

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

Evaluate clean/noisy robustness:

```bash
python src/evaluate_robustness.py
```

Run detailed robustness analysis:

```bash
python src/analyze_robustness.py
```

Explore confidence thresholds:

```bash
python src/confidence_threshold_analysis.py
```

Select the threshold using noisy validation data:

```bash
python src/select_confidence_threshold.py
```

Run the final locked-threshold test:

```bash
python src/final_threshold_test.py
```

Export per-image noisy-test predictions and generate the final-report figures:

```bash
python src/export_noisy_predictions.py
python src/make_report_figures.py
```

## Next Steps

The classification stage is largely complete. The highest-value remaining work is:

1. inspect and save representative misclassified images for qualitative failure analysis;
2. generate a small number of publication-quality summary figures for the final report;
3. document classification limitations and dataset-domain limitations;
4. identify or create a supplementary dataset with pixel-level lithology masks;
5. implement a U-Net-based lithology segmentation extension;
6. evaluate segmentation using mIoU and Dice score;
7. compare classification and segmentation as complementary outputs for preliminary drill-core analysis;
8. integrate results into the ENGG2112 final report and demonstration.

Optional classification extensions include:

- probability calibration;
- limited RWDA retraining or augmentation experiments;
- DCID-35 evaluation if time and scope permit.

These are secondary to the segmentation extension and should not be added if they distract from the main project scope.

## Intended Use and Limitations

The current model is a preliminary classification system trained and evaluated on benchmark drill-core images.

High benchmark performance does not demonstrate reliable deployment across all:

- geological sites;
- rock weathering states;
- rock-strength conditions;
- camera systems;
- lighting conditions;
- image acquisition procedures;
- field environments.

The noisy/RWDA evaluation improves the robustness analysis but does not reproduce every real worksite condition.

The current model predicts lithology classes only. It does not directly estimate engineering properties such as weathering grade, rock strength, fracture condition, or site stability.

Predictions should therefore be treated as **decision support only**. Uncertain or safety-relevant cases should remain subject to review by qualified engineers or geologists.

## Citation

If using DCID, cite the original dataset paper:

> J.-Y. Li, J.-Z. Tang, X.-Z. Zhao, B. Fan, W.-Y. Jiang, S.-Y. Song, J.-B. Li, K.-D. Chen, and Z.-G. Zhao, “A large-scale, high-quality dataset for lithology identification: Construction and applications,” *Petroleum Science*, 2025. DOI: 10.1016/j.petsci.2025.04.013.
