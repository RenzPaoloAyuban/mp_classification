# Microplastic Fiber vs Fragment Classification Using Fine-Tuned ResNet18

A computer vision classification pipeline built with PyTorch and torchvision that classifies cropped microplastic particles into two morphological categories: **fiber** and **fragment**.

This repository contains a standalone computer vision portfolio project developed out of a broader research interest in microplastic analysis. It provides an end-to-end workflow covering dataset extraction from COCO annotations, transfer learning with ResNet18, controlled ablation experiments, model evaluation on a held-out test set, confidence analysis, and Grad-CAM interpretability visualization.

---

## Overview

Microplastic particles present distinct morphological characteristics under optical microscopy and smartphone imaging. Classifying particle types automatically can support environmental monitoring workflows.

This project implements a binary image classifier targeting two primary microplastic shapes:
* **Fiber**: Elongated, thread-like synthetic structures.
* **Fragment**: Irregular, rigid, angular or sheet-like synthetic particles.

Using a transfer learning strategy based on ResNet18, the pipeline achieves strong performance on the held-out test split, reaching an overall test accuracy of **97.33%** and a Macro F1-score of **97.11%** (73 correct predictions out of 75 test particles).

---

## Dataset

The model is trained and evaluated using annotations from a publicly available dataset on Mendeley Data:

> **Title**: Research data for Microplastic quantification and chemical characterization in salt samples, using stereomicroscopy, smartphone camera and supervised machine learning tools  
> **Source**: Mendeley Data  
> **DOI**: [10.17632/5c4wfd99w8.2](https://doi.org/10.17632/5c4wfd99w8.2)  
> **Version**: 2  
> **Published**: June 15, 2026  
> **License**: [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

### Dataset Attribution and Redistribution Notice
Per the terms of the CC BY-NC-ND 4.0 license, the original images and derived image crops are **not redistributed** in this repository. Users who wish to run the pipeline locally must download the raw dataset directly from Mendeley Data and place it into the project directory as described in the setup instructions below.

---

## Data Processing

The original dataset provides full-frame microscopy and smartphone camera images accompanied by COCO-format JSON annotation files (`_annotations.coco.json`). 

The preprocessing script [`crop_particles.py`](file:///d:/mp_classification/crop_particles.py) automates the extraction of individual particle crops:
1. Parses COCO annotation files for `train`, `valid`, and `test` splits.
2. Filters for target category IDs:
   * Category 3: `fiber`
   * Category 4: `fragment`
3. Extracts bounding boxes `[x, y, width, height]` and converts coordinates to pixel boundaries.
4. Crops bounding box regions from the source images and saves individual PNG files into class-specific directories.

```
data/
├── train/
│   ├── fiber/
│   └── fragment/
├── valid/
│   ├── fiber/
│   └── fragment/
└── test/
    ├── fiber/
    └── fragment/
```

### Dataset Statistics

* **Total full-frame images**: 1,065
* **Total COCO annotations**: 6,375
* **Total extracted target crops**: 720 (440 fibers, 280 fragments)

| Split | Fiber Crops | Fragment Crops | Total Crops |
| :--- | :---: | :---: | :---: |
| **Train** | 305 | 194 | 499 |
| **Validation** | 88 | 58 | 146 |
| **Test** | 47 | 28 | 75 |
| **Total** | **440** | **280** | **720** |

---

## Methodology

The end-to-end technical pipeline follows a structured machine learning workflow:

```
Raw Dataset & COCO JSON
       │
       ▼
Category Filtering (Fiber / Fragment)
       │
       ▼
Bounding Box Cropping (crop_particles.py)
       │
       ▼
Image Preprocessing & Data Augmentation
       │
       ▼
ResNet18 Transfer Learning Experiments
 ├── Baseline (Frozen backbone)
 ├── Class-Weighted Loss (Frozen backbone)
 └── Fine-Tuned (Layer4 + FC unfrozen, Class-Weighted Loss)
       │
       ▼
Validation Set Selection & Model Saving
       │
       ▼
Held-Out Test Set Evaluation
       │
       ▼
Error Analysis & Interpretability
 ├── Confusion Matrix & Classification Report
 ├── Softmax Confidence Analysis (confidence_analysis.py)
 └── Grad-CAM Visualizations (gradcam.py)
```

---

## Model Architecture

The classifier is built on the **ResNet18** architecture pretrained on ImageNet weights (`ResNet18_Weights.DEFAULT`).

### Input Preprocessing and Augmentation
* **Input Resolution**: Resized to \(224 \times 224\) pixels.
* **Normalization**: ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
* **Training Data Augmentation**:
  * Random Horizontal Flip
  * Random Rotation (up to 10 degrees)
* **Validation and Test Pipeline**: Deterministic resizing and normalization without random transformations.

---

## Controlled Experiments

To evaluate the effect of fine-tuning and class weighting on performance, three controlled experiments were conducted in [`experiments.py`](file:///d:/mp_classification/experiments.py):

1. **Baseline ResNet18**:
   * Pretrained feature extraction layers frozen.
   * Only the final linear classifier (`fc`) trained.
   * Standard unweighted `CrossEntropyLoss`.
   * Learning rate: \(1 \times 10^{-4}\), Adam optimizer, 10 epochs.

2. **Class-Weighted ResNet18**:
   * Pretrained feature extraction layers frozen.
   * Only the final linear classifier trained.
   * Inverse-frequency class-weighted `CrossEntropyLoss` applied to compensate for the dataset imbalance (305 train fibers vs 194 train fragments).
   * Learning rate: \(1 \times 10^{-4}\), Adam optimizer, 10 epochs.
   * *Outcome*: Class weighting alone on a frozen backbone degraded generalization performance on this dataset.

3. **Fine-Tuned ResNet18 (Best Model)**:
   * Backbone unfrozen at `layer4` (the final residual block) and the classification head.
   * Inverse-frequency class-weighted `CrossEntropyLoss`.
   * Lower learning rate for fine-tuning: \(1 \times 10^{-5}\), Adam optimizer, 10 epochs.
   * *Outcome*: Unfreezing higher-level spatial feature maps allowed the network to adapt to microplastic textures, yielding superior accuracy.

---

## Results

### Model Comparison on Held-Out Test Set

| Experiment Model | Overall Accuracy | Macro F1-Score | Correct / Total Test Images |
| :--- | :---: | :---: | :---: |
| **Baseline** | 84.00% | 81.62% | 63 / 75 |
| **Class-Weighted** | 76.00% | 72.43% | 57 / 75 |
| **Fine-Tuned (Best)** | **97.33%** | **97.11%** | **73 / 75** |

### Fine-Tuned Model Classification Report

```text
              precision    recall  f1-score   support

       fiber     0.9592    1.0000    0.9792        47
    fragment     1.0000    0.9286    0.9630        28

    accuracy                         0.9733        75
   macro avg     0.9796    0.9643    0.9711        75
weighted avg     0.9744    0.9733    0.9731        75
```

The fine-tuned model achieved perfect recall (100.00%) for fibers and perfect precision (100.00%) for fragments. Both misclassifications were fragments incorrectly predicted as fibers.

---

## Error Analysis and Confidence Analysis

Running [`confidence_analysis.py`](file:///d:/mp_classification/confidence_analysis.py) on the fine-tuned model identified the two misclassified test instances:

### Misclassified Sample 1
* **Image File**: `NextCamera_20230627_220545_jpg.rf.a749fa7fd1c06231c496a3ae8016ee20_9.png`
* **Ground Truth**: `fragment`
* **Predicted Class**: `fiber`
* **Fiber Probability**: 79.36%
* **Fragment Probability**: 20.64%
* **Prediction Confidence**: 79.36%

### Misclassified Sample 2
* **Image File**: `NextCamera_20230627_222358_jpg.rf.374a9f493dd18b3393af392078425f1b_42.png`
* **Ground Truth**: `fragment`
* **Predicted Class**: `fiber`
* **Fiber Probability**: 72.14%
* **Fragment Probability**: 27.86%
* **Prediction Confidence**: 72.14%

---

## Grad-CAM Interpretability Analysis

Grad-CAM (Gradient-weighted Class Activation Mapping) was generated using [`gradcam.py`](file:///d:/mp_classification/gradcam.py) by targeting the final convolutional layer of `layer4` (`model.layer4[-1].conv2`).

Grad-CAM visualizes spatial regions that contribute to the network's final output score. It is an interpretability diagnostic tool rather than a definitive proof of causal reasoning.

```
Original Image                  Grad-CAM Heatmap                Overlay Visualization
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│                       │       │       .::.            │       │       (::)            │
│    Microplastic       │  ──►  │     .::::::.          │  ──►  │    Microplastic       │
│    Particle Crop      │       │       '::'            │       │    Particle Crop      │
│                       │       │                       │       │                       │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

### Visual Findings on Errors

1. **First Misclassification**:
   * Grad-CAM activation focused primarily on background elements adjacent to the particle edge rather than strictly within the particle boundaries.
   * This observation indicates potential reliance on non-particle visual features or background artifacts present in the cropped image.

2. **Second Misclassification**:
   * Grad-CAM displayed diffuse, low-intensity activations spread across the crop without a strongly localized focal region.
   * This behavior suggests that the network lacked a distinct, localized morphological feature to differentiate the particle shape, leading to a default leaning toward the majority class (fiber).

---

## Visualizations and Generated Artifacts

All experimental plots and analytical artifacts are saved in the [`results/`](file:///d:/mp_classification/results) directory:

* **Training and Validation Plots**:
  * [`results/fine_tuned_accuracy.png`](file:///d:/mp_classification/results/fine_tuned_accuracy.png): Accuracy progression per epoch for the fine-tuned model.
  * [`results/fine_tuned_loss.png`](file:///d:/mp_classification/results/fine_tuned_loss.png): Training and validation loss curves for the fine-tuned model.
  * [`results/baseline_accuracy.png`](file:///d:/mp_classification/results/baseline_accuracy.png) & [`results/baseline_loss.png`](file:///d:/mp_classification/results/baseline_loss.png): Curves for the baseline model.
  * [`results/class_weighted_accuracy.png`](file:///d:/mp_classification/results/class_weighted_accuracy.png) & [`results/class_weighted_loss.png`](file:///d:/mp_classification/results/class_weighted_loss.png): Curves for the class-weighted model.

* **Confusion Matrices**:
  * [`results/fine_tuned_confusion_matrix.png`](file:///d:/mp_classification/results/fine_tuned_confusion_matrix.png): Heatmap showing test predictions (73 correct, 2 misclassified).
  * [`results/baseline_confusion_matrix.png`](file:///d:/mp_classification/results/baseline_confusion_matrix.png) & [`results/class_weighted_confusion_matrix.png`](file:///d:/mp_classification/results/class_weighted_confusion_matrix.png): Comparative matrices.

* **Grad-CAM Visualizations**:
  * [`results/gradcam/NextCamera_20230627_220545_jpg.rf.a749fa7fd1c06231c496a3ae8016ee20_9_gradcam.png`](file:///d:/mp_classification/results/gradcam/NextCamera_20230627_220545_jpg.rf.a749fa7fd1c06231c496a3ae8016ee20_9_gradcam.png)
  * [`results/gradcam/NextCamera_20230627_222358_jpg.rf.374a9f493dd18b3393af392078425f1b_42_gradcam.png`](file:///d:/mp_classification/results/gradcam/NextCamera_20230627_222358_jpg.rf.374a9f493dd18b3393af392078425f1b_42_gradcam.png)

* **Saved Weights & Misclassified Crops**:
  * [`results/fine_tuned.pth`](file:///d:/mp_classification/results/fine_tuned.pth): Saved PyTorch state dictionary for the best fine-tuned model.
  * [`results/misclassified/fine_tuned/`](file:///d:/mp_classification/results/misclassified/fine_tuned): Extracted image crops of the misclassified test samples.

---

## Project Structure

```
mp_classification/
├── inspect_dataset.py       # Helper script to inspect COCO category annotations
├── crop_particles.py        # Preprocessing script to extract bounding box crops
├── view_samples.py          # Utility script to display random dataset crops
├── train.py                 # Standalone training script for ResNet18
├── experiments.py           # Controlled evaluation script (Baseline vs Weighted vs Fine-Tuned)
├── gradcam.py               # Grad-CAM heatmap generation script for error interpretability
├── confidence_analysis.py   # Softmax probability analysis for misclassified samples
├── mp/                      # Original Mendeley dataset directory (user supplied)
│   ├── train/
│   ├── valid/
│   └── test/
├── data/                    # Generated cropped particle dataset
│   ├── train/
│   ├── valid/
│   └── test/
└── results/                 # Evaluation output directory
    ├── baseline.pth
    ├── class_weighted.pth
    ├── fine_tuned.pth
    ├── fine_tuned_accuracy.png
    ├── fine_tuned_loss.png
    ├── fine_tuned_confusion_matrix.png
    ├── gradcam/             # Generated Grad-CAM overlay visual outputs
    └── misclassified/       # Misclassified sample crops per model
```

---

## Technology Stack

* **Programming Language**: Python 3.10+
* **Deep Learning Framework**: PyTorch (`torch`, `torchvision`)
* **Computer Vision & Image Processing**: OpenCV (`cv2`), Pillow (`PIL`)
* **Data Evaluation & Metrics**: scikit-learn (`sklearn`), NumPy (`numpy`)
* **Visualization & Plotting**: Matplotlib (`matplotlib`), Seaborn (`seaborn`)

---

## Installation and Setup

### 1. Environment Setup (Windows PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install Required Dependencies

Install the necessary Python packages:

```powershell
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn Pillow
```

---

## Reproduction Workflow

Follow these steps to reproduce the dataset extraction, training experiments, interpretability visualizations, and evaluation results.

### Step 1: Download and Place the Dataset
1. Download the dataset from Mendeley Data: [DOI: 10.17632/5c4wfd99w8.2](https://data.mendeley.com/datasets/5c4wfd99w8.2).
2. Extract the downloaded files into an `mp` folder in the project root:
   ```
   d:\mp_classification\mp\
   ├── train\
   │   ├── _annotations.coco.json
   │   └── *.jpg
   ├── valid\
   │   ├── _annotations.coco.json
   │   └── *.jpg
   └── test\
       ├── _annotations.coco.json
       └── *.jpg
   ```

### Step 2: Crop Particle Annotations
Run the preprocessing script to parse COCO annotations and generate image crops in `data/`:

```powershell
python crop_particles.py
```

### Step 3: Run Model Training & Experiments
Execute the controlled experiment suite (Baseline, Class-Weighted, Fine-Tuned):

```powershell
python experiments.py
```

*Alternatively, to run the standalone trainer:*
```powershell
python train.py
```

### Step 4: Generate Grad-CAM Interpretability Visualizations
Produce Grad-CAM activation heatmaps for misclassified test instances:

```powershell
python gradcam.py
```

### Step 5: Run Confidence Analysis
Print softmax class probabilities for misclassified samples:

```powershell
python confidence_analysis.py
```

---

## Limitations

* **Dataset Scale**: The model is trained on a small dataset (720 total particle crops and 75 test samples). Deep learning models generally benefit from larger sample volumes.
* **Scope of Classification**: The model performs binary morphological classification (fiber vs fragment). It does not perform polymer type identification (e.g., FTIR spectroscopy analysis) nor does it detect or classify other shapes such as beads, pellets, or films.
* **Dataset Dependence**: Reported accuracy reflects performance on this specific held-out test split. Performance may vary on imagery captured under different lighting, magnification, optical settings, or environmental backgrounds.
* **Background Sensitivity**: Bounding box cropping includes small surrounding background regions. As highlighted by Grad-CAM, background artifacts can influence network predictions.
* **No Real-World Deployment Claim**: This project demonstrates computer vision methodology for portfolio evaluation and is not presented as a production-ready or field-tested microplastic quantification system.

---

## Future Work

* **Expanded Morphology Classes**: Incorporate additional categories such as beads, pellets, films, and foams.
* **Background Removal & Segmentation**: Implement semantic segmentation (e.g., Mask R-CNN or U-Net) prior to classification to isolate particles from background slide noise.
* **Cross-Dataset Validation**: Evaluate model generalization across independent external microplastic image databases.
* **Architectural Benchmarking**: Compare ResNet18 against alternative lightweight CNN architectures (EfficientNet, MobileNetV3) and Vision Transformers (ViT).
* **Advanced Data Augmentation**: Apply color jittering, background swapping, and synthetic noise injection to improve robust feature learning.
* **Handling Visually Ambiguous Particles**: Explore multi-label classification or distance-based uncertainty estimation for boundary cases between fibers and elongated fragments.

---

## License and Attribution

* **Code License**: MIT License
* **Dataset Attribution**: The dataset used in this project is provided by Mendeley Data under CC BY-NC-ND 4.0. Refer to [Mendeley Data DOI: 10.17632/5c4wfd99w8.2](https://data.mendeley.com/datasets/5c4wfd99w8.2) for the original research context and citation.
