# Microplastic Fiber vs Fragment Classification Using Fine-Tuned ResNet18

A computer vision classification pipeline built with PyTorch and torchvision that classifies cropped microplastic particles into two morphological categories: **fiber** and **fragment**.

This repository contains a standalone computer vision portfolio project developed as part of a broader research interest in microplastic analysis. It provides an end-to-end workflow covering dataset preprocessing from COCO annotations, transfer learning with ResNet18, controlled ablation experiments, held-out test evaluation, confidence analysis, and Grad-CAM interpretability visualization.

---

## Overview

Microplastic pollution analysis requires identifying distinct particle morphologies under microscopy and digital imaging. Automatically classifying microplastic shapes can streamline environmental monitoring and dataset processing.

This project implements a binary image classifier targeting two primary microplastic particle shapes:
* **Fiber**: Elongated, thread-like synthetic structures.
* **Fragment**: Irregular, rigid, angular or sheet-like synthetic particles.

Using a transfer learning strategy based on ResNet18, the fine-tuned model achieved **97.33% accuracy** and a Macro F1-score of **97.11%** on the project's held-out test set (73 correct predictions out of 75 test particles).

---

## Dataset

The dataset was obtained from Roboflow Universe:

* **Platform**: Roboflow Universe
* **Dataset Name**: Microplastic v2
* **Version**: Version 2
* **Roboflow Workspace**: `research-new-things-m0fiq/microplastic-v2-wowak`
* **Dataset Page**: [Roboflow Universe Dataset Link](https://universe.roboflow.com/research-new-things-m0fiq/microplastic-v2-wowak/dataset/2)

### Dataset Attribution and License Terms
The dataset metadata file (`mp/README.dataset.txt`) indicates that the dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Per Roboflow Universe distribution guidelines, raw dataset images and cropped files are not included in this repository. Please consult the original Roboflow dataset page linked above for official license details and dataset updates.

---

## Dataset Processing

The dataset contains full-frame microscopy and camera images with COCO-format JSON annotations (`_annotations.coco.json`).

The custom preprocessing script [`crop_particles.py`](file:///d:/mp_classification/crop_particles.py) automates particle extraction:
1. Parses COCO annotation files across `train`, `valid`, and `test` splits.
2. Filters target category IDs:
   * Category 3: `fiber`
   * Category 4: `fragment`
3. Extracts bounding boxes `[x, y, width, height]` and converts coordinates to pixel boundaries.
4. Crops bounding box regions from source images and saves individual PNG files into structured directories:

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

* **Total full-frame images**: 1,477 (Train: 1,065 | Valid: 343 | Test: 69)
* **Total COCO annotations**: 8,497 (Train: 6,375 | Valid: 1,985 | Test: 137)
* **Total extracted target crops**: 720 (440 fibers, 280 fragments)

| Split | Fiber Crops | Fragment Crops | Total Crops |
| :--- | :---: | :---: | :---: |
| **Train** | 305 | 194 | 499 |
| **Validation** | 88 | 58 | 146 |
| **Test** | 47 | 28 | 75 |
| **Total** | **440** | **280** | **720** |

---

## Methodology

The pipeline follows a systematic computer vision methodology from raw dataset extraction to model interpretability:

```
Roboflow Universe Dataset
           │
           ▼
    COCO Annotations
           │
           ▼
Filter Fiber & Fragment Categories
           │
           ▼
Bounding-Box Particle Cropping (crop_particles.py)
           │
           ▼
Train / Validation / Test Dataset Splits
           │
           ▼
Image Preprocessing & Augmentation
           │
           ▼
ResNet18 Transfer Learning Experiments
 ├── Baseline Experiment (Frozen backbone)
 ├── Class-Weighted Experiment (Frozen backbone)
 └── Fine-Tuning Experiment (Layer4 + FC unfrozen, Class-Weighted)
           │
           ▼
Held-Out Test Set Evaluation
           │
           ▼
Confusion Matrix & Classification Metrics
           │
           ▼
Confidence Analysis (confidence_analysis.py)
           │
           ▼
Grad-CAM Error Analysis (gradcam.py)
```

### Why Fine-Tuning Works
Pretrained convolutional neural networks learn hierarchical visual representations. Early layers detect generic low-level features such as simple edges and color transitions, whereas deeper layers learn high-level, task-specific representations.

In this pipeline, unfreezing `layer4` (the final residual block) alongside the classification head allowed the high-level spatial representations to adapt specifically to the visual textures and geometric features of microplastic fibers and fragments.

---

## Model Architecture

The classifier uses a **ResNet18** backbone pretrained on ImageNet weights (`ResNet18_Weights.DEFAULT`).

### Input Preprocessing and Data Augmentation
* **Input Resolution**: Resized to \(224 \times 224\) pixels.
* **Normalization**: Standard ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
* **Training Data Augmentation**:
  * Random Horizontal Flip
  * Random Rotation (up to 10 degrees)
* **Validation and Test Transformations**: Deterministic resizing and normalization without random transformations.

---

## Loss Function Explanation

The project uses **CrossEntropyLoss** for training classification models.

### Understanding CrossEntropyLoss
CrossEntropyLoss evaluates how well predicted class probability distributions match actual target labels.
* A **confident correct prediction** yields a loss close to zero.
* A **confident incorrect prediction** yields a significantly higher penalty loss.

During backward propagation, calculated loss gradients update trainable model parameters via the Adam optimizer to minimize classification loss in subsequent training iterations.

### Class-Weighted Loss
Due to class imbalance in training data (305 fibers vs 194 fragments), inverse-frequency class weighting was integrated into CrossEntropyLoss:
* Higher loss weight assigned to minority class (fragment) errors.
* Lower loss weight assigned to majority class (fiber) errors.

Class weighting adjusts loss penalties during training without altering underlying training image counts or class labels.

---

## Controlled Experiments

Three controlled experiments were conducted in [`experiments.py`](file:///d:/mp_classification/experiments.py):

1. **Baseline ResNet18**:
   * Pretrained feature layers frozen (`requires_grad = False`).
   * Final linear classifier (`fc`) trained.
   * Standard unweighted CrossEntropyLoss.
   * Learning rate: \(1 \times 10^{-4}\), Adam optimizer, 10 epochs.

2. **Class-Weighted ResNet18**:
   * Pretrained feature layers frozen.
   * Final linear classifier trained.
   * Class-weighted CrossEntropyLoss applied.
   * Learning rate: \(1 \times 10^{-4}\), Adam optimizer, 10 epochs.
   * *Finding*: Class weighting alone on a frozen backbone did not improve performance in this setup and resulted in lower generalization accuracy.

3. **Fine-Tuned ResNet18 (Best Model)**:
   * Pretrained `layer4` and final classifier (`fc`) unfrozen.
   * Class-weighted CrossEntropyLoss applied.
   * Lower learning rate for fine-tuning: \(1 \times 10^{-5}\), Adam optimizer, 10 epochs.
   * *Finding*: Fine-tuning deeper representations significantly improved feature discrimination, making it the top-performing model.

---

## Results

### Model Comparison on Held-Out Test Set

| Model | Accuracy | Macro F1 | Correct / Total Test Images |
| :--- | :---: | :---: | :---: |
| **Baseline** | 84.00% | 81.62% | 63 / 75 |
| **Class Weighted** | 76.00% | 72.43% | 57 / 75 |
| **Fine Tuned (Best)** | **97.33%** | **97.11%** | **73 / 75** |

The fine-tuned ResNet18 achieved **97.33% accuracy** on the held-out test set, correctly classifying 73 out of 75 test particles.

### Fine-Tuned Model Classification Report

```text
              precision    recall  f1-score   support

       fiber     0.9592    1.0000    0.9792        47
    fragment     1.0000    0.9286    0.9630        28

    accuracy                         0.9733        75
   macro avg     0.9796    0.9643    0.9711        75
weighted avg     0.9744    0.9733    0.9731        75
```

---

## Error Analysis

The fine-tuned model produced two misclassifications out of 75 test images. In both instances, actual fragment particles were misclassified as fibers.

Softmax probability analysis using [`confidence_analysis.py`](file:///d:/mp_classification/confidence_analysis.py) showed moderately high-confidence incorrect predictions:

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

## Grad-CAM Analysis

Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps were generated via [`gradcam.py`](file:///d:/mp_classification/gradcam.py) targeting the final convolutional layer of `layer4` (`model.layer4[-1].conv2`).

Grad-CAM highlights spatial regions associated with model predictions. It serves as an interpretability diagnostic tool rather than proof of causal reasoning.

```
Original Image                  Grad-CAM Heatmap                Overlay Visualization
┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
│                       │       │       .::.            │       │       (::)            │
│    Microplastic       │  ──►  │     .::::::.          │  ──►  │    Microplastic       │
│    Particle Crop      │       │       '::'            │       │    Particle Crop      │
│                       │       │                       │       │                       │
└───────────────────────┘       └───────────────────────┘       └───────────────────────┘
```

### Visual Insights from Misclassified Samples

1. **Misclassified Image 1**:
   * Grad-CAM activation focused primarily on background region features surrounding the particle rather than strictly within particle boundaries.
   * This observation indicates potential reliance on non-particle visual features or background artifacts present in the cropped sample.

2. **Misclassified Image 2**:
   * Grad-CAM displayed weak, diffuse activation without a clear, strongly highlighted focal region.
   * This indicates that the sample was visually ambiguous for the learned representation, leading the model to default toward the majority fiber class.

---

## Visualizations and Generated Artifacts

All visual outputs and trained models are stored in the [`results/`](file:///d:/mp_classification/results) directory:

* **Training and Validation Curves**:
  * [`results/fine_tuned_accuracy.png`](file:///d:/mp_classification/results/fine_tuned_accuracy.png) & [`results/fine_tuned_loss.png`](file:///d:/mp_classification/results/fine_tuned_loss.png)
  * [`results/baseline_accuracy.png`](file:///d:/mp_classification/results/baseline_accuracy.png) & [`results/baseline_loss.png`](file:///d:/mp_classification/results/baseline_loss.png)
  * [`results/class_weighted_accuracy.png`](file:///d:/mp_classification/results/class_weighted_accuracy.png) & [`results/class_weighted_loss.png`](file:///d:/mp_classification/results/class_weighted_loss.png)

* **Confusion Matrices**:
  * [`results/fine_tuned_confusion_matrix.png`](file:///d:/mp_classification/results/fine_tuned_confusion_matrix.png)
  * [`results/baseline_confusion_matrix.png`](file:///d:/mp_classification/results/baseline_confusion_matrix.png)
  * [`results/class_weighted_confusion_matrix.png`](file:///d:/mp_classification/results/class_weighted_confusion_matrix.png)

* **Grad-CAM Visual Outputs**:
  * [`results/gradcam/NextCamera_20230627_220545_jpg.rf.a749fa7fd1c06231c496a3ae8016ee20_9_gradcam.png`](file:///d:/mp_classification/results/gradcam/NextCamera_20230627_220545_jpg.rf.a749fa7fd1c06231c496a3ae8016ee20_9_gradcam.png)
  * [`results/gradcam/NextCamera_20230627_222358_jpg.rf.374a9f493dd18b3393af392078425f1b_42_gradcam.png`](file:///d:/mp_classification/results/gradcam/NextCamera_20230627_222358_jpg.rf.374a9f493dd18b3393af392078425f1b_42_gradcam.png)

* **Saved Weights & Misclassified Crops**:
  * [`results/fine_tuned.pth`](file:///d:/mp_classification/results/fine_tuned.pth)
  * [`results/misclassified/fine_tuned/`](file:///d:/mp_classification/results/misclassified/fine_tuned)

---

## Project Structure

```
mp_classification/
├── inspect_dataset.py       # Utility to inspect COCO category annotations
├── crop_particles.py        # Script to crop particles using COCO bounding boxes
├── view_samples.py          # Script to display random cropped particle samples
├── train.py                 # Standalone ResNet18 training script
├── experiments.py           # Evaluation script comparing Baseline, Weighted, Fine-Tuned models
├── gradcam.py               # Grad-CAM heatmap generation script
├── confidence_analysis.py   # Softmax probability analyzer for misclassifications
├── mp/                      # Roboflow dataset directory (downloaded by user)
│   ├── train/
│   ├── valid/
│   └── test/
├── data/                    # Generated cropped particle dataset
│   ├── train/
│   ├── valid/
│   └── test/
└── results/                 # Output models, evaluation charts, and heatmaps
    ├── baseline.pth
    ├── class_weighted.pth
    ├── fine_tuned.pth
    ├── fine_tuned_accuracy.png
    ├── fine_tuned_loss.png
    ├── fine_tuned_confusion_matrix.png
    ├── gradcam/             # Generated Grad-CAM overlay outputs
    └── misclassified/       # Extracted misclassified sample images
```

---

## Technology Stack

* **Programming Language**: Python 3.10+
* **Deep Learning Framework**: PyTorch (`torch`, `torchvision`)
* **Computer Vision & Image Processing**: OpenCV (`cv2`), Pillow (`PIL`)
* **Data Metrics & Processing**: scikit-learn (`sklearn`), NumPy (`numpy`)
* **Data Visualization**: Matplotlib (`matplotlib`), Seaborn (`seaborn`)

---

## Installation

### 1. Environment Setup (Windows PowerShell)

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install Project Dependencies

Install required packages:

```powershell
pip install torch torchvision opencv-python numpy matplotlib seaborn scikit-learn Pillow
```

---

## Reproduction

Follow these steps to replicate the experiment pipeline:

1. **Obtain Dataset**: Download the Microplastic v2 Version 2 dataset from [Roboflow Universe](https://universe.roboflow.com/research-new-things-m0fiq/microplastic-v2-wowak/dataset/2).
2. **Extract Files**: Place the dataset files inside an `mp/` folder in the project root:
   ```
   mp/
   ├── train/
   │   ├── _annotations.coco.json
   │   └── *.jpg
   ├── valid/
   │   ├── _annotations.coco.json
   │   └── *.jpg
   └── test/
       ├── _annotations.coco.json
       └── *.jpg
   ```
3. **Crop Particles**: Extract bounding box crops into `data/`:
   ```powershell
   python crop_particles.py
   ```
4. **Run Training Experiments**: Execute the model training and ablation comparison:
   ```powershell
   python experiments.py
   ```
   *(Or run standalone training via `python train.py`)*
5. **Generate Grad-CAM Visualizations**: Create activation heatmaps:
   ```powershell
   python gradcam.py
   ```
6. **Run Confidence Analysis**: Inspect misclassified sample probability scores:
   ```powershell
   python confidence_analysis.py
   ```

---

## Limitations

* **Dataset Size**: The dataset contains 720 total particle crops and a held-out test set of 75 particles.
* **Dataset-Specific Performance**: The reported 97.33% accuracy reflects evaluation strictly on this project's held-out test set and does not imply general performance across all real-world microplastic samples.
* **Scope of Classification**: The model performs binary morphological classification (fiber vs fragment). It is not a complete microplastic detection system and does not perform polymer identification.
* **Background Artifact Sensitivity**: Cropped images include small background regions around particles, which can influence model predictions as observed in Grad-CAM outputs.
* **No External Validation**: The model has not been validated on an independent external microplastic image dataset.
* **Interpretability Scope**: Grad-CAM heatmaps provide visual interpretability evidence but do not establish definitive causal explanations.

---

## Future Work

* **Larger and More Diverse Datasets**: Expand particle sample counts across varied background surfaces and optical conditions.
* **Independent External Validation**: Evaluate generalization capability on separate external microplastic benchmarks.
* **Particle Segmentation**: Implement semantic or instance segmentation prior to classification to isolate particle masks from background slide artifacts.
* **Additional Morphology Classes**: Include extra particle categories such as beads, pellets, films, and foams.
* **Model Comparisons**: Benchmark ResNet18 performance against alternative architectures like EfficientNet, MobileNetV3, and Vision Transformers.
* **Handling Visually Ambiguous Particles**: Incorporate uncertainty estimation for borderline particle shapes.

---

## Dataset Attribution

* **Dataset Source**: [Roboflow Universe - Microplastic v2 Version 2](https://universe.roboflow.com/research-new-things-m0fiq/microplastic-v2-wowak/dataset/2)
* **Roboflow Workspace**: `research-new-things-m0fiq/microplastic-v2-wowak`
* **Dataset Name**: Microplastic v2 (Version 2)

For dataset licensing details and terms of use, please visit the original Roboflow Universe dataset page.

---

## License

This project code is licensed under the MIT License.
