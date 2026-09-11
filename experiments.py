import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
EPOCHS = 10

BASELINE_LR = 0.0001
FINETUNE_LR = 0.00001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MICROPLASTIC FIBER vs FRAGMENT EXPERIMENTS")
print("=" * 60)
print("Device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MISCLASSIFIED_DIR = RESULTS_DIR / "misclassified"
MISCLASSIFIED_DIR.mkdir(exist_ok=True)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    "data/train",
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    "data/valid",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    "data/test",
    transform=eval_transform
)

print("\nClasses:", train_dataset.classes)
print("Train:", len(train_dataset))
print("Valid:", len(valid_dataset))
print("Test:", len(test_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_dataset.targets
)

print("\nClass counts:")
for i, class_name in enumerate(train_dataset.classes):
    print(
        f"  {class_name}: {class_counts[i]}"
    )

# Inverse-frequency weighting
class_weights = len(train_dataset) / (
    len(class_counts) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:", class_weights.cpu().numpy())


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(finetune=False):

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze everything first
    for parameter in model.parameters():
        parameter.requires_grad = False

    if finetune:

        # Fine-tune the last ResNet block
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    # Replace classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    # Classifier must always be trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model.to(DEVICE)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(
    model,
    criterion,
    optimizer,
    model_name
):

    print("\n")
    print("=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_acc": [],
        "valid_acc": []
    }

    best_valid_accuracy = 0.0
    best_weights = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(EPOCHS):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        valid_loss_total = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():

            for images, labels in valid_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                valid_loss_total += (
                    loss.item() * images.size(0)
                )

                predictions = outputs.argmax(
                    dim=1
                )

                valid_correct += (
                    predictions == labels
                ).sum().item()

                valid_total += labels.size(0)

        valid_loss = (
            valid_loss_total / valid_total
        )

        valid_accuracy = (
            valid_correct / valid_total
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["valid_loss"].append(
            valid_loss
        )

        history["train_acc"].append(
            train_accuracy
        )

        history["valid_acc"].append(
            valid_accuracy
        )


        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_accuracy:.4f}"
        )


        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if valid_accuracy > best_valid_accuracy:

            best_valid_accuracy = valid_accuracy

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            print("  → New best model")


    # Restore best validation model
    model.load_state_dict(best_weights)

    # Save model
    filename = (
        RESULTS_DIR
        / f"{model_name}.pth"
    )

    torch.save(
        model.state_dict(),
        filename
    )

    print(
        f"\nSaved: {filename}"
    )

    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    model_name,
    save_misclassified=False
):

    model.eval()

    predictions = []
    labels_all = []
    paths_all = []

    # ImageFolder keeps samples as:
    # (path, class_index)
    for path, label in test_dataset.samples:

        image = Image.open(
            path
        ).convert("RGB")

        image_tensor = eval_transform(
            image
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            output = model(
                image_tensor
            )

            prediction = output.argmax(
                dim=1
            ).item()

        predictions.append(prediction)
        labels_all.append(label)
        paths_all.append(path)


    accuracy = accuracy_score(
        labels_all,
        predictions
    )

    macro_f1 = f1_score(
        labels_all,
        predictions,
        average="macro"
    )

    print("\n")
    print("=" * 60)
    print("TEST RESULTS:", model_name)
    print("=" * 60)

    print(
        classification_report(
            labels_all,
            predictions,
            target_names=train_dataset.classes,
            digits=4
        )
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        labels_all,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )

    plt.title(
        f"{model_name} - Confusion Matrix"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    cm_path = (
        RESULTS_DIR
        / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {cm_path}"
    )


    # --------------------------------------------------------
    # MISCLASSIFIED IMAGES
    # --------------------------------------------------------

    if save_misclassified:

        model_dir = (
            MISCLASSIFIED_DIR
            / model_name
        )

        model_dir.mkdir(
            exist_ok=True
        )

        wrong = []

        for i in range(len(labels_all)):

            if labels_all[i] != predictions[i]:

                wrong.append(i)

                original_path = Path(
                    paths_all[i]
                )

                actual_name = (
                    train_dataset.classes[
                        labels_all[i]
                    ]
                )

                predicted_name = (
                    train_dataset.classes[
                        predictions[i]
                    ]
                )

                filename = (
                    f"{i:03d}_"
                    f"actual-{actual_name}_"
                    f"predicted-{predicted_name}_"
                    f"{original_path.name}"
                )

                destination = (
                    model_dir / filename
                )

                Image.open(
                    original_path
                ).convert("RGB").save(
                    destination
                )

        print(
            f"Misclassified images: {len(wrong)}"
        )

        print(
            f"Saved to: {model_dir}"
        )


    return accuracy, macro_f1


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_history(
    history,
    model_name
):

    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )


    # Accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["valid_acc"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        f"{model_name} - Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULTS_DIR
        / f"{model_name}_accuracy.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["valid_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        f"{model_name} - Loss"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        RESULTS_DIR
        / f"{model_name}_loss.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()


# ============================================================
# EXPERIMENT 1
# BASELINE
# ============================================================

baseline_model = create_model(
    finetune=False
)

baseline_criterion = nn.CrossEntropyLoss()

baseline_optimizer = torch.optim.Adam(
    baseline_model.fc.parameters(),
    lr=BASELINE_LR
)

baseline_model, baseline_history = train_model(
    baseline_model,
    baseline_criterion,
    baseline_optimizer,
    "baseline"
)

plot_history(
    baseline_history,
    "baseline"
)

baseline_accuracy, baseline_f1 = evaluate_model(
    baseline_model,
    "baseline"
)


# ============================================================
# EXPERIMENT 2
# CLASS-WEIGHTED
# ============================================================

weighted_model = create_model(
    finetune=False
)

weighted_criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

weighted_optimizer = torch.optim.Adam(
    weighted_model.fc.parameters(),
    lr=BASELINE_LR
)

weighted_model, weighted_history = train_model(
    weighted_model,
    weighted_criterion,
    weighted_optimizer,
    "class_weighted"
)

plot_history(
    weighted_history,
    "class_weighted"
)

weighted_accuracy, weighted_f1 = evaluate_model(
    weighted_model,
    "class_weighted"
)


# ============================================================
# EXPERIMENT 3
# FINE-TUNED
# ============================================================

finetuned_model = create_model(
    finetune=True
)

finetuned_criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

finetuned_parameters = [
    parameter
    for parameter in finetuned_model.parameters()
    if parameter.requires_grad
]

finetuned_optimizer = torch.optim.Adam(
    finetuned_parameters,
    lr=FINETUNE_LR
)

finetuned_model, finetuned_history = train_model(
    finetuned_model,
    finetuned_criterion,
    finetuned_optimizer,
    "fine_tuned"
)

plot_history(
    finetuned_history,
    "fine_tuned"
)

finetuned_accuracy, finetuned_f1 = evaluate_model(
    finetuned_model,
    "fine_tuned",
    save_misclassified=True
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"{'Model':<20}"
    f"{'Accuracy':<15}"
    f"{'Macro F1':<15}"
)

print("-" * 50)

print(
    f"{'Baseline':<20}"
    f"{baseline_accuracy:.4f}"
    f"         {baseline_f1:.4f}"
)

print(
    f"{'Class Weighted':<20}"
    f"{weighted_accuracy:.4f}"
    f"         {weighted_f1:.4f}"
)

print(
    f"{'Fine Tuned':<20}"
    f"{finetuned_accuracy:.4f}"
    f"         {finetuned_f1:.4f}"
)

print("\nAll results saved in:", RESULTS_DIR.resolve())