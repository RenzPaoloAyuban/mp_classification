import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# =========================
# SETTINGS
# =========================

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# =========================
# IMAGE TRANSFORMS
# =========================

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

valid_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# DATASETS
# =========================

train_dataset = datasets.ImageFolder(
    "data/train",
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    "data/valid",
    transform=valid_test_transform
)

test_dataset = datasets.ImageFolder(
    "data/test",
    transform=valid_test_transform
)

print("\nClasses:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(valid_dataset))
print("Test images:", len(test_dataset))

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

# =========================
# MODEL
# =========================

print("\nLoading ResNet18...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace final layer
model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model = model.to(DEVICE)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# =========================
# TRAINING
# =========================

best_valid_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total

    # =====================
    # VALIDATION
    # =====================

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    valid_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {running_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Valid Acc: {valid_accuracy:.4f}"
    )

    # Save best model
    if valid_accuracy > best_valid_accuracy:

        best_valid_accuracy = valid_accuracy

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("  → Saved best model")

# =========================
# TEST
# =========================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=DEVICE
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

# =========================
# RESULTS
# =========================

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=train_dataset.classes,
        digits=4
    )
)

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Fiber vs Fragment - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    train_dataset.classes
)

plt.yticks(
    [0, 1],
    train_dataset.classes
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nSaved:")
print("  best_model.pth")
print("  confusion_matrix.png")