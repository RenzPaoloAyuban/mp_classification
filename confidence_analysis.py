import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import os


# =========================
# SETTINGS
# =========================

TEST_DIR = "data/test"
MODEL_PATH = "results/fine_tuned.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASET
# =========================

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

class_names = dataset.classes


# =========================
# LOAD MODEL
# =========================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()


# =========================
# FIND MISCLASSIFICATIONS
# =========================

print("=" * 60)
print("MISCLASSIFIED IMAGE CONFIDENCE ANALYSIS")
print("=" * 60)

errors = 0

for idx in range(len(dataset)):

    image_tensor, true_label = dataset[idx]

    image_batch = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(image_batch)

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

        predicted_label = probabilities.argmax().item()

    # Only analyze wrong predictions
    if predicted_label != true_label:

        errors += 1

        path, _ = dataset.samples[idx]

        fiber_prob = probabilities[
            class_names.index("fiber")
        ].item()

        fragment_prob = probabilities[
            class_names.index("fragment")
        ].item()

        confidence = probabilities[
            predicted_label
        ].item()

        print("\n" + "-" * 60)

        print(
            f"Image: {os.path.basename(path)}"
        )

        print(
            f"Actual class:      {class_names[true_label]}"
        )

        print(
            f"Predicted class:   {class_names[predicted_label]}"
        )

        print(
            f"Fiber probability:     {fiber_prob * 100:.2f}%"
        )

        print(
            f"Fragment probability:  {fragment_prob * 100:.2f}%"
        )

        print(
            f"Prediction confidence: {confidence * 100:.2f}%"
        )


print("\n" + "=" * 60)

print(
    f"Total misclassified images: {errors}"
)

print("=" * 60)