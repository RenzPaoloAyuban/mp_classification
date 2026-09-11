import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms, models
import torch.nn as nn


# =========================
# SETTINGS
# =========================

TEST_DIR = "data/test"
MODEL_PATH = "results/fine_tuned.pth"
OUTPUT_DIR = "results/gradcam"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

print("Classes:", class_names)
print("Test images:", len(dataset))


# =========================
# LOAD MODEL
# =========================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()


# =========================
# GRAD-CAM HOOKS
# =========================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Use the last convolutional layer
target_layer = model.layer4[-1].conv2

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# =========================
# GRAD-CAM FUNCTION
# =========================

def generate_gradcam(image_tensor, target_class):

    global activations, gradients

    model.zero_grad()

    output = model(image_tensor)

    score = output[0, target_class]

    score.backward()

    # Average gradients spatially
    weights = gradients.mean(
        dim=(2, 3),
        keepdim=True
    )

    # Weighted combination of feature maps
    cam = (weights * activations).sum(dim=1)

    # ReLU: only keep positive influence
    cam = torch.relu(cam)

    # Convert to numpy
    cam = cam.detach().cpu().numpy()[0]

    # Normalize
    cam -= cam.min()

    if cam.max() != 0:
        cam /= cam.max()

    return cam


# =========================
# SELECT EXAMPLES
# =========================

examples = []

for idx in range(len(dataset)):

    image_tensor, true_label = dataset[idx]

    image_batch = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_batch)

    predicted_label = output.argmax(dim=1).item()

    # Keep misclassified images
    if predicted_label != true_label:
        examples.append(
            (idx, true_label, predicted_label)
        )


# If there are no errors, take first few images
if len(examples) == 0:
    examples = [
        (
            i,
            dataset[i][1],
            model(
                dataset[i][0]
                .unsqueeze(0)
                .to(DEVICE)
            ).argmax(dim=1).item()
        )
        for i in range(min(4, len(dataset)))
    ]


print("\nGrad-CAM examples:")

for idx, true_label, predicted_label in examples:

    path, _ = dataset.samples[idx]

    print(
        os.path.basename(path),
        "| Actual:",
        class_names[true_label],
        "| Predicted:",
        class_names[predicted_label]
    )


# =========================
# GENERATE HEATMAPS
# =========================

for idx, true_label, predicted_label in examples:

    image_tensor, _ = dataset[idx]

    image_batch = image_tensor.unsqueeze(0).to(DEVICE)

    # Generate Grad-CAM for predicted class
    cam = generate_gradcam(
        image_batch,
        predicted_label
    )

    # Original image
    path, _ = dataset.samples[idx]

    original = cv2.imread(path)

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    # Resize CAM
    cam_resized = cv2.resize(
        cam,
        (original.shape[1], original.shape[0])
    )

    # Convert to heatmap
    heatmap = np.uint8(
        255 * cam_resized
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    # Save
    filename = os.path.splitext(
        os.path.basename(path)
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        filename + "_gradcam.png"
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title(
        f"Original\nActual: {class_names[true_label]}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap="jet")
    plt.title(
        f"Grad-CAM\nPredicted: {class_names[predicted_label]}"
    )
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output_path)


print("\nDone!")