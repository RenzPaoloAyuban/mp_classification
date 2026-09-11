from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

classes = ["fiber", "fragment"]

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for row, class_name in enumerate(classes):
    folder = Path("data/train") / class_name
    images = list(folder.glob("*.png"))

    samples = random.sample(images, min(5, len(images)))

    for col, image_path in enumerate(samples):
        image = Image.open(image_path)

        axes[row, col].imshow(image)
        axes[row, col].set_title(class_name)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()