import json
from pathlib import Path
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

# Only these COCO categories will be used
TARGET_CATEGORIES = {
    3: "fiber",
    4: "fragment"
}

SPLITS = ["train", "valid", "test"]

OUTPUT_DIR = Path("data")


# ============================================================
# FUNCTION: PROCESS ONE SPLIT
# ============================================================

def process_split(split):
    split_dir = Path("mp") / split
    annotation_file = split_dir / "_annotations.coco.json"

    if not annotation_file.exists():
        print(f"[WARNING] Annotation file not found: {annotation_file}")
        return

    print(f"\nProcessing {split}...")

    # Load COCO annotations
    with open(annotation_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Map image ID -> image information
    images = {
        image["id"]: image
        for image in coco["images"]
    }

    # Create output directories
    for category_name in TARGET_CATEGORIES.values():
        (OUTPUT_DIR / split / category_name).mkdir(
            parents=True,
            exist_ok=True
        )

    counts = {
        "fiber": 0,
        "fragment": 0
    }

    skipped = 0

    # Process every annotation
    for annotation in coco["annotations"]:

        category_id = annotation["category_id"]

        # Ignore bead, Microplastic, and other categories
        if category_id not in TARGET_CATEGORIES:
            continue

        category_name = TARGET_CATEGORIES[category_id]

        image_id = annotation["image_id"]

        if image_id not in images:
            skipped += 1
            continue

        image_info = images[image_id]

        image_path = split_dir / image_info["file_name"]

        if not image_path.exists():
            print(f"[WARNING] Image not found: {image_path}")
            skipped += 1
            continue

        try:
            # Open image
            image = Image.open(image_path).convert("RGB")

            # COCO bbox format:
            # [x, y, width, height]
            x, y, width, height = annotation["bbox"]

            # Convert coordinates to integers
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(image.width, int(x + width))
            y2 = min(image.height, int(y + height))

            # Ignore invalid bounding boxes
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue

            # Crop particle
            crop = image.crop((x1, y1, x2, y2))

            # Generate unique filename
            crop_number = counts[category_name]

            output_filename = (
                f"{image_path.stem}_"
                f"{annotation['id']}.png"
            )

            output_path = (
                OUTPUT_DIR
                / split
                / category_name
                / output_filename
            )

            # Save crop
            crop.save(output_path)

            counts[category_name] += 1

        except Exception as e:
            print(
                f"[ERROR] Could not process "
                f"{image_path}: {e}"
            )
            skipped += 1

    # Summary
    print(f"  Fiber:    {counts['fiber']}")
    print(f"  Fragment: {counts['fragment']}")
    print(f"  Skipped:  {skipped}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MICROPLASTIC PARTICLE CROPPER")
    print("=" * 60)

    for split in SPLITS:
        process_split(split)

    print("\nDone!")
    print(f"Output saved to: {OUTPUT_DIR.resolve()}")