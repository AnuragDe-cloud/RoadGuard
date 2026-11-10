
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets/road_damage/yolo"

CLASS_NAMES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_split(split):
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels = list(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    errors = []
    class_counts = Counter()
    total_objects = 0
    empty_labels = 0

    for label_path in labels:
        for line_number, line in enumerate(
            label_path.read_text().splitlines(), start=1
        ):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label_path.name}:{line_number} - "
                    "Expected 5 values"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label_path.name}:{line_number} - "
                    "Invalid numeric value"
                )
                continue

            if not 0 <= class_id < len(CLASS_NAMES):
                errors.append(
                    f"{label_path.name}:{line_number} - "
                    f"Invalid class ID: {class_id}"
                )
                continue

            if not all(
                0 <= value <= 1
                for value in (x, y, width, height)
            ):
                errors.append(
                    f"{label_path.name}:{line_number} - "
                    "Coordinates outside [0, 1]"
                )
                continue

            if width <= 0 or height <= 0:
                errors.append(
                    f"{label_path.name}:{line_number} - "
                    "Invalid bounding box dimensions"
                )
                continue

            if (
                x - width / 2 < -1e-6
                or x + width / 2 > 1 + 1e-6
                or y - height / 2 < -1e-6
                or y + height / 2 > 1 + 1e-6
            ):
                errors.append(
                    f"{label_path.name}:{line_number} - "
                    "Bounding box exceeds image boundaries"
                )
                continue

            class_counts[class_id] += 1
            total_objects += 1

        if not label_path.read_text().strip():
            empty_labels += 1

    print(f"\n========== {split.upper()} VALIDATION ==========")
    print(f"Images: {len(images)}")
    print(f"Label files: {len(labels)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Empty label files: {empty_labels}")
    print(f"Total valid objects: {total_objects}")
    print(f"Validation errors: {len(errors)}")

    print("\nClass distribution:")

    for class_id, name in enumerate(CLASS_NAMES):
        print(f"{name}: {class_counts[class_id]}")

    if errors:
        print("\nFirst 10 errors:")

        for error in errors[:10]:
            print(error)

    return (
        not missing_labels
        and not missing_images
        and not errors
    )


def main():
    results = []

    for split in ["train", "val"]:
        results.append(validate_split(split))

    print("\n========== FINAL RESULT ==========")

    if all(results):
        print("All YOLO labels passed validation!")
    else:
        print("Some problems were found. Review the output.")


if __name__ == "__main__":
    main()
