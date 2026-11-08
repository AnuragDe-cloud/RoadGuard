
import json
import random
import shutil
from pathlib import Path

# CONFIGURATION

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "datasets/road_damage/raw/RDD2022_sample"
OUTPUT = ROOT / "datasets/road_damage/yolo"

TRAIN_IMAGES = SOURCE / "train/img"
TRAIN_ANN = SOURCE / "train/ann"

VAL_RATIO = 0.20
RANDOM_SEED = 42

CLASS_NAMES = [
    "longitudinal crack",
    "transverse crack",
    "alligator crack",
    "other corruption",
    "pothole"
]

CLASS_MAP = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}


# --------------------------------------------------
# CONVERT ONE ANNOTATION
# --------------------------------------------------

def convert_annotation(annotation_path, image_path):
    with open(annotation_path, "r") as file:
        data = json.load(file)

    width = data["size"]["width"]
    height = data["size"]["height"]

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions: {annotation_path}")

    labels = []

    for obj in data.get("objects", []):
        class_name = obj.get("classTitle")

        if class_name not in CLASS_MAP:
            raise ValueError(
                f"Unknown class '{class_name}' in {annotation_path}"
            )

        if obj.get("geometryType") != "rectangle":
            raise ValueError(
                f"Unsupported geometry '{obj.get('geometryType')}' "
                f"in {annotation_path}"
            )

        points = obj.get("points", {}).get("exterior", [])

        if len(points) < 2:
            raise ValueError(
                f"Invalid bounding box in {annotation_path}"
            )

        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        if not (
            0 <= x_min < x_max <= width
            and 0 <= y_min < y_max <= height
        ):
            raise ValueError(
                f"Bounding box outside image boundaries: {annotation_path}"
            )

        # Convert to normalized YOLO format
        x_center = ((x_min + x_max) / 2) / width
        y_center = ((y_min + y_max) / 2) / height

        box_width = (x_max - x_min) / width
        box_height = (y_max - y_min) / height

        class_id = CLASS_MAP[class_name]

        labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    return labels


# --------------------------------------------------
# PREPARE OUTPUT DIRECTORIES
# --------------------------------------------------

def create_directories():
    for split in ["train", "val"]:
        (OUTPUT / "images" / split).mkdir(
            parents=True, exist_ok=True
        )

        (OUTPUT / "labels" / split).mkdir(
            parents=True, exist_ok=True
        )


# --------------------------------------------------
# MAIN CONVERSION
# --------------------------------------------------

def main():
    if not TRAIN_IMAGES.exists() or not TRAIN_ANN.exists():
        raise FileNotFoundError("Training dataset folders not found.")

    annotation_files = sorted(TRAIN_ANN.glob("*.json"))

    if not annotation_files:
        raise FileNotFoundError("No annotation JSON files found.")

    samples = []

    for annotation_path in annotation_files:
        # Example: Japan_004597.jpg.json -> Japan_004597.jpg
        image_name = annotation_path.name[:-5]
        image_path = TRAIN_IMAGES / image_name

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image missing for annotation: {annotation_path.name}"
            )

        samples.append((image_path, annotation_path))

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    val_count = round(len(samples) * VAL_RATIO)

    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    create_directories()

    class_counts = {name: 0 for name in CLASS_NAMES}

    for split, split_samples in [
        ("train", train_samples),
        ("val", val_samples)
    ]:
        for image_path, annotation_path in split_samples:
            labels = convert_annotation(
                annotation_path, image_path
            )

            destination_image = OUTPUT / "images" / split / image_path.name
            destination_label = (
                OUTPUT / "labels" / split /
                f"{image_path.stem}.txt"
            )

            shutil.copy2(image_path, destination_image)

            destination_label.write_text(
                "\n".join(labels) + ("\n" if labels else "")
            )

            with open(annotation_path, "r") as file:
                data = json.load(file)

            for obj in data.get("objects", []):
                class_counts[obj["classTitle"]] += 1

        # CREATE YOLO DATA CONFIGURATION

    yaml_content = (
        f"path: {OUTPUT.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
    )

    for idx, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {idx}: '{name}'\n"

    (OUTPUT / "data.yaml").write_text(yaml_content)

    print("\n========== YOLO CONVERSION COMPLETE ==========")
    print(f"Total images: {len(samples)}")
    print(f"Training images: {len(train_samples)}")
    print(f"Validation images: {len(val_samples)}")

    print("\nClass distribution:")
    for name, count in class_counts.items():
        print(f"{name}: {count}")

    print(f"\nOutput directory: {OUTPUT}")


if __name__ == "__main__":
    main()
