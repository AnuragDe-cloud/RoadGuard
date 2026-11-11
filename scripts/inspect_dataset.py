
import json
from pathlib import Path
from collections import Counter

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "road_damage"
    / "raw"
    / "RDD2022_sample"
)


def inspect_split(split_name):

    split_path = DATASET_ROOT / split_name

    image_path = split_path / "img"
    annotation_path = split_path / "ann"

    images = list(image_path.glob("*.jpg"))
    annotations = list(annotation_path.glob("*.json"))

    classes = Counter()
    countries = Counter()

    missing_annotations = []
    invalid_boxes = 0
    total_objects = 0

    annotation_names = {file.name for file in annotations}

    for image in images:

        annotation_file = annotation_path / f"{image.name}.json"

        if annotation_file.name not in annotation_names:
            missing_annotations.append(image.name)
            continue

        try:
            with open(annotation_file, "r") as file:
                data = json.load(file)

            country = image.name.split("_")[0]
            countries[country] += 1

            width = data["size"]["width"]
            height = data["size"]["height"]

            for obj in data.get("objects", []):

                class_name = obj.get("classTitle", "Unknown")
                classes[class_name] += 1
                total_objects += 1

                points = obj.get("points", {})
                exterior = points.get("exterior", [])

                if len(exterior) != 2:
                    invalid_boxes += 1
                    continue

                (x1, y1), (x2, y2) = exterior

                if not (
                    0 <= x1 < x2 <= width
                    and 0 <= y1 < y2 <= height
                ):
                    invalid_boxes += 1

        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"Invalid annotation: {annotation_file.name}")

    print(f"\n{'=' * 45}")
    print(f"{split_name.upper()} DATASET SUMMARY")
    print(f"{'=' * 45}")

    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Missing annotations: {len(missing_annotations)}")
    print(f"Total damage objects: {total_objects}")
    print(f"Invalid bounding boxes: {invalid_boxes}")

    print("\nDamage class distribution:")

    for name, count in classes.most_common():
        print(f"{name}: {count}")

    print("\nCountry distribution:")

    for name, count in countries.most_common():
        print(f"{name}: {count}")


if __name__ == "__main__":

    print("ROADGUARD - DATASET INSPECTOR")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_ROOT}"
        )

    inspect_split("train")
    inspect_split("test")
