
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets/road_damage/yolo"

CLASS_NAMES = [
    "Longitudinal crack",
    "Transverse crack",
    "Alligator crack",
    "Other corruption",
    "Pothole"
]

def visualize_samples(split="train", count=6):

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = list(image_dir.glob("*.jpg"))

    if not images:
        raise FileNotFoundError("No images found!")

    samples = random.sample(images, min(count, len(images)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, image_path in zip(axes, samples):

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width, _ = image.shape

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():

            for line in label_path.read_text().splitlines():

                if not line.strip():
                    continue

                class_id, x, y, w, h = map(
                    float, line.split()
                )

                x_center = x * width
                y_center = y * height

                box_width = w * width
                box_height = h * height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                name = CLASS_NAMES[int(class_id)]

                cv2.putText(
                    image,
                    name,
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

        ax.imshow(image)
        ax.set_title(image_path.name)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    output_path = ROOT / "datasets/road_damage/yolo/visualization.jpg"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    visualize_samples("train", 6)
