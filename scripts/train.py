
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = ROOT / "datasets/road_damage/yolo/data.yaml"

PROJECT_DIR = ROOT / "experiments/detection"

def main():

    print("Starting RoadGuard YOLO training...")

    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        patience=10,
        project=str(PROJECT_DIR),
        name="roadguard_baseline",
        seed=42,
        plots=True,
        workers=4
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
