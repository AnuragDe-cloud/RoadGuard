
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL = ROOT / "experiments/detection/roadguard_baseline/weights/best.pt"

SOURCE = ROOT / "datasets/road_damage/yolo/images/val"

OUTPUT = ROOT / "experiments/detection/predictions"


def main():
    model = YOLO(str(MODEL))

    model.predict(
        source=str(SOURCE),
        conf=0.25,
        imgsz=640,
        device=0,
        save=True,
        project=str(OUTPUT),
        name="roadguard_predictions"
    )

    print("RoadGuard inference completed!")


if __name__ == "__main__":
    main()
