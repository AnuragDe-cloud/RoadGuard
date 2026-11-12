
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT / "experiments/detection/roadguard_baseline/weights/best.pt"
)

DATA_YAML = ROOT / "datasets/road_damage/yolo/data.yaml"

OUTPUT_DIR = ROOT / "experiments/detection/evaluation"


def main():
    print("Evaluating RoadGuard baseline model...")

    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        conf=0.001,
        plots=True,
        project=str(OUTPUT_DIR),
        name="baseline_evaluation"
    )

    print("\n========== ROADGUARD EVALUATION ==========")

    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
