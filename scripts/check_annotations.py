import json
from pathlib import Path

root = Path("datasets/road_damage/raw/RDD2022_sample")

for split in ["train", "test"]:

    folder = root / split / "ann"
    files = list(folder.glob("*.json"))

    print(f"\n--- {split.upper()} ---")

    for file in files[:2]:

        with open(file, "r") as f:
            data = json.load(f)

        print(f"\nFile: {file.name}")
        print("Keys:", list(data.keys()))
        print("Objects:", len(data.get("objects", [])))
        print(json.dumps(data, indent=2)[:1000])
