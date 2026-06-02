from pathlib import Path
import csv
from collections import defaultdict

PROJECT_ROOT = Path.cwd()

DATASETS_DIR = PROJECT_ROOT / "datasets"
KAGGLE_UPLOAD_DIR = PROJECT_ROOT / "kaggle_upload"
SPLITS_DIR = KAGGLE_UPLOAD_DIR / "splits"

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

manifest_path = SPLITS_DIR / "split_manifest.csv"
summary_path = SPLITS_DIR / "split_summary.csv"

rows = []
summary = defaultdict(lambda: defaultdict(int))

for split_name in ["train", "val", "test"]:
    split_dir = DATASETS_DIR / split_name

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() not in IMG_EXTENSIONS:
                continue

            kaggle_full_path = f"full/{class_name}/{image_path.name}"

            rows.append({
                "split": split_name,
                "class_name": class_name,
                "filename": image_path.name,
                "relative_path": kaggle_full_path
            })

            summary[class_name][split_name] += 1

with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["split", "class_name", "filename", "relative_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["class_name", "train", "val", "test", "total"]
    )
    writer.writeheader()

    for class_name in sorted(summary.keys()):
        train_count = summary[class_name]["train"]
        val_count = summary[class_name]["val"]
        test_count = summary[class_name]["test"]

        writer.writerow({
            "class_name": class_name,
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": train_count + val_count + test_count
        })

print("Created:", manifest_path)
print("Created:", summary_path)
print("Total rows:", len(rows))