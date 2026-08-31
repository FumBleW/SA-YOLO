import os
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = Path(
    os.getenv("SA_YOLO_WEIGHTS", ROOT / "runs" / "detect" / "sa_yolo" / "weights" / "best.pt")
).expanduser()
DATA_YAML = Path(os.getenv("SA_YOLO_DATA", ROOT / "datasets" / "kaist_barcode" / "data.yaml")).expanduser()
OUTPUT_DIR = Path(
    os.getenv("SA_YOLO_TEST_OUTPUT", ROOT / "outputs" / "test_set_evaluation_results")
).expanduser()

VISUALIZE_SAMPLES = 50
CONFIDENCE = 0.25


def main():
    if not WEIGHTS.is_file():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS}\nSet SA_YOLO_WEIGHTS to override this path.")
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}\nSet SA_YOLO_DATA to override this path.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    visualization_dir = OUTPUT_DIR / "visualizations"
    visualization_dir.mkdir(exist_ok=True)

    model = YOLO(str(WEIGHTS))
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=32,
        conf=0.001,
        iou=0.6,
        device=0,
        plots=True,
        save_json=True,
        project=OUTPUT_DIR,
        name="quantitative_eval",
    )

    model_info = model.info()
    report = (
        f"Model: {WEIGHTS}\n"
        f"Dataset: {DATA_YAML}\n\n"
        f"Precision: {metrics.box.mp:.4f}\n"
        f"Recall: {metrics.box.mr:.4f}\n"
        f"mAP50: {metrics.box.map50:.4f}\n"
        f"mAP50-95: {metrics.box.map:.4f}\n\n"
        f"Parameters: {model_info[0] / 1e6:.2f} M\n"
        f"GFLOPs: {model_info[1]:.2f}\n"
        f"Inference: {metrics.speed['inference']:.2f} ms/image\n"
    )
    print(report)
    (OUTPUT_DIR / "test_set_metrics.txt").write_text(report, encoding="utf-8")

    with DATA_YAML.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    dataset_root = Path(data_config["path"])
    if not dataset_root.is_absolute():
        dataset_root = DATA_YAML.parent / dataset_root
    test_image_dir = dataset_root / data_config["test"]

    image_paths = [
        path for path in test_image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ][:VISUALIZE_SAMPLES]

    for image_path in tqdm(image_paths, desc="Visualizing"):
        predictions = model.predict(
            source=image_path,
            conf=CONFIDENCE,
            iou=0.6,
            imgsz=640,
            device=0,
            verbose=False,
        )
        for prediction in predictions:
            cv2.imwrite(str(visualization_dir / f"det_{image_path.name}"), prediction.plot())

    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
