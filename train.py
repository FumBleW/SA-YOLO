import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = Path(os.getenv("SA_YOLO_DATA", ROOT / "datasets" / "kaist_barcode" / "data.yaml")).expanduser()
MODEL_YAML = Path(
    os.getenv(
        "SA_YOLO_MODEL_CONFIG",
        ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11-c2fema-msscem.yaml",
    )
).expanduser()
OUTPUT_DIR = ROOT / "runs" / "detect"
RUN_NAME = "sa_yolo"


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}\nSet SA_YOLO_DATA to override this path.")
    if not MODEL_YAML.is_file():
        raise FileNotFoundError(f"Model YAML not found: {MODEL_YAML}")

    model = YOLO(str(MODEL_YAML))
    model.train(
        data=DATA_YAML,
        epochs=200,
        batch=16,
        imgsz=640,
        workers=0,
        device=0,
        optimizer="SGD",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.001,
        momentum=0.937,
        patience=20,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        exist_ok=False,
        seed=42,
        deterministic=True,
        box=10.0,
        cls=0.2,
        dfl=1.5,
        degrees=180.0,
        shear=15.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.5,
        mosaic=1.0,
        close_mosaic=0,
        mixup=0.5,
        copy_paste=0.0,
        project=OUTPUT_DIR,
        name=RUN_NAME,
    )

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.6,
        plots=True,
    )

    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Output:    {OUTPUT_DIR / RUN_NAME}")


if __name__ == "__main__":
    main()
