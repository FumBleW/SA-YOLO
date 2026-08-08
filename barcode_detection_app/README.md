# Barcode Detection Demonstration System

This desktop application demonstrates the improved YOLO11 barcode detector used in the research project.

## Features

- Load a custom `best.pt` model.
- Upload product images and detect barcodes.
- Display original and annotated images side by side.
- Show class labels, confidence scores, and bounding-box coordinates.
- Optionally attempt to decode the detected barcode region with pyzbar/ZBar.
- Export an annotated image, a CSV detection table, and a JSON detection record.
- Run a real-time webcam demonstration.
- Display the reported project metrics:
  - Precision: 94.1%
  - Recall: 95.0%
  - mAP@0.5: 98.3%
  - mAP@0.5:0.95: 61.1%

## Before Running

Use the same Python environment that was used for YOLO training, for example:

```text
D:\Anaconda\envs\pytorch\python.exe
```

Install the required packages:

```text
ultralytics
opencv-python
Pillow
```

To enable barcode decoding, also install:

```text
pyzbar
```

> On Windows, pyzbar may also require a local ZBar dynamic library. Barcode detection itself does not depend on this optional decoding feature.

## How to Run

### Option 1: Run in PyCharm or VS Code

1. Open `barcode_detection_app.py`.
2. Check the `DEFAULT_WEIGHTS_PATH` setting at the top of the file.
3. Run the file directly.

### Option 2: Use the Batch File

Edit the Python path in `run_app.bat` if needed, then double-click the file.

## Suggested Workflow

1. Click **Load Model**.
2. Click **Select Image** and choose a product image.
3. Review Confidence, IoU threshold, and Image size.
4. Click **Run Detection**.
5. Review the detection table on the right side.
6. Click **Export Current Result** to save the output files.

## Suggested Research Demonstration Cases

Use examples with:

1. Text-dense package backgrounds;
2. Small or distant barcodes;
3. Rotated or cylindrical-package barcodes;
4. Reflection, shadow, or partial occlusion.

## Example Paper Description

> To demonstrate the practical applicability of the proposed method, a barcode detection demonstration system was developed using Python and Tkinter. The system integrates the improved YOLO11 model and supports image upload, real-time webcam detection, visualized bounding boxes, confidence and coordinate display, and result export. The prototype provides an intuitive interface for demonstrating the proposed model in retail product identification, inventory management, and automated inspection scenarios.
