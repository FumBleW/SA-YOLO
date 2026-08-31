# Barcode detection demo

A Tkinter interface for running a trained SA-YOLO checkpoint on images or a webcam.

## Run

From the repository root:

```bash
pip install -r barcode_detection_app/requirements.txt
python barcode_detection_app/barcode_detection_app.py
```

The checkpoint can be selected in the interface. To provide a default path, set `SA_YOLO_WEIGHTS` before starting the application.

The application can export the annotated image and detection records in CSV and JSON formats. Barcode decoding uses `pyzbar`/ZBar and is optional.

On Windows, `run_app.bat` starts the application with the active Python environment.
