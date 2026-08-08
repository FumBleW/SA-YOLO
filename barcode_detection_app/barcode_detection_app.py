# -*- coding: utf-8 -*-
"""
Barcode Detection Demonstration System

This desktop application demonstrates the improved YOLO11 barcode detector
described in the accompanying research project.

Main functions:
1. Load a trained YOLO .pt model.
2. Upload an image and detect product barcodes.
3. Display original and annotated images side by side.
4. Show class, confidence, and bounding-box coordinates.
5. Optionally attempt barcode decoding with pyzbar/ZBar.
6. Export annotated images, CSV details, and JSON records.
7. Run a real-time webcam demonstration.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from ultralytics import YOLO


# =============================================================================
# USER CONFIGURATION: edit only this section if needed
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Path to the trained proposed-model weights.
# You can also choose a weight file from the interface after the program starts.
DEFAULT_WEIGHTS_PATH = os.getenv(
    "SA_YOLO_WEIGHTS",
    str(PROJECT_ROOT / "runs" / "detect" / "sa_yolo" / "weights" / "best.pt"),
)

# Default directory for exported images, CSV files, and JSON files.
DEFAULT_OUTPUT_DIR = os.getenv(
    "SA_YOLO_APP_OUTPUT",
    str(PROJECT_ROOT / "outputs" / "barcode_detection_app"),
)

# Final reported test metrics shown in the application dashboard.
PAPER_METRICS = {
    "Precision": "94.1%",
    "Recall": "95.0%",
    "mAP@0.5": "98.3%",
    "mAP@0.5:0.95": "61.1%",
}

# Default inference settings.
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.60
DEFAULT_DEVICE = "0"  # Use GPU 0. Set this value to "cpu" to run on CPU.

# Webcam settings.
DEFAULT_CAMERA_ID = 0
CAMERA_INFERENCE_INTERVAL = 5  # Run model inference once every N frames.


# =============================================================================
# INTERNAL CODE: no changes are normally needed below this line
# =============================================================================

APP_TITLE = "Barcode Detection Demonstration System"
MODEL_DESCRIPTION = "Proposed YOLO11: C2f-EMA + MS-SCEM + BC-IoU"


class BarcodeDetectionApp(tk.Tk):
    """Tkinter-based desktop application for barcode detection."""

    def __init__(self) -> None:
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1500x930")
        self.minsize(1180, 760)
        self.configure(bg="#F4F7FB")

        self.model: Optional[YOLO] = None
        self.current_image_path: Optional[Path] = None
        self.current_original_bgr: Optional[np.ndarray] = None
        self.last_annotated_bgr: Optional[np.ndarray] = None
        self.last_detections: list[dict[str, Any]] = []
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.camera_running = False
        self.camera_frame_count = 0
        self.cached_camera_bgr: Optional[np.ndarray] = None
        self._original_photo: Optional[ImageTk.PhotoImage] = None
        self._result_photo: Optional[ImageTk.PhotoImage] = None

        self.weights_path_var = tk.StringVar(value=DEFAULT_WEIGHTS_PATH)
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.device_var = tk.StringVar(value=DEFAULT_DEVICE)
        self.imgsz_var = tk.StringVar(value=str(DEFAULT_IMAGE_SIZE))
        self.conf_var = tk.StringVar(value=str(DEFAULT_CONFIDENCE))
        self.iou_var = tk.StringVar(value=str(DEFAULT_IOU))
        self.decode_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(
            value="Select a model weight file and load the model to begin."
        )

        self._configure_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------------------------------------------------------------------------
    # User interface
    # -------------------------------------------------------------------------

    def _configure_styles(self) -> None:
        """Configure application visual styles."""
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("Header.TFrame", background="#163A5F")
        style.configure(
            "HeaderTitle.TLabel",
            background="#163A5F",
            foreground="white",
            font=("Segoe UI", 19, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background="#163A5F",
            foreground="#D8E7F5",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Card.TLabelframe",
            background="white",
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background="white",
            foreground="#163A5F",
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "MetricName.TLabel",
            background="white",
            foreground="#5B6B7A",
            font=("Segoe UI", 9),
        )
        style.configure(
            "MetricValue.TLabel",
            background="white",
            foreground="#0D4D8B",
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 7),
        )
        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            padding=(8, 6),
        )
        style.configure(
            "Status.TLabel",
            background="#EAF2FA",
            foreground="#164B7B",
            font=("Segoe UI", 9),
            padding=(10, 7),
        )

    def _build_ui(self) -> None:
        """Build the complete user interface."""
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill="x")

        header_left = ttk.Frame(header, style="Header.TFrame")
        header_left.pack(side="left", padx=20, pady=13)

        ttk.Label(
            header_left,
            text=APP_TITLE,
            style="HeaderTitle.TLabel",
        ).pack(anchor="w")

        ttk.Label(
            header_left,
            text=MODEL_DESCRIPTION,
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=12, pady=12)

        self._build_left_panel(main)
        self._build_center_panel(main)
        self._build_right_panel(main)

        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            style="Status.TLabel",
        )
        status_bar.pack(fill="x", padx=12, pady=(0, 12))

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        """Build the model, inference, and control panels."""
        left = ttk.Frame(parent)
        left.pack(side="left", fill="y", padx=(0, 10))

        model_frame = ttk.LabelFrame(left, text="Model Configuration", style="Card.TLabelframe")
        model_frame.pack(fill="x", pady=(0, 10), ipadx=8, ipady=8)

        ttk.Label(model_frame, text="Model weights (.pt)").pack(anchor="w", padx=8, pady=(4, 2))

        weight_row = ttk.Frame(model_frame)
        weight_row.pack(fill="x", padx=8)

        ttk.Entry(weight_row, textvariable=self.weights_path_var, width=31).pack(
            side="left",
            fill="x",
            expand=True,
        )
        ttk.Button(
            weight_row,
            text="Browse",
            style="Secondary.TButton",
            command=self._browse_weights,
        ).pack(side="left", padx=(6, 0))

        ttk.Button(
            model_frame,
            text="Load Model",
            style="Primary.TButton",
            command=self._load_model,
        ).pack(fill="x", padx=8, pady=(8, 4))

        self.model_state_label = ttk.Label(
            model_frame,
            text="Model status: Not loaded",
            foreground="#8A2C2C",
            background="white",
            font=("Segoe UI", 9),
        )
        self.model_state_label.pack(anchor="w", padx=8, pady=(2, 4))

        setting_frame = ttk.LabelFrame(left, text="Inference Settings", style="Card.TLabelframe")
        setting_frame.pack(fill="x", pady=(0, 10), ipadx=8, ipady=8)

        self._add_setting_row(setting_frame, "Device", self.device_var)
        self._add_setting_row(setting_frame, "Image size", self.imgsz_var)
        self._add_setting_row(setting_frame, "Confidence", self.conf_var)
        self._add_setting_row(setting_frame, "IoU threshold", self.iou_var)

        ttk.Checkbutton(
            setting_frame,
            text="Attempt barcode decoding (optional)",
            variable=self.decode_var,
        ).pack(anchor="w", padx=8, pady=(6, 2))

        control_frame = ttk.LabelFrame(left, text="Controls", style="Card.TLabelframe")
        control_frame.pack(fill="x", pady=(0, 10), ipadx=8, ipady=8)

        ttk.Button(
            control_frame,
            text="Select Image",
            style="Primary.TButton",
            command=self._select_image,
        ).pack(fill="x", padx=8, pady=(3, 5))

        ttk.Button(
            control_frame,
            text="Run Detection",
            style="Primary.TButton",
            command=self._detect_current_image,
        ).pack(fill="x", padx=8, pady=5)

        camera_row = ttk.Frame(control_frame)
        camera_row.pack(fill="x", padx=8, pady=5)

        ttk.Button(
            camera_row,
            text="Start Camera",
            style="Secondary.TButton",
            command=self._start_camera,
        ).pack(side="left", fill="x", expand=True)

        ttk.Button(
            camera_row,
            text="Stop Camera",
            style="Secondary.TButton",
            command=self._stop_camera,
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Button(
            control_frame,
            text="Export Current Result",
            style="Secondary.TButton",
            command=self._export_current_result,
        ).pack(fill="x", padx=8, pady=5)

        ttk.Button(
            control_frame,
            text="Clear Screen",
            style="Secondary.TButton",
            command=self._clear_display,
        ).pack(fill="x", padx=8, pady=(5, 3))

        metric_frame = ttk.LabelFrame(left, text="Reported Test Metrics", style="Card.TLabelframe")
        metric_frame.pack(fill="x", pady=(0, 10), ipadx=8, ipady=8)

        metric_grid = ttk.Frame(metric_frame)
        metric_grid.pack(fill="x", padx=8, pady=5)

        for index, (name, value) in enumerate(PAPER_METRICS.items()):
            cell = ttk.Frame(metric_grid)
            cell.grid(row=index // 2, column=index % 2, sticky="nsew", padx=6, pady=5)
            ttk.Label(cell, text=name, style="MetricName.TLabel").pack(anchor="center")
            ttk.Label(cell, text=value, style="MetricValue.TLabel").pack(anchor="center")

        metric_grid.columnconfigure(0, weight=1)
        metric_grid.columnconfigure(1, weight=1)

    def _build_center_panel(self, parent: ttk.Frame) -> None:
        """Build the image preview panel."""
        center = ttk.Frame(parent)
        center.pack(side="left", fill="both", expand=True, padx=(0, 10))

        preview_frame = ttk.LabelFrame(center, text="Detection Visualization", style="Card.TLabelframe")
        preview_frame.pack(fill="both", expand=True, ipadx=8, ipady=8)

        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(
            preview_frame,
            text="Original Image",
            background="white",
            foreground="#163A5F",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="n", pady=(8, 4))

        ttk.Label(
            preview_frame,
            text="Detection Result",
            background="white",
            foreground="#163A5F",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=1, sticky="n", pady=(8, 4))

        self.original_preview_label = tk.Label(
            preview_frame,
            text="Select a product image",
            bg="#F7F9FC",
            fg="#8190A0",
            font=("Segoe UI", 12),
            relief="solid",
            bd=1,
        )
        self.original_preview_label.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(8, 4),
            pady=(4, 8),
        )

        self.result_preview_label = tk.Label(
            preview_frame,
            text="The detection result will be displayed here",
            bg="#F7F9FC",
            fg="#8190A0",
            font=("Segoe UI", 12),
            relief="solid",
            bd=1,
        )
        self.result_preview_label.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(4, 8),
            pady=(4, 8),
        )

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        """Build the detection table and information panel."""
        right = ttk.Frame(parent)
        right.pack(side="right", fill="y")

        result_frame = ttk.LabelFrame(right, text="Detection Details", style="Card.TLabelframe")
        result_frame.pack(fill="both", expand=True, ipadx=8, ipady=8)

        columns = ("id", "class", "confidence", "bbox", "decode")
        self.result_tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            height=16,
        )

        headings = {
            "id": "ID",
            "class": "Class",
            "confidence": "Confidence",
            "bbox": "Bounding Box",
            "decode": "Decode Result",
        }

        widths = {
            "id": 45,
            "class": 80,
            "confidence": 90,
            "bbox": 170,
            "decode": 160,
        }

        for column in columns:
            self.result_tree.heading(column, text=headings[column])
            self.result_tree.column(
                column,
                width=widths[column],
                anchor="center",
                stretch=False,
            )

        y_scroll = ttk.Scrollbar(
            result_frame,
            orient="vertical",
            command=self.result_tree.yview,
        )
        self.result_tree.configure(yscrollcommand=y_scroll.set)

        self.result_tree.pack(side="left", fill="both", expand=True, padx=(5, 0), pady=5)
        y_scroll.pack(side="right", fill="y", padx=(0, 5), pady=5)

        info_frame = ttk.LabelFrame(right, text="System Notes", style="Card.TLabelframe")
        info_frame.pack(fill="x", pady=(10, 0), ipadx=8, ipady=8)

        info_text = (
            "• Input: product images or webcam frames\n"
            "• Output: barcode class, confidence, and box coordinates\n"
            "• Optional: decode a detected barcode ROI\n"
            "• Export: annotated image, CSV details, and JSON record\n\n"
            "Recommended demonstration cases:\n"
            "1. Text-dense packaging;\n"
            "2. Small or distant barcodes;\n"
            "3. Rotated or cylindrical-package barcodes;\n"
            "4. Reflective or partially occluded barcodes."
        )

        ttk.Label(
            info_frame,
            text=info_text,
            background="white",
            foreground="#455A6E",
            justify="left",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=8, pady=8)

    def _add_setting_row(
        self,
        parent: ttk.LabelFrame,
        name: str,
        variable: tk.StringVar,
    ) -> None:
        """Add one labeled inference-setting row."""
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)

        ttk.Label(row, text=name, width=14).pack(side="left")
        ttk.Entry(row, textvariable=variable, width=15).pack(
            side="right",
            fill="x",
            expand=True,
        )

    # -------------------------------------------------------------------------
    # Model loading and image input
    # -------------------------------------------------------------------------

    def _browse_weights(self) -> None:
        """Choose a model weight file."""
        path = filedialog.askopenfilename(
            title="Select YOLO Model Weights",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.weights_path_var.set(path)

    def _load_model(self) -> bool:
        """Load the selected YOLO model."""
        weights_path = Path(self.weights_path_var.get().strip())

        if not weights_path.exists():
            messagebox.showerror(
                "Invalid Path",
                f"Model weights were not found:\n{weights_path}",
            )
            self.status_var.set("Model loading failed: weight file not found.")
            return False

        try:
            self.status_var.set("Loading model. Please wait...")
            self.update_idletasks()

            self.model = YOLO(str(weights_path))
            self.model_state_label.config(
                text=f"Model status: Loaded\n{weights_path.name}",
                foreground="#177245",
            )
            self.status_var.set(f"Model loaded successfully: {weights_path.name}")
            return True

        except Exception as error:
            self.model = None
            self.model_state_label.config(
                text="Model status: Loading failed",
                foreground="#8A2C2C",
            )
            messagebox.showerror("Model Loading Failed", str(error))
            self.status_var.set("Model loading failed. Check the Python environment and weight path.")
            return False

    def _select_image(self) -> None:
        """Select a single image for barcode detection."""
        self._stop_camera()

        file_path = filedialog.askopenfilename(
            title="Select Product Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror(
                "Image Read Failed",
                "The selected image could not be read. Please check the file.",
            )
            return

        self.current_image_path = Path(file_path)
        self.current_original_bgr = image
        self.last_annotated_bgr = None
        self.last_detections = []
        self._clear_tree()

        self._display_bgr(image, self.original_preview_label, original=True)
        self.result_preview_label.configure(
            image="",
            text="Image loaded. Click 'Run Detection' to continue.",
        )
        self.status_var.set(f"Image loaded: {self.current_image_path.name}")

    # -------------------------------------------------------------------------
    # Model inference
    # -------------------------------------------------------------------------

    def _get_inference_settings(self) -> tuple[int, float, float, str]:
        """Read and validate inference parameters from the interface."""
        try:
            imgsz = int(self.imgsz_var.get())
            conf = float(self.conf_var.get())
            iou = float(self.iou_var.get())
            device = self.device_var.get().strip() or "cpu"
        except ValueError as error:
            raise ValueError(
                "Image size, confidence, and IoU threshold must be valid numeric values."
            ) from error

        if imgsz <= 0:
            raise ValueError("Image size must be greater than zero.")
        if not 0 <= conf <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not 0 <= iou <= 1:
            raise ValueError("IoU threshold must be between 0 and 1.")

        return imgsz, conf, iou, device

    def _detect_current_image(self) -> None:
        """Run barcode detection on the selected image."""
        if self.current_original_bgr is None:
            messagebox.showwarning("No Image Selected", "Select an image before running detection.")
            return

        if self.model is None and not self._load_model():
            return

        try:
            imgsz, conf, iou, device = self._get_inference_settings()
            self.status_var.set("Running barcode detection...")
            self.update_idletasks()

            results = self.model.predict(
                source=self.current_original_bgr,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )

            self._process_result(results[0], self.current_original_bgr)
            self.status_var.set(
                f"Detection complete: {len(self.last_detections)} barcode target(s) found."
            )

        except Exception as error:
            messagebox.showerror("Detection Failed", str(error))
            self.status_var.set("Detection failed. Check the model, device, and inference settings.")

    def _process_result(self, result: Any, original_bgr: np.ndarray) -> None:
        """Convert a YOLO result into images and table records."""
        annotated_bgr = result.plot()
        self.last_annotated_bgr = annotated_bgr
        self.last_detections = []

        self._clear_tree()
        self._display_bgr(original_bgr, self.original_preview_label, original=True)
        self._display_bgr(annotated_bgr, self.result_preview_label, original=False)

        if result.boxes is None or len(result.boxes) == 0:
            self.result_preview_label.configure(text="No barcode was detected.")
            return

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names

        for index, (box, confidence, class_id) in enumerate(
            zip(xyxy, confidences, class_ids),
            start=1,
        ):
            x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
            class_name = names[class_id] if isinstance(names, dict) else names[class_id]
            decoded_text = self._try_decode_roi(original_bgr, (x1, y1, x2, y2))

            record = {
                "id": index,
                "class_id": int(class_id),
                "class_name": str(class_name),
                "confidence": float(confidence),
                "bbox_xyxy": [x1, y1, x2, y2],
                "decode": decoded_text,
            }
            self.last_detections.append(record)

            bbox_text = f"({x1}, {y1}, {x2}, {y2})"
            self.result_tree.insert(
                "",
                "end",
                values=(
                    index,
                    class_name,
                    f"{confidence:.3f}",
                    bbox_text,
                    decoded_text,
                ),
            )

    def _try_decode_roi(
        self,
        image_bgr: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> str:
        """Optionally decode the detected ROI with pyzbar/ZBar."""
        if not self.decode_var.get():
            return "--"

        try:
            from pyzbar.pyzbar import decode as zbar_decode
        except Exception:
            return "pyzbar unavailable"

        x1, y1, x2, y2 = box
        image_height, image_width = image_bgr.shape[:2]
        padding = 8

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_width, x2 + padding)
        y2 = min(image_height, y2 + padding)

        if x2 <= x1 or y2 <= y1:
            return "invalid ROI"

        roi = image_bgr[y1:y2, x1:x2]
        decoded_items = zbar_decode(roi)

        if not decoded_items and roi.shape[1] < roi.shape[0]:
            roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
            decoded_items = zbar_decode(roi)

        if not decoded_items:
            return "not decoded"

        values = []
        for item in decoded_items:
            value = item.data.decode("utf-8", errors="replace")
            values.append(f"{item.type}: {value}")

        return " | ".join(values)

    # -------------------------------------------------------------------------
    # Webcam support
    # -------------------------------------------------------------------------

    def _start_camera(self) -> None:
        """Start real-time barcode detection from a webcam."""
        if self.camera_running:
            return

        if self.model is None and not self._load_model():
            return

        self._stop_camera()

        camera = cv2.VideoCapture(DEFAULT_CAMERA_ID)
        if not camera.isOpened():
            messagebox.showerror(
                "Camera Error",
                "The camera could not be opened. Check whether it is in use by another application "
                "or change DEFAULT_CAMERA_ID.",
            )
            return

        self.video_capture = camera
        self.camera_running = True
        self.camera_frame_count = 0
        self.cached_camera_bgr = None
        self.status_var.set("Real-time camera detection started.")
        self._camera_loop()

    def _camera_loop(self) -> None:
        """Process one webcam frame and schedule the next update."""
        if not self.camera_running or self.video_capture is None:
            return

        success, frame = self.video_capture.read()
        if not success:
            self.status_var.set("Camera read failed. The camera has been stopped.")
            self._stop_camera()
            return

        self.camera_frame_count += 1
        display_bgr = frame

        try:
            if self.camera_frame_count % CAMERA_INFERENCE_INTERVAL == 0:
                imgsz, conf, iou, device = self._get_inference_settings()
                results = self.model.predict(
                    source=frame,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    device=device,
                    verbose=False,
                )
                self.cached_camera_bgr = results[0].plot()
                self.last_annotated_bgr = self.cached_camera_bgr.copy()

            if self.cached_camera_bgr is not None:
                display_bgr = self.cached_camera_bgr

            self._display_bgr(frame, self.original_preview_label, original=True)
            self._display_bgr(display_bgr, self.result_preview_label, original=False)

        except Exception as error:
            self.status_var.set(f"Camera inference error: {error}")
            self._stop_camera()
            return

        self.after(20, self._camera_loop)

    def _stop_camera(self) -> None:
        """Stop webcam detection and release camera resources."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        if self.camera_running:
            self.status_var.set("Real-time camera detection stopped.")

        self.camera_running = False
        self.cached_camera_bgr = None

    # -------------------------------------------------------------------------
    # Display and export helpers
    # -------------------------------------------------------------------------

    def _display_bgr(
        self,
        image_bgr: np.ndarray,
        label: tk.Label,
        original: bool,
    ) -> None:
        """Show an OpenCV BGR image in a Tkinter label."""
        if image_bgr is None:
            return

        label.update_idletasks()
        target_width = max(label.winfo_width(), 420)
        target_height = max(label.winfo_height(), 420)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail(
            (target_width - 20, target_height - 20),
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(pil_image)

        label.configure(image=photo, text="")
        if original:
            self._original_photo = photo
        else:
            self._result_photo = photo

    def _clear_tree(self) -> None:
        """Remove all detection rows from the result table."""
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

    def _clear_display(self) -> None:
        """Clear loaded images, results, and tables."""
        self._stop_camera()

        self.current_image_path = None
        self.current_original_bgr = None
        self.last_annotated_bgr = None
        self.last_detections = []

        self._original_photo = None
        self._result_photo = None

        self.original_preview_label.configure(image="", text="Select a product image")
        self.result_preview_label.configure(
            image="",
            text="The detection result will be displayed here",
        )
        self._clear_tree()
        self.status_var.set("The application view has been cleared.")

    def _export_current_result(self) -> None:
        """Export the annotated image and detection records."""
        if self.last_annotated_bgr is None:
            messagebox.showwarning(
                "No Result Available",
                "Run image detection or webcam detection before exporting a result.",
            )
            return

        output_dir = Path(self.output_dir_var.get().strip() or DEFAULT_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        source_stem = self.current_image_path.stem if self.current_image_path else "camera_frame"
        output_stem = f"{source_stem}_{timestamp}"

        image_path = output_dir / f"{output_stem}_detected.jpg"
        csv_path = output_dir / f"{output_stem}_detections.csv"
        json_path = output_dir / f"{output_stem}_detections.json"

        cv2.imwrite(str(image_path), self.last_annotated_bgr)

        with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "id",
                    "class_id",
                    "class_name",
                    "confidence",
                    "bbox_xyxy",
                    "decode",
                ],
            )
            writer.writeheader()
            writer.writerows(self.last_detections)

        payload = {
            "application": APP_TITLE,
            "model": MODEL_DESCRIPTION,
            "weights": self.weights_path_var.get().strip(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "inference_settings": {
                "device": self.device_var.get().strip(),
                "imgsz": self.imgsz_var.get().strip(),
                "confidence": self.conf_var.get().strip(),
                "iou": self.iou_var.get().strip(),
            },
            "detections": self.last_detections,
        }

        with json_path.open("w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, ensure_ascii=False, indent=2)

        messagebox.showinfo(
            "Export Complete",
            "Files were exported successfully:\n"
            f"{image_path}\n"
            f"{csv_path}\n"
            f"{json_path}",
        )
        self.status_var.set(f"Results exported to: {output_dir}")

    def _on_close(self) -> None:
        """Release resources before closing the application."""
        self._stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = BarcodeDetectionApp()
    app.mainloop()
