# SA-YOLO



**Shape-Aware Product Barcode Detection with Multi-Scale Attention and Strip Convolutions in Complex Retail Scenes**



SA-YOLO is a shape-aware object detection framework designed for **1D product barcode detection in complex retail scenes**. It is built on a YOLO11-style detector and introduces three task-specific components to improve barcode feature discrimination, multi-scale directional representation, and bounding-box localization:



- **C2f-EMA** — integrates Efficient Multi-Scale Attention (EMA) to strengthen barcode stripes, edges, and directional features while suppressing background interference.

- **MS-SCEM** — a Multi-Scale Strip Convolution Enhancement Module that uses asymmetric strip convolutions to model elongated barcode structures at different scales.

- **BC-IoU** — a Barcode-CIoU regression loss designed for extreme-aspect-ratio targets by adding width/height consistency constraints and aspect-ratio-aware weighting.



On the **KAIST 1D Barcode** benchmark, SA-YOLO achieves **94.1% Precision, 95.0% Recall, 98.3% mAP@0.5, and 61.1% mAP@0.5:0.95**.



> This repository focuses on improving **detection accuracy and localization quality** for product barcodes. SA-YOLO is not presented as a lightweight model, and this project does not make claims about parameter reduction, FLOPs reduction, or inference-speed improvement.



---



## Contents



- [Highlights](#highlights)

- [Motivation](#motivation)

- [Method Overview](#method-overview)

- [Model Architecture](#model-architecture)

- [Main Components](#main-components)

- [Experimental Results](#experimental-results)

- [Dataset](#dataset)

- [Environment](#environment)

- [Installation](#installation)

- [Custom Ultralytics Integration](#custom-ultralytics-integration)

- [Dataset Preparation](#dataset-preparation)

- [Training](#training)

- [Validation](#validation)

- [Inference](#inference)

- [Application Demo](#application-demo)

- [Repository Structure](#repository-structure)

- [Reproducibility Notes](#reproducibility-notes)

- [Citation](#citation)

- [Acknowledgements](#acknowledgements)

- [License](#license)

- [Contact](#contact)



---



## Highlights



SA-YOLO is designed around the visual and geometric characteristics of product barcodes.



### 1. Shape-aware feature enhancement



Product barcodes contain repeated parallel stripes, narrow edges, and strong directional structures. In complex retail packaging, these patterns can be confused with text, decorative lines, and repeated background textures.



**C2f-EMA** enhances informative barcode features through grouped attention, directional pooling, and cross-spatial interaction.



### 2. Multi-scale strip convolution



Standard square kernels are not always well suited to extremely elongated barcode targets.



**MS-SCEM** uses multi-branch asymmetric strip convolutions to capture horizontal and vertical structures over different receptive fields.



### 3. Barcode-oriented box regression



A small width or height error can strongly affect a long and narrow barcode crop even when the overall IoU remains relatively high.



**BC-IoU** modifies CIoU for barcode-shaped targets by introducing a normalized width/height consistency penalty and an aspect-ratio-aware weighting strategy.



### 4. Practical barcode detection prototype



A Python desktop application is included in the project design to support:



- model weight loading;

- image upload;

- webcam detection;

- inference parameter adjustment;

- visualization of detection results;

- confidence and bounding-box display;

- JPG / CSV / JSON export;

- optional barcode decoding for demonstration.



The optional decoding function is an application feature and is **not included in the model performance evaluation**.



---



## Motivation



Barcode detection in real retail environments is more difficult than detection under controlled scanning conditions. Common challenges include:



- extreme width-to-height ratios;

- small barcode regions;

- arbitrary orientations;

- perspective distortion;

- motion blur;

- uneven illumination;

- partial occlusion;

- reflective packaging;

- dense text and graphics;

- background patterns visually similar to barcode stripes.



SA-YOLO addresses these issues through a coordinated optimization pipeline:



```text

Feature Selection

      ↓

C2f-EMA

      ↓

Directional Structure Enhancement

      ↓

MS-SCEM

      ↓

Barcode-Oriented Box Refinement

      ↓

BC-IoU

```



---



## Method Overview



The detector follows a YOLO11-style **Backbone–Neck–Head** pipeline.



```text

Input Image

    │

    ▼

┌───────────────────────────────┐

│           Backbone            │

│                               │

│ Conv → Conv                   │

│   ↓                           │

│ C2f-EMA                       │

│   ↓                           │

│ Conv → C2f-EMA                │

│   ↓                           │

│ Conv → C2f-EMA                │

│   ↓                           │

│ Conv → C2f-EMA                │

│   ↓                           │

│ SPPF → C2PSA                  │

└───────────────┬───────────────┘

                │

                ▼

┌───────────────────────────────┐

│             Neck              │

│                               │

│ Upsample / Concat             │

│      ↓                        │

│ C2f-EMA → MS-SCEM             │

│      ↓                        │

│ Upsample / Concat             │

│      ↓                        │

│ C2f-EMA → MS-SCEM             │

│      ↓                        │

│ Downsample / Concat           │

│      ↓                        │

│ C2f-EMA → MS-SCEM             │

│      ↓                        │

│ Downsample / Concat           │

│      ↓                        │

│ C2f-EMA → MS-SCEM             │

└───────────────┬───────────────┘

                │

                ▼

┌───────────────────────────────┐

│         Detection Head        │

│                               │

│   P3/8   P4/16   P5/32        │

│     \\      |      /           │

│          Detect               │

└───────────────┬───────────────┘

                │

                ▼

        Barcode Bounding Boxes

        + Confidence Scores

```



The provided model configuration is:



```text

yolo11-c2fema-msscem.yaml

```



The model is configured as a **single-class detector**:



```yaml

nc: 1

```



---



## Model Architecture



The current YAML configuration uses three detection scales:



| Scale | Feature Level | Primary Role |

|---|---|---|

| P3 | 1/8 | small barcode targets |

| P4 | 1/16 | medium barcode targets |

| P5 | 1/32 | large barcode targets |



The detection head receives the enhanced feature maps produced after MS-SCEM:



```yaml

- [[18, 22, 26], 1, Detect, [nc]]

```



In the current configuration:



- **C2f-EMA** is used in the backbone and feature-fusion path;

- **MS-SCEM** is inserted after feature fusion at P3, P4, and P5 scales;

- **BC-IoU** is used for barcode-oriented bounding-box regression.



---



## Main Components



### C2f-EMA



C2f-EMA combines the feature reuse and cross-stage information flow of C2f with **Efficient Multi-Scale Attention (EMA)**.



For an input feature map, EMA divides channels into groups and learns attention through:



- horizontal directional pooling;

- vertical directional pooling;

- local convolutional representation;

- cross-spatial feature interaction.



This design helps the detector emphasize:



- continuous barcode stripes;

- thin barcode boundaries;

- directional structures;

- weak barcode features under cluttered packaging backgrounds.



At the same time, it reduces the influence of text, decorative lines, and other barcode-like patterns.



---



### MS-SCEM



**MS-SCEM** stands for **Multi-Scale Strip Convolution Enhancement Module**.



The module contains four feature branches:



```text

Input

&#x20;├── 1×1 Conv

&#x20;├── 1×5 Strip Conv

&#x20;├── 5×1 Strip Conv

&#x20;└── 1×9 Strip Conv → 9×1 Strip Conv

          │

          ▼

       Fusion

          │

       1×1 Conv

          │

        Output

```



The branches serve different purposes:



| Branch | Purpose |

|---|---|

| `1×1` | preserves compact semantic information |

| `1×5` | models horizontal directional structures |

| `5×1` | models vertical directional structures |

| `1×9 → 9×1` | expands directional receptive fields for small, rotated, or distorted barcodes |



The outputs are fused along the channel dimension and processed by a final `1×1` convolution.



---



### BC-IoU



**BC-IoU (Barcode-CIoU)** is designed for bounding-box regression of elongated barcode targets.



Starting from CIoU, BC-IoU introduces an additional normalized width/height consistency term:



```text

L_BC-IoU =

    1 - IoU

    + center-distance term

    + aspect-ratio term

    + β × size-consistency term

```



The additional size-consistency term is based on the normalized absolute difference between predicted and ground-truth width and height.



In the experiments:



```text

β = 0.25

```



An aspect-ratio-aware weight is applied to place stronger regression emphasis on extremely elongated targets. The threshold used in the experimental design is a ground-truth aspect ratio greater than `5`.



The goal is to improve the fit between predicted boxes and long, narrow barcode regions, especially under stricter IoU thresholds.



---



## Experimental Results



### Overall Performance



Experiments are conducted on the **KAIST 1D Barcode** benchmark.



| Model | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |

|---|---:|---:|---:|---:|

| YOLO11n | 92.6 | 92.1 | 95.6 | 58.2 |

| **SA-YOLO** | **94.1** | **95.0** | **98.3** | **61.1** |

| Improvement | **+1.5** | **+2.9** | **+2.7** | **+2.9** |



SA-YOLO improves both target discovery and strict bounding-box localization compared with the YOLO11n baseline.



---



### Comparison with Representative Detectors



| Model | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |

|---|---:|---:|---:|---:|

| YOLOv5n | 90.9 | 89.6 | 95.2 | 53.7 |

| YOLOv8n | 92.2 | 82.7 | 91.8 | 55.5 |

| YOLO11n | 92.6 | 92.1 | 95.6 | 58.2 |

| Faster R-CNN | 81.4 | 93.8 | 97.5 | 60.8 |

| Integrated DL-Geo | 93.3 | 90.0 | 96.0 | 58.6 |

| SSD | 90.0 | 91.0 | 95.0 | 53.0 |

| LBC-YOLO11 | 92.2 | 92.6 | 96.1 | 57.3 |

| FWL-YOLO | 93.1 | 93.4 | 97.0 | 59.1 |

| RT-DETR-R18 | 90.5 | 91.3 | 94.8 | 56.1 |

| **SA-YOLO** | **94.1** | **95.0** | **98.3** | **61.1** |



> All values above correspond to the experimental comparison reported in the project manuscript under a unified dataset split and evaluation protocol.



---



### Ablation Study



| C2f-EMA | MS-SCEM | BC-IoU | Precision (%) | Recall (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |

|:---:|:---:|:---:|---:|---:|---:|---:|

|  |  |  | 92.6 | 92.1 | 95.6 | 58.2 |

| ✓ |  |  | 93.6 | 94.3 | 97.7 | 59.7 |

|  | ✓ |  | 93.1 | 92.7 | 97.6 | 58.6 |

|  |  | ✓ | 92.5 | 92.5 | 96.9 | 59.0 |

| ✓ | ✓ |  | 93.6 | 94.9 | 97.7 | 59.1 |

|  | ✓ | ✓ | 90.7 | 92.6 | 96.5 | 57.5 |

| ✓ |  | ✓ | 94.0 | 93.4 | 98.0 | 60.5 |

| ✓ | ✓ | ✓ | **94.1** | **95.0** | **98.3** | **61.1** |



The full SA-YOLO configuration achieves the best overall performance, indicating complementary effects between feature selection, directional structure enhancement, and barcode-oriented box regression.



---



## Dataset



This project uses the **KAIST 1D Barcode** benchmark distributed through the QuickBrowser research project.



The dataset contains real product barcode images captured under diverse retail conditions, including variations in:



- lighting;

- viewpoint;

- object scale;

- orientation;

- geometric distortion;

- product packaging;

- background texture.



The task is formulated as **single-class object detection**, where each target is labeled as:



```text

barcode

```



The official dataset split is preserved for training, validation, and final testing.



### Important



The dataset itself should **not** be committed to this repository unless its original license explicitly permits redistribution.



Please download the dataset from its official source and organize it locally.



---



## Environment



The experimental environment reported in the project is:



| Component | Configuration |

|---|---|

| Operating System | Windows 11 |

| CPU | Intel Core i7-13650HX |

| GPU | NVIDIA GeForce RTX 4060 |

| GPU Memory | 8 GB |

| Python | 3.13.7 |

| PyTorch | 2.8.0 |

| CUDA | 12.6 |

| Ultralytics | 8.4.8 |



Core training settings:



| Parameter | Value |

|---|---:|

| Input size | 640 × 640 |

| Epochs | 200 |

| Batch size | 16 |

| Initial learning rate (`lr0`) | 0.001 |

| Final LR factor (`lrf`) | 0.01 |

| Optimizer | SGD |

| Weight decay | 0.001 |

| Mosaic | 1.0 |

| Early stopping patience | 20 epochs |



The experiments are designed to train models **from random initialization without pretrained weights**.



---



## Installation



### 1. Clone the repository



```bash

git clone https://github.com/FumBleW/SA-YOLO.git

cd SA-YOLO

```



### 2. Create a virtual environment



Windows:



```bash

python -m venv .venv

.venv\\Scripts\\activate

```



Linux / macOS:



```bash

python3 -m venv .venv

source .venv/bin/activate

```



### 3. Install dependencies



Install the runtime dependencies, then install the bundled Ultralytics fork in editable mode:



```bash

pip install -r requirements.txt

pip install -e . --no-deps

```



For the optional desktop barcode-detection demo, install its additional dependencies:



```bash

pip install -r barcode_detection_app/requirements.txt

```



Optional dependency for barcode decoding in the desktop demo:



```text

pyzbar

```



> `pyzbar` may additionally require the system-level ZBar library depending on the operating system.



---



## Custom Ultralytics Integration



SA-YOLO is **not only a YAML configuration**. It contains custom modules and a custom regression loss.



Therefore, a standard unmodified installation of Ultralytics may not recognize:



```text

C2f_EMA

MS_SCEM

BC-IoU

```



Before training, make sure that the custom implementation has been correctly integrated into the Ultralytics source code.



At minimum, the following must be available to the model parser and training pipeline:



```text

C2f_EMA

MS_SCEM

BC-IoU regression logic

```



Typical integration steps include:



1. implement `EMA`, `C2f_EMA`, and `MS_SCEM`;

2. export the custom modules from the relevant Ultralytics module package;

3. register the custom modules so the YAML model parser can resolve them;

4. integrate BC-IoU into the bounding-box loss calculation;

5. verify that `yolo11-c2fema-msscem.yaml` can be parsed successfully;

6. run a small forward pass before starting full training.



A successful model build should recognize the custom architecture without errors such as:



```text

KeyError: 'C2f_EMA'

```



or:



```text

KeyError: 'MS_SCEM'

```



---



## Dataset Preparation



A recommended YOLO-format directory structure is:



```text

datasets/

└── kaist_barcode/

    ├── images/

    │   ├── train/

    │   ├── val/

    │   └── test/

    ├── labels/

    │   ├── train/

    │   ├── val/

    │   └── test/

    └── data.yaml

```



Each YOLO label line follows:



```text

class_id x_center y_center width height

```



All coordinates are normalized to `[0, 1]`.



Because the project contains only one class:



```text

class_id = 0

```



Example `data.yaml`:



```yaml

path: datasets/kaist_barcode



train: images/train

val: images/val

test: images/test



names:

  0: barcode

```



Before training, check:



- every image has a matching label file;

- class IDs are valid;

- bounding-box coordinates lie within the expected range;

- there are no zero-area boxes;

- no test images are used during model development.



---



## Data Augmentation



Data augmentation is applied **only to the training split**.



The project uses augmentation strategies including:



- random rotation;

- random horizontal / vertical flipping;

- shear transformation;

- HSV color perturbation;

- Mosaic;

- MixUp.



Reported augmentation settings include:



```text

horizontal flip probability = 0.5

vertical flip probability   = 0.5

HSV hue                      = 0.015

HSV saturation               = 0.7

HSV value                    = 0.4

```



Because barcode information is encoded by repeated stripe structures, overly aggressive geometric augmentation should be avoided. Strong cropping, deformation, or interpolation can destroy useful barcode texture.



Validation and test images should remain unaugmented to ensure objective evaluation.



---



## Training



### Python API



A reproducible training configuration can be written as follows:



```python

from ultralytics import YOLO



MODEL_YAML = "ultralytics/cfg/models/11/yolo11-c2fema-msscem.yaml"

DATA_YAML = "datasets/kaist_barcode/data.yaml"



model = YOLO(MODEL_YAML)



model.train(

    data=DATA_YAML,

    imgsz=640,

    epochs=200,

    batch=16,

    lr0=0.001,

    lrf=0.01,

    optimizer="SGD",

    weight_decay=0.001,

    mosaic=1.0,

    patience=20,

    pretrained=False,

    device=0,

    project="barcode_detection",

    name="sa_yolo"

)

```



### Important: no pretrained weights



To reproduce the experimental protocol, initialize the model directly from the YAML architecture:



```python

model = YOLO("ultralytics/cfg/models/11/yolo11-c2fema-msscem.yaml")

```



Do **not** load pretrained weights such as:



```python

model = YOLO("yolo11n.pt")

```



and do not call:



```python

model.load("yolo11n.pt")

```



when reproducing the reported from-scratch training setting.



---



## Validation



After training:



```python

from ultralytics import YOLO



model = YOLO("barcode_detection/sa_yolo/weights/best.pt")



metrics = model.val(

    data="datasets/kaist_barcode/data.yaml",

    imgsz=640,

    split="test",

    device=0

)



print(metrics)

```



Primary evaluation metrics:



- **Precision**

- **Recall**

- **mAP@0.5**

- **mAP@0.5:0.95**



For final reporting, use the held-out test split only after model design and hyperparameter selection are complete.



---



## Inference



### Image inference



```python

from ultralytics import YOLO



model = YOLO("barcode_detection/sa_yolo/weights/best.pt")



results = model.predict(

    source="test.jpg",

    imgsz=640,

    conf=0.25,

    device=0,

    save=True

)

```



### Folder inference



```python

results = model.predict(

    source="path/to/images",

    imgsz=640,

    conf=0.25,

    device=0,

    save=True

)

```



### Webcam inference



```python

results = model.predict(

    source=0,

    imgsz=640,

    conf=0.25,

    device=0,

    show=True

)

```



Adjust the confidence threshold according to the deployment scenario.



---



## Application Demo



The project also includes a desktop barcode-detection prototype developed with:



```text

Python

Tkinter

Ultralytics YOLO

OpenCV

Pillow

```



Supported functions include:



```text

Model loading

    ↓

Image / camera input

    ↓

SA-YOLO inference

    ↓

Bounding-box visualization

    ↓

Class + confidence + coordinates

    ↓

JPG / CSV / JSON export

    ↓

Optional barcode decoding

```



The application can display the original image and detection result side by side.



Example exported fields may include:



```text

class

confidence

x1

y1

x2

y2

```



Optional decoding can be implemented using `pyzbar/ZBar`.



> Barcode decoding is provided only as a system demonstration. The quantitative results reported for SA-YOLO evaluate barcode **detection/localization**, not decoding success rate.



---



## Repository Structure



The public repository is organized as follows:



```text

SA-YOLO/

├── README.md                 # Project overview and reproduction guide
├── requirements.txt         # Runtime dependencies
├── train.py                 # Training and final test evaluation
├── test.py                  # Standalone evaluation and visualization
├── barcode_detection_app/   # Desktop demonstration application
├── ultralytics/             # Bundled fork with SA-YOLO modules and BC-IoU
│   └── cfg/models/11/
│       └── yolo11-c2fema-msscem.yaml
├── examples/                # Usage examples
└── tests/                   # Upstream compatibility tests

```



Datasets, trained weights, experiment outputs, and local caches are intentionally excluded from Git.



---



## Reproducibility Notes



For reproducible experiments:



1. use the same official dataset split;

2. keep the task single-class;

3. keep the input resolution at `640 × 640`;

4. train for up to `200` epochs;

5. use batch size `16`;

6. use SGD with the reported learning-rate settings;

7. use early stopping with patience `20`;

8. do not use pretrained weights when reproducing the reported setting;

9. apply augmentation only to the training split;

10. keep the validation and test sets unchanged;

11. make sure the same custom module and BC-IoU implementations are used;

12. report Precision, Recall, mAP@0.5, and mAP@0.5:0.95.



For fair comparison with other detectors, use the same:



```text

dataset split

input size

training protocol

evaluation split

evaluation metrics

```



---



## Citation



If this project is useful for your research, please cite the SA-YOLO project/manuscript.



```bibtex

@article{fang2026sayolo,

  title   = {SA-YOLO: Shape-Aware Product Barcode Detection with Multi-Scale Attention and Strip Convolutions in Complex Retail Scenes},

  author  = {Fang, Bowen},

  year    = {2026},

  note    = {Manuscript}

}

```



The citation information should be updated after the paper receives its final publication information.



---



## Related Work



The design of this project is related to research on:



- YOLO-style one-stage object detection;

- efficient multi-scale attention;

- strip convolution and long-range directional modeling;

- IoU-based bounding-box regression;

- deep-learning-based barcode localization.



Representative references include:



1. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, **“You Only Look Once: Unified, Real-Time Object Detection,”** CVPR, 2016.

2. D. Ouyang *et al.*, **“Efficient Multi-Scale Attention Module with Cross-Spatial Learning,”** ICASSP, 2023.

3. Z. Zheng *et al.*, **“Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression,”** AAAI, 2020.

4. Y.-F. Zhang *et al.*, **“Focal and Efficient IoU Loss for Accurate Bounding Box Regression,”** Neurocomputing, 2022.

5. A. Sharif *et al.*, **“An Accurate and Efficient 1-D Barcode Detector for Medium of Deployment in IoT Systems,”** IEEE Internet of Things Journal, 2021.

6. J. Jia *et al.*, **“Tiny-BDN: An Efficient and Compact Barcode Detection Network,”** IEEE JSTSP, 2020.

7. J. Jia *et al.*, **“EMBDN: An Efficient Multiclass Barcode Detection Network for Complicated Environments,”** IEEE Internet of Things Journal, 2019.



Please refer to the project paper for the complete bibliography.



---



## Acknowledgements



This project builds on and is inspired by the following open research ecosystems and datasets:



- **Ultralytics YOLO** for the object-detection framework;

- **KAIST 1D Barcode / QuickBrowser** for the barcode benchmark;

- the authors of **EMA** for efficient multi-scale attention;

- the broader barcode-detection and IoU-loss research communities.



Please follow the licenses and citation requirements of all third-party code and datasets used in your local setup.



---



## License



This repository includes modified Ultralytics source code and is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See [LICENSE](LICENSE) for the full terms.



Datasets, model weights, and other third-party resources are not included in this repository and remain subject to their respective licenses and terms of use.



---



## Contact



**Bowen Fang**<br>

School of Economics and Management<br>

Beijing Jiaotong University<br>

Beijing, China



E-mail: `24711006@bjtu.edu.cn`



For questions related to SA-YOLO, experimental reproduction, or the barcode-detection application prototype, please open a GitHub Issue or contact the author by email.
