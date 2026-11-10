#  RoadGuard
### AI-Powered Road Damage Detection & Infrastructure Intelligence

**Turning road images into actionable infrastructure intelligence.**

RoadGuard is an AI-powered road monitoring and damage intelligence platform designed to detect, analyze, locate, and track road surface damage using computer vision and geospatial technologies.

By combining deep learning, image processing, and geospatial analysis, RoadGuard aims to streamline road inspection, identify critical damage locations, and support data-driven infrastructure maintenance.

---

## 📌 Overview

Traditional road inspection often relies on manual surveys and periodic assessments. RoadGuard aims to automate this process through an intelligent detection and monitoring pipeline.

The platform is being developed to:

- Automatically identify road damage from images and video.
- Classify different types of road surface deterioration.
- Associate detected damage with geographical coordinates.
- Analyze damage severity and identify high-priority locations.
- Support infrastructure authorities in monitoring and maintenance workflows.

---

## 🚀 Key Features

### 1. AI-Based Road Damage Detection
- YOLO-based object detection pipeline.
- Identification of potholes and different road crack patterns.
- Bounding-box localization of detected damage.
- Image-based inference and model evaluation.

### 2. Geospatial Intelligence *(In Development)*
- GPS-based road damage localization.
- Spatial clustering of nearby damage reports.
- Identification of road damage hotspots.
- Geographical visualization of reported issues.

### 3. Damage Analysis & Prioritization *(Planned)*
- Damage severity assessment.
- Priority scoring for maintenance requirements.
- Aggregation of nearby damage reports.
- Data-driven maintenance recommendations.

### 4. Intelligent Maintenance Tracking *(Planned)*
- Centralized road damage reporting.
- Issue verification and assignment.
- Maintenance status monitoring.
- Repair progress tracking.

---

## 🧠 AI & Machine Learning

RoadGuard's AI pipeline uses computer vision and deep learning to identify road surface damage.

**Current detection classes:**

| Class ID | Damage Category |
|---|---|
| 0 | Longitudinal Crack |
| 1 | Transverse Crack |
| 2 | Alligator Crack |
| 3 | Other Corruption |
| 4 | Pothole |

### Current Dataset

The initial detection pipeline uses a sample of the RDD2022 road damage dataset.

| Dataset Split | Images |
|---|---:|
| Training | 643 |
| Validation | 161 |
| Original Test | 190 |

### Baseline Model

- **Architecture:** YOLO11n
- **Approach:** Transfer Learning
- **Training Epochs:** 50
- **Image Size:** 640 × 640
- **Hardware:** NVIDIA GeForce RTX 4050

**Initial Validation Results:**

| Metric | Score |
|---|---:|
| Precision | 51.2% |
| Recall | 33.6% |
| mAP@50 | 32.8% |
| mAP@50–95 | 14.2% |

*These are preliminary baseline results. Further training, evaluation, and optimization are planned.*

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Object Detection | Ultralytics YOLO |
| Deep Learning | PyTorch |
| Backend API | FastAPI |
| Database | PostgreSQL |
| Geospatial Database | PostGIS |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
RoadGuard/
│
├── ai_engine/
│   ├── detection/
│   ├── processing/
│   └── tracking/
│
├── backend/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   └── main.py
│
├── database/
│
├── datasets/
│   └── road_damage/
│
├── docs/
│
├── experiments/
│   └── detection/
│       ├── roadguard_baseline/
│       └── predictions/
│
├── intelligence/
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── check_annotations.py
│   ├── convert_to_yolo.py
│   ├── validate_yolo.py
│   ├── visualize_yolo.py
│   ├── train.py
│   └── predict.py
│
├── tests/
├── workers/
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.12
- Git
- NVIDIA GPU with compatible CUDA-enabled PyTorch (optional, for GPU acceleration)

### 1. Clone the repository

```bash
git clone https://github.com/AnuragDe-cloud/RoadGuard.git

cd RoadGuard
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip

pip install ultralytics opencv-python matplotlib fastapi
```

### 4. Prepare the dataset

Place the RDD2022 sample dataset in:

```text
datasets/road_damage/raw/RDD2022_sample/
```

Run the dataset inspection and conversion scripts:

```bash
python scripts/inspect_dataset.py

python scripts/check_annotations.py

python scripts/convert_to_yolo.py

python scripts/validate_yolo.py
```

### 5. Train the baseline detection model

```bash
python scripts/train.py
```

### 6. Run inference

After training:

```bash
python scripts/predict.py
```

*Note: The dataset and trained model weights are not included in the Git repository. You must provide the dataset locally before running the complete pipeline.*

---

## 📊 Development Roadmap

- [x] Initial project structure and development environment.
- [x] RDD2022 sample dataset preparation.
- [x] Dataset inspection and annotation analysis.
- [x] JSON-to-YOLO annotation conversion.
- [x] Training and validation dataset split.
- [x] YOLO annotation validation.
- [x] Bounding-box visualization.
- [x] First YOLO11n baseline training.
- [x] Initial validation and experiment reports.
- [x] Image inference pipeline.
- [ ] Improve detection precision and recall.
- [ ] Experiment with larger YOLO architectures.
- [ ] Implement video-based road damage detection.
- [ ] Implement GPS-based damage localization.
- [ ] Develop severity scoring and hotspot analysis.
- [ ] Integrate AI services with the application backend.
- [ ] Develop end-to-end maintenance tracking.

---

## 👥 Contributors

- **Anurag De** — AI & Systems Lead
- **Daksh** — Application & Platform Lead

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>RoadGuard — Building smarter roads through AI and geospatial intelligence.</b>
</p>
