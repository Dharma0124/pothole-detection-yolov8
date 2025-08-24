# Pothole Detection using YOLOv8

## Overview

This project detects potholes using the YOLOv8 (You Only Look Once) deep learning model. The primary objective is to automate the identification of potholes from static images to improve road safety and facilitate maintenance. The model is pre-trained using a custom dataset and focuses on delivering accurate detection results without relying on real-time video feeds. Large files, including model weights and datasets, are tracked using Git LFS.

## Table of Contents

- [Overview](#overview)
- [Setup and Installation](#setup-and-installation)
- [Model Training](#model-training)
- [Running the Detection](#running-the-detection)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Setup and Installation

### Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8 package
- Git
- Git LFS (for large files)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Dharma0124/pothole-detection-yolov8.git
cd pothole-detection-yolov8
````

2. Create a virtual environment:

```bash
python -m venv yolov8-env
# Windows
yolov8-env\Scripts\activate
# Linux/macOS
source yolov8-env/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

---

## Model Training

1. Use the provided Google Colab Notebook to train the YOLOv8 model.
2. Download the trained weights (`best.pt`) and place it in the root directory.
3. Ensure Git LFS is installed to handle large files such as model weights and datasets.

---

## Running the Detection

1. Make sure `best.pt` is in the root directory.
2. Run the main detection script:

```bash
python pothole_detection_yolov8.py
```

3. Output images with detected potholes will be saved in the `Output images/` folder.
4. Additional scripts such as `integration.py` can be used for Firebase or other integrations.

---

## Project Structure

```text
pothole-detection-yolov8/
├── Implementation/                # Core Python scripts and modules
├── CombinedDataset_roboflow/      # Dataset used for training
├── Output images/                 # Directory to save detection results
├── test_images/                   # Test images for detection
├── valid_images/                  # Validation images
├── README.md                      # Project documentation
├── best.pt                        # Pre-trained YOLOv8 model weights (Git LFS)
├── firebase-key.json              # Firebase credentials for integration
├── integration.py                 # Script for Firebase integration
├── pothole_detection_yolov8.py    # Main detection script
├── .gitattributes                 # Git LFS attributes
├── .gitignore                     # Ignored files and folders
├── detected_output.jpg            # Example output image
├── potholes_map.html              # HTML visualization of detected potholes
└── requirements.txt               # Python dependencies
```

---

## Results

The detection results are evaluated using standard metrics like accuracy and precision. Example outputs:

* `detected_output.jpg` → Sample image with detected potholes
* `potholes_map.html` → Visualization of detected potholes

### Training Results

![Train\_model](https://github.com/user-attachments/assets/57760c74-e10f-4371-b600-c693e637cc29)

### Confusion Matrix

![Confusion matrix](https://github.com/user-attachments/assets/e09efce9-dd1c-4220-b3ee-e187adb7f3e4)

---
