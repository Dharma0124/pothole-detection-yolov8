# Pothole Detection using YOLOv8

## Overview
This project aims to detect potholes using the YOLOv8 (You Only Look Once) deep learning model. The primary objective is to automate the identification of potholes from static test images to improve road safety and maintenance. The model is pre-trained using a custom dataset in Google Colab, and it focuses on delivering accurate detection results without relying on real-time video feeds.

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
- Google Colab (for training)
- Git

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Dharma0124/pothole-detection-yolov8.git
   cd pothole-detection-yolov8

2. Create a virtual environment (recommended):
   ```bash
   python -m venv yolov8-env
   source yolov8-env/bin/activate

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

### Model Training
1. Use the provided Google Colab Notebook to train the YOLOv8 model.
2. Download the trained weights (best.pt) from Google Colab:
   Go to the Colab Notebook.
   Right-click on the trained weights file (best.pt) in the file explorer and download it.
   Save the file in the root directory of this project.
### Running the Detection
1. Make sure your best.pt file is in the root directory.
2. Run the Python script to perform pothole detection on test images:
   ```bash
   python pds_yolo.py

3. Ensure the path to the weights file is correctly set in the script:
   ```python
   model = YOLO('path/to/your/best.pt')  # Update the path

### Project Structure
1. The output images with detected potholes will be saved in the specified output directory.
   ```bash
   pothole-detection-yolov8/
   ├── best.pt               # Pre-trained YOLOv8 model weights
   ├── requirements.txt      # List of dependencies
   ├── pds_yolo.py           # Python script for pothole detection
   ├── images/               # Directory containing test images
   ├── output/               # Directory to save output images with detections
   └── README.md             # Project documentation (this file)

### Results
The detection results are evaluated using several performance metrics, including accuracy and precision. Below are the training results and confusion matrix:

### Training Results
![Train_model](https://github.com/user-attachments/assets/57760c74-e10f-4371-b600-c693e637cc29)

### Confusion Matrix
![Confusion matrix](https://github.com/user-attachments/assets/e09efce9-dd1c-4220-b3ee-e187adb7f3e4)

