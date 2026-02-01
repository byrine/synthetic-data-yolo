# End-to-End Synthetic Data Pipeline & YOLOv8 Optimization

## 🚀 Project Overview
This project demonstrates an automated pipeline to solve data scarcity in Computer Vision. I built a system that programmatically generates synthetic training data using **OpenCV**, trains a **YOLOv8** object detection model, and optimizes the inference pipeline using **ONNX Runtime** for production environments.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Computer Vision:** OpenCV, Albumentations
* **Deep Learning:** YOLOv8 (Ultralytics), PyTorch
* **Deployment:** ONNX Runtime (CPU/Edge Optimization)

## 📂 Project Structure
* `generate_data.py`: The engine. Generates thousands of synthetic images with randomized domain adaptation (rain, fog, blur).
* `prepare_dataset.py`: Automates the split of Train/Val sets and configures the YOLO yaml files.
* `infer_onnx.py`: A lightweight, dependency-free inference script that runs the model without PyTorch.

## ⚡ How to Run
1.  **Generate Data:**
    ```bash
    python generate_data.py
    ```
2.  **Train Model:**
    ```bash
    yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=10
    ```
3.  **Run Inference (ONNX):**
    ```bash
    python infer_onnx.py
    ```