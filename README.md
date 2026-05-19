# 🛑 Synthetic Data Pipeline for YOLOv8

A demonstration of how synthetic data generation can solve data scarcity in Computer Vision. Using a single stop sign image (no background) and a collection of background images, this pipeline programmatically generates a labeled training dataset, trains a YOLOv8 model, and achieves near-perfect detection — with only 50 synthetic images.

---

## 💡 Motivation

Collecting and annotating real-world data is expensive and time-consuming. This project shows that with the right augmentation strategy, a small synthetic dataset can be enough to train a highly accurate object detector.

---

## 🛠️ Tech Stack

- **Computer Vision:** OpenCV, Albumentations
- **Deep Learning:** YOLOv8 (Ultralytics)
- **Inference:** ONNX Runtime (CPU/edge optimized)
- **Language:** Python 3.9+

---

## ⚙️ How It Works

1. **Synthetic image generation** — a single cutout of a stop sign is composited onto diverse background images with randomized position, scale, and domain augmentations (rain, fog, blur)
2. **Auto-labeling** — bounding box annotations are generated automatically during image generation (no manual labeling needed)
3. **Dataset preparation** — images are split into train/val sets with YOLO-compatible config files
4. **Training** — YOLOv8n is fine-tuned on the synthetic dataset
5. **Inference** — the trained model is exported to ONNX for lightweight, dependency-free deployment

---

## 📊 Results

Trained and validated on **50 synthetic images**:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **Stop Sign** | 0.997 | 1.000 | 0.995 | 0.995 |

Near-perfect detection achieved with zero manual annotation.

---

## 📁 Project Structure

```
synthetic-data-yolo/
├── generate_data.py      # Generates synthetic images with augmentations
├── prepare_dataset.py    # Splits data and creates YOLO config
├── infer_onnx.py         # Lightweight ONNX inference (no PyTorch needed)
├── visualize_data.py     # Visualize generated dataset
├── assets/               # Background images and stop sign cutout
└── models/               # Trained model weights
```

---

## ▶️ How to Run

### 1. Generate Synthetic Data
```bash
python generate_data.py
```

### 2. Train the Model
```bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=10
```

### 3. Run Inference (ONNX)
```bash
python infer_onnx.py
```
