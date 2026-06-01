# 🔆 Solar Panel Inspector — AI Defect Detection

An AI-powered desktop application for intelligent inspection of photovoltaic solar panels using drone imagery and deep learning.

---

## 📋 Overview

Solar Panel Inspector automatically detects and classifies visual defects on solar panels from thermal or RGB images. It uses a fine-tuned MobileNetV2 model to identify 6 defect categories, localizes the damaged zone with bounding boxes, and provides simulated GPS coordinates of the detected fault.

---

## ✨ Features

- 🤖 **AI-based defect detection** — MobileNetV2 transfer learning model trained on solar panel imagery
- 🎯 **Defect localization** — color-aware bounding box detection drawn on the image
- 🗺️ **Simulated GPS geolocation** — latitude/longitude and precision per defect type
- 📊 **Confidence scores** — per-class probability displayed with a visual bar
- 🔍 **Smart correction logic** — filters false positives (fog, shadows, reflections, sunlight glare)
- 🖥️ **Tkinter GUI** — clean desktop interface, no browser required

---

## 🗂️ Defect Classes

| Class | Label | Danger Level |
|-------|-------|-------------|
| 0 | Clean | None |
| 1 | Dusty | Low |
| 2 | Bird Drop | Medium |
| 3 | Electrical Damage | Critical |
| 4 | Physical Damage | Serious |
| 5 | Snow Covered | Medium |

Additional inferred states: **Sunlight glare**, **Fog / blurry image**, **Cloud reflection**, **Structural shadow**.

---

## 📁 Project Structure

```
solar-panel-inspector/
├── app.py                   # Main GUI application
├── train_model.py           # Model training script
├── model/
│   ├── solar_defect_model.keras   # Trained model (not included in repo)
│   └── resultats.png              # Training curves (generated after training)
├── dataset/                 # Training images organized by class
│   ├── Clean/
│   ├── Dusty/
│   ├── Bird Drop/
│   ├── Electrical Damage/
│   ├── Physical Damage/
│   └── Snow Covered/
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV (`cv2`)
- Pillow (PIL)
- NumPy
- Matplotlib
- Tkinter (included with Python on most platforms)

Install dependencies:

```bash
pip install tensorflow opencv-python pillow numpy matplotlib
```

---

## 🚀 Usage

### 1. Train the model

Organize your dataset in the `dataset/` folder with one subfolder per class, then run:

```bash
python train_model.py
```

Training takes approximately 10–20 minutes on CPU. The model is saved to `model/solar_defect_model.keras` and training curves are saved to `model/resultats.png`.

### 2. Launch the application

```bash
python app.py
```

Click **"Charger une image et analyser"** to load a solar panel image. The app will:

1. Run inference with the trained model
2. Apply false-positive correction heuristics
3. Detect and draw the defective zone
4. Display GPS coordinates, confidence score, and per-class probabilities

---

## 🧠 Model Architecture

- **Base model**: MobileNetV2 (pretrained on ImageNet, frozen)
- **Head**: GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.3) → Dense(6, Softmax)
- **Input size**: 224 × 224 × 3
- **Optimizer**: Adam
- **Loss**: Categorical cross-entropy
- **Training split**: 80% train / 20% validation

---

## 🔧 Defect Localization Method

Bounding boxes are detected using color-based HSV masking adapted to each defect type:

| Defect | Detection strategy |
|--------|--------------------|
| Electrical Damage | Orange/red HSV range |
| Bird Drop | White HSV range |
| Dusty | Light gray HSV range |
| Physical Damage | Bright + dark threshold (cracks/chips) |
| Snow Covered | Near-white HSV range |
| Sunlight glare | Brightest compact region |

Morphological operations (close + open) clean the mask before contour extraction. Up to 3 bounding boxes are drawn, sorted by area.

---

## 📌 Notes

- GPS coordinates are **simulated** and based on a fixed reference location (Toulouse, France). In a production system, these would come from drone telemetry.
- The model file (`solar_defect_model.keras`) is not included due to size. Train it locally using `train_model.py`.
- The app requires the model file to exist at `model/solar_defect_model.keras` before launching.

---

## 👥 Authors
Elaa HAMDANI - Samar GUIZANI
Projet PFA 2025/2026 — Inspection photovoltaïque par Drone & Intelligence Artificielle

---

## 📄 License

This project is for academic and educational use.

