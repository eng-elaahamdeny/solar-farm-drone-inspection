# 🛸 Solar Farm Drone Inspection

> AI-powered desktop application for intelligent photovoltaic panel defect detection — trained on thermal drone imagery using Deep Learning (MobileNetV2).

---

## 📌 Overview

This project combines drone-captured thermal imagery with a deep learning model to automatically detect and classify solar panel defects. A clean desktop GUI allows operators to load any panel image and get an instant AI-powered diagnosis with confidence scores per defect class.

---

## 🎯 Detected Defect Classes

| Class | Severity | Description |
|---|---|---|
| 🔥 `Hotspot` | **Critical** | Localized overheating on a cell |
| 💥 `Crack` | **Serious** | Physical micro-fracture in the panel |
| 🐦 `Bird Drop` | **Medium** | Soiling from bird droppings |
| 🌫️ `Dirty` | **Medium** | Dust or dirt accumulation |
| ❄️ `Snow Covered` | **Low** | Panel covered by snow |
| ✅ `Normal` | **None** | Panel operating correctly |

---

## 🧠 Model Architecture

- **Base model:** MobileNetV2 (pretrained on ImageNet)
- **Fine-tuning:** Custom classification head (GlobalAveragePooling → Dense 128 → Dropout → Softmax)
- **Input size:** 224×224 RGB
- **Training split:** 80% train / 20% validation
- **Data augmentation:** Rotation, horizontal flip, normalization
- **Optimizer:** Adam | **Loss:** Categorical Crossentropy
- **Epochs:** 10

---

## 🗂️ Project Structure

```
solar-farm-drone-inspection/
│
├── dataset/                        # Kaggle dataset (solar_augmented_dataset)
│   ├── Bird_Drop/
│   ├── Snow_Covered/
│   ├── Crack/
│   ├── Dirty/
│   ├── Hotspot/
│   └── Normal/
│
├── model/
│   ├── solar_defect_model.keras    # Saved trained model
│   └── resultats.png               # Training accuracy/loss curves
│
├── train_model.py                  # Model training script
├── interface.py                    # Desktop GUI application
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/solar-farm-drone-inspection.git
cd solar-farm-drone-inspection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download **solar_augmented_dataset** from [Kaggle](https://www.kaggle.com/) and place the class folders inside `dataset/`:

```
dataset/
├── Bird_Drop/
├── Snow_Covered/
├── Crack/
├── Dirty/
├── Hotspot/
└── Normal/
```

### 4. Train the model

```bash
python train_model.py
```

This will:
- Load and augment the dataset automatically
- Train MobileNetV2 for 10 epochs (~10–20 min depending on hardware)
- Save the model to `model/solar_defect_model.keras`
- Generate and save accuracy/loss curves to `model/resultats.png`

### 5. Launch the desktop app

```bash
python interface.py
```

---

## 🖥️ Application Features

The desktop GUI (built with Tkinter) lets you:

- 📂 **Load** any thermal or RGB panel image from disk
- 🤖 **Analyze** it instantly with the trained AI model
- 📊 **View** a confidence bar + scores for all 6 defect classes
- 🎨 **Color-coded** severity levels — green (safe) → orange → red (critical)

---

## 📦 Requirements

```
tensorflow>=2.10
numpy
Pillow
matplotlib
```

Install all at once:

```bash
pip install tensorflow numpy Pillow matplotlib
```

---

## 📊 Training Output

After training, two plots are saved to `model/resultats.png`:

- **Accuracy curve** — Train vs Validation accuracy per epoch
- **Loss curve** — Train vs Validation loss per epoch

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋 Author
**Elaa Hamdani**  
Engineering Student at INSAT – Instrumentation & Industrial Maintenance Engineering  
Specialized in AI & Aerodynamics

> *MobileNetV2 · TensorFlow · Tkinter · Deep Learning · Thermal Imaging*
