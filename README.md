# 👁️ Retinal Eye Disease Detection System
### AI-Powered OCT Scan Classification using Deep Learning & Streamlit

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red?style=for-the-badge&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green?style=for-the-badge&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

</p>

---

# 📖 Overview

Retinal diseases are among the leading causes of vision impairment worldwide. Early diagnosis plays a crucial role in preventing permanent vision loss.

The **Retinal Eye Disease Detection System** is an end-to-end Deep Learning application that automatically analyzes **Optical Coherence Tomography (OCT)** retinal scans and predicts the presence of retinal abnormalities.

The system combines:

- 🧠 Deep Learning (CNN)
- 📷 Medical Image Processing
- ⚡ Real-time Prediction
- 🌐 Interactive Streamlit Dashboard
- 📊 Clinical Recommendation Engine

to provide instant disease classification along with clinical interpretation and suggested precautions.

---

# 🚀 Features

✅ Upload retinal OCT images

✅ Automatic Image Preprocessing

- Resize
- Grayscale Conversion
- CLAHE Enhancement
- Normalization

✅ CNN-based Disease Prediction

✅ Softmax Confidence Scores

✅ Disease Description

✅ Severity Analysis

✅ Recommended Next Steps

✅ Clean Streamlit Dashboard

✅ Interactive Clinical Alerts

---

# 🩺 Diseases Detected

| Disease | Description |
|----------|-------------|
| 🔴 CNV | Choroidal Neovascularization |
| 🟠 DME | Diabetic Macular Edema |
| 🟡 DRUSEN | Early Age-related Macular Degeneration |
| 🟢 NORMAL | Healthy Retina |

---

# 🧠 Deep Learning Pipeline

```
                OCT Image
                     │
                     ▼
          Image Preprocessing
      (Resize + CLAHE + Normalize)
                     │
                     ▼
        Convolutional Neural Network
                     │
         Feature Extraction Layers
                     │
                     ▼
          Dense Classification Layer
                     │
                     ▼
        Softmax Probability Scores
                     │
                     ▼
     Disease Recommendation Engine
                     │
                     ▼
         Streamlit User Dashboard
```

---

# 🏗 Project Architecture

```
                    ┌──────────────────┐
                    │ OCT Retinal Scan │
                    └─────────┬────────┘
                              │
                              ▼
                 Image Preprocessing Module
             (CLAHE + Resize + Normalization)
                              │
                              ▼
                 Convolutional Neural Network
                              │
                              ▼
                  Disease Classification
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
      Confidence Score                  Disease Class
            │                                   │
            └──────────────┬────────────────────┘
                           ▼
              Clinical Recommendation Engine
                           │
                           ▼
               Interactive Streamlit Dashboard
```

---

# 🧠 CNN Architecture

Input Shape

```
224 × 224 × 1
```

### Layer Configuration

| Layer | Configuration |
|---------|--------------|
| Conv2D | 32 Filters (3×3), ReLU |
| MaxPooling | 2×2 |
| Conv2D | 64 Filters (3×3), ReLU |
| MaxPooling | 2×2 |
| Flatten | Feature Vector |
| Dense | 128 Neurons, ReLU |
| Output | Dense(4), Softmax |

---

# 📷 Image Processing Pipeline

The uploaded OCT image undergoes multiple enhancement stages before inference.

### Step 1

Resize image

```
224 × 224
```

↓

### Step 2

Convert to grayscale

↓

### Step 3

Apply CLAHE

Contrast Limited Adaptive Histogram Equalization improves retinal texture visibility.

↓

### Step 4

Normalize pixel values

↓

### Step 5

CNN Prediction

---

# 📂 Project Structure

```
Eye_Disease_Detection_Project
│
├── config
│   └── requirements.txt
│
├── models
│   └── eye_disease_model.keras
│
├── sample_data
│   ├── temp_upload.jpg
│   └── test_eye.jpg
│
├── src
│   ├── app.py
│   ├── cnn_architecture.py
│   ├── evaluator.py
│   ├── preprocessor.py
│   ├── recommendation.py
│   └── train.py
│
├── README.md
└── LICENSE
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/vedantvarma19/Eye_Disease_Detection_Project.git
```

```bash
cd Eye_Disease_Detection_Project
```

---

## Create Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r config/requirements.txt
```

---

# ▶ Running the Application

```bash
streamlit run src/app.py
```

Open

```
http://localhost:8501
```

---

# 🖥 Application Workflow

```
Upload OCT Image
        │
        ▼
Image Enhancement
        │
        ▼
CNN Prediction
        │
        ▼
Disease Detection
        │
        ▼
Confidence Score
        │
        ▼
Clinical Recommendation
```

---

# 📸 Application Preview

## Dashboard

```
📌 Add Screenshot Here
```

---

## Upload Screen

```
📌 Add Screenshot Here
```

---

## Prediction Result

```
📌 Add Screenshot Here
```

---

# 📊 Prediction Output Example

```
Prediction

Disease:
CNV

Confidence:
98.47%

Severity:
High

Recommendation:

Immediate consultation with an ophthalmologist is strongly advised.

Potential retinal neovascularization detected.
```

---

# 🩺 Clinical Recommendation Engine

The recommendation module automatically maps predictions to medical insights.

Example

```
Prediction

↓

Disease Description

↓

Severity

↓

Suggested Action

↓

Emergency Indicator
```

Example Output

| Disease | Severity | Recommendation |
|----------|------------|----------------|
| NORMAL | Low | Routine eye check-up |
| DRUSEN | Moderate | Ophthalmologist Consultation |
| DME | High | Immediate Medical Evaluation |
| CNV | Critical | Emergency Retina Specialist |

---

# 📈 Model Performance

The trained CNN is evaluated using standard classification metrics.

Metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 🛠 Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning |
| OpenCV | Image Processing |
| NumPy | Numerical Operations |
| Streamlit | Web Interface |
| PIL | Image Handling |

---

# 💡 Future Improvements

- Transfer Learning (EfficientNet)
- Mobile Deployment
- Docker Containerization
- Cloud Deployment
- Multi-Class Probability Visualization
- Explainable AI (Grad-CAM)
- PDF Medical Report Generation
- REST API Support
- User Authentication
- Medical Database Integration

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Added feature"
```

4. Push

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## Vedant Varma

Information Science Engineering Student

AI • Machine Learning • Deep Learning • Full Stack Development

GitHub

https://github.com/vedantvarma19

LinkedIn

https://www.linkedin.com/in/vedant-varma-14952232a/

---

# ⭐ Support

If you found this project useful,

please consider giving it a ⭐ on GitHub.

It motivates me to build more open-source AI projects.

---

<p align="center">

⭐ Star this Repository • 🍴 Fork it • 🚀 Build Something Amazing

</p>
