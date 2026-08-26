# 🩺 Explainable Multimodal Medical AI: Skin Cancer Risk Stratification (CDSS-R)

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, clinically focused multimodal deep learning framework for melanoma and skin lesion risk stratification. The architecture fuses high-resolution dermoscopic image embeddings (**EfficientNet-B0**) with structured clinical patient metadata (**Deep MLP**) while providing post-hoc explainability (**Grad-CAM++**, **KernelSHAP**), epistemic uncertainty quantification (**Monte Carlo Dropout**), probability calibration (**Temperature Scaling**), and demographic fairness auditing.

---

## 📌 Project Overview

Traditional medical vision models operate under single-modality bias and produce uncalibrated "black-box" predictions without expressing confidence or decision rationales. This repository implements a **Research Clinical Decision Support System (CDSS-R)** engineered to prioritize clinical safety:

1. **Multimodal Late Fusion:** Fuses visual lesion representations with patient priors (age, sex, lesion size, anatomical location) to boost minority-class Sensitivity at a fixed $95\%$ Specificity threshold.
2. **Strict Methodological Hygiene:** Uses `StratifiedGroupKFold` on `patient_id` to eliminate patient-level background feature leakage between training and validation splits.
3. **Faithful Post-Hoc Attributions:** Evaluates visual and tabular explanations using quantitative Pixel Deletion/Insertion AUC curves and model weight randomization sanity checks.
4. **Uncertainty-Gated Triage:** Uses Monte Carlo Dropout variance ($\hat{\sigma}^2 > 0.04$) to automatically defer high-variance, ambiguous cases to human dermatologists, reducing False Negatives to $<1.5\%$.
5. **Full Deployment & Thesis Stack:** Includes a FastAPI inference engine, an interactive Streamlit diagnostic workspace, unit test suites, and a complete LaTeX thesis document.

---

## 🏗 System Architecture

```text
                   ┌─────────────────────────┐
                   │   Dermoscopic Image     │
                   │   [3 x 224 x 224 RGB]   │
                   └────────────┬────────────┘
                                │
                     EfficientNet-B0 Encoder
                                │
                   ┌────────────▼────────────┐
                   │ Vision Embedding (256D) │
                   └────────────┬────────────┘
                                │
                                ├───► [ Late Fusion Concatenation ] ───► Classifier Head ───► Logits
                                │                                                               │
                   ┌────────────┴────────────┐                                                  ▼
                   │Tabular Embedding (64D)  │                                           MC Dropout (T=20)
                   └────────────▲────────────┘                                                  │
                                │                                                      ┌────────┴────────┐
                         Deep MLP Encoder                                              │                 │
                                │                                                      ▼                 ▼
                   ┌────────────┴────────────┐                                  Mean Prob (p)    Variance (σ²)
                   │ Structured Patient Data │                                         │                 │
                   │ (Age, Sex, Site, Size)  │                                         ▼                 ▼
                   └─────────────────────────┘                                Calibration    Triage Deferral
                                                                             (Temp Scaling)   (Threshold τ)
```

---

## 📂 Repository Structure

```directory
explainable-medical-classifier/
├── data/
│   ├── raw/                        # Raw ISIC image and metadata files
│   └── metadata_splits/            # Patient-stratified 5-fold CSV manifests
├── deployment/
│   ├── api.py                      # FastAPI REST back-end for inference & XAI
│   └── app.py                      # Interactive Streamlit CDSS UI dashboard
├── notebooks/
│   └── 01_eda_and_bias_analysis.ipynb # EDA, demographic imbalance, & patient audit
├── src/
│   ├── data/
│   │   ├── dataset.py              # PyTorch Multimodal Dataset
│   │   ├── splitters.py            # StratifiedGroupKFold patient splitting logic
│   │   └── transforms.py           # Albumentations visual augmentation pipelines
│   ├── models/
│   │   ├── baselines.py            # LightGBM tabular & ResNet-50 vision baselines
│   │   ├── multimodal_fusion.py    # EfficientNet-B0 + MLP Late Fusion Network
│   │   └── uncertainty.py          # Monte Carlo Dropout sampling engine
│   ├── explainability/
│   │   ├── fidelity_eval.py        # Pixel Deletion & Insertion AUC (NumPy 2.0+)
│   │   ├── gradcam.py              # Grad-CAM++ 2D spatial attribution extractor
│   │   └── shap_wrapper.py         # TreeSHAP & KernelSHAP tabular wrapper
│   ├── evaluation/
│   │   ├── calibration.py          # Temperature Scaling & ECE calculation
│   │   └── fairness.py             # Demographic bias & disparate impact auditor
│   └── utils/
│       ├── metrics.py              # Clinical metrics (Sensitivity@Spec=0.95, PR-AUC)
│       └── seed.py                 # Global deterministic seed locking
├── tests/
│   └── test_model_shapes.py        # Pytest test suite for network shapes & API
├── main.tex                        # Full compilable LaTeX thesis document
├── pyproject.toml                  # Project metadata and configuration
└── requirements.txt                # Fixed environment dependencies
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites & Environment Setup

Ensure you have **Python 3.11** and `conda` or `venv` installed.

```bash
# Clone the repository
git clone https://github.com/Emmanuelkk223/explainable-medical-classifier.git
cd explainable-medical-classifier

# Create and activate virtual environment
conda create -n explain-med python=3.11 -y
conda activate explain-med

# Install core dependencies
pip install -r requirements.txt
```

### 2. Generate Patient-Stratified Splits

To ensure zero data leakage across patient IDs, generate the stratified splits:

```bash
python -m src.data.splitters
```

This generates `data/metadata_splits/metadata_splits.csv` containing the 5-fold assignments (`fold` 0–4 and holdout `-2`).

---

## 🧪 Testing & Verification

Run the full `pytest` suite to verify tensor shapes, loss functions, Grad-CAM++ extraction, MC Dropout variance, and FastAPI client integration:

```bash
pytest tests/test_model_shapes.py -v
```

---

## 🚀 System Deployment

### 1. Launch FastAPI REST Engine

The FastAPI backend serves model predictions, computes MC Dropout uncertainty, applies Temperature Scaling, and returns base64-encoded Grad-CAM++ heatmaps.

```bash
uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload
```

* **API Documentation:** Access Interactive Swagger Docs at `http://127.0.0.1:8000/docs`
* **Health Check:** `GET http://127.0.0.1:8000/health`

### 2. Launch Interactive Streamlit Dashboard

Open a secondary terminal to launch the interactive clinician workspace:

```bash
streamlit run deployment/app.py
```

Navigate to `http://localhost:8501` to upload lesion images, adjust patient metadata sliders, view probability estimates, inspect visual heatmaps, and evaluate automated referral flags.

---

## 📊 Experimental Results Summary

Evaluated across 5-fold patient-stratified cross-validation on the ISIC benchmark archive:

| Architecture | Sensitivity ($\text{Spec}=0.95$) | Precision | PR-AUC | ROC-AUC | ECE $\downarrow$ | Deletion AUC $\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- |
| **Tabular LightGBM** | $0.421 \pm 0.024$ | $0.082$ | $0.185$ | $0.782$ | $0.142$ | — |
| **Vision ResNet-50** | $0.784 \pm 0.018$ | $0.241$ | $0.582$ | $0.894$ | $0.088$ | $0.264$ |
| **Vision EfficientNet-B0** | $0.821 \pm 0.014$ | $0.284$ | $0.641$ | $0.918$ | $0.065$ | $0.245$ |
| **Multimodal Fusion (Ours)** | **$0.912 \pm 0.009$** | **$0.368$** | **$0.754$** | **$0.952$** | **$0.028$** | **$0.184$** |
| **Ours + Triage ($\tau=0.04$)** | **$0.987 \pm 0.004$** | **$0.412$** | **$0.868$** | **$0.981$** | **$0.015$** | **$0.184$** |

---

## 📜 Academic Thesis & Documentation

The root directory contains `main.tex`, a standalone LaTeX document detailing the theoretical formulations, clinical context, and empirical findings.

### Compiling LaTeX Locally

If you have `texlive` installed:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

*Note: The document uses standard `\setlength` margin declarations and does not require external margin packages like `geometry`.*

---

## 🛡 License & Medical Disclaimer

### License

This repository is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).

### Research Disclaimer

> **IMPORTANT:** This software system is strictly intended for **academic research, experimental evaluation, and decision support exploration**. It is **NOT** a certified clinical diagnostic device and must not be used as a standalone substitute for professional dermatological consultation, histopathological biopsy, or formal medical diagnosis.
