# 🧠 Neuroscan: Automated EEG Seizure Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-94.30%25-brightgreen)

## 📌 Project Overview
Neuroscan is an end-to-end Machine Learning diagnostic tool designed to analyze continuous raw EEG (Electroencephalogram) brainwave data and automatically detect epileptic seizure activity in real-time. 

Built as a proof-of-concept for MedTech deployment, the project condenses a complex Digital Signal Processing (DSP) and Machine Learning pipeline into a clean, interactive web dashboard that medical professionals can easily use.

## 🗄️ Dataset
This model was trained on the **University of Bonn EEG Database**, a globally recognized gold-standard dataset for epileptic seizure research. The data was pre-segmented into 23.6-second intervals and binarized for classification:
* **Label 1:** Epileptic Seizure (Ictal phase)
* **Label 0:** Normal Brain Activity (Healthy, Tumorous, Eyes Open, Eyes Closed)

## ⚙️ Technical Architecture
The system processes data through a strict 4-phase pipeline:

### 1. Preprocessing (Noise Reduction)
Raw EEG signals are notoriously noisy. Neuroscan utilizes a **4th-Order Butterworth Bandpass Filter** (0.5 Hz - 40.0 Hz) to eliminate high-frequency electrical interference and low-frequency muscle movement artifacts, isolating pure brainwave frequencies.

### 2. Feature Extraction (DSP)
Passing 178 raw voltage points to an ML model causes overfitting. Instead, the system applies a **Discrete Wavelet Transform (DWT)** using the Daubechies 4 (`db4`) wavelet. It compresses the time-series signal into two powerful mathematical features:
* **Energy:** The overall power and magnitude of the brainwave.
* **Shannon Entropy:** The level of chaos and unpredictability in the signal.

### 3. Machine Learning Model
The extracted features are standardized using `StandardScaler` and passed to a **Support Vector Machine (SVM)** with an RBF kernel. 
* **Training Environment:** Kaggle Cloud Compute
* **Model Accuracy:** 94.30%

### 4. Web Dashboard (UI)
The backend model is deployed via **Streamlit**. The UI handles file uploading, live pipeline processing, status alerting (Normal vs. Seizure), and renders a beautiful, interactive visualization of the patient's cleaned brainwave using `matplotlib`.

## 🚀 How to Run Locally

### Prerequisites
Make sure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
