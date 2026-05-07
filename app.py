import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis

st.set_page_config(page_title="Neuroscan Diagnostic AI", layout="centered")
st.title("Neuroscan: EEG Seizure Detection")
st.write("Upload a raw EEG signal file (.txt) to instantly detect ictal (seizure) activity using our robust Machine Learning model.")

@st.cache_resource # This caches the model so it doesn't reload every time you click a button
def load_models():
    model = joblib.load('robust_seizure_model.pkl')
    scaler = joblib.load('eeg_scaler.pkl')
    return model, scaler

model, scaler = load_models()

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=173.6):
    nyq = 0.5 * fs
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signal, fs=173.6):
    feats = []
    feats.extend([np.mean(signal), np.std(signal), np.var(signal), skew(signal), kurtosis(signal)])
    feats.extend([np.max(np.abs(signal)), np.sum(signal**2), np.sqrt(np.mean(signal**2))])
    feats.append(np.sum(np.diff(np.sign(signal)) != 0))
    feats.append(np.ptp(signal))
    
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    power = np.abs(np.fft.rfft(signal))**2
    bands = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,40)}
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        bp = np.sum(power[mask])
        feats.append(bp)
        feats.append(bp / (np.sum(power) + 1e-8))
    return feats

uploaded_file = st.file_uploader("Upload EEG Segment (.txt file)", type=['txt'])

if uploaded_file is not None:
    # Read the data
    raw_signal = np.loadtxt(uploaded_file)
    
    st.subheader("1. Signal Visualization")
    # Plot the raw signal
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(raw_signal, color='#1f77b4', linewidth=0.8)
    ax.set_title("Uploaded Raw EEG Signal")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time (Samples)")
    st.pyplot(fig)
    
    with st.spinner("Processing Signal and Extracting Features..."):
        # Process data
        filtered_signal = bandpass_filter(raw_signal)
        normalized_signal = (filtered_signal - np.mean(filtered_signal)) / (np.std(filtered_signal) + 1e-8)
        
        # Extract and scale features
        features = np.array([extract_features(normalized_signal)])
        scaled_features = scaler.transform(features)
        
        # Predict
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]

    st.subheader("2. Diagnostic Report")
    
    # Display Results beautifully
    if prediction == 1:
        st.error(f"**ALERT: Seizure Activity Detected!**")
        st.write(f"**Confidence Score:** {probability[1] * 100:.2f}%")
    else:
        st.success(f"**NORMAL: No Seizure Activity Detected.**")
        st.write(f"**Confidence Score:** {probability[0] * 100:.2f}%")
        
    st.info("Note: This tool uses a robust Random Forest model trained with Gaussian Noise augmentation, achieving a 0.9942 AUC score.")