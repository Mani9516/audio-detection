import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import streamlit as st

# Helper Functions
def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract MFCC features from an audio file.
    """
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        st.error(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)


def create_dataset(directory, label):
    """
    Create a dataset with features and labels from a directory of audio files.
    """
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    if not audio_files:
        st.error(f"No .wav files found in directory: {directory}")
    else:
        st.write(f"Processing {len(audio_files)} files from {directory}")

    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
    return X, y


def train_model(X, y):
    """
    Train an SVM model and save the trained model and scaler on the backend.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_scaled, y)

    # Define backend file paths for saving the model and scaler
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"

    # Save the model and scaler to the backend
    joblib.dump(svm_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)

    st.success("Model and scaler saved successfully!")


def analyze_audio(input_audio_path):
    """
    Analyze a single audio file and classify it as genuine or deepfake.
    """
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"

    # Check if the model and scaler files exist in the backend
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        return "Error: Model or scaler file not found. Train the model first."

    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is None:
        return "Error: Unable to process the input audio."

    scaler = joblib.load(scaler_filename)
    mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

    svm_classifier = joblib.load(model_filename)
    prediction = svm_classifier.predict(mfcc_features_scaled)

    if prediction[0] == 0:
        return "The input audio is classified as genuine 📈."
    else:
        return "The input audio is classified as deepfake ☠️, Be Safe."

# Streamlit UI
st.title("🎹 Audio Deepfake Detection Tool")
st.sidebar.title("⚙️ Options")

# Training Section
if st.sidebar.checkbox("Train Model"):
    st.header("Train the Model")
    st.write("The training process will use pre-defined directories for genuine and deepfake audio files.")

    if st.button("Train"):
        # Predefined directories for genuine and deepfake audio files
        genuine_dir = "C:/Users/mani chourasiya/Downloads/real"
        deepfake_dir = "C:/Users/mani chourasiya/Downloads/fake"

        X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
        X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)

        if len(X_genuine) > 0 and len(X_deepfake) > 0:
            X = np.vstack((X_genuine, X_deepfake))
            y = np.hstack((y_genuine, y_deepfake))
            train_model(X, y)
        else:
            st.error("Insufficient data to train the model. Ensure the directories contain valid .wav files.")

# Prediction Section
st.header("🎧 Analyze Audio File")
uploaded_file = st.file_uploader("Upload a .wav file for analysis", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Play the audio file
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Analyze"):
        # Analyze and display result
        result = analyze_audio(temp_path)
        st.success(result)

        # Remove temporary file
        os.remove(temp_path)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with Lenscan.AI ⚡")
