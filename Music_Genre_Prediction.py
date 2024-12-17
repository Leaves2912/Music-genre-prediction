# Importing necessary libraries
import pandas as pd
import joblib
import requests
import librosa
import numpy as np
import streamlit as st
import os
import tempfile

st.markdown(
    """
    <style>
    * {
        font-family: 'Times New Roman', Times, serif;  
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Function to download Music file from a URL
def download_Music_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False

# Function to extract Music features
def extract_Music_features(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sample_rate)
    harmony = np.mean(librosa.effects.harmonic(y=x))
    perceptr_mean = np.mean(librosa.feature.spectral_centroid(y=x))
    perceptr_var = np.var(librosa.feature.spectral_centroid(y=x))
    mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13)
    mfcc1_mean = np.mean(mfcc[0])
    mfcc3_var = np.var(mfcc[2])
    mfcc4_mean = np.mean(mfcc[3])
    mfcc5_var = np.var(mfcc[4])
    mfcc6_mean = np.mean(mfcc[5])
    mfcc6_var = np.var(mfcc[5])
    mfcc8_var = np.var(mfcc[7])
    mfcc9_mean = np.mean(mfcc[8])
    length = librosa.get_duration(y=x, sr=sample_rate)

    features = [
        length,
        np.mean(chroma_stft),
        np.var(chroma_stft),
        np.mean(spectral_bandwidth),
        harmony,
        perceptr_mean,
        perceptr_var,
        mfcc1_mean,
        mfcc3_var,
        mfcc4_mean,
        mfcc5_var,
        mfcc6_mean,
        mfcc6_var,
        mfcc8_var,
        mfcc9_mean
    ]
    return features

# Cached function to load the model
@st.cache_resource
def load_model():
    return joblib.load('best_rf_model.pkl')

# Load the model
rfc = load_model()

# Streamlit UI
st.markdown("<h1 style='font-family: Times New Roman; font-size: 40px;'>Music Genre Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family: Times New Roman; font-size: 25px;'>This project trains a Random Forest model to classify audio genres using feature extraction from audio datasets. It includes a Streamlit-based app that predicts the genre of an audio file provided via URL.</p>", unsafe_allow_html=True)


# Use HTML to customize the label and control spacing
st.markdown( """ <h2 style='font-family: Times New Roman; font-size: 28px; margin-bottom: -50px;'>Enter the Music File URL:</h2> """, unsafe_allow_html=True )

# Create the text input field
Music_url = st.text_input("")

# Define save_path using a temporary directory
temp_dir = tempfile.gettempdir()
save_path = os.path.join(temp_dir, "TEST_FILE.wav")

if st.button("Download and Predict"):
    if download_Music_file(Music_url, save_path):
        st.markdown(f"<p style='font-size: 18px; margin-bottom: 40px;'>Music File Downloaded Successfully!</p>", unsafe_allow_html=True)
        
        # Extract features and predict genre
        extracted_features = [extract_Music_features(save_path)]
        predicted_genre = rfc.predict(extracted_features)
        
        # Display predicted genre with increased font size
        formatted_genre = predicted_genre[0].capitalize()

        # Use HTML to customize the label and control spacing
        st.markdown(f"<h2 style='font-family: Times New Roman; font-size: 28px; margin-bottom: -25px;'>Predicted Genre:</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size: 25px;'>{formatted_genre}</p>", unsafe_allow_html=True)
    else:
        st.error("Failed To Download The Music File. Please Check The URL.")

# Clean up: Remove the downloaded file after prediction
if os.path.exists(save_path):
    os.remove(save_path)
