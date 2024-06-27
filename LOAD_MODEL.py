import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from librosa.util import normalize
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation, RPComputation
from pyrqa.opencl import OpenCL
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_features(audio_file, sr, hop_length):
    y, sr = librosa.load(audio_file, sr=sr)
    normalized_y = normalize(y)
    time_series = TimeSeries(normalized_y.tolist(), embedding_dimension=2, time_delay=4)
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(0.9),
                        similarity_measure=EuclideanMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings, opencl=OpenCL(platform_id=0, device_ids=(0,)), verbose=True)
    hasil_rqa = rqa.run()
    
    # Extract features from RQA result
    recurrence_rate = hasil_rqa.recurrence_rate
    determinism = hasil_rqa.determinism
    entropy_diagonal = hasil_rqa.entropy_diagonal_lines
    entropy_vertical = hasil_rqa.entropy_vertical_lines
    entropy_white_vertical = hasil_rqa.entropy_white_vertical_lines
    laminarity = hasil_rqa.laminarity
    trapping_time = hasil_rqa.trapping_time
    divergence = hasil_rqa.divergence
    L_max = hasil_rqa.longest_diagonal_line
    L_min = hasil_rqa.min_diagonal_line_length
    V_min = hasil_rqa.min_vertical_line_length
    W_min = hasil_rqa.min_white_vertical_line_length
    W_max = hasil_rqa.longest_vertical_line
    W_div = hasil_rqa.longest_white_vertical_line_inverse
    
    features = np.array([recurrence_rate, determinism, entropy_diagonal, entropy_vertical, entropy_white_vertical, laminarity, trapping_time, divergence, L_max, L_min, V_min, W_min, W_max, W_div])
    #Indeks fitur yang ingin diambil 
    selected_features = [0, 2, 4, 6, 12, 13]

    # Membuat array baru dengan fitur yang dipilih
    features = features[selected_features]
    
    return features

def normalize_data(X, scaler=None):
    #X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled.reshape(1, -1, 1), scaler
    #return X_scaled.reshape(1, -1, 1), scaler

def predict_parkinson(audio_file, model_path, scaler_path):
    # Load the saved model
    model = load_model(model_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    #model.summary()
    
    # Extract features from the audio file
    features = extract_features(audio_file, 22050, 128)
    features = features.reshape(1, -1)
    #print(features)
    
    # Normalize data
    features, _ = normalize_data(features, scaler)
    print(features)
    
    # Make prediction
    prediction = model.predict(features)
    #print(prediction)
    
    # Decode the predictions
    predicted_label = np.argmax(prediction, axis=1)
    #predicted_label = le.inverse_transform(predicted_label)
    #print(predicted_label)
    
    return predicted_label, prediction
    #return le.inverse_transform(predicted_label)[0]

# Path to the saved model file
model_path = 'kodingan RQA/FIX/TEST MODELS/TEST_MODEL1'

# Path to the saved scaler
scaler_path = 'kodingan RQA/FIX/TEST MODELS/TEST_MODEL1.pkl'

new_audio_paths = [
    "dataset/validation/2024-05-13 13-29-59.wav",
    "dataset/validation/2024-05-13 13-30-56.wav",
    "dataset/validation/2024-05-13 13-32-02.wav",
    "dataset/Dataset Suara/DATA BUAT UJI COBA/PARKINSON/D2rlouscsi77F2605161823.wav",
    "dataset/Dataset Suara/DATA BUAT UJI COBA/PARKINSON/D2rriovbie49M2605161843.wav",
    "dataset/Dataset Suara/DATA BUAT UJI COBA/PARKINSON/D2sncihcio44M1606161719.wav"
]

for audio_path in new_audio_paths:
    predicted_label, prediction = predict_parkinson(audio_path, model_path, scaler_path)
    label_map = {0: "Healthy", 1: "Parkinson"}
    result = label_map[predicted_label[0]]
    print(f"Predicted Label for {os.path.basename(audio_path)}: {result}")
    print(f"Prediction Confidence: {prediction}")
