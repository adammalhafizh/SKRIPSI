import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from librosa.util import normalize
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.opencl import OpenCL
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.layers import  LeakyReLU, Conv1D, MaxPooling1D, BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D, Dense, Dropout
def extract_features(audio_file, sr, hop_length):
    y, sr = librosa.load(audio_file, sr=sr)
    normalized_y = normalize(y)
    time_series = TimeSeries(normalized_y.tolist(), embedding_dimension=2, time_delay=4)
    settings = Settings(time_series, analysis_type=Classic, neighbourhood=FixedRadius(0.9),
                        similarity_measure=EuclideanMetric, theiler_corrector=1)
    rqa = RQAComputation.create(settings, opencl=OpenCL(platform_id=0, device_ids=(0,)), verbose=True)
    hasil_rqa = rqa.run()
    
    # Ekstraksi fitur dari hasil RQA
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
    selected_features = [0, 1, 2, 4, 5, 6, 7, 8, 12, 13]
    features = features[selected_features]
    
    return features

def plot_history(history):
    # Plot nilai akurasi dan loss pelatihan & validasi
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    # Plot nilai loss pelatihan & validasi
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.show()
    
def augment_data(X, Y):
    augmented_X, augmented_Y = [], []
    for x, y in zip(X, Y):
        augmented_X.append(x)
        augmented_Y.append(y)
        noise = np.random.normal(0, 0.005, x.shape)
        augmented_X.append(x + noise)
        augmented_Y.append(y)
    return np.array(augmented_X), np.array(augmented_Y)

def normalize_data(X, scaler=None):
    X_reshaped = X.reshape(X.shape[0], -1)
    if scaler is None:
        scaler = StandardScaler().fit(X_reshaped)
    X_scaled = scaler.transform(X_reshaped)
    return X_scaled.reshape(X.shape[0], X.shape[1], 1), scaler

def load_data():
    # Memuat array dari file
    fitur_normal = np.load('D:/Codingan/skripsi/fitur_normal.npy')
    label_normal = np.load('D:/Codingan/skripsi/label_normal.npy')

    fitur_parkinson = np.load('D:/Codingan/skripsi/features1.npy')
    label_parkinson = np.load('D:/Codingan/skripsi/label1.npy')

    # Indeks fitur yang ingin diambil 
    selected_features = [0, 2, 4, 6, 12, 13]
    print(fitur_normal.shape)
    # Membuat array baru dengan fitur yang dipilih
    fitur_normal = fitur_normal[:, selected_features]
    fitur_parkinson = fitur_parkinson[:, selected_features]

    # Menggabungkan fitur dan label dari kedua kelompok
    X = np.vstack((fitur_normal, fitur_parkinson))
    y = np.hstack((label_normal, label_parkinson))
    
    return X, y

def build_model(X_train):
    model = Sequential([
        
        Conv1D(128, 3, padding='same', input_shape=(X_train.shape[1], 1)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2, strides=1),
        SpatialDropout1D(0.3),
        
        Conv1D(256, 3,  padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        MaxPooling1D(2, strides=1),
        SpatialDropout1D(0.3),
        
        Conv1D(512, 3,  padding='same'),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        
        GlobalAveragePooling1D(),
        #Flatten(),
        
        Dense(1024, kernel_regularizer=l2(0.02)),
        LeakyReLU(alpha=0.1),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(2, activation='softmax')
    ])

    # Model summary
    model.summary()

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess_and_train(X, y, epochs=250):
    # Reshape dataset_x untuk memasukkan dimensi saluran
    dataset_x = X.reshape((X.shape[0], X.shape[1], 1))
    print("Shape X: :", X.shape)
    # Encode labels ke one-hot encoding jika mereka berupa kategori
    le = LabelEncoder()
    dataset_y_encoded = le.fit_transform(y)
    dataset_y_onehot = to_categorical(dataset_y_encoded)

    # Normalisasi fitur
    dataset_x, scaler = normalize_data(dataset_x)
    
    # Simpan scaler
    joblib.dump(scaler, 'kodingan RQA/FIX/TEST MODELS/TEST_MODEL3.pkl')

    # Data Augmentation
    X_augmented, Y_augmented = augment_data(dataset_x, dataset_y_onehot)

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_augmented, Y_augmented, test_size=0.2, random_state=42, stratify=Y_augmented)
    
    model = build_model(X_train)
    
    # Membuat callback untuk early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint("kodingan RQA/FIX/TEST MODELS/TEST_MODEL3", monitor='val_accuracy', save_best_only=True)

    # Latih model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=[early_stopping, reduce_lr, model_checkpoint])
    return model, history


# Load data
X, y = load_data()

# Train model
trained_model, history = preprocess_and_train(X, y)

# Onehot encoding
y_onehot = to_categorical(LabelEncoder().fit_transform(y))

# Normalize data (X_scaled only, not returning scaler)
X_scaled, _ = normalize_data(X.reshape((X.shape[0], X.shape[1], 1)))

# Evaluate model on entire dataset
plot_history(history)
evaluation = trained_model.evaluate(X_scaled, y_onehot)
print(f"Final Loss: {evaluation[0]}")
print(f"Final Accuracy: {evaluation[1]}")

# Load and evaluate the best model
best_model = load_model("kodingan RQA/FIX/TEST MODELS/TEST_MODEL3")
best_evaluation = best_model.evaluate(X_scaled, y_onehot)
print(f"Best Model Final Loss: {best_evaluation[0]}")
print(f"Best Model Final Accuracy: {best_evaluation[1]}")
