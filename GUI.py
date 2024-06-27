import tkinter as tk
import pyaudio
import wave
import time
import numpy as np
import joblib

from keras.models import load_model
from tkinter import messagebox
from threading import Thread
from sklearn.preprocessing import StandardScaler
from librosa.util import normalize
from pyrqa.settings import Settings
from pyrqa.time_series import TimeSeries
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.opencl import OpenCL

def extract_features(audio_file, sr, hop_length):
    import librosa
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
    if scaler is None:
        scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled.reshape(1, -1, 1), scaler

class ParkinsonDetectionApp:
    
    def __init__(self, master):
        self.master = master
        master.title("Sistem Deteksi Parkinson")
        master.geometry("650x420")
        master.configure(bg="#2C3E50")

        self.page1 = Page1(master, self)
        self.page2 = Page2(master, self)

        self.show_page1()

    def show_page1(self):
        self.page2.hide()
        self.page1.show()

    def show_page2(self, prediction_result, confidence):
        self.page1.hide()
        self.page2.show(prediction_result, confidence)

class Page1:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master, bg="#2C3E50")
        self.frame.pack(expand=True)

        self.label = tk.Label(self.frame, text="Rekam suara anda", font=("Helvetica", 28, "bold"), bg="#2C3E50", fg="#ECF0F1")
        self.record_button = tk.Button(self.frame, text="Mulai Rekam", font=("Helvetica", 20, "bold"), bg="#E74C3C", fg="#ECF0F1", command=self.record_audio, borderwidth=2, relief="raised")

        self.label.pack(pady=20)
        self.record_button.pack(pady=50)

    def show(self):
        self.frame.pack(expand=True)

    def hide(self):
        self.frame.pack_forget()

    def record_audio(self):
        for i in range(3, 0, -1):
            self.label.config(text=f"Perekaman dimulai dalam {i}")
            self.master.update()
            time.sleep(1)

        self.label.config(text="Perekaman suara...")
        self.master.update()

        recording_thread = Thread(target=self.simulate_recording)
        recording_thread.start()

    def simulate_recording(self):
        try:
            duration = 4
            file_name = "recorded_audio.wav"

            p = pyaudio.PyAudio()

            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=44100,
                            input=True,
                            frames_per_buffer=1024)

            frames = []

            for i in range(0, int(44100 / 1024 * duration)):
                data = stream.read(1024)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()

            with wave.open(file_name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(44100)
                wf.writeframes(b''.join(frames))

            self.label.config(text="Rekaman suara anda berhasil")
            self.master.update()

            model_path = "kodingan RQA/FIX/Model_86%"
            scaler_path = 'kodingan RQA/FIX/Scaler_86%.pkl'

            predicted_label, prediction = predict_new_audio(file_name, model_path, scaler_path)
            label_map = {0: "Non-Parkinson", 1: "Parkinson"}
            result = label_map[predicted_label[0]]
            confidence = prediction[0][predicted_label[0]]

            self.app.show_page2(result, confidence)

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan saat merekam audio: {str(e)}")

class Page2:
    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.frame = tk.Frame(master, bg="#2C3E50")
        self.frame.pack(expand=True)

        self.label = tk.Label(self.frame, text="Hasil Prediksi:", font=("Helvetica", 28, "bold"), bg="#2C3E50", fg="#ECF0F1")
        self.prediction_label = tk.Label(self.frame, text="", font=("Helvetica", 24), bg="#2C3E50", fg="#ECF0F1")
        self.confidence_label = tk.Label(self.frame, text="", font=("Helvetica", 20), bg="#2C3E50", fg="#ECF0F1")

        self.back_button = tk.Button(self.frame, text="Back", font=("Helvetica", 20, "bold"), bg="#E74C3C", fg="#ECF0F1", command=app.show_page1, borderwidth=2, relief="raised")

        self.label.pack(pady=20)
        self.prediction_label.pack(pady=10)
        self.confidence_label.pack(pady=10)
        self.back_button.pack(pady=30)

    def show(self, prediction_result, confidence):
        self.prediction_label.config(text=f"{prediction_result}")
        self.confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")
        self.frame.pack(expand=True)

    def hide(self):
        self.frame.pack_forget()

def predict_new_audio(audio_path, model_path, scaler_path):
    # Load model
    model = load_model(model_path)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Ekstraksi fitur dari file audio baru
    features = extract_features(audio_path, 22050, 128)
    
    # Normalisasi data
    features = features.reshape(1, -1)
    features, _ = normalize_data(features, scaler)
    
    # Prediksi
    prediction = model.predict(features)
    
    # Konversi prediksi ke label
    predicted_label = np.argmax(prediction, axis=1)
    
    return predicted_label, prediction

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkinsonDetectionApp(root)
    root.mainloop()
