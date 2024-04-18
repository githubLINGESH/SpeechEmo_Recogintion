import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from pygame import mixer
import librosa
import numpy as np
from keras.models import load_model
import joblib


mixer.init()

# Load your saved models
scaler = joblib.load('D:/voice-emo/models/D_scaler.pkl')
pca = joblib.load('D:/voice-emo/models/D_pca.pkl')
model = load_model('D:/voice-emo/models/D_emotion-model.h5')
emotion_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise', 'pleasant_suprised']

# Initialize the main window
root = tk.Tk()
root.title('Emotion Detector')
root.geometry('500x350')

# Variable to hold the selected file name
file_name = ''


# Function to load and preprocess the audio file
def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    audio = librosa.effects.trim(audio)[0]  # Trim leading and trailing silence
    return audio, sr


def extract_features(audio, sr, n_mfcc=13, n_chroma=12, max_pad_length=400, fmin=200):
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma)
    
    # Extract Pitch (using librosa's piptrack)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=fmin)
    pitch_mean = np.mean(pitches, axis=0)  # Taking the mean pitch over time
    pitch_mean_padded = np.pad(pitch_mean, (0, max_pad_length - pitch_mean.shape[0]), mode='constant')
    
    # Determine the maximum length among the features
    max_length = max(mfccs.shape[1], chroma.shape[1], pitch_mean.shape[0], max_pad_length)
    
    # Pad the features
    mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    chroma_padded = np.pad(chroma, pad_width=((0, 0), (0, max_length - chroma.shape[1])), mode='constant')
    
    # Flatten and concatenate the features
    features_flattened = np.concatenate((
        mfccs_padded.flatten(),
        chroma_padded.flatten(),
        pitch_mean_padded.flatten()
    ))
    
    return features_flattened


# Function to predict the emotion of an audio file
def predict_emotion(file_path):
    audio, sr = load_and_preprocess_audio(file_path)
    features = extract_features(audio, sr)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    prediction = model.predict(features_pca)
    predicted_class = np.argmax(prediction)
    return emotion_names[predicted_class]


# Function to handle file selection
def select_file():
    global file_name
    file_name = filedialog.askopenfilename(initialdir='/', title='Select an Audio File',
                                           filetypes=(("WAV files", "*.wav"), ("All files", "*.*")))
    status_label.config(text="File selected: " + file_name)


# Function to handle the prediction
def make_prediction():
    if not file_name:
        messagebox.showerror("Error", "Please select a file first")
        return
    
    predicted_emotion = predict_emotion(file_name)
    prediction_label.config(text="Predicted emotion: " + predicted_emotion)


# Setup the GUI layout
status_label = tk.Label(root, text="Select an audio file", relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(fill=tk.X, side=tk.BOTTOM, ipady=2)

prediction_label = tk.Label(root, text="Emotion will be displayed here", font=('bold', 14))
prediction_label.pack(pady=20)

select_button = tk.Button(root, text="Select File", command=select_file)
select_button.pack()

predict_button = tk.Button(root, text="Predict Emotion", command=make_prediction)
predict_button.pack()

# Run the GUI event loop
root.mainloop()
