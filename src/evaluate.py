import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from preprocessing import load_and_preprocess_audio
from feature_extraction import extract_features
import joblib
import re

# Load the SVM model, scaler, and label encoder
svm_model = joblib.load('emotion_svm_model.pkl')
scaler = joblib.load('emotion_scaler_svm.pkl')
label_encoder = joblib.load('emotion_label_encoder_svm.pkl')


def parse_features(feature_str):
    numbers_str = re.findall(r"[-+]?\d*\.\d+|\d+", feature_str)
    numbers_float = [float(num) for num in numbers_str]
    fixed_length = 20800
    padded_array = np.zeros(fixed_length)
    padded_array[:len(numbers_float)] = numbers_float[:fixed_length]
    return padded_array

def predict_emotion(audio_path, model, scaler, label_encoder):
    audio, sr = load_and_preprocess_audio(audio_path)
    features = extract_features(audio, sr)
    scaled_features = scaler.transform([features])
    predicted_label_index = model.predict(scaled_features)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    return predicted_label

def process_and_predict_test_files(test_data_folder, model, scaler, label_encoder):
    y_true = []
    y_pred = []
    for emotion_folder in os.listdir(test_data_folder):
        emotion_path = os.path.join(test_data_folder, emotion_folder)
        for file_name in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file_name)
            true_label = emotion_folder
            predicted_label = predict_emotion(file_path, model, scaler, label_encoder)
            y_true.append(true_label)
            y_pred.append(predicted_label)
    return y_true, y_pred

def evaluate_predictions(y_true, y_pred):
    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(y_true)
    y_pred_encoded = label_encoder.transform(y_pred)
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    print(classification_report(y_true_encoded, y_pred_encoded,
                                target_names=label_encoder.classes_))

# Evaluate the model
test_data_folder = 'D:/voice-emo/data_sep'
y_true, y_pred = process_and_predict_test_files(test_data_folder, svm_model, scaler, label_encoder)
evaluate_predictions(y_true, y_pred)
