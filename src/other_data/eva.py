import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Assuming feature_extraction.py and preprocessing.py are correctly set up
from feature_extraction import extract_features
from preprocessing import load_and_preprocess_audio

from sklearn.preprocessing import LabelEncoder
import numpy as np

class CustomLabelEncoder(LabelEncoder):
    def __init__(self):
        super().__init__()
        self.unknown_label = 'Unknown'

    def fit(self, y):
        super().fit(y)
        if self.unknown_label not in self.classes_:
            self.classes_ = np.append(self.classes_, self.unknown_label)

    def transform(self, y):
        y = np.asarray(y)
        unseen_labels = ~np.isin(y, self.classes_)
        y[unseen_labels] = self.unknown_label  # Assign 'Unknown' label to unseen values
        return super().transform(y)



# Load the KMeans model and scaler
kmeans = joblib.load('emotion_kmeans_model.pkl')
scaler = joblib.load('emotion_scaler.pkl')

# Function to map cluster numbers to emotion labels
# Make sure this matches with your dataset and KMeans model
cluster_to_emotion_map = {0: "Angry", 1: "Sad", 2: "Disgusted", 3: "Neutral", 4: "Happy", 5: "Fear", 6: "Surprised"}

def predict_emotion(audio_path, kmeans_model, scaler, cluster_to_emotion_map):
    audio, sr = load_and_preprocess_audio(audio_path)
    features = extract_features(audio, sr)
    features_scaled = scaler.transform([features])
    predicted_cluster = kmeans_model.predict(features_scaled)[0]
    predicted_emotion = cluster_to_emotion_map.get(predicted_cluster, "Unknown")
    return predicted_emotion

def process_and_predict_test_files(test_data_folder, kmeans_model, scaler, cluster_to_emotion_map):
    y_true = []
    y_pred = []
    for emotion_folder in os.listdir(test_data_folder):
        emotion_path = os.path.join(test_data_folder, emotion_folder)
        for file_name in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file_name)
            true_label = emotion_folder
            predicted_emotion = predict_emotion(file_path, kmeans_model, scaler, cluster_to_emotion_map)
            y_true.append(true_label)
            y_pred.append(predicted_emotion)
    return y_true, y_pred

def evaluate_predictions(y_true, y_pred):
    label_encoder = CustomLabelEncoder()
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
test_data_folder = 'D:/voice-emo/dat'
y_true, y_pred = process_and_predict_test_files(test_data_folder, kmeans, scaler, cluster_to_emotion_map)
evaluate_predictions(y_true, y_pred)
