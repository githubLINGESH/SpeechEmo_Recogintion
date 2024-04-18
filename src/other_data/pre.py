import librosa
import joblib
from feature_extraction import extract_features

# Load the KMeans model and the scaler
kmeans = joblib.load('emotion_kmeans_model.pkl')
scaler = joblib.load('emotion_scaler.pkl')

# Function to map cluster numbers to emotion labels
# You will need to create this mapping based on your dataset
# and cluster analysis
cluster_to_emotion = {0: "Angry", 1: "Disgusted", 2: "Fear", 3: "Happy",
                      4: "Neutral",5: "sad",6: "Surprised"}


def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    audio = librosa.effects.trim(audio)[0]  # Trim leading and trailing silence
    return audio, sr


def predict_emotion(audio_path, kmeans_model, scaler, cluster_to_emotion_map):
    # Extract features from the audio file
    
    audio, sr = load_and_preprocess_audio(audio_path)
    # Extract features
    features = extract_features(audio, sr)
    
    # Scale the extracted features
    features_scaled = scaler.transform([features])
    
    # Predict the closest cluster
    predicted_cluster = kmeans_model.predict(features_scaled)[0]
    print(predicted_cluster)
    
    # Map the cluster to an emotion label
    predicted_emotion = cluster_to_emotion_map.get(predicted_cluster, "Unknown")
    
    return predicted_emotion


# Example usage
audio_path = 'D:/voice-emo/src/samples/sadd.wav'
predicted_emotion = predict_emotion(audio_path, kmeans, scaler,
                                    cluster_to_emotion)
print(f"The predicted emotion is: {predicted_emotion}")
