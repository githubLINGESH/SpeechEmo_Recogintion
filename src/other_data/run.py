import os
from feature_extraction import extract_audio_features
from normalize import normalize_features
from cluster import perform_clustering
from cluster_analysis import analyze_clusters
import numpy as np

data_root = 'D:/voice-emo/Indian_Languages_Audio_Dataset/'

def list_wav_files(root_dir):
    audio_files = []
    # Traverse through all subdirectories and collect WAV files
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)
    return audio_files


# List all WAV files in the directory structure
audio_files = list_wav_files(data_root)

# Print the list of found WAV files
for file_path in audio_files:
    print(file_path)


# Extract features for each audio file
features_list = []
for file_path in audio_files:
    features = extract_audio_features(file_path)
    if features is not None:
        features_list.append(features)

# Create a feature matrix (2D array)
if features_list:
    features_matrix = np.vstack(features_list)
else:
    print("No valid features extracted. Check your audio files and feature extraction process.")
    exit(1)  # Exit or handle the case when no valid features are extracted

# Normalize features
normalized_features = normalize_features(features_matrix)


# Create a feature matrix (2D array)
features_matrix = np.vstack(features_list) 

# Normalize features
normalized_features = normalize_features(features_matrix)

# Perform clustering
num_clusters = 3  # Example: Number of clusters
cluster_labels = perform_clustering(normalized_features, num_clusters)

# Analyze clusters
clusters = analyze_clusters(audio_files, cluster_labels)

def classify_emotion(cluster_features):
    tone_mean = np.mean(cluster_features[:, 0])  # Assuming tone feature is at index 0
    intensity_mean = np.mean(cluster_features[:, 1])  # Assuming intensity feature is at index 1
    pitch_mean = np.mean(cluster_features[:, 2])  # Assuming pitch feature is at index 2
    volume_mean = np.mean(cluster_features[:, 3])  # Assuming volume feature is at index 3
    
    if tone_mean > 0.5 and intensity_mean > 0.5 and pitch_mean > 0.5 and volume_mean > 0.5:
        return "Angry"
    elif tone_mean < 0.5 and intensity_mean < 0.5 and pitch_mean < 0.5 and volume_mean < 0.5:
        return "Sad"
    elif tone_mean > 0.3 and intensity_mean > 0.3 and pitch_mean > 0.3 and volume_mean > 0.6:
        return "Happy"
    elif tone_mean < 0.4 and intensity_mean < 0.4 and pitch_mean < 0.4 and volume_mean > 0.4:
        return "Calm"
    elif tone_mean > 0.5 and intensity_mean > 0.5 and pitch_mean > 0.5 and volume_mean > 0.5:
        return "Excited"
    elif tone_mean < 0.4 and intensity_mean < 0.4 and pitch_mean < 0.4 and volume_mean < 0.4:
        return "Relaxed"
    else:
        return "Surprised"

# Assuming `cluster_features` is a 2D array where each row represents features of a cluster
# `cluster_features` should have the shape (num_clusters, num_features)

# Example usage to classify clusters
for cluster_id in clusters:
    cluster_data = np.array([normalized_features[i] for i, label in enumerate(cluster_labels) if label == cluster_id])
    if len(cluster_data) > 0:
        emotion_label = classify_emotion(cluster_data)
        print(f"Cluster {cluster_id}: Emotion - {emotion_label}")
    else:
        print(f"Cluster {cluster_id}: No data points")


