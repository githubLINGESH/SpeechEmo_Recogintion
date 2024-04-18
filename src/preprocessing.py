import os
import librosa
import numpy as np
import pandas as pd
from feature_extraction import extract_features


# Function to load the dataset file paths and their respective labels
def load_dataset(data_root):
    dataset = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.lower().endswith('.wav'):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                # The label is the directory name
                dataset.append((file_path, label))
    return dataset


# Function to load and preprocess an audio file
def load_and_preprocess_audio(file_path, target_sr=22050, target_dBFS=-20,top_db=20):
    audio, sr = librosa.load(file_path, sr=target_sr)
    audio = normalize_volume(audio, target_dBFS)
    audio = trim_silence(audio, top_db)
    return audio, sr


# Function to normalize the volume of an audio file
def normalize_volume(audio, target_dBFS):
    mean_volume = np.mean(librosa.core.amplitude_to_db(audio))
    return librosa.db_to_amplitude(target_dBFS - mean_volume) * audio


# Function to trim silence from an audio file
def trim_silence(audio, top_db):
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed_audio


def main():
    # The path to the folder where the separated emotion folders are located
    data_root = 'D:/voice-emo/dat/'

    all_features = []
    all_labels = []

    # Iterate over each emotion folder
    for emotion in os.listdir(data_root):
        emotion_folder = os.path.join(data_root, emotion)
        print(f"Processing emotion: {emotion}")

        # Process each file within the emotion folder
        for file_name in os.listdir(emotion_folder):
            file_path = os.path.join(emotion_folder, file_name)
            try:
                # Load and preprocess the audio
                audio, sr = load_and_preprocess_audio(file_path)
                # Extract features
                features = extract_features(audio, sr)
                # Convert the features to a properly formatted string
                # that looks like a list
                
                features_str = str(list(features)).replace(' ', '')
                # Append the features and the emotion as the label
                all_features.append(features_str)
                all_labels.append(emotion)
                print(f"Processed {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Save the features and labels to a CSV
    df = pd.DataFrame({'Features': all_features, 'Label': all_labels})
    df.to_csv('f_processed_features.csv', index=False)


if __name__ == '__main__':
    main()
