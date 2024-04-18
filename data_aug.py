import os
import librosa
import numpy as np
import soundfile as sf

data_root = 'D:/voice-emo/data_sep/'

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def pitch_shift(y, sr, n_steps):
    # Correctly return the shifted audio
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def save_wav_file(augmented_data, sample_rate, file_path):
    sf.write(file_path, augmented_data, sample_rate)

for emotion in os.listdir(data_root):
    emotion_folder = os.path.join(data_root, emotion)
    print(f"Processing emotion: {emotion}")
    
    for file_name in os.listdir(emotion_folder):
        file_path = os.path.join(emotion_folder, file_name)
        data, sample_rate = librosa.load(file_path, sr=None)
        
        # Augmentations
        noise_data = add_noise(data)
        pitched_data = pitch_shift(data, sample_rate, 5)
        stretched_data = time_stretch(data , 0.8)

        # Save augmented files
        augmentations = {'noise': noise_data, 'pitch': pitched_data, 'stretch': stretched_data}
        for aug_type, aug_data in augmentations.items():
            augmented_file_name = f"{os.path.splitext(file_name)[0]}_{aug_type}_augmented{os.path.splitext(file_name)[1]}"
            augmented_file_path = os.path.join(emotion_folder, augmented_file_name)
            save_wav_file(aug_data, sample_rate, augmented_file_path)
            print(f"Augmented file saved: {augmented_file_path}")
