import librosa
import numpy as np

def extract_audio_features(file_path):
    try:
        audio, sr = librosa.load(file_path)
        
        # Extract MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Calculate mean along each MFCC coefficient to get a 1D feature vector
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]  # Extract only the first array from the result
        # Calculate mean of spectral centroid to get a scalar feature
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        # Extract zero-crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)[0]  # Extract only the first array from the result
        # Calculate mean of zero-crossing rate to get a scalar feature
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)
        
        # Combine features into a single feature vector
        features = np.concatenate((mfcc_mean, [spectral_centroid_mean, zero_crossing_rate_mean]))
        
        return features
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None
