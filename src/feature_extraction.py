import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_features(audio, sr, n_mfcc=13, n_chroma=12, max_pad_length=400, fmin=200):
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_delta = librosa.feature.delta(mfccs)
    mfccs_delta2 = librosa.feature.delta(mfccs, order=2)
    
    # Extract Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=n_chroma)
    
    # Extract Pitch (using librosa's piptrack)
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=fmin)
    pitch_mean = np.mean(pitches, axis=0)  # Taking the mean pitch over time
    
    # Normalization (example for MFCCs, extend this to other features as needed)
    mfccs = StandardScaler().fit_transform(mfccs.T).T  # Normalize MFCCs
    mfccs_delta = StandardScaler().fit_transform(mfccs_delta.T).T  # Normalize deltas
    mfccs_delta2 = StandardScaler().fit_transform(mfccs_delta2.T).T  # Normalize delta-deltas
    
    # Pad features
    max_length = max(mfccs.shape[1], chroma.shape[1], pitch_mean.shape[0], max_pad_length)
    mfccs_padded = np.pad(mfccs, pad_width=((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
    mfccs_delta_padded = np.pad(mfccs_delta, pad_width=((0, 0), (0, max_length - mfccs_delta.shape[1])), mode='constant')
    mfccs_delta2_padded = np.pad(mfccs_delta2, pad_width=((0, 0), (0, max_length - mfccs_delta2.shape[1])), mode='constant')
    chroma_padded = np.pad(chroma, pad_width=((0, 0), (0, max_length - chroma.shape[1])), mode='constant')
    pitch_mean_padded = np.pad(pitch_mean, (0, max_length - pitch_mean.shape[0]), mode='constant')
    
    # Concatenate features
    features = np.concatenate((
        mfccs_padded.flatten(),
        mfccs_delta_padded.flatten(),
        mfccs_delta2_padded.flatten(),
        chroma_padded.flatten(),
        pitch_mean_padded.flatten()
    ))

    return features
