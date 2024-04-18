import os
import shutil

# Define the paths to your datasets and the target folder for separated emotions
data_roots = [
    'D:/voice-emo/data/TESS Toronto emotional speech set data',
    'D:/voice-emo/data/AudioWav',
    'D:/voice-emo/data/ALL'
]
target_root = 'D:/voice-emo/data_sep'

# Emotion mapping from file naming conventions to folder names
emotion_mapping = {
    'angry': ['angry', 'ANG'],
    'disgust': ['disgust', 'DIS'],
    'fear': ['fear', 'FEA', 'Fear'],
    'happy': ['happy', 'HAP'],
    'neutral': ['neutral', 'NEU'],
    'sad': ['sad', 'SAD', 'Sad'],
    'surprise': ['surprise', 'SUR'],
    'pleasant_suprised': ['pleasant_suprised', 'Pleasant_suprise']
}

# Additional single-letter identifiers for the '/ALL' dataset
all_dataset_identifiers = {
    'angry': ['a'],
    'disgust': ['d'],
    'fear': ['f'],
    'happy': ['h'],
    'neutral': ['n'],
    'sad': ['s'],
    'surprise': ['su']  # Assuming 'su' for surprise
}


# Function to find the emotion category based on the file name and dataset
def find_emotion_category(file_name, data_root):
    if 'ALL' in data_root:
        for emotion, ids in all_dataset_identifiers.items():
            if any(id.lower() in file_name.lower() for id in ids):
                return emotion
    for emotion, ids in emotion_mapping.items():
        if any(id.lower() in file_name.lower() for id in ids):
            return emotion
    return None

# Create target directories if they don't exist
for emotion in emotion_mapping.keys():
    os.makedirs(os.path.join(target_root, emotion), exist_ok=True)

# Iterate through each dataset and move files to the respective emotion folder
for data_root in data_roots:
    for root, dirs, files in os.walk(data_root):
        for file in files:
            emotion_category = find_emotion_category(file, data_root)
            if emotion_category:
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_root, emotion_category, file)
                shutil.move(source_path, target_path)
                print(f"Moved: {file} to {emotion_category} folder.")
