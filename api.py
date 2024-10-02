from torch import nn
import torch
import soundfile as sf
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Model
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from huggingface_hub import hf_hub_download


app = Flask(__name__)

UPLOAD_FOLDER = 'temp_audio_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)


# Define the FineTunedWav2Vec2Model class
class FineTunedWav2Vec2Model(nn.Module):
    def __init__(self, wav2vec2_model, output_size):
        super(FineTunedWav2Vec2Model, self).__init__()
        self.wav2vec2 = wav2vec2_model
        self.fc = nn.Linear(self.wav2vec2.config.hidden_size, output_size)

    def forward(self, x):
        # Convert parameters to Double data type
        self.wav2vec2 = self.wav2vec2.double()
        self.fc = self.fc.double()

        out = self.wav2vec2(x.double()).last_hidden_state
        out = self.fc(out[:, 0, :])  # Taking the first output token
        return out


# Function to preprocess audio
def preprocess_audio(audio_file):
    waveform, _ = sf.read(audio_file)
    resampler = Resample(orig_freq=16000, new_freq=16000)
    waveform = resampler(torch.Tensor(waveform)).numpy()
    return waveform

model_path = hf_hub_download(repo_id="Lingeshg/SpeechEmotionDetector", filename="model.pth")
    
model = FineTunedWav2Vec2Model(wav2vec2_model, 7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Function to perform inference
def predict(audio_file):

    # Preprocess the audio file
    waveform = preprocess_audio(audio_file)
    waveform = torch.tensor(waveform, dtype=torch.float64).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(waveform)

    predicted_label = torch.argmax(output, dim=1).item()
    return predicted_label


@app.route('/classify-emotion', methods=['POST'])
def classify_emotion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    local_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(local_file_path)
    
    try:
        # Perform prediction
        predicted_label = predict(local_file_path)  # Change here
        
        # Return the prediction result
        return jsonify({'emotion': predicted_label})
    except Exception as e:
        print(f"Error during voice cloning: {e}")  # Log the error message
        import traceback
        print(traceback.format_exc())  # Log the full traceback
        os.remove(local_file_path)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=8001)
