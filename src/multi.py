# Load model directly
from transformers import AutoProcessor, Wav2Vec2Model

processor = AutoProcessor.from_pretrained("auditi31/wav2vec2-large-xlsr-53-multilingual-audio-emotion")
model = Wav2Vec2Model.from_pretrained("auditi31/wav2vec2-large-xlsr-53-multilingual-audio-emotion")

print(model.predict(["sam.wav"]))


