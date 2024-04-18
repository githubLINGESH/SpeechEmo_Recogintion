import numpy as np
from dotenv import load_dotenv
from pyht import Client
from pyht.client import TTSOptions
import pyaudio

load_dotenv()

# Initialize the Play.ht client
client = Client(
    user_id="nTx2MoNL1tXcxYbuGbuzwg0vNNw1",
    api_key="a2d236fb9d6f4f99ab6845003bb2d0fa"
)

# Define the text to be converted to speech
text = "Can you tell me your account email or, ah your phone number?"

# Configure TTS options with voice details
options = TTSOptions(
    voice="s3://voice-cloning-zero-shot/6ce020f3-2c42-4e5f-9e71-9579d2e5abf4/enhanced/manifest.json",
)

# Initialize an in-memory byte stream to store the audio data
audio_stream = bytearray()

# Generate speech from text and collect the audio data in the byte stream
for chunk in client.tts(text, options):
    audio_stream.extend(chunk)

# Use PyAudio to play the audio stream in real-time
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,  # 16-bit audio
    channels=1,                         # Mono audio
    rate=20000,                         # Sample rate of 44100 Hz
    output=True,
    
)

# Play the audio stream
stream.write(np.array(audio_stream))

# Close the stream and terminate PyAudio
stream.stop_stream()
stream.close()
p.terminate()

print("Speech synthesized successfully! Audio played.")
