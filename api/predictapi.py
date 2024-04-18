from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from speechbrain.inference.interfaces import foreign_class
import boto3

app = Flask(__name__)

# Assuming you have AWS credentials set up properly in your environment
s3_client = boto3.client('s3')
S3_BUCKET_NAME = 'audioemo'

# Specify a directory to temporarily save uploaded files
UPLOAD_FOLDER = 'temp_audio_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/classify-emotion', methods=['POST'])
def classify_emotion():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Securely save the uploaded file locally
    filename = secure_filename(file.filename)
    local_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(local_file_path)
    
    try:
        # Initialize your classifier here
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        
        # Classify the emotion of the uploaded file
        out_prob, score, index, text_lab = classifier.classify_file(local_file_path)
        
        # Optionally, upload processed file to S3 or do additional processing here
        
        # Clean up: remove the file after processing
        os.remove(local_file_path)
        
        return jsonify({'emotion': text_lab})
    except Exception as e:
        # If something goes wrong, remove the uploaded file and return an error
        os.remove(local_file_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)


