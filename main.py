from fastapi import FastAPI, status
from fastapi import FastAPI, File, UploadFile
from pydub import AudioSegment
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
import io

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

app = FastAPI(title="REST API using FastAPI PostgreSQL Async EndPoints")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def extract_features(file_path):
        try:
            audio, sample_rate = librosa.librosa.load(file_path, res_type='kaiser_fast')             
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            return mfccs
        except Exception as e:
            print("Error encountered while parsing file:", file_path)
            return None

@app.post("/process_voice/")
async def process_voice(upload_file: UploadFile = File(...)):
    # Save the uploaded voice sample to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_filename = temp_file.name
        temp_file.write(await upload_file.read())

        
  
    audio1 = AudioSegment.from_wav(temp_filename)
    duration = len(audio1) / 1000  # Duration in seconds

    audio2, sample_rate = librosa.load(temp_filename) 
    mfccs = np.mean(librosa.feature.mfcc(y=audio2, sr=audio1.frame_rate, n_mfcc=40).T, axis=0)
    loaded_model = joblib.load("models/random_forest_model.joblib")
    
    if mfccs is not None:
          prediction = loaded_model.predict([mfccs])
          if prediction[0] == 1 :
             isHumanVoice= "true"
             isFabicatedVoice= "false"
          else :
             isHumanVoice= "false"
             isFabicatedVoice= "true"
               
    else:
          return "Error extracting features from the example file."
    
    # Generate JSON response
    response = {
        "isHumanVoice": isHumanVoice,
        "isFabicatedVoice":isFabicatedVoice,
        "duration_seconds": duration,
        "sample_rate": audio1.frame_rate,
        "channels": audio1.channels,
        "file_size_bytes": os.path.getsize(temp_filename)
    }

    # Clean up the temporary file
    os.unlink(temp_filename)

    return response