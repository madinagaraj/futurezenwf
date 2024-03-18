from fastapi import FastAPI, status
from fastapi import FastAPI, File, UploadFile
from pydub import AudioSegment
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
import time


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
        
@app.post("/voice/analyse/ping")
async def healthcheck():
     respose ={
          "Health Check Sucessful"
     }
     return respose

@app.post("/voice/analyse")
async def process_voice(sample: UploadFile = File(...)):
   
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_filename = temp_file.name
        temp_file.write(await sample.read())

    start_time = time.time()    
    status="false"
    detectedVoice="false"
    emotionalTone="unableToDetect"
    backgroundNoiseLevel="unableToDetect"
    audio1 = AudioSegment.from_wav(temp_filename)
    duration = len(audio1) / 1000  # Duration in seconds

    audio2, sample_rate = librosa.load(temp_filename) 
    mfccs = np.mean(librosa.feature.mfcc(y=audio2, sr=audio1.frame_rate, n_mfcc=40).T, axis=0)
    loaded_model = joblib.load("models/random_forest_model.joblib")
    
    if mfccs is not None:
          prediction = loaded_model.predict([mfccs])
          status="success"
          detectedVoice="true"
          if prediction[0] == 1 :
              voiceType = "Human"
              aiProbability = "0"
              humanProbability= "100"

          else :
             voiceType = "Artificail"
             aiProbability = "100"
             humanProbability= "0" 
                        
    else:
          return "Error extracting features from the example file."
    
    responseTime= time.time()- start_time 
    
   
    response = {
         "status": status,              
          "analysis":{
                     "detectedVoice": detectedVoice,
                     "voiceType" : voiceType,
                     "confidenceScore": {
                          "aiProbability": aiProbability,
                          "humanProbability": humanProbability
                                        },
                     
                     "additionalInfo": {
                     "emotionalTone": emotionalTone,
                     "backgroundNoiseLevel": backgroundNoiseLevel
                            }
                     },
          "responseTime": responseTime
 
                }
   

    # Clean up the temporary file
    os.unlink(temp_filename)

    return response