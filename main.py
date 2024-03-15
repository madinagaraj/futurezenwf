from fastapi import FastAPI, status
from fastapi import FastAPI, File, UploadFile
from pydub import AudioSegment
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="REST API using FastAPI PostgreSQL Async EndPoints")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/process_voice/")
async def process_voice(upload_file: UploadFile = File(...)):
    # Save the uploaded voice sample to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        temp_filename = temp_file.name
        temp_file.write(await upload_file.read())

    # Process the voice sample (Here, we're just simulating processing)
    audio = AudioSegment.from_wav(temp_filename)
    duration = len(audio) / 1000  # Duration in seconds

    # Generate JSON response
    response = {
        "duration_seconds": duration,
        "sample_rate": audio.frame_rate,
        "channels": audio.channels,
        "file_size_bytes": os.path.getsize(temp_filename)
    }

    # Clean up the temporary file
    os.unlink(temp_filename)

    return response