from fastapi import FastAPI, UploadFile, File
import uvicorn
import os
from keras.models import load_model
from audio_detector import detect_fake
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = Path(__file__).resolve().parent / "cnn_full_model_13.h5"
model = load_model(model_path)


@app.get("/")
def home():
    return {"message": "API de detección de audios deepfake"}


@app.post("/upload-audio")
async def uploadAudio(file: UploadFile = File(...)):
    fileLocation = f"temp_{file.filename}"
    with open(fileLocation, "wb") as buffer:
        buffer.write(await file.read())

    result, confidence = detect_fake(fileLocation, model)

    if os.path.exists(fileLocation):
        os.remove(fileLocation)

    return {
        "Nombre de archivo": file.filename,
        "Real o Deepfake": result,
        "Intervalo de confianza": confidence,
    }
