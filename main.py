import base64
import io
import librosa
import torch
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
app = FastAPI()
model_name = "facebook/wav2vec2-base-960h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
class AudioInput(BaseModel):
    audioFormat: str
    audioBase64: str
def verify_key(api_key: str = Header(None)):
    if x_api_key != "SynxsOG": 
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

@app.post("/api/voice-detection")
async def detect_voice(data: AudioInput, key: str = Depends(verify_key)):
    try:
        SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        if data.language not in SUPPORTED_LANGUAGES:
            return {"status": "error", "message": "Unsupported language"}  
        clean_base64 = data.audioBase64.strip().split(',')[-1]
        audio_bytes = base64.b64decode(clean_base64)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence_score = torch.max(probs).item() 
        if confidence_score > 0.9: 
            prediction = "AI_GENERATED"
        else:
            prediction = "HUMAN"
        return {
            "status": "success",
            "result": prediction,
            "language_detected": data.language,
            "confidence": round(confidence_score, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")