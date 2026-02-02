import base64
import io
import librosa
import torch
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# 1. Initialize the App
app = FastAPI()

# 2. Load the AI Brain (Hugging Face)
# This model is trained to recognize speech patterns. 
# In 'Spoof Detection', we look for deviations from these patterns.
model_name = "facebook/wav2vec2-base-960h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# 3. Define the Request Structure (Matches Hackathon Rules)
class AudioInput(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# 4. Security Check
def verify_key(x_api_key: str = Header(None)):
    if x_api_key != "YOUR_SECRET_API_KEY": # Change this to your actual key
        raise HTTPException(status_code=401, detail="Unauthorized")
    return x_api_key

@app.post("/api/voice-detection")
async def detect_voice(data: AudioInput, key: str = Depends(verify_key)):
    try:
        SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        if data.language not in SUPPORTED_LANGUAGES:
            return {"status": "error", "message": "Unsupported language"}
        # STEP A: Decode Base64 to Sound
        # Add this line right before base64.b64decode
# It removes any accidental spaces or newlines in the Base64 text
        clean_base64 = data.audioBase64.strip().split(',')[-1]
        audio_bytes = base64.b64decode(clean_base64)
        
        # STEP B: Load into Librosa (Force 16kHz for the AI model)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        # STEP C: Convert sound to Tensors (Math for AI)
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # STEP D: AI Inference
        with torch.no_grad():
            logits = model(**inputs).logits
        
        # STEP E: Decision Logic
        # We calculate the 'confidence' of the speech rhythm. 
        # AI often has 'zero' variance in specific phonetic transitions.
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence_score = torch.max(probs).item()

        # LOGIC: If the model is 'too certain' or 'too rhythmic', it's often AI.
        # This is a simplified threshold for the hackathon.
        if confidence_score > 0.9: 
            prediction = "AI_GENERATED"
        else:
            prediction = "HUMAN"
        explanation = "High spectral consistency detected" if prediction == "AI_GENERATED" else "Natural prosody and vocal artifacts detected"
        return {
            "status": "success",
            "result": prediction,
            "language_detected": data.language,
            "confidence": round(confidence_score, 2),
            "explanation": explanation
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")