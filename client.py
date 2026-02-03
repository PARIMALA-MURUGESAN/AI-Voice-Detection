import requests
import base64

# 1. Your API Details
URL = "https://parimala007-voice-brain07.hf.space/api/voice-detection"
API_KEY = "YOUR_SECRET_API_KEY"

def check_voice(file_path, lang="English"):
    # 2. AUTOMATICALLY convert file to Base64
    with open(file_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')

    # 3. Create JSON
    payload = {
        "language": lang,
        "audioFormat": "mp3",
        "audioBase64": encoded_string
    }
    
    headers = {"x-api-key": API_KEY}

    # 4. Send to Cloud
    print(f"Analyzing {file_path}...")
    response = requests.post(URL, json=payload, headers=headers)
    
    # 5. Show Result
    print(response.json())

# Just run this one line!
check_voice("my_recording.mp3", lang="Tamil")