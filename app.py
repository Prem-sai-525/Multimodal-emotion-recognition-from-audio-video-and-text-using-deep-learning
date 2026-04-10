import os
import cv2
import torch
import librosa
import numpy as np
import tensorflow as tf
import speech_recognition as sr
import webrtcvad
import uuid
from flask import Flask, render_template, request, jsonify
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor
)
from moviepy import VideoFileClip

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJI = {
    'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 
    'Happy': '😄', 'Sad': '😢', 'Surprise': '😲', 'Neutral': '😐'
}
RMS_THRESHOLD = 0.005  
CONF_THRESHOLD = 0.40  

# Load Models
face_model = tf.keras.models.load_model('video_mobilenetv2.h5')
text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
text_model.load_state_dict(torch.load('emotion_model_bert.pth', map_location=device))
text_model.to(device).eval()

audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=7)
audio_model.load_state_dict(torch.load('audio_model.pth', map_location=device))
audio_model.to(device).eval()

def align_probs(probs):
    return np.array([probs[0], probs[1], probs[2], probs[3], probs[5], probs[6], probs[4]])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('video')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
        
    unique_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.webm")
    file.save(video_path)
    
    analysis_logs = []
    modality_probs = []
    video_clip = None

    try:
        # --- BRANCH 1: VISION (FACE) ---
        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 6)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                roi = cv2.resize(frame[y:y+h, x:x+w], (224, 224)) / 255.0
                f_raw = face_model.predict(np.expand_dims(roi, axis=0), verbose=0)[0]
                f_probs = align_probs(f_raw)
                
                if np.max(f_probs) > CONF_THRESHOLD:
                    label = EMOTION_LABELS[np.argmax(f_probs)]
                    modality_probs.append(f_probs)
                    analysis_logs.append({
                        "mode": "FACE", 
                        "label": f"{label} (Frame {frame_idx})", 
                        "emoji": EMOTION_EMOJI[label]
                    })
                    break 
        cap.release()

        # --- BRANCH 2: AUDIO & TEXT ---
        video_clip = VideoFileClip(video_path)
        temp_audio = f"temp_{unique_id}.wav"
        
        if video_clip.audio:
            video_clip.audio.write_audiofile(temp_audio, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
            y, _ = librosa.load(temp_audio, sr=16000)
            
            # VAD Pre-check
            audio_int16 = (y * 32767).astype(np.int16).tobytes()
            vad = webrtcvad.Vad(1)
            
            is_speech_present = False
            scan_limit = min(len(audio_int16), 16000 * 2) # Scan first 1s
            for i in range(0, scan_limit, 960): # 30ms frames
                frame = audio_int16[i : i + 960]
                if len(frame) < 960: break
                if vad.is_speech(frame, 16000):
                    is_speech_present = True
                    break

            # ONLY PROCEED IF VAD DETECTS SOMETHING
            if is_speech_present:
                r = sr.Recognizer()
                text_detected = False
                with sr.AudioFile(temp_audio) as src:
                    try:
                        audio_data = r.record(src)
                        text_ip = r.recognize_google(audio_data)
                        
                        # --- 1. TEXT BRANCH ---
                        inputs = text_tokenizer(text_ip, return_tensors="pt", padding=True, truncation=True).to(device)
                        with torch.no_grad():
                            t_raw = torch.softmax(text_model(**inputs).logits, dim=1).cpu().numpy()[0]
                            t_probs = align_probs(t_raw)
                            modality_probs.append(t_probs)
                            label = EMOTION_LABELS[np.argmax(t_probs)]
                            analysis_logs.append({
                                "mode": "TEXT", "transcription": text_ip, 
                                "label": label, "emoji": EMOTION_EMOJI[label]
                            })
                            text_detected = True

                        # --- 2. AUDIO BRANCH (Runs ONLY if Text worked) ---
                        if text_detected:
                            a_inputs = audio_extractor(y, sampling_rate=16000, return_tensors="pt").to(device)
                            with torch.no_grad():
                                a_raw = torch.softmax(audio_model(a_inputs.input_values).logits, dim=1).cpu().numpy()[0]
                                a_probs = align_probs(a_raw)
                                modality_probs.append(a_probs)
                                label = EMOTION_LABELS[np.argmax(a_probs)]
                                analysis_logs.append({
                                    "mode": "AUDIO", "label": label, "emoji": EMOTION_EMOJI[label]
                                })

                    except (sr.UnknownValueError, sr.RequestError):
                        analysis_logs.append({"mode": "TEXT", "label": "No text detected", "emoji": "😶"})
                        analysis_logs.append({"mode": "AUDIO", "label": "Skipped (No speech found)", "emoji": "🔇"})
            else:
                analysis_logs.append({"mode": "AUDIO", "label": "Silence filtered", "emoji": "🔇"})
            
            if os.path.exists(temp_audio): os.remove(temp_audio)

        # --- FINAL FUSION ---
        if not modality_probs:
            final_res = "NOTHING TO RECOGNIZE"
            final_emoji = "❓"
        else:
            final_idx = np.argmax(np.mean(modality_probs, axis=0))
            final_label = EMOTION_LABELS[final_idx]
            final_res = final_label.upper()
            final_emoji = EMOTION_EMOJI[final_label]

        return jsonify({"final": final_res, "emoji": final_emoji, "logs": analysis_logs})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if video_clip: video_clip.close()
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except: pass

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)