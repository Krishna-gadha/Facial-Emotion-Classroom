"""
STEP 3 — REAL-TIME INFERENCE (FIXED)
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime

# ── Load emotion model ──
MODEL_PATH = "../models/emotion_model_final.h5"
IMG_SIZE   = 96
EMOTIONS   = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
ENGAGEMENT_SCORE = {
    'happy': 0.9, 'surprise': 0.8, 'neutral': 0.6,
    'sad': 0.3, 'fear': 0.3, 'angry': 0.2, 'disgust': 0.1
}

print("⏳ Loading emotion model...")
from tensorflow.keras.models import load_model
emotion_model = load_model(MODEL_PATH)
print("✅ Model loaded!")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

os.makedirs("../data/logs", exist_ok=True)


def predict_emotion(face_roi):
    try:
        face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        preds = emotion_model.predict(face, verbose=0)[0]
        return EMOTIONS[np.argmax(preds)], float(np.max(preds))
    except Exception as e:
        print(f"Predict error: {e}")
        return "unknown", 0.0


COLORS = {
    'happy':    (0, 255, 100),
    'surprise': (0, 200, 255),
    'neutral':  (200, 200, 200),
    'sad':      (255, 100, 100),
    'fear':     (150, 0, 255),
    'angry':    (0, 0, 255),
    'disgust':  (0, 140, 255),
    'unknown':  (128, 128, 128)
}

print("🎥 Opening webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("✅ Webcam open! Press 'q' to quit.")
start_time = time.time()
session_log = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame.")
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        face_roi          = frame[y:y+h, x:x+w]
        emotion, conf     = predict_emotion(face_roi)
        color             = COLORS.get(emotion, (255, 255, 255))
        engagement        = ENGAGEMENT_SCORE.get(emotion, 0.5)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{emotion} {conf:.0%}",
                    (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"engage: {engagement:.0%}",
                    (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        session_log.append({
            "time":       round(time.time() - start_time, 2),
            "emotion":    emotion,
            "confidence": round(conf, 3),
            "engagement": engagement
        })

    elapsed = round(time.time() - start_time, 1)
    cv2.putText(frame, f"Session: {elapsed}s | Faces: {len(faces)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Classroom Emotion Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save session log
if session_log:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"../data/logs/session_{ts}.json"
    with open(path, 'w') as f:
        json.dump(session_log, f, indent=2)
    print(f"💾 Session saved to {path}")

print("✅ Done!")