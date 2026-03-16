"""
STEP 4 — FLASK BACKEND
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import json
import os
import glob
from datetime import datetime
import tf_keras
from tf_keras.models import load_model
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

MODEL_PATH = "../models/emotion_model_v2.keras"
REGISTERED_FACES = "../data/registered_faces"
LOGS_DIR         = "../data/logs"
IMG_SIZE         = 96

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
ENGAGEMENT_SCORE = {
    'happy': 0.9, 'surprise': 0.8, 'neutral': 0.6,
    'sad': 0.3, 'fear': 0.3, 'angry': 0.2, 'disgust': 0.1
}

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization

def build_model():
    base = MobileNetV2(input_shape=(96,96,3), include_top=False, weights=None)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(7, activation='softmax')(x)
    return Model(inputs=base.input, outputs=out)

print("⏳ Loading model...")
emotion_model = build_model()
emotion_model.load_weights('../models/emotion_weights.weights.h5')
print("✅ Ready.")


current_session = {"active": False, "data": {}}
frame_num = 0


def predict_emotion(face_roi):
    face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    preds = emotion_model.predict(face, verbose=0)[0]
    return EMOTIONS[np.argmax(preds)], float(np.max(preds))


def identify_student(face_roi):
    try:
        temp = "../data/temp_face.jpg"
        cv2.imwrite(temp, face_roi)
        result = DeepFace.find(temp, db_path=REGISTERED_FACES,
                               enforce_detection=False, silent=True)
        if result and len(result[0]) > 0:
            match = result[0].iloc[0]
            if match['distance'] < 0.4:
                return os.path.basename(match['identity']).replace('.jpg', '')
    except Exception:
        pass
    return "Unknown"


def generate_frames():
    global frame_num
    camera    = cv2.VideoCapture(0)
    frame_num = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame_num += 1
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if emotion_model:
                emotion, conf = predict_emotion(face_roi)
                if frame_num % 10 == 0:
                    student = identify_student(face_roi)
                    if current_session["active"] and student != "Unknown":
                        if student not in current_session["data"]:
                            current_session["data"][student] = []
                        current_session["data"][student].append({
                            "emotion":    emotion,
                            "engagement": ENGAGEMENT_SCORE.get(emotion, 0.5)
                        })
                else:
                    student = ""

                color = (0,255,100) if emotion=='happy' else \
                        (0,200,255) if emotion=='surprise' else \
                        (0,0,255)   if emotion=='angry' else (200,200,200)
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                label = f"{student} | {emotion} {conf:.0%}" if student else f"{emotion} {conf:.0%}"
                cv2.putText(frame, label, (x, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/session/start', methods=['POST'])
def start_session():
    current_session.update({"active": True, "data": {}})
    return jsonify({"status": "started"})

@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    current_session["active"] = False
    summary = {}
    for student, logs in current_session["data"].items():
        if logs:
            emotions = [l['emotion'] for l in logs]
            summary[student] = {
                "dominant_emotion": max(set(emotions), key=emotions.count),
                "avg_engagement":   round(np.mean([l['engagement'] for l in logs]), 3),
                "total_frames":     len(logs)
            }
    os.makedirs(LOGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(LOGS_DIR, f"session_{ts}.json"), 'w') as f:
        json.dump({"session_id": ts, "summary": summary}, f, indent=2)
    return jsonify({"status": "stopped", "summary": summary})

@app.route('/api/session/live')
def live_stats():
    stats = {}
    for student, logs in current_session["data"].items():
        if logs:
            recent = logs[-20:]
            stats[student] = {
                "current_emotion": recent[-1]['emotion'],
                "avg_engagement":  round(np.mean([l['engagement'] for l in recent]), 3),
                "emotion_counts":  {e: [l['emotion'] for l in logs].count(e)
                                    for e in EMOTIONS}
            }
    return jsonify({"active": current_session["active"], "students": stats})

@app.route('/api/sessions')
def get_sessions():
    sessions = []
    for f in sorted(glob.glob(os.path.join(LOGS_DIR, "*.json")), reverse=True)[:10]:
        with open(f) as fp:
            sessions.append(json.load(fp))
    return jsonify(sessions)


if __name__ == "__main__":
    print("=" * 50)
    print("  FLASK DASHBOARD — http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)