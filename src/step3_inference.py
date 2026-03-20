import os
import cv2
import numpy as np
import json
import time
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND'] = 'tensorflow'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from deepface import DeepFace

# ── Config ──
MODEL_PATH     = "../models/emotion_model_v2.keras"
REGISTERED_DIR = "../data/registered_faces"
IMG_SIZE       = 96
EMOTIONS       = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
ENGAGEMENT_SCORE = {
    'happy': 0.9, 'surprise': 0.8, 'neutral': 0.6,
    'sad': 0.3, 'fear': 0.3, 'angry': 0.2, 'disgust': 0.1
}
EMOTION_EMOJI = {
    'happy': 'Happy', 'surprise': 'Surprised', 'neutral': 'Neutral',
    'sad': 'Sad', 'fear': 'Fearful', 'angry': 'Angry',
    'disgust': 'Disgusted', 'unknown': 'Unknown'
}
COLORS = {
    'happy':    (0, 220, 90),
    'surprise': (0, 180, 255),
    'neutral':  (180, 180, 180),
    'sad':      (255, 80,  80),
    'fear':     (180, 0,   255),
    'angry':    (0,   0,   255),
    'disgust':  (0,   120, 255),
    'unknown':  (120, 120, 120)
}

# ── Load emotion model ──
print("Loading emotion model...")
emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded!")

# ── Face detector ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Load student embeddings ──
print("Loading student faces...")
registered_embeddings = {}

for fname in os.listdir(REGISTERED_DIR):
    if not fname.lower().endswith('.npy'):
        continue
    name = os.path.splitext(fname)[0]
    registered_embeddings[name] = np.load(os.path.join(REGISTERED_DIR, fname))
    print(f"  [AVG] {name}")

for fname in os.listdir(REGISTERED_DIR):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    name = os.path.splitext(fname)[0]
    if name in registered_embeddings:
        continue
    img_path = os.path.join(REGISTERED_DIR, fname)
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name="Facenet",
            enforce_detection=False
        )
        registered_embeddings[name] = np.array(result[0]["embedding"])
        print(f"  [OK]  {name}")
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")

print(f"\nTotal students loaded: {len(registered_embeddings)}")
names_list = list(registered_embeddings.keys())

os.makedirs("../data/logs", exist_ok=True)

# ── Helper: draw text with background ──
def draw_label(frame, text, pos, font_scale=0.6, thickness=2, color=(255,255,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Dark background rectangle behind text
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

# ── Functions ──
def predict_emotion(face_roi):
    try:
        face = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        face = np.expand_dims(face, axis=0)
        preds = emotion_model.predict(face, verbose=0)[0]
        return EMOTIONS[np.argmax(preds)], float(np.max(preds))
    except:
        return "unknown", 0.0

def identify_student(face_roi):
    try:
        tmp = "/tmp/tmp_face.jpg"
        cv2.imwrite(tmp, face_roi)
        result = DeepFace.represent(
            img_path=tmp,
            model_name="Facenet",
            enforce_detection=False
        )
        emb = np.array(result[0]["embedding"])
        best_name  = "Unknown"
        best_score = -1
        for name, reg_emb in registered_embeddings.items():
            score = np.dot(emb, reg_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(reg_emb)
            )
            if score > best_score:
                best_score = score
                best_name  = name
        return (best_name, best_score) if best_score > 0.4 else ("Unknown", best_score)
    except:
        return "Unknown", 0.0

# ── Webcam ──
print("\nOpening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

# Set higher resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam open! Press Q to quit.")

start_time      = time.time()
session_log     = []
frame_count     = 0
current_student = "Unknown"
face_history    = []  # for smoothing false detections

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Stricter detection — minSize bigger, scaleFactor tighter
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,       # increased from 5 — reduces false positives
        minSize=(80, 80)      # increased from 60 — ignores tiny false boxes
    )

    # Keep only the LARGEST face if multiple detected
    # (useful when you're alone — avoids phantom boxes)
    if len(faces) > 1:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[:1]
        faces = np.array(faces)

    h_frame, w_frame = frame.shape[:2]

    for (x, y, w, h) in faces:
        face_roi      = frame[y:y+h, x:x+w]
        emotion, conf = predict_emotion(face_roi)
        color         = COLORS.get(emotion, (255, 255, 255))
        engagement    = ENGAGEMENT_SCORE.get(emotion, 0.5)
        emotion_label = EMOTION_EMOJI.get(emotion, emotion)

        if frame_count % 15 == 0:
            current_student, _ = identify_student(face_roi)

        # Draw rounded-style box (thicker, cleaner)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # Corner accents for professional look
        corner_len = 20
        corner_t   = 4
        # Top-left
        cv2.line(frame, (x, y), (x+corner_len, y), color, corner_t)
        cv2.line(frame, (x, y), (x, y+corner_len), color, corner_t)
        # Top-right
        cv2.line(frame, (x+w, y), (x+w-corner_len, y), color, corner_t)
        cv2.line(frame, (x+w, y), (x+w, y+corner_len), color, corner_t)
        # Bottom-left
        cv2.line(frame, (x, y+h), (x+corner_len, y+h), color, corner_t)
        cv2.line(frame, (x, y+h), (x, y+h-corner_len), color, corner_t)
        # Bottom-right
        cv2.line(frame, (x+w, y+h), (x+w-corner_len, y+h), color, corner_t)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_len), color, corner_t)

        # Student name — large, yellow, with dark background
        draw_label(frame, current_student,
                   (x, y - 38), font_scale=0.75, thickness=2, color=(255, 255, 0))

        # Emotion + confidence
        draw_label(frame, f"{emotion_label}  {conf:.0%}",
                   (x, y - 10), font_scale=0.65, thickness=2, color=color)

        # Engagement bar
        bar_w = w
        bar_h = 8
        bar_y = y + h + 10
        cv2.rectangle(frame, (x, bar_y), (x+bar_w, bar_y+bar_h), (50,50,50), -1)
        cv2.rectangle(frame, (x, bar_y), (x+int(bar_w*engagement), bar_y+bar_h), color, -1)
        draw_label(frame, f"Engagement {engagement:.0%}",
                   (x, bar_y + 28), font_scale=0.55, thickness=1, color=(200, 200, 200))

        session_log.append({
            "time":       round(time.time() - start_time, 2),
            "student":    current_student,
            "emotion":    emotion,
            "confidence": round(conf, 3),
            "engagement": engagement
        })

    # ── Header bar ──
    cv2.rectangle(frame, (0, 0), (w_frame, 45), (20, 20, 20), -1)
    elapsed = round(time.time() - start_time, 1)
    header  = f"Smart Classroom Analytics  |  Students: {len(registered_embeddings)}  |  Session: {elapsed}s  |  Faces: {len(faces)}"
    cv2.putText(frame, header, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 180), 2)

    cv2.imshow("Smart Classroom Emotion Analytics", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if session_log:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"../data/logs/session_{ts}.json"
    with open(path, 'w') as f:
        json.dump(session_log, f, indent=2)
    print(f"Session saved to {path}")

print("Done!")