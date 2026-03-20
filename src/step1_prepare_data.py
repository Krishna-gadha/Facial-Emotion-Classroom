"""
STEP 1 — DATA PREPARATION
Downloads FER2013 from Kaggle and preps your student images.
Run this first before anything else.
"""

import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

STUDENT_DATA_PATH = "../data/students/students_data"
FER_DATA_PATH     = "../data/fer2013"
OUTPUT_PATH       = "../data/processed"

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def process_fer2013():
    csv_path = os.path.join(FER_DATA_PATH, "fer2013.csv")

    if not os.path.exists(csv_path):
        print("❌ fer2013.csv not found.")
        print("   Download from: https://www.kaggle.com/datasets/msambare/fer2013")
        print(f"   Place it in: {FER_DATA_PATH}")
        return

    print("📂 Processing FER2013...")
    df = pd.read_csv(csv_path)

    for split in ['train', 'val', 'test']:
        for emotion in EMOTIONS:
            os.makedirs(f"{OUTPUT_PATH}/fer/{split}/{emotion}", exist_ok=True)

    split_map = {
        'Training':    'train',
        'PublicTest':  'val',
        'PrivateTest': 'test'
    }

    counts = {s: {e: 0 for e in EMOTIONS} for s in ['train', 'val', 'test']}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting FER2013"):
        emotion_label = EMOTIONS[int(row['emotion'])]
        split         = split_map.get(row['Usage'], 'train')
        pixels = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        img    = Image.fromarray(pixels)
        save_path = f"{OUTPUT_PATH}/fer/{split}/{emotion_label}/{idx}.png"
        img.save(save_path)
        counts[split][emotion_label] += 1

    print("\n✅ FER2013 processed.")
    for split, emotions in counts.items():
        print(f"   {split}: {sum(emotions.values())} images")


def register_students():
    if not os.path.exists(STUDENT_DATA_PATH):
        print(f"❌ Student data not found at {STUDENT_DATA_PATH}")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    registered_dir = "../data/registered_faces"
    os.makedirs(registered_dir, exist_ok=True)

    students   = [d for d in os.listdir(STUDENT_DATA_PATH)
                  if os.path.isdir(os.path.join(STUDENT_DATA_PATH, d))]
    registered = 0
    failed     = []

    print(f"\n👤 Registering {len(students)} students...")

    for student in tqdm(students, desc="Processing students"):
        student_folder = os.path.join(STUDENT_DATA_PATH, student)
        images = [f for f in os.listdir(student_folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        best_image = None

        for img_file in images:
            img_path = os.path.join(student_folder, img_file)
            img      = cv2.imread(img_path)
            if img is None:
                continue
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 1:
                best_image = img_path
                break

        if best_image:
            shutil.copy2(best_image, os.path.join(registered_dir, f"{student}.jpg"))
            registered += 1
        else:
            failed.append(student)

    print(f"\n✅ Registered: {registered}/{len(students)} students")
    if failed:
        print(f"⚠️  Could not find clear face for: {failed}")


if __name__ == "__main__":
    print("=" * 50)
    print("  STEP 1 — DATA PREPARATION")
    print("=" * 50)
    process_fer2013()
    register_students()
    print("\n✅ Step 1 complete. Run step2_train_emotion.py next.")