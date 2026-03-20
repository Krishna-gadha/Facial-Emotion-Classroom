# 🎓 Smart Classroom Emotion Analytics

Real-time facial emotion detection and student recognition system built for smart classroom monitoring.

## 🚀 Features
- Real-time face detection using OpenCV
- Emotion recognition (7 classes) using MobileNetV2 trained on FER2013
- Student identity recognition using DeepFace + Facenet embeddings
- Engagement scoring per student
- Session logging to JSON
- Flask web dashboard

## 🛠️ Tech Stack
- Python, TensorFlow, Keras
- OpenCV, DeepFace, Facenet
- Flask, HTML/CSS

## 📊 Model
- Architecture: MobileNetV2 (transfer learning)
- Dataset: FER2013
- Accuracy: ~54% (7-class emotion classification)
- Emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## ⚙️ Setup
```bash
pip install -r requirements.txt
cd src
python3 step3_inference.py
```

## 📁 Project Structure
```
facial-emotion-classroom/
├── src/
│   ├── step1_prepare_data.py
│   ├── step2_train_emotion.py
│   └── step3_inference.py
├── app/
│   ├── app.py
│   └── templates/
├── data/
│   └── logs/
├── models/
└── requirements.txt
```

## ⚠️ Privacy Note
Student images and trained model files are excluded from this repository via .gitignore.

## 👩‍💻 Author
Gadha — B.Tech AI/ML Student