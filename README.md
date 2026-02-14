🧠 Computer Vision Suite
Real-Time Face Detection, Recognition & Emotion Analysis System.
Computer Vision Suite is a modular AI-based system that performs:
✅ Real-time Face Detection
✅ Face Recognition using trained datasets
✅ Emotion Detection from facial expressions
📌 Project Overview
This repository includes the following modules:
1️⃣ Face Detection
Detects human faces in real-time using Haar Cascade classifiers.
Key Features:
Real-time face detection using webcam
Uses OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
Draws bounding boxes around detected faces

2️⃣ Face Recognition
Recognizes and identifies faces after training on a dataset.
Key Features:
Face dataset creation and training
Face recognition using trained model
Stores trained data in trainer/
Supports real-time recognition via webcam
Files related:
train_faces.py
face_recognize.py
saved_faces/
trainer/

3️⃣ Emotion Detection
Detects facial emotions from live video input.
Key Features:
Real-time emotion classification
Detects facial expressions such as happy, sad, angry, etc.
Uses trained emotion recognition model
File:
emotion_detection.py

🛠 Technologies Used
Python
OpenCV
NumPy
Machine Learning concepts
Haar Cascade Classifier

📂 Project Structure
computer_vision_projects/
│
├── saved_faces/
├── trainer/
├── emotion_detection.py
├── face_detection.ipynb
├── face_recognize.py
├── train_faces.py
├── haarcascade_frontalface_default.xml
└── README.md

▶️ How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/your-username/computer_vision_projects.git
cd computer_vision_projects

Step 2: Install Required Libraries
pip install opencv-python numpy

Step 3: Run Any Module
For Face Detection:
python face_detection.ipynb

For Face Recognition:
 python train_faces.py
python face_recognize.py

For Emotion Detection:
python emotion_detection.py

🚀 Features
🎥 Live Camera Face Detection
🧍 Face Recognition with Custom Dataset
😊 Emotion Classification (Happy, Sad, Angry, etc.)
💾 Capture & Save Faces using Key Press
📦 Modular Project Structure
🔁 Scalable and Extendable Design

🧠 How It Works
Face Detection
Uses Haar Cascade to locate faces in real-time video frames.

Face Recognition
Compares detected faces with trained dataset embeddings.

Emotion Detection
Uses a trained model to classify facial expressions.

📈 Future Improvements
🔗  Combine detection + recognition + emotion in one pipeline
🌐  Build web version using Flask or Streamlit
📊  Add dashboard for recognition logs
☁️  Deploy as API service
🧠  Improve accuracy using Deep Learning models

🎯  Learning Outcomes
This project demonstrates:
a. Real-time image processing
b. Dataset handling
c. Model training & inference
d. Modular Python project design
e. Version control with Git

📄 License
This project is open-source and available for educational and research purposes.

👩‍💻 Author
Umra Jamadar
Aspiring AI Engineer | Computer Vision Enthusiast

