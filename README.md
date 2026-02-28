
Real-Time Driver Drowsiness Detection

📌 Overview

This project is a real-time AI-based system that detects driver drowsiness using computer vision techniques.
It monitors the driver's eyes through a webcam and triggers an alarm if the eyes remain closed for a specific duration.

The goal is to help prevent road accidents caused by driver fatigue.

I initially experimented with a deep learning eye-state classifier trained on an eye dataset. However, in real-time conditions (different lighting, camera angles, and eye-cropping variability),

the model was not reliable enough and produced inconsistent closed-eye detection.

For robustness and real-time stability, I switched to an Eye Aspect Ratio (EAR) landmark-based approach, which provided more consistent performance with minimal latency.

Deep learning-based classification remains a planned future upgrade once a stronger dataset and better preprocessing pipeline are available.

نص

🎯 Problem Statement

Driver drowsiness is a major cause of road accidents worldwide.
When a driver becomes sleepy, reaction time decreases and attention drops.

This project detects drowsiness by analyzing:

Eye Aspect Ratio (EAR)

Eye closure duration

Real-time face landmarks detection

If the eyes stay closed longer than a defined threshold, an alarm sound is activated.

🧠 Technologies Used

Python

OpenCV

MediaPipe Face Mesh

NumPy

Pygame (for alarm sound)

⚙️ How It Works

Webcam captures live video.

MediaPipe detects facial landmarks.

Eye Aspect Ratio (EAR) is calculated.

If EAR < threshold for several seconds:

🚨 Alarm is triggered.

📂 Project Structure
Real-Time-Driver-Drowsiness-Detection/
│
├── test.py
├── requirements.txt
├── PROGRESSIVE_BLEEP_xvo.wav
└── README.md
▶️ Installation

Clone the repository:

git clone https://github.com/yourusername/Real-Time-Driver-Drowsiness-Detection.git
cd Real-Time-Driver-Drowsiness-Detection

Install dependencies:

pip install -r requirements.txt

Run the project:

python test.py
📊 Features

✔ Real-time eye tracking
✔ EAR-based drowsiness detection
✔ Automatic alarm system
✔ Lightweight & efficient

🔥 Future Improvements

Deep Learning-based eye state classification but with  better real dataset from me 

Head pose estimation

Mobile deployment

Integration with IoT alert systems


Ahmad Nachar
Software Engineering Student
Interested in AI & Computer Vision
