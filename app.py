import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_TRT_ALLOW_SOFT_PLACEMENT'] = '1'

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import flask
    import cv2
    import numpy
    import tensorflow
except ImportError:
    install("Flask==2.0.1")
    install("opencv-python-headless==4.5.3.56")
    install("numpy==1.21.0")
    install("tensorflow==2.5.0")

from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Define the image dimensions
img_height = 48
img_width = 48

# Load the pre-trained model
model_path = 'facialemotionmodel.h5'
try:
    model = load_model(model_path)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define colors for each emotion
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (255, 0, 255),
    'Happy': (0, 255, 255),
    'Sad': (128, 128, 128),
    'Surprise': (255, 255, 0),
    'Neutral': (255, 255, 255)
}

# Function to preprocess input image for the model
def preprocess_input(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (img_height, img_width))
    face = face.astype('float32') / 255
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)
    return face

def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW backend for video capture
    while True:
        try:
            ret, frame = camera.read()
            if not ret:
                logging.error("Failed to capture frame from webcam")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_input = preprocess_input(face)
                emotion_prediction = model.predict(face_input)
                max_index = np.argmax(emotion_prediction[0])
                emotion = emotion_labels[max_index]
                color = emotion_colors[emotion]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logging.error(f"Error in gen_frames: {e}")

    camera.release()  # Release the camera when done

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
