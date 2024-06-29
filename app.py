import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['TF_TRT_ALLOW_SOFT_PLACEMENT'] = '1'  # Allow soft placement for TensorRT

from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.python.compiler.tensorrt import trt_convert as trt

app = Flask(__name__)

# Define the image dimensions
img_height = 48
img_width = 48

# Load the pre-trained model
model_path = 'facialemotionmodel.h5'

# Convert the Keras model to a TensorFlow SavedModel
model = load_model(model_path)
saved_model_dir = 'saved_model'
model.save(saved_model_dir)

# Convert the SavedModel to a TensorRT optimized model
conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=trt.TrtPrecisionMode.FP16)
converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, conversion_params=conversion_params)
converter.convert()
converter.save(saved_model_dir)

# Load the TensorRT optimized model
model = tf.saved_model.load(saved_model_dir)

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

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_input = preprocess_input(face)
            emotion_prediction = model(face_input)
            max_index = np.argmax(emotion_prediction[0])
            emotion = emotion_labels[max_index]
            color = emotion_colors[emotion]
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
