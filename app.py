# app.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
from threading import Lock

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


modelo = joblib.load('modelo_posturas.pkl')
encoder = joblib.load('encoder.pkl')


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)


thread_lock = Lock()
thread = None

def classify_posture(frame):
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        
        landmarks = [coord for landmark in results.pose_landmarks.landmark 
                    for coord in [landmark.x, landmark.y, landmark.z]]
        
        
        posture = encoder.inverse_transform(modelo.predict([landmarks]))[0]
        
        
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        return frame, posture
    return frame, None

def video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame, posture = classify_posture(frame)
        
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        
        with thread_lock:
            socketio.emit('video_feed', {
                'image': f"data:image/jpeg;base64,{frame_base64}",
                'posture': posture or "No detectado"
            })
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(video_stream)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
