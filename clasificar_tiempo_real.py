import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


try:
    modelo = joblib.load('modelo_posturas.pkl')
    encoder = joblib.load('encoder.pkl')
except FileNotFoundError:
    print("Error: No se encontraron los archivos del modelo")
    print("Ejecuta primero 'entrenar_modelo.py'")
    exit()


base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara")
    exit()

# Colores para cada postura
colores = {
    'recto': (0, 255, 0),      # Verde
    'deslizado': (0, 255, 255), # Amarillo
    'encorvado': (0, 0, 255),   # Rojo
    'cruzado': (255, 0, 0),     # Azul
    'inclinado': (255, 0, 255)  # Magenta
}

print("\nIniciando clasificación en tiempo real...")
print("Presiona ESC para salir")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir frame a formato MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Detectar postura
    resultado = detector.detect(imagen_mp)
    
    if resultado.pose_landmarks:
        # Convertir a NormalizedLandmarkList
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in resultado.pose_landmarks[0]
        ])
        
        # Extraer landmarks para el modelo
        landmarks_flatten = [coord for landmark in resultado.pose_landmarks[0] 
                           for coord in [landmark.x, landmark.y, landmark.z]]
        
        # Predecir postura (ignorar advertencia de nombres de características)
        with np.errstate(divide='ignore', invalid='ignore'):
            clase_num = modelo.predict([landmarks_flatten])[0]
            postura = encoder.inverse_transform([clase_num])[0]
            color = colores.get(postura, (255, 255, 255))
        
        # Dibujar resultado
        cv2.putText(frame, f"Postura: {postura}", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Dibujar landmarks y conexiones
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=color, thickness=2, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                color=color, thickness=2))
    
    cv2.imshow('Clasificador de Posturas', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()