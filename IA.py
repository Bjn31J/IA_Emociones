#=========================================
#Emociones que detecta DeepFace:
#Emoción	    Descripción breve
#happy--->	    Feliz, sonriente, alegre
#sad--->        Triste, mirada caída, labios hacia abajo
#angry--->      Enojado, ceño fruncido, mirada intensa
#surprise--->   Sorprendido, ojos muy abiertos, cejas alzadas
#fear--->       Miedo, mirada tensa, rostro rígido
#disgust--->	Asco, fruncimiento de nariz y boca
#neutral--->	Expresión sin emociones visibles

# ======================================
# SISTEMA DE DETECCION DE EMOCIONES Y FATIGA PERSONALIZADO POR PERSONA
# Funcionalidades:
# - Deteccion de emociones (DeepFace)
# - Deteccion de ojos cerrados y cabeza baja (MediaPipe)
# - Identificacion facial por imagen (DeepFace + dataset)
# - Registro en MongoDB (emocion, fatiga, persona, timestamp)
# - Visualizacion en tiempo real con OpenCV
# - Graficas por persona con pandas + matplotlib
# ======================================

import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
from pymongo import MongoClient
import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import threading
import tkinter as tk
from tkinter import simpledialog

# === Inicializar interfaz Tkinter para inputs ===
root = tk.Tk()
root.withdraw()

# === Cargar dataset existente de personas ===
dataset_path = "dataset"
personas_registradas = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
print("[INFO] Personas en el dataset:", personas_registradas)

# === Conexion a MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["monitor_emocional"]
coleccion = db["registros"]

# === Inicializar MediaPipe para landmarks faciales ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)

# === Funciones auxiliares ===
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def registrar_nueva_persona(nombre, frame):
    persona_dir = os.path.join(dataset_path, nombre)
    os.makedirs(persona_dir, exist_ok=True)
    cv2.imwrite(os.path.join(persona_dir, f"{nombre}_1.jpg"), frame)
    print(f"[INFO] Persona {nombre} registrada.")

# === Indices de landmarks para ojos y cabeza ===
LEFT_EYE_IDX = [362, 385]
RIGHT_EYE_IDX = [159, 145]
CHIN_IDX = 152
NOSE_IDX = 1

# === Configuracion del sistema ===
contador_fatiga = 0
umbral_fatiga = 10

# === Tiempos para procesamiento optimizado ===
tiempo_ultima_emocion = 0
intervalo_emocion = 5

tiempo_ultima_identificacion = 0
intervalo_identificacion = 10

# === Variables iniciales ===
emocion = "no_detectada"
persona = "no_detectado"

# === Captura de video ===
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    resized_frame = cv2.resize(frame, (640, 480))

    if result.multi_face_landmarks:
        if time.time() - tiempo_ultima_emocion > intervalo_emocion:
            def analizar_emocion():
                global emocion, tiempo_ultima_emocion
                try:
                    analysis = DeepFace.analyze(resized_frame, actions=['emotion'], enforce_detection=False)
                    emocion = analysis[0]['dominant_emotion']
                except:
                    emocion = "no_detectada"
                tiempo_ultima_emocion = time.time()
            threading.Thread(target=analizar_emocion).start()

        if time.time() - tiempo_ultima_identificacion > intervalo_identificacion:
            def identificar_persona():
                global persona, tiempo_ultima_identificacion
                try:
                    found = DeepFace.find(img_path=resized_frame, db_path=dataset_path, model_name="Facenet512", detector_backend="retinaface", enforce_detection=True)
                    if len(found[0]) > 0 and found[0].iloc[0]['distance'] < 0.4:
                        persona = os.path.basename(found[0].iloc[0]['identity']).split(os.sep)[-2]
                    else:
                        persona = "desconocido"
                except:
                    persona = "no_detectado"
                tiempo_ultima_identificacion = time.time()
            threading.Thread(target=identificar_persona).start()

    ojos_cerrados = False
    cabeza_baja = False

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        left_eye_top = (int(landmarks[LEFT_EYE_IDX[0]].x * w), int(landmarks[LEFT_EYE_IDX[0]].y * h))
        left_eye_bottom = (int(landmarks[LEFT_EYE_IDX[1]].x * w), int(landmarks[LEFT_EYE_IDX[1]].y * h))
        right_eye_top = (int(landmarks[RIGHT_EYE_IDX[0]].x * w), int(landmarks[RIGHT_EYE_IDX[0]].y * h))
        right_eye_bottom = (int(landmarks[RIGHT_EYE_IDX[1]].x * w), int(landmarks[RIGHT_EYE_IDX[1]].y * h))

        left_eye_dist = euclidean(left_eye_top, left_eye_bottom)
        right_eye_dist = euclidean(right_eye_top, right_eye_bottom)
        avg_eye_dist = (left_eye_dist + right_eye_dist) / 2

        if avg_eye_dist < 5:
            ojos_cerrados = True

        chin = (landmarks[CHIN_IDX].x * w, landmarks[CHIN_IDX].y * h)
        nose = (landmarks[NOSE_IDX].x * w, landmarks[NOSE_IDX].y * h)
        if chin[1] - nose[1] < 40:
            cabeza_baja = True

    registro = {
        "timestamp": datetime.datetime.now(),
        "persona": persona,
        "emocion": emocion,
        "ojos_cerrados": ojos_cerrados,
        "cabeza_baja": cabeza_baja
    }
    coleccion.insert_one(registro)

    if ojos_cerrados or cabeza_baja:
        contador_fatiga += 1
    else:
        contador_fatiga = 0

    if contador_fatiga >= umbral_fatiga:
        cv2.putText(frame, "⚠️ Fatiga sostenida detectada", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    fps = 1 / (time.time() - start_time)
    start_time = time.time()

    cv2.putText(frame, f'FPS: {fps:.2f}', (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f'Persona: {persona}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Emocion: {emocion}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if ojos_cerrados:
        cv2.putText(frame, "Ojos cerrados", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if cabeza_baja:
        cv2.putText(frame, "Cabeza baja", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Monitor emocional por persona", frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord("r"):
        nombre = simpledialog.askstring("Registro de persona", "Nombre de la persona:")
        if nombre:
            registrar_nueva_persona(nombre, frame)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# === Análisis por persona ===
registros = list(coleccion.find())
df = pd.DataFrame(registros)
df['timestamp'] = pd.to_datetime(df['timestamp'])

personas = df['persona'].unique()

for persona in personas:
    if persona in ["no_detectado", "desconocido"]:
        continue

    df_persona = df[df['persona'] == persona].copy()
    df_persona.set_index('timestamp', inplace=True)
    df_persona['fatiga'] = df_persona['ojos_cerrados'] | df_persona['cabeza_baja']

    emocion_counts = df_persona['emocion'].value_counts()
    fatiga_counts = df_persona['fatiga'].value_counts()

    if emocion_counts.empty or fatiga_counts.empty:
        print(f"[!] Saltando a {persona}: no hay suficientes datos para graficar.")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Análisis emocional de {persona}', fontsize=14)

    emocion_counts.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Emociones')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_xlabel('Emoción')

    fatiga_counts.plot(kind='bar', ax=axes[1], color='salmon')
    axes[1].set_title('Eventos con signos de fatiga')
    axes[1].set_ylabel('Cantidad')
    axes[1].set_xticklabels(['Sin fatiga', 'Con fatiga'], rotation=0)

    plt.tight_layout()
    plt.show()