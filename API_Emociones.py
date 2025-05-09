import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pymongo import MongoClient
import datetime
import time
from collections import deque
from flask import Flask, jsonify
import threading

# === Variables globales compartidas ===
emocion = "no_detectada"
contador_fatiga = 0
umbral_fatiga = 10

# === Iniciar app Flask ===
app = Flask(__name__)

@app.route('/estado')
def estado():
    # Devuelve emoción y estado de fatiga como JSON
    return jsonify({
        "emocion": emocion,
        "fatiga": contador_fatiga >= umbral_fatiga
    })

# === Función principal: detección emocional y fatiga ===
def detectar_emociones():
    global emocion, contador_fatiga

    # Modelo entrenado y etiquetas
    modelo = tf.keras.models.load_model("modelo_emociones_mobilenetv2_optimo.h5")
    emociones_list = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # Conexión a MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["monitor_emocional"]
    coleccion = db["registros"]

    # MediaPipe FaceMesh
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
    CHIN_IDX = 152
    NOSE_IDX = 1
    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

    # Captura de video
    cap = cv2.VideoCapture(0)

    # Variables auxiliares
    cola_emociones = deque(maxlen=5)
    registro_guardado = False
    tiempo_registro_guardado = 0
    tiempo_reinicio_registro = 30  # segundos

    # Función para calcular EAR (Eye Aspect Ratio)
    def calcular_ear(p):
        A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
        B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
        C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
        return (A + B) / (2.0 * C)

    # Bucle principal
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        ojos_cerrados = False
        cabeza_baja = False
        emocion_actual = "no_detectada"
        prob = 0.0

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX]
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX]
            ear = (calcular_ear(right_eye) + calcular_ear(left_eye)) / 2
            ojos_cerrados = ear < 0.21

            chin = (landmarks[CHIN_IDX].x * w, landmarks[CHIN_IDX].y * h)
            nose = (landmarks[NOSE_IDX].x * w, landmarks[NOSE_IDX].y * h)
            cabeza_baja = chin[1] - nose[1] < 40

            # Recorte y predicción de emoción
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = max(min(xs)-20, 0), min(max(xs)+20, w)
            y_min, y_max = max(min(ys)-20, 0), min(max(ys)+20, h)
            rostro = frame[y_min:y_max, x_min:x_max]

            if rostro.size > 0:
                rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
                rostro_resized = cv2.resize(rostro_rgb, (48, 48)) / 255.0
                rostro_array = np.expand_dims(rostro_resized, axis=0)
                pred = modelo.predict(rostro_array)
                idx = int(np.argmax(pred))  # Asegurar tipo int nativo
                emocion_predicha = emociones_list[idx]
                prob = float(pred[0][idx])  # Asegurar tipo float nativo
                cola_emociones.append(emocion_predicha)
                emocion_actual = max(set(cola_emociones), key=cola_emociones.count)

        # Actualizar estado global
        emocion = emocion_actual
        if ojos_cerrados or cabeza_baja:
            contador_fatiga += 1
        else:
            contador_fatiga = 0

        # Mostrar resultados en pantalla
        cv2.putText(frame, f'Emocion: {emocion_actual} ({prob*100:.1f}%)', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if ojos_cerrados:
            cv2.putText(frame, "Ojos cerrados", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if cabeza_baja:
            cv2.putText(frame, "Cabeza baja", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if contador_fatiga >= umbral_fatiga:
            cv2.putText(frame, "⚠️ Fatiga sostenida", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Emociones y Fatiga", frame)

        # Guardado en MongoDB cada 30 segundos como máximo
        tiempo_actual = time.time()
        if registro_guardado and (tiempo_actual - tiempo_registro_guardado > tiempo_reinicio_registro):
            registro_guardado = False

        if not registro_guardado and emocion_actual != "no_detectada":
            doc = {
                "timestamp": datetime.datetime.now(),
                "emocion": emocion_actual,
                "probabilidad": float(prob),
                "ojos_cerrados": bool(ojos_cerrados),
                "cabeza_baja": bool(cabeza_baja),
                "fatiga_sostenida": bool(contador_fatiga >= umbral_fatiga)
            }
            coleccion.insert_one(doc)
            registro_guardado = True
            tiempo_registro_guardado = tiempo_actual

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# === Iniciar la detección en un hilo paralelo ===
threading.Thread(target=detectar_emociones).start()

# === Iniciar la API Flask en el hilo principal ===
app.run(port=5000)
