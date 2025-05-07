import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from pymongo import MongoClient
import datetime
import time
from collections import deque

# === Cargar modelo entrenado ===
modelo = tf.keras.models.load_model("modelo_emociones_mobilenetv2_optimo.h5")
emociones = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# === MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["monitor_emocional"]
coleccion = db["registros"]

# === MediaPipe FaceMesh ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)
CHIN_IDX = 152
NOSE_IDX = 1

# === Puntos para EAR (MediaPipe indices) ===
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# === Inicializar cámara ===
cap = cv2.VideoCapture(0)

# === Variables ===
ultima_emocion = None
tiempo_ultimo_registro = 0
contador_fatiga = 0
umbral_fatiga = 10
cola_emociones = deque(maxlen=5)

def calcular_ear(puntos):
    A = np.linalg.norm(np.array(puntos[1]) - np.array(puntos[5]))
    B = np.linalg.norm(np.array(puntos[2]) - np.array(puntos[4]))
    C = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[3]))
    return (A + B) / (2.0 * C)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    ojos_cerrados = False
    cabeza_baja = False
    emocion = "no_detectada"
    prob = 0

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # === EAR para detectar ojos cerrados ===
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX]
        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX]

        ear_right = calcular_ear(right_eye)
        ear_left = calcular_ear(left_eye)
        ear_avg = (ear_right + ear_left) / 2

        if ear_avg < 0.21:
            ojos_cerrados = True

        # === Cabeza baja ===
        chin = (landmarks[CHIN_IDX].x * w, landmarks[CHIN_IDX].y * h)
        nose = (landmarks[NOSE_IDX].x * w, landmarks[NOSE_IDX].y * h)
        if chin[1] - nose[1] < 40:
            cabeza_baja = True

        # === Recortar rostro y predecir emoción ===
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(min(xs)-20, 0), min(max(xs)+20, w)
        y_min, y_max = max(min(ys)-20, 0), min(max(ys)+20, h)
        rostro = frame[y_min:y_max, x_min:x_max]

        if rostro.size > 0:
            rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
            rostro_resized = cv2.resize(rostro_rgb, (48, 48)) / 255.0  # 👈 Ajuste a tu modelo
            rostro_array = np.expand_dims(rostro_resized, axis=0)
            pred = modelo.predict(rostro_array)
            idx = np.argmax(pred)
            emocion_predicha = emociones[idx]
            prob = float(pred[0][idx])

            cola_emociones.append(emocion_predicha)
            emocion = max(set(cola_emociones), key=cola_emociones.count)

    # === Fatiga sostenida ===
    if ojos_cerrados or cabeza_baja:
        contador_fatiga += 1
    else:
        contador_fatiga = 0

    if contador_fatiga >= umbral_fatiga:
        cv2.putText(frame, "⚠️ Fatiga sostenida detectada", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === Mostrar resultados ===
    cv2.putText(frame, f'Emocion: {emocion} ({prob*100:.1f}%)', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if ojos_cerrados:
        cv2.putText(frame, "Ojos cerrados (EAR)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if cabeza_baja:
        cv2.putText(frame, "Cabeza baja", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Detección de emociones y fatiga", frame)

    # === Registro en MongoDB ===
    cambio = emocion != ultima_emocion
    pasaron_5_seg = time.time() - tiempo_ultimo_registro > 5

    if cambio or pasaron_5_seg:
        doc = {
            "timestamp": datetime.datetime.now(),
            "emocion": emocion,
            "probabilidad": prob,
            "ojos_cerrados": ojos_cerrados,
            "cabeza_baja": cabeza_baja,
            "fatiga_sostenida": contador_fatiga >= umbral_fatiga
        }
        coleccion.insert_one(doc)
        ultima_emocion = emocion
        tiempo_ultimo_registro = time.time()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



