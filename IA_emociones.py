# === Importación de librerías ===
import cv2                          # OpenCV para captura de video y procesamiento de imágenes
import numpy as np                 # Operaciones con matrices y arrays
import tensorflow as tf            # Para cargar y usar el modelo de emociones
import mediapipe as mp             # Detección de puntos faciales
from pymongo import MongoClient    # Conexión a base de datos MongoDB
import datetime                    # Tiempos para registrar fechas
import time                        # Medición de tiempo para control de eventos
from collections import deque      # Cola para suavizar predicciones de emociones

# === Cargar modelo entrenado de emociones ===
modelo = tf.keras.models.load_model("modelo_emociones_mobilenetv2_optimo.h5")
emociones = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # Etiquetas del modelo

# === Conexión a base de datos MongoDB ===
client = MongoClient("mongodb://localhost:27017/")
db = client["monitor_emocional"]               # Base de datos
coleccion = db["registros"]                    # Colección donde se guardan los datos

# === Inicialización de MediaPipe para reconocimiento facial ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1)  # Solo un rostro

# Índices específicos de MediaPipe para mentón y nariz
CHIN_IDX = 152
NOSE_IDX = 1

# Índices de puntos del ojo izquierdo y derecho para calcular EAR
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# === Inicializar cámara web ===
cap = cv2.VideoCapture(0)

# === Variables auxiliares ===
ultima_emocion = None                     # Última emoción registrada
tiempo_ultimo_registro = 0               # Última vez que se guardó en MongoDB
contador_fatiga = 0                      # Contador para fatiga sostenida
umbral_fatiga = 10                       # Número de frames consecutivos para considerar fatiga
cola_emociones = deque(maxlen=5)         # Suavizado de emociones con últimas 5

# === Función para calcular EAR (Eye Aspect Ratio) ===
def calcular_ear(puntos):
    A = np.linalg.norm(np.array(puntos[1]) - np.array(puntos[5]))
    B = np.linalg.norm(np.array(puntos[2]) - np.array(puntos[4]))
    C = np.linalg.norm(np.array(puntos[0]) - np.array(puntos[3]))
    return (A + B) / (2.0 * C)  # Fórmula de EAR

# === Bucle principal ===
while True:
    ret, frame = cap.read()          # Captura frame de la cámara
    if not ret:
        break                        # Si falla, termina

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB para MediaPipe
    result = face_mesh.process(rgb)               # Procesar frame para obtener landmarks

    ojos_cerrados = False
    cabeza_baja = False
    emocion = "no_detectada"
    prob = 0

    # === Si se detecta un rostro ===
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Obtener coordenadas de puntos para ambos ojos
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX]
        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX]

        # Calcular EAR promedio
        ear_right = calcular_ear(right_eye)
        ear_left = calcular_ear(left_eye)
        ear_avg = (ear_right + ear_left) / 2

        # Considerar ojos cerrados si EAR bajo
        if ear_avg < 0.21:
            ojos_cerrados = True

        # === Detección de cabeza baja ===
        chin = (landmarks[CHIN_IDX].x * w, landmarks[CHIN_IDX].y * h)
        nose = (landmarks[NOSE_IDX].x * w, landmarks[NOSE_IDX].y * h)
        if chin[1] - nose[1] < 40:
            cabeza_baja = True

        # === Recorte del rostro para predecir emoción ===
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(min(xs)-20, 0), min(max(xs)+20, w)
        y_min, y_max = max(min(ys)-20, 0), min(max(ys)+20, h)
        rostro = frame[y_min:y_max, x_min:x_max]

        # Si el rostro tiene datos válidos
        if rostro.size > 0:
            rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)           # Convertir a RGB
            rostro_resized = cv2.resize(rostro_rgb, (48, 48)) / 255.0      # Redimensionar y normalizar
            rostro_array = np.expand_dims(rostro_resized, axis=0)          # Expandir dimensión
            pred = modelo.predict(rostro_array)                            # Predecir emoción
            idx = np.argmax(pred)                                          # Índice de mayor probabilidad
            emocion_predicha = emociones[idx]                              # Obtener emoción
            prob = float(pred[0][idx])                                     # Probabilidad
            cola_emociones.append(emocion_predicha)                        # Guardar emoción en cola
            emocion = max(set(cola_emociones), key=cola_emociones.count)  # Emoción más frecuente

    # === Verificación de fatiga sostenida ===
    if ojos_cerrados or cabeza_baja:
        contador_fatiga += 1  # Aumentar contador si hay signos de fatiga
    else:
        contador_fatiga = 0   # Reiniciar si no hay fatiga

    # Mostrar alerta si se supera el umbral
    if contador_fatiga >= umbral_fatiga:
        cv2.putText(frame, "⚠️ Fatiga sostenida detectada", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === Mostrar texto en la ventana de la cámara ===
    cv2.putText(frame, f'Emocion: {emocion} ({prob*100:.1f}%)', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if ojos_cerrados:
        cv2.putText(frame, "Ojos cerrados (EAR)", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if cabeza_baja:
        cv2.putText(frame, "Cabeza baja", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar ventana de OpenCV con la imagen
    cv2.imshow("Detección de emociones y fatiga", frame)

    # === Condición para registrar en MongoDB ===
    cambio = emocion != ultima_emocion
    pasaron_5_seg = time.time() - tiempo_ultimo_registro > 5

    # Si hubo cambio de emoción o ya pasaron 5 segundos, registrar
    if cambio or pasaron_5_seg:
        doc = {
            "timestamp": datetime.datetime.now(),
            "emocion": emocion,
            "probabilidad": prob,
            "ojos_cerrados": ojos_cerrados,
            "cabeza_baja": cabeza_baja,
            "fatiga_sostenida": contador_fatiga >= umbral_fatiga
        }
        coleccion.insert_one(doc)              # Guardar en MongoDB
        ultima_emocion = emocion               # Actualizar última emoción
        tiempo_ultimo_registro = time.time()   # Actualizar tiempo

    # Salir si se presiona la tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Liberar recursos ===
cap.release()
cv2.destroyAllWindows()



