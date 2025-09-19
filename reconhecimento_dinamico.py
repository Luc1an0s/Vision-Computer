import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pickle
import os
from collections import deque
from tensorflow.keras.models import load_model

MODELOS_DIR = "modelos"
modelo_path = os.path.join(MODELOS_DIR, "modelo_gestos_dinamicos.h5")
encoder_path = os.path.join(MODELOS_DIR, "label_encoder.pkl")

model = load_model(modelo_path)
with open(encoder_path, "rb") as f:
    encoder = pickle.load(f)

SEQ_LEN = 20
buffer_seq = deque(maxlen=SEQ_LEN)
ultimo_gesto = None
CONF_THRESH = 0.7

voz = pyttsx3.init()
voz.setProperty("rate", 160)
voz.setProperty("volume", 1.0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("Reconhecimento dinamico iniciado — 'q' para sair.")

while True:
    ok, img = cap.read()
    if not ok:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_present = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_present = True
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            if len(data) == 63:
                buffer_seq.append(data)

    if len(buffer_seq) == SEQ_LEN:
        X = np.array(buffer_seq, dtype=np.float32).reshape(1, SEQ_LEN, 63)
        probs = model.predict(X, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        gesto = encoder.inverse_transform([idx])[0]

        cv2.putText(img, f"{gesto}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        altura, largura = img.shape[:2]
        cv2.putText(img, f"Confianca: {conf:.2f}", (10, altura - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if conf >= CONF_THRESH and gesto != ultimo_gesto:
            voz.say(gesto)
            voz.runAndWait()
            ultimo_gesto = gesto

    else:
        cv2.putText(img, f"Aguardando {SEQ_LEN - len(buffer_seq)} frames...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)

    if not hand_present:
        ultimo_gesto = None
        buffer_seq.clear()
        cv2.putText(img, "Mao nao detectada", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Reconhecimento de Gestos (Dinamico)", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()