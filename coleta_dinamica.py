import cv2
import mediapipe as mp
import csv
import os
import time


GESTURE_NAME = "OI"  
SAVE_DIR = "gestos_dinamicos"
RECORD_SECONDS = 2.0  


os.makedirs(SAVE_DIR, exist_ok=True)
csv_path = os.path.join(SAVE_DIR, f"{GESTURE_NAME}.csv")
csv_file = open(csv_path, mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
gravando = False
tempo_inicio = None

print(f"Coletando gesto dinamico: {GESTURE_NAME} — 'r' para gravar {RECORD_SECONDS}s, 'q' para sair")

while True:
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

    
    if gravando:
        cv2.putText(img, "Gravando...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        restante = max(0, RECORD_SECONDS - (time.time() - tempo_inicio))
        cv2.putText(img, f"{restante:.1f}s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if data is not None and len(data) == 63:
            csv_writer.writerow(data)
        
        if time.time() - tempo_inicio >= RECORD_SECONDS:
            gravando = False
            print("Gravacao encerrada.")

    else:
        cv2.putText(img, "Pressione 'r' para gravar", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        if not results.multi_hand_landmarks:
            cv2.putText(img, "Mao nao detectada", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Coletor de Gestos (Dinamicos)", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        gravando = True
        tempo_inicio = time.time()
        print("Gravando gesto...")
    elif key == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
