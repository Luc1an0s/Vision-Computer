import cv2
import face_recognition
import numpy as np

# Inicia a webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Erro: Webcam não foi aberta.")
    exit()

cv2.namedWindow("Linhas do Rosto - Webcam", cv2.WINDOW_NORMAL)

while True:
    ret, frame = video_capture.read()
    if not ret or frame is None:
        print("Erro ao capturar frame.")
        continue

    # Converte para RGB corretamente
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    try:
        # Detecta landmarks faciais
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
    except Exception as e:
        print("Erro ao detectar rosto:", e)
        continue

    # Desenha as linhas faciais
    for face_landmarks in face_landmarks_list:
        for feature, points in face_landmarks.items():
            for i in range(len(points) - 1):
                pt1 = points[i]
                pt2 = points[i + 1]
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # Exibe o vídeo com linhas desenhadas
    cv2.imshow("Linhas do Rosto - Webcam", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(10) & 0xFF == ord("q"):
        print("Encerrando...")
        break

video_capture.release()
cv2.destroyAllWindows()
