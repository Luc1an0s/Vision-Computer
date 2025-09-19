import cv2
import face_recognition
import numpy as np

# Carrega imagem de referência e gera encoding
ref_image = face_recognition.load_image_file("luciano.jpg")
ref_encoding = face_recognition.face_encodings(ref_image)[0]

# Inicia webcam
video_capture = cv2.VideoCapture(0)
cv2.namedWindow("Reconhecimento Facial", cv2.WINDOW_NORMAL)

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # Converte para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Localiza rostos e gera encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compara com rosto de referência
        match = face_recognition.compare_faces([ref_encoding], face_encoding)[0]

        # Desenha retângulo e texto
        color = (0, 255, 0) if match else (0, 0, 255)
        label = "Luciano" if match else "Desconhecido"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Reconhecimento Facial", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
