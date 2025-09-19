import face_recognition
import cv2

# Carrega imagem de referência
known_image = face_recognition.load_image_file("luciano.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Inicia webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    rgb_frame = frame[:, :, ::-1]

    # Detecta rostos
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
        label = "Luciano" if match else "Desconhecido"

        # Desenha retângulo e nome
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

