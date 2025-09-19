import face_recognition
import numpy as np

# Carrega a imagem
image = face_recognition.load_image_file("luciano.jpg")

# Garante que o array seja contíguo e compatível
image = np.ascontiguousarray(image)

# Detecta rostos
face_locations = face_recognition.face_locations(image)
print("Rostos detectados:", len(face_locations))
