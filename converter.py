from PIL import Image
import numpy as np

# Carrega e converte para RGB
pil_image = Image.open("luciano.jpg").convert("RGB")
known_image = np.array(pil_image)

# Garante que é RGB 8-bit
print("Shape:", known_image.shape)
print("Dtype:", known_image.dtype)