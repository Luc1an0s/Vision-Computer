from PIL import Image

# Abre e converte para RGB puro
img = Image.open("luciano.jpg").convert("RGB")

# Salva como nova imagem limpa
img.save("luciano_clean.jpg", format="JPEG")
