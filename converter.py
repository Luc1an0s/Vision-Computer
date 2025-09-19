from PIL import Image


img = Image.open("luciano.jpg").convert("RGB")

img.save("luciano_clean.jpg", format="JPEG")
