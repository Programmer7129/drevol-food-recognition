import pytesseract
from PIL import Image

img = Image.open('Number images/images.png')
text = pytesseract.image_to_string(img, config='--psm 6')  # psm 6 is for single-block text
print(text)
