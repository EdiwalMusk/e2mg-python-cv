from PIL import Image
import pytesseract

# 下载https://digi.bib.uni-mannheim.de/tesseract/
# 安装pytesseract
img = Image.open("../../img/opencv.png")
text = pytesseract.image_to_string(img, lang='eng')
print(text)