from PIL import Image  # Importa a biblioteca Pillow para abrir imagens
import matplotlib.pyplot as plt  # Para exibir a imagem

# Abre a imagem usando Pillow
img_PID_2025R = Image.open('./Captura de tela 2025-03-12 114819.png')
img_PID_2025R.show()

nova_imagem = img_PID_2025R.resize((500, 200))  # Largura x Altura
nova_imagem.show()

cinza = img_PID_2025R.convert("L") # tudo cinza
cinza.show()

rotacionada = img_PID_2025R.rotate(45) # rotate
rotacionada.show()

corte = img_PID_2025R.crop((50, 50, 500, 500)) # corte
corte.show()

from PIL import ImageFilter

borrada = img_PID_2025R.filter(ImageFilter.BLUR)
borrada.show()

from PIL import ImageDraw

desenho = ImageDraw.Draw(img_PID_2025R)
desenho.rectangle([300, 300, 600, 600], outline="red", width=3)
img_PID_2025R.show()

from PIL import ImageFont, ImageDraw

desenho = ImageDraw.Draw(img_PID_2025R)
fonte = ImageFont.load_default()
desenho.text((600, 300), "Olá, Mundo!", font=fonte, fill="blue")
img_PID_2025R.show()

largura, altura = img_PID_2025R.size
formato = img_PID_2025R.format
modo = img_PID_2025R.mode

print(f"Largura: {largura}, Altura: {altura}, Formato: {formato}, Modo: {modo}")