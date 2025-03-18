'''
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_PID_2025R = cv.imread('./Captura de tela 2025-03-12 114819.png')

plt.imshow(img_PID_2025R, cmap='gray', vmin=0, vmax=255)
plt.show()
'''
from PIL import Image  # Importa a biblioteca Pillow para abrir imagens
import matplotlib.pyplot as plt  # Para exibir a imagem

# Abre a imagem usando Pillow
img_PID_2025R = Image.open('./Captura de tela 2025-03-12 114819.png')

# Exibe a imagem usando matplotlib
plt.imshow(img_PID_2025R, cmap='gray', vmin=0, vmax=255)
plt.show()