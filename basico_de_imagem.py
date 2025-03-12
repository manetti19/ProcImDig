import cv2
import matplotlib.pyplot as plt
import numpy as np

img_PID_2025 = cv2.imread("C:/Users/dinos/OneDrive/Imagens/Capturas de tela/Captura de tela 2025-03-12 114819.png", cv2.IMREAD_GRAYSCALE)

img_PID_2025R = cv2.imread("C:/Users/dinos/OneDrive/Imagens/Capturas de tela/Captura de tela 2025-03-12 114819.png")

plt.imshow(img_PID_2025, cmap='gray', vmin=0, vmax=255)