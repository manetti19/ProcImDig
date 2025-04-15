import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open('./arroz.png').convert('RGB')
img.show()
r, g, b = img.split()
r.show()
g.show()
b.show()



img_np = np.array(img)

# Separa canais
rr, gg, bb = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]

# Cria imagens coloridas por canal
red_img_np = np.zeros_like(img_np)
red_img_np[:,:,0] = rr

green_img_np = np.zeros_like(img_np)
green_img_np[:,:,1] = gg

blue_img_np = np.zeros_like(img_np)
blue_img_np[:,:,2] = bb

# Converte de volta para imagem PIL
red_img = Image.fromarray(red_img_np)
green_img = Image.fromarray(green_img_np)
blue_img = Image.fromarray(blue_img_np)

# Exibe ou salva
red_img.show()
green_img.show()
blue_img.show()
