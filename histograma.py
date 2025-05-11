from PIL import Image
import matplotlib.pyplot as plt

# Abre a imagem e converte para RGB
img = Image.open('./arroz.png').convert('RGB')

# Obtém o histograma (lista com 256 valores por canal: R, G e B)
hist = img.histogram()

# Separa os canais
r = hist[0:256]
g = hist[256:512]
b = hist[512:768]

# Plota os histogramas
plt.figure(figsize=(10, 4))
plt.plot(r, color='red')
plt.plot(g, color='green')
plt.plot(b, color='blue')
plt.title('Histograma RGB')
plt.xlabel('Valor do pixel')
plt.ylabel('Frequência')
plt.grid()
plt.show()


# Abre a imagem e converte para tons de cinza
img2 = Image.open('./arroz.png').convert('L')

# Obtém o histograma (256 valores para níveis de cinza de 0 a 255)
hist = img2.histogram()

# Plota o histograma
plt.figure(figsize=(8, 4))
plt.bar(range(256), hist, color='gray', width=1)
plt.title('Histograma em tons de cinza')
plt.xlabel('Valor de pixel (0-255)')
plt.ylabel('Frequência')
plt.grid(True, linestyle='--', alpha=0.4)
plt.show()
