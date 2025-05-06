from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt

# Carrega imagem e converte para PIL RGB
image_pil = Image.open('./arroz.png').convert('RGB')

# Converte imagem para array NumPy
image_np = np.array(image_pil)

# Função para adicionar ruído sal e pimenta
def salt_and_pepper_noise(img_np, prob):
    output = np.copy(img_np)
    rnd = np.random.rand(*img_np.shape[:2])

    # Aplica sal e pimenta nos 3 canais RGB
    output[rnd < prob / 2] = 0        # Pimenta (preto)
    output[rnd > 1 - prob / 2] = 255  # Sal (branco)
    return output

# Adiciona ruído e converte de volta para imagem PIL
noisy_np = salt_and_pepper_noise(image_np, 0.05).astype(np.uint8)
noisy_pil = Image.fromarray(noisy_np)

# Aplica os filtros
blur_pil = noisy_pil.filter(ImageFilter.BLUR)                        # Filtro de média
gaussian_pil = noisy_pil.filter(ImageFilter.GaussianBlur(radius=2)) # Filtro Gaussiano
median_pil = noisy_pil.filter(ImageFilter.MedianFilter(size=5))     # Filtro de Mediana

# Detecção de bordas (substituto simples para Canny)
edges_pil = image_pil.filter(ImageFilter.FIND_EDGES)

# Exibição lado a lado
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image_pil)
axs[0, 0].set_title("Original")
axs[0, 0].axis('off')

axs[0, 1].imshow(noisy_pil)
axs[0, 1].set_title("Imagem com Ruído S&P")
axs[0, 1].axis('off')

axs[0, 2].imshow(blur_pil)
axs[0, 2].set_title("Filtro de Média (BLUR)")
axs[0, 2].axis('off')

axs[1, 0].imshow(gaussian_pil)
axs[1, 0].set_title("Filtro Gaussiano")
axs[1, 0].axis('off')

axs[1, 1].imshow(median_pil)
axs[1, 1].set_title("Filtro Mediana")
axs[1, 1].axis('off')

axs[1, 2].imshow(edges_pil)
axs[1, 2].set_title("Detector de Bordas (FIND_EDGES)")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
