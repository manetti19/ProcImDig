from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Cores RGB
verde = (0, 255, 0)
vermelho = (255, 0, 0)

# Tamanho do canvas
n = 300
# Criar imagem em tons de cinza e depois converter para RGB para permitir cores
canvas = Image.new("RGB", (n, n), (0, 0, 0))  # fundo preto
draw = ImageDraw.Draw(canvas)

# Desenha retângulo com borda verde
draw.rectangle([10, 70, 90, 190], outline=verde)
plt.imshow(canvas)
plt.title("Retângulo com borda verde")
plt.axis("off")
plt.show()

# Desenha retângulo preenchido de vermelho
draw.rectangle([250, 50, 300, 125], fill=vermelho)
plt.imshow(canvas)
plt.title("Retângulo preenchido de vermelho")
plt.axis("off")
plt.show()
