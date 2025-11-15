import numpy as np
from PIL import Image


# =======================================================
# CRIA FILTRO DE MÉDIA m×m
# =======================================================
def criar_filtro_media(m):
    return np.ones((m, m)) / (m * m)


# =======================================================
# CONVOLUÇÃO (MESMO MÉTODO DA TAREFA 2)
# =======================================================
def aplicar_convolucao(matriz, filtro):
    H, W = matriz.shape  # altura e largura reais
    m = filtro.shape[0]
    pad = m // 2

    # padding
    matriz_padded = np.pad(matriz, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    # saída com mesmas dimensões da imagem
    saida = np.zeros((H, W), dtype=float)

    # convolução
    for i in range(H):
        for j in range(W):
            janela = matriz_padded[i:i+m, j:j+m]

            # SE a janela não tiver tamanho completo, pule
            if janela.shape != filtro.shape:
                continue

            saida[i, j] = np.sum(janela * filtro)

    return saida



# =======================================================
# PROGRAMA PRINCIPAL DA TAREFA 3
# =======================================================

# --------------------------------------------------
# 1) Gerar matriz 100x100
# --------------------------------------------------
matriz = np.random.randint(1, 100*100 + 1, size=(100, 100))
np.savetxt("matriz_100x100_t3.txt", matriz, fmt="%d")
print("-> matriz_100x100_t3.txt gerada")

# --------------------------------------------------
# 2) Filtrar por 3 filtros diferentes
# --------------------------------------------------
filtros = [3, 10, 25]

for m in filtros:
    filtro = criar_filtro_media(m)
    matriz_filtrada = aplicar_convolucao(matriz, filtro)
    np.savetxt(f"matriz_100x100_t3_filtrada_{m}x{m}.txt", matriz_filtrada, fmt="%.5f")
    print(f"-> matriz_100x100_t3_filtrada_{m}x{m}.txt gerada")


# --------------------------------------------------
# 3) Filtrar imagem arroz.png com filtro 25×25
# --------------------------------------------------
print("\nProcessando imagem arroz.png ...")

# abre imagem e converte para escala de cinza
img = Image.open('./arroz.png').convert("L")
img_np = np.array(img, dtype=float)

filtro25 = criar_filtro_media(25)
img_filtrada_np = aplicar_convolucao(img_np, filtro25)

# normaliza valores para salvar como PNG
img_norm = (img_filtrada_np - img_filtrada_np.min()) / (img_filtrada_np.max() - img_filtrada_np.min())
img_norm = (img_norm * 255).astype(np.uint8)

img_out = Image.fromarray(img_norm)
img_out.save("arroz_filtrada_25x25.png")

print("-> arroz_filtrada_25x25.png gerada")
print("\nTarefa 3 concluída!")
