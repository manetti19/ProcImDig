import cv2
import numpy as np


CAMINHO_IMAGEM = "top_mosaic_09cm_area23.tif"
SAIDA_ESQUERDA = "imagem_esquerda.png"
SAIDA_DIREITA = "imagem_direita.png"
SAIDA_ANAGLIFO = "anaglifo_vermelho_ciano.png"
DESLOCAMENTO = 10


def gerar_imagens_estereo(imagem, deslocamento):
    altura, largura = imagem.shape[:2]

    matriz_esquerda = np.float32([[1, 0, -deslocamento], [0, 1, 0]])
    matriz_direita = np.float32([[1, 0, deslocamento], [0, 1, 0]])

    imagem_esquerda = cv2.warpAffine(imagem, matriz_esquerda, (largura, altura))
    imagem_direita = cv2.warpAffine(imagem, matriz_direita, (largura, altura))

    return imagem_esquerda, imagem_direita


def criar_anaglifo(imagem_esquerda, imagem_direita):
    azul_dir, verde_dir, _ = cv2.split(imagem_direita)
    _, _, vermelho_esq = cv2.split(imagem_esquerda)

    return cv2.merge((azul_dir, verde_dir, vermelho_esq))


imagem_base = cv2.imread(CAMINHO_IMAGEM)

if imagem_base is None:
    raise FileNotFoundError(
        f"Nao foi possivel carregar a imagem base: {CAMINHO_IMAGEM}"
    )

imagem_esquerda, imagem_direita = gerar_imagens_estereo(imagem_base, DESLOCAMENTO)
anaglifo = criar_anaglifo(imagem_esquerda, imagem_direita)

cv2.imwrite(SAIDA_ESQUERDA, imagem_esquerda)
cv2.imwrite(SAIDA_DIREITA, imagem_direita)
cv2.imwrite(SAIDA_ANAGLIFO, anaglifo)

cv2.imshow("Imagem Esquerda", imagem_esquerda)
cv2.imshow("Imagem Direita", imagem_direita)
cv2.imshow("Imagem Anaglifo Vermelho-Ciano", anaglifo)
cv2.waitKey(0)
cv2.destroyAllWindows()
