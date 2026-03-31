from PIL import Image
import numpy as np
from scipy.ndimage import binary_hit_or_miss


# ============================================================

def carregar_imagem_binaria(caminho_imagem):
    img = Image.open(caminho_imagem)
    img = img.convert("L")
    img_array = np.array(img)
    img_binaria = img_array > 127
    return img_binaria


def refletir_elemento_estruturante(elemento):
    return np.flipud(np.fliplr(elemento))


def criar_elemento_estruturante_quadrado(tamanho):
    return np.ones((tamanho, tamanho), dtype=bool)



def salvar_imagem_binaria(img_binaria, caminho_saida):
    img_uint8 = (img_binaria.astype(np.uint8)) * 255
    img = Image.fromarray(img_uint8)
    img.save(caminho_saida)



# ============================================================



def hit_or_miss_skimage(img_binaria, elemento_objeto, elemento_fundo):
    return binary_hit_or_miss(
        img_binaria,
        structure1=elemento_objeto,
        structure2=elemento_fundo
    )


# ============================================================

caminho_entrada = "./lena-Color.png"

img = carregar_imagem_binaria(caminho_entrada)

elemento = criar_elemento_estruturante_quadrado(3)

elemento_objeto = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=bool)

elemento_fundo = np.array([
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
], dtype=bool)

elemento_objeto_2 = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=bool)

elemento_fundo_2 = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
], dtype=bool)


# ----------------------------
# Hit-or-Miss
# ----------------------------
resultado_hitmiss_lib = hit_or_miss_skimage(img, elemento_objeto, elemento_fundo)
resultado_hitmiss_lib_2 = hit_or_miss_skimage(img, elemento_objeto_2, elemento_fundo_2)

salvar_imagem_binaria(resultado_hitmiss_lib, "hit_or_miss_biblioteca.png")
salvar_imagem_binaria(resultado_hitmiss_lib_2, "hit_or_miss_biblioteca_2.png")
