from PIL import Image
import numpy as np
import time

from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_closing
)

from scipy.ndimage import (
    binary_fill_holes,
    binary_hit_or_miss
)

from PIL import Image
import numpy as np


# ============================================================
# FUNÇÕES AUXILIARES
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
# IMPLEMENTAÇÕES MANUAIS
# ============================================================

def dilatacao_binaria(img_binaria, elemento_estruturante):
    linhas_img = img_binaria.shape[0]
    colunas_img = img_binaria.shape[1]

    linhas_elem = elemento_estruturante.shape[0]
    colunas_elem = elemento_estruturante.shape[1]

    centro_linha = linhas_elem // 2
    centro_coluna = colunas_elem // 2

    elemento_refletido = refletir_elemento_estruturante(elemento_estruturante)

    resultado = np.zeros_like(img_binaria, dtype=bool)

    for i in range(linhas_img):
        for j in range(colunas_img):
            if img_binaria[i, j]:
                for m in range(linhas_elem):
                    for n in range(colunas_elem):
                        if elemento_refletido[m, n]:
                            x = i + (m - centro_linha)
                            y = j + (n - centro_coluna)

                            if 0 <= x < linhas_img and 0 <= y < colunas_img:
                                resultado[x, y] = True

    return resultado


def erosao_binaria(img_binaria, elemento_estruturante):
    linhas_img = img_binaria.shape[0]
    colunas_img = img_binaria.shape[1]

    linhas_elem = elemento_estruturante.shape[0]
    colunas_elem = elemento_estruturante.shape[1]

    centro_linha = linhas_elem // 2
    centro_coluna = colunas_elem // 2

    resultado = np.zeros_like(img_binaria, dtype=bool)

    for i in range(linhas_img):
        for j in range(colunas_img):
            encaixa_totalmente = True

            for m in range(linhas_elem):
                for n in range(colunas_elem):
                    if elemento_estruturante[m, n]:
                        x = i + (m - centro_linha)
                        y = j + (n - centro_coluna)

                        if not (0 <= x < linhas_img and 0 <= y < colunas_img):
                            encaixa_totalmente = False
                            break

                        if not img_binaria[x, y]:
                            encaixa_totalmente = False
                            break

                if not encaixa_totalmente:
                    break

            if encaixa_totalmente:
                resultado[i, j] = True

    return resultado


def abertura_binaria(img_binaria, elemento_estruturante):
    img_erodida = erosao_binaria(img_binaria, elemento_estruturante)
    img_aberta = dilatacao_binaria(img_erodida, elemento_estruturante)
    return img_aberta


def fechamento_binaria(img_binaria, elemento_estruturante):
    img_dilatada = dilatacao_binaria(img_binaria, elemento_estruturante)
    img_fechada = erosao_binaria(img_dilatada, elemento_estruturante)
    return img_fechada


def hit_or_miss_manual(img_binaria, elemento_objeto, elemento_fundo):
    erosao_objeto = erosao_binaria(img_binaria, elemento_objeto)
    complemento = np.logical_not(img_binaria)
    erosao_fundo = erosao_binaria(complemento, elemento_fundo)
    resultado = np.logical_and(erosao_objeto, erosao_fundo)
    return resultado



def preenchimento_buracos_manual(img_binaria, elemento_estruturante):
    # Calcula o complemento da imagem:
    # True onde há fundo da imagem original
    complemento = np.logical_not(img_binaria)

    # Cria a imagem inicial que vai marcar o fundo conectado à borda
    marcador = np.zeros_like(img_binaria, dtype=bool)

    # Número de linhas da imagem
    linhas = img_binaria.shape[0]

    # Número de colunas da imagem
    colunas = img_binaria.shape[1]

    # Marca os pixels de fundo da borda superior
    marcador[0, :] = complemento[0, :]

    # Marca os pixels de fundo da borda inferior
    marcador[linhas - 1, :] = complemento[linhas - 1, :]

    # Marca os pixels de fundo da borda esquerda
    marcador[:, 0] = np.logical_or(marcador[:, 0], complemento[:, 0])

    # Marca os pixels de fundo da borda direita
    marcador[:, colunas - 1] = np.logical_or(marcador[:, colunas - 1], complemento[:, colunas - 1])

    # Processo iterativo para reconstruir o fundo conectado à borda
    while True:
        # Dilata o marcador atual
        marcador_dilatado = dilatacao_binaria(marcador, elemento_estruturante)

        # Restringe o crescimento ao complemento da imagem original
        novo_marcador = np.logical_and(marcador_dilatado, complemento)

        # Se não houve alteração, encerra
        if np.array_equal(novo_marcador, marcador):
            break

        # Atualiza o marcador
        marcador = novo_marcador

    # Os buracos são as partes do complemento que NÃO estão conectadas à borda
    buracos = np.logical_and(complemento, np.logical_not(marcador))

    # Preenche os buracos somando-os à imagem original
    resultado = np.logical_or(img_binaria, buracos)

    # Retorna a imagem final
    return resultado


# ============================================================
# FUNÇÕES DA BIBLIOTECA
# ============================================================

def dilatacao_skimage(img_binaria, elemento_estruturante):
    return binary_dilation(img_binaria, footprint=elemento_estruturante)


def erosao_skimage(img_binaria, elemento_estruturante):
    return binary_erosion(img_binaria, footprint=elemento_estruturante)


def abertura_skimage(img_binaria, elemento_estruturante):
    return binary_opening(img_binaria, footprint=elemento_estruturante)


def fechamento_skimage(img_binaria, elemento_estruturante):
    return binary_closing(img_binaria, footprint=elemento_estruturante)


def hit_or_miss_skimage(img_binaria, elemento_objeto, elemento_fundo):
    return binary_hit_or_miss(
        img_binaria,
        structure1=elemento_objeto,
        structure2=elemento_fundo
    )

def preenchimento_buracos_scipy(img_binaria):
    return binary_fill_holes(img_binaria)


# ============================================================
# MEDIÇÃO DE TEMPO
# ============================================================

def medir_tempo(funcao, *args, repeticoes=5):
    tempos = []

    for _ in range(repeticoes):
        inicio = time.perf_counter()
        funcao(*args)
        fim = time.perf_counter()
        tempos.append(fim - inicio)

    return min(tempos), sum(tempos) / len(tempos), max(tempos)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

caminho_entrada = "./lena-Color.png"

img = carregar_imagem_binaria(caminho_entrada)

elemento = criar_elemento_estruturante_quadrado(3)

elemento_objeto = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=bool)

elemento_fundo = np.array([
    [1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]
], dtype=bool)

elemento_objeto_2 = np.array([
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0]
], dtype=bool)

elemento_fundo_2 = np.array([
    [1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 1]
], dtype=bool)

resultados = []

# Dilatação
tempo_manual = medir_tempo(dilatacao_binaria, img, elemento, repeticoes=5)
tempo_lib = medir_tempo(dilatacao_skimage, img, elemento, repeticoes=5)
resultados.append(("Dilatação", tempo_manual, tempo_lib))

# Erosão
tempo_manual = medir_tempo(erosao_binaria, img, elemento, repeticoes=5)
tempo_lib = medir_tempo(erosao_skimage, img, elemento, repeticoes=5)
resultados.append(("Erosão", tempo_manual, tempo_lib))

# Abertura
tempo_manual = medir_tempo(abertura_binaria, img, elemento, repeticoes=5)
tempo_lib = medir_tempo(abertura_skimage, img, elemento, repeticoes=5)
resultados.append(("Abertura", tempo_manual, tempo_lib))

# Fechamento
tempo_manual = medir_tempo(fechamento_binaria, img, elemento, repeticoes=5)
tempo_lib = medir_tempo(fechamento_skimage, img, elemento, repeticoes=5)
resultados.append(("Fechamento", tempo_manual, tempo_lib))

# Hit-or-Miss
tempo_manual = medir_tempo(hit_or_miss_manual, img, elemento_objeto, elemento_fundo, repeticoes=5)
tempo_lib = medir_tempo(hit_or_miss_skimage, img, elemento_objeto, elemento_fundo, repeticoes=5)
tempo_manual = medir_tempo(hit_or_miss_manual, img, elemento_objeto_2, elemento_fundo_2, repeticoes=5)
tempo_lib = medir_tempo(hit_or_miss_skimage, img, elemento_objeto_2, elemento_fundo_2, repeticoes=5)

resultados.append(("Hit-or-Miss", tempo_manual, tempo_lib))

# Preenchimento de buracos
tempo_manual = medir_tempo(preenchimento_buracos_manual, img, elemento, repeticoes=5)
tempo_lib = medir_tempo(preenchimento_buracos_scipy, img, repeticoes=5)
resultados.append(("Preenchimento de buracos", tempo_manual, tempo_lib))

# Impressão dos resultados
print("\nCOMPARAÇÃO DE TEMPOS\n")

for nome_operacao, tempo_manual, tempo_lib in resultados:
    manual_min, manual_medio, manual_max = tempo_manual
    lib_min, lib_medio, lib_max = tempo_lib

    aceleracao = manual_medio / lib_medio if lib_medio > 0 else float("inf")

    print(f"Operação: {nome_operacao}")
    print(f"  Manual   -> min: {manual_min:.6f}s | médio: {manual_medio:.6f}s | max: {manual_max:.6f}s")
    print(f"  Biblioteca -> min: {lib_min:.6f}s | médio: {lib_medio:.6f}s | max: {lib_max:.6f}s")
    print(f"  Fator de aceleração (manual / biblioteca): {aceleracao:.2f}x")
    print("-" * 70)


    # ============================================================
# GERAR E SALVAR RESULTADOS VISUAIS
# ============================================================

# Salva a imagem original binária
salvar_imagem_binaria(img, "original_binaria.png")

# ----------------------------
# Dilatação
# ----------------------------
resultado_dilatacao_manual = dilatacao_binaria(img, elemento)
resultado_dilatacao_lib = dilatacao_skimage(img, elemento)

salvar_imagem_binaria(resultado_dilatacao_manual, "dilatacao_manual.png")
salvar_imagem_binaria(resultado_dilatacao_lib, "dilatacao_biblioteca.png")

# ----------------------------
# Erosão
# ----------------------------
resultado_erosao_manual = erosao_binaria(img, elemento)
resultado_erosao_lib = erosao_skimage(img, elemento)

salvar_imagem_binaria(resultado_erosao_manual, "erosao_manual.png")
salvar_imagem_binaria(resultado_erosao_lib, "erosao_biblioteca.png")

# ----------------------------
# Abertura
# ----------------------------
resultado_abertura_manual = abertura_binaria(img, elemento)
resultado_abertura_lib = abertura_skimage(img, elemento)

salvar_imagem_binaria(resultado_abertura_manual, "abertura_manual.png")
salvar_imagem_binaria(resultado_abertura_lib, "abertura_biblioteca.png")

# ----------------------------
# Fechamento
# ----------------------------
resultado_fechamento_manual = fechamento_binaria(img, elemento)
resultado_fechamento_lib = fechamento_skimage(img, elemento)

salvar_imagem_binaria(resultado_fechamento_manual, "fechamento_manual.png")
salvar_imagem_binaria(resultado_fechamento_lib, "fechamento_biblioteca.png")

# ----------------------------
# Hit-or-Miss
# ----------------------------
resultado_hitmiss_manual = hit_or_miss_manual(img, elemento_objeto, elemento_fundo)
resultado_hitmiss_lib = hit_or_miss_skimage(img, elemento_objeto, elemento_fundo)
resultado_hitmiss_manual_2 = hit_or_miss_manual(img, elemento_objeto_2, elemento_fundo_2)
resultado_hitmiss_lib_2 = hit_or_miss_skimage(img, elemento_objeto_2, elemento_fundo_2)

salvar_imagem_binaria(resultado_hitmiss_manual, "hit_or_miss_manual.png")
salvar_imagem_binaria(resultado_hitmiss_lib, "hit_or_miss_biblioteca.png")
salvar_imagem_binaria(resultado_hitmiss_manual_2, "hit_or_miss_manual_2.png")
salvar_imagem_binaria(resultado_hitmiss_lib_2, "hit_or_miss_biblioteca_2.png")

# ----------------------------
# Preenchimento de buracos
# ----------------------------
resultado_preenchimento_manual = preenchimento_buracos_manual(img, elemento)
resultado_preenchimento_lib = preenchimento_buracos_scipy(img)

salvar_imagem_binaria(resultado_preenchimento_manual, "preenchimento_manual.png")
salvar_imagem_binaria(resultado_preenchimento_lib, "preenchimento_biblioteca.png")