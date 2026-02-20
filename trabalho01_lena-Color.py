# 1) NumPy para trabalhar com matrizes
import numpy as np

# 2) Matplotlib para ler/salvar imagens PNG facilmente
import matplotlib.pyplot as plt

# 3) scikit-image para converter RGB -> Lab (base de ΔE)
from skimage import color


# 4) Função: lê CSV simples (sem pandas) no formato code,r,g,b
def ler_ral_csv(caminho_csv):
    # 4.1) lê todas as linhas do arquivo
    linhas = open(caminho_csv, "r", encoding="utf-8").read().splitlines()

    # 4.2) separa cabeçalho em colunas
    cab = linhas[0].split(",")

    # 4.3) encontra as posições das colunas necessárias
    idx_code = cab.index("code")
    idx_r = cab.index("r")
    idx_g = cab.index("g")
    idx_b = cab.index("b")

    # 4.4) cria listas para armazenar códigos e RGB
    codigos = []
    rgbs = []

    # 4.5) percorre as linhas de dados
    for linha in linhas[1:]:
        # 4.6) ignora linhas vazias
        if not linha.strip():
            continue

        # 4.7) separa por vírgula
        partes = linha.split(",")

        # 4.8) lê código e valores RGB
        codigo = partes[idx_code].strip()
        r = int(partes[idx_r])
        g = int(partes[idx_g])
        b = int(partes[idx_b])

        # 4.9) salva nas listas
        codigos.append(codigo)
        rgbs.append([r, g, b])

    # 4.10) converte a lista de RGB para array NumPy (N,3)
    rgbs = np.array(rgbs, dtype=np.float32)

    # 4.11) retorna códigos e RGB
    return codigos, rgbs


# 5) Função: RGB (0..1) -> CMY (0..1)
def rgb_para_cmy(rgb_01):
    # 5.1) C = 1 - R
    C = 1.0 - rgb_01[:, :, 0]

    # 5.2) M = 1 - G
    M = 1.0 - rgb_01[:, :, 1]

    # 5.3) Y = 1 - B
    Y = 1.0 - rgb_01[:, :, 2]

    # 5.4) empilha (C, M, Y)
    return np.dstack([C, M, Y])


# 6) Função: RGB (0..1) -> HSI (H em graus, S e I em 0..1)
def rgb_para_hsi(rgb_01):
    # 6.1) separa canais
    R = rgb_01[:, :, 0]
    G = rgb_01[:, :, 1]
    B = rgb_01[:, :, 2]

    # 6.2) intensidade
    I = (R + G + B) / 3.0

    # 6.3) saturação
    min_rgb = np.minimum(np.minimum(R, G), B)
    soma = R + G + B
    eps = 1e-12
    S = 1.0 - (3.0 * min_rgb / (soma + eps))

    # 6.4) hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + eps
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    H = np.where(B <= G, theta, (2.0 * np.pi - theta))
    H = np.degrees(H)

    # 6.5) retorna HSI
    return np.dstack([H, S, I])


# 7) Função: quantiza RGB para paleta RAL via ΔE76 em CIE Lab
def quantizar_para_ral(rgb_01, ral_rgb_255):
    # 7.1) paleta RAL em 0..1
    ral_rgb_01 = ral_rgb_255 / 255.0

    # 7.2) converte paleta para Lab
    ral_lab = color.rgb2lab(ral_rgb_01.reshape(1, -1, 3)).reshape(-1, 3)

    # 7.3) converte imagem para Lab
    img_lab = color.rgb2lab(rgb_01)

    # 7.4) achata para (P,3)
    Hh, Ww, _ = img_lab.shape
    px = img_lab.reshape(-1, 3)

    # 7.5) saída: índice do RAL mais próximo
    idx_out = np.empty((px.shape[0],), dtype=np.int32)

    # 7.6) processa em blocos para evitar muita memória
    bloco = 200000

    # 7.7) loop por blocos
    for ini in range(0, px.shape[0], bloco):
        fim = min(ini + bloco, px.shape[0])
        parte = px[ini:fim, :]

        # 7.8) distância ΔE76 para todas as cores RAL
        dist = np.sqrt(
            (parte[:, None, 0] - ral_lab[None, :, 0]) ** 2 +
            (parte[:, None, 1] - ral_lab[None, :, 1]) ** 2 +
            (parte[:, None, 2] - ral_lab[None, :, 2]) ** 2
        )

        # 7.9) menor distância -> índice
        idx_out[ini:fim] = np.argmin(dist, axis=1)

    # 7.10) remodela para imagem (H,W)
    idx_img = idx_out.reshape(Hh, Ww)

    # 7.11) gera RGB quantizado (0..255)
    rgb_quant_255 = ral_rgb_255[idx_img]

    # 7.12) volta para 0..1
    rgb_quant_01 = rgb_quant_255 / 255.0

    # 7.13) retorna índice e imagem quantizada
    return idx_img, rgb_quant_01


# =========================
# 8) AJUSTE OS NOMES AQUI
# =========================

# 8.1) arquivo PNG da Lena
caminho_lena = r".\lena-Color.png"

# 8.2) tabela RAL (CSV) na mesma pasta
caminho_ral_csv = r".\ral_classic_complete.csv"


# 9) lê a Lena com matplotlib (pode vir como float 0..1 ou uint8)
img = plt.imread(caminho_lena)

# 10) garante que está em float 0..1
if img.dtype != np.float32 and img.dtype != np.float64:
    img = img.astype(np.float32) / 255.0

# 11) se tiver canal alfa (RGBA), remove e fica só RGB
if img.shape[2] == 4:
    img = img[:, :, :3]

# 12) garante faixa 0..1
rgb = np.clip(img, 0.0, 1.0)

# =========================
# d.1) Separar RGB
# =========================
R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

plt.imsave("lena_R_nova.png", R, cmap="gray")
plt.imsave("lena_G_nova.png", G, cmap="gray")
plt.imsave("lena_B_nova.png", B, cmap="gray")

# =========================
# d.2) Separar CMY
# =========================
cmy = rgb_para_cmy(rgb)
C = cmy[:, :, 0]
M = cmy[:, :, 1]
Y = cmy[:, :, 2]

plt.imsave("lena_C_nova.png", C, cmap="gray")
plt.imsave("lena_M_nova.png", M, cmap="gray")
plt.imsave("lena_Y_nova.png", Y, cmap="gray")

# =========================
# d.3) Recompor RGB
# =========================
rgb_recomposta = np.dstack([R, G, B])
plt.imsave("lena_rgb_recomposta_nova.png", np.clip(rgb_recomposta, 0.0, 1.0))

# =========================
# d.4) RGB -> HSI
# =========================
hsi = rgb_para_hsi(rgb)
H = hsi[:, :, 0]          # 0..360
S = hsi[:, :, 1]          # 0..1
I = hsi[:, :, 2]          # 0..1

plt.imsave("lena_H_nova.png", (H / 360.0), cmap="gray")   # normaliza H para 0..1 só pra visualizar
plt.imsave("lena_S_nova.png", np.clip(S, 0.0, 1.0), cmap="gray")
plt.imsave("lena_I_nova.png", np.clip(I, 0.0, 1.0), cmap="gray")

# =========================
# d.5) RGB -> RAL (quantização)
# =========================

# 13) lê a paleta RAL do CSV
codigos_ral, ral_rgb_255 = ler_ral_csv(caminho_ral_csv)

# 14) quantiza a Lena para RAL
idx_ral, lena_ral = quantizar_para_ral(rgb, ral_rgb_255)

# 15) salva imagem final quantizada para RAL
plt.imsave("lena_ral_nova.png", np.clip(lena_ral, 0.0, 1.0))

# 16) salva raster de índices em formato simples (NumPy) para checagem
np.save("lena_indice_ral_nova.npy", idx_ral)

# 17) salva legenda (índice -> código RAL)
with open("lena_indice_ral_legenda_nova.txt", "w", encoding="utf-8") as f:
    for i, cod in enumerate(codigos_ral):
        f.write(f"{i} = {cod}\n")

print("Pronto! Gerou PNGs dos canais, HSI e lena_ral.png + índice/legenda.")
