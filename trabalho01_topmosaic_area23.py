# 1) Importa o NumPy para trabalhar com matrizes (imagens)
import numpy as np

# 2) Importa rasterio para ler TIFF/GeoTIFF (multibanda, comum em cartografia)
import rasterio

# 3) Importa matplotlib para visualizar e salvar imagens
import matplotlib.pyplot as plt

# 4) scikit-image para converter RGB -> Lab de forma padrão (sRGB)
from skimage import color


# 4) Função auxiliar: converte uma banda (2D) para o intervalo 0..1 (float)
def normalizar_0_1(banda_2d):
    # 4.1) Converte para float32 para permitir contas e evitar overflow
    banda = banda_2d.astype(np.float32)

    # 4.2) Encontra mínimo da banda
    mn = np.nanmin(banda)

    # 4.3) Encontra máximo da banda
    mx = np.nanmax(banda)

    # 4.4) Evita divisão por zero quando a imagem é constante
    eps = 1e-12

    # 4.5) Normaliza para 0..1
    return (banda - mn) / (mx - mn + eps)


# 5) Função: converte RGB (0..1) para HSI
def rgb_para_hsi(rgb_01):
    # 5.1) Separa R, G, B (cada um 2D) a partir do array (H, W, 3)
    R = rgb_01[:, :, 0]
    G = rgb_01[:, :, 1]
    B = rgb_01[:, :, 2]

    # 5.2) Calcula Intensidade I = (R+G+B)/3
    I = (R + G + B) / 3.0

    # 5.3) Calcula o mínimo por pixel entre R, G, B (para S)
    min_rgb = np.minimum(np.minimum(R, G), B)

    # 5.4) Soma RGB por pixel (para S)
    soma = R + G + B

    # 5.5) Evita divisão por zero quando soma=0 (pixel preto)
    eps = 1e-12

    # 5.6) Saturação: S = 1 - 3 * min(R,G,B) / (R+G+B)
    S = 1.0 - (3.0 * min_rgb / (soma + eps))

    # 5.7) Termos para Hue (H) usando arccos
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + eps

    # 5.8) Ângulo theta em radianos
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    # 5.9) Se B <= G então H = theta; senão H = 2π - theta
    H = np.where(B <= G, theta, (2.0 * np.pi - theta))

    # 5.10) Converte Hue de radianos para graus (0..360)
    H = np.degrees(H)

    # 5.11) Empilha H, S, I em um array (H, W, 3)
    hsi = np.dstack([H, S, I])

    # 5.12) Retorna o HSI
    return hsi


# 6) Função: converte RGB (0..1) para CMY (0..1) via modelo subtrativo simples
def rgb_para_cmy(rgb_01):
    # 6.1) C = 1 - R
    C = 1.0 - rgb_01[:, :, 0]

    # 6.2) M = 1 - G
    M = 1.0 - rgb_01[:, :, 1]

    # 6.3) Y = 1 - B
    Y = 1.0 - rgb_01[:, :, 2]

    # 6.4) Empilha (C, M, Y)
    cmy = np.dstack([C, M, Y])

    # 6.5) Retorna CMY
    return cmy


# 8) Função auxiliar: salva uma imagem 2D (grayscale) em PNG
def salvar_banda_png(banda_2d_01, titulo, caminho_saida):
    # 8.1) Cria figura
    plt.figure()

    # 8.2) Mostra banda em tons de cinza
    plt.imshow(banda_2d_01, cmap="gray")

    # 8.3) Coloca título
    plt.title(titulo)

    # 8.4) Remove eixos
    plt.axis("off")

    # 8.5) Salva em arquivo
    plt.savefig(caminho_saida, dpi=200, bbox_inches="tight")

    # 8.6) Fecha figura para não consumir memória
    plt.close()


# 9) Função auxiliar: salva uma imagem RGB (0..1) em PNG
def salvar_rgb_png(rgb_01, titulo, caminho_saida):
    # 9.1) Cria figura
    plt.figure()

    # 9.2) Mostra RGB
    plt.imshow(np.clip(rgb_01, 0.0, 1.0))

    # 9.3) Coloca título
    plt.title(titulo)

    # 9.4) Remove eixos
    plt.axis("off")

    # 9.5) Salva em arquivo
    plt.savefig(caminho_saida, dpi=200, bbox_inches="tight")

    # 9.6) Fecha figura
    plt.close()


# 10) Caminho do seu TIFF (ajuste aqui)
caminho_tiff = r".\top_mosaic_09cm_area23.tif"

# 11) Abre o TIFF com rasterio
with rasterio.open(caminho_tiff) as src:
    # 11.1) Lê todas as bandas como array (bandas, altura, largura)
    arr = src.read()

# 12) Checa quantas bandas existem
num_bandas = arr.shape[0]

# 13) Garante que temos pelo menos 3 bandas para RGB
if num_bandas < 3:
    raise ValueError("Seu TIFF tem menos de 3 bandas. Não dá para formar um RGB diretamente.")

# 14) Extrai as 3 primeiras bandas como R, G, B (ajuste se seu RGB estiver em outra ordem)
R_raw = arr[0, :, :]
G_raw = arr[1, :, :]
B_raw = arr[2, :, :]

# 15) Normaliza cada banda para 0..1 (isso ajuda muito quando é 16-bit)
R = normalizar_0_1(R_raw)
G = normalizar_0_1(G_raw)
B = normalizar_0_1(B_raw)

# 16) Monta a imagem RGB (H, W, 3)
rgb = np.dstack([R, G, B])

# =========================
# d.1) Separar nos canais RGB
# =========================

# 17) Salva cada canal RGB separado (grayscale)
salvar_banda_png(R, "Canal R", "canal_R_novo.png")
salvar_banda_png(G, "Canal G", "canal_G_novo.png")
salvar_banda_png(B, "Canal B", "canal_B_novo.png")

# =========================
# d.2) Separar nos canais CMY
# =========================

# 18) Converte RGB para CMY (0..1)
cmy = rgb_para_cmy(rgb)

# 19) Separa C, M, Y
C = cmy[:, :, 0]
M = cmy[:, :, 1]
Yc = cmy[:, :, 2]

# 20) Salva C, M, Y separados
salvar_banda_png(C, "Canal C (Ciano)", "canal_C_novo.png")
salvar_banda_png(M, "Canal M (Magenta)", "canal_M_novo.png")
salvar_banda_png(Yc, "Canal Y (Amarelo)", "canal_Y_novo.png")

# =========================
# d.3) A partir da decomposição RGB, gerar uma imagem colorida
# (recomposição: volta a juntar R,G,B)
# =========================

# 21) Recomposição RGB (na prática, é o próprio rgb)
rgb_recomposta = np.dstack([R, G, B])

# 22) Salva a imagem recomposta
salvar_rgb_png(rgb_recomposta, "RGB Recomposta", "rgb_recomposta_novo.png")

# =========================
# d.4) Transformar RGB -> HSI
# =========================

# 23) Converte RGB (0..1) para HSI
hsi = rgb_para_hsi(rgb)

# 24) Separa H, S, I
H = hsi[:, :, 0]  # graus (0..360)
S = hsi[:, :, 1]  # 0..1
I = hsi[:, :, 2]  # 0..1 (pois RGB está em 0..1)

# 25) Para salvar Hue como imagem, normaliza H (0..360) para 0..1
H_01 = H / 360.0

# 26) Salva H, S, I
salvar_banda_png(H_01, "Hue (H) normalizado", "canal_H_novo.png")
salvar_banda_png(S, "Saturation (S)", "canal_S_novo.png")
salvar_banda_png(I, "Intensity (I)", "canal_I_novo.png")



# 5) Função: normaliza uma banda para 0..1 (float), útil se for 16-bit
def normalizar_0_1(banda_2d):
    # 5.1) converte para float
    banda = banda_2d.astype(np.float32)

    # 5.2) mínimo e máximo ignorando NaN
    mn = np.nanmin(banda)
    mx = np.nanmax(banda)

    # 5.3) epsilon para evitar divisão por zero
    eps = 1e-12

    # 5.4) normaliza
    return (banda - mn) / (mx - mn + eps)


# 6) Função: lê CSV simples sem pandas (para evitar dependência extra)
def ler_ral_csv(caminho_csv):
    # 6.1) carrega todas as linhas do arquivo (texto)
    linhas = open(caminho_csv, "r", encoding="utf-8").read().splitlines()

    # 6.2) remove cabeçalho
    cab = linhas[0].split(",")

    # 6.3) acha índices das colunas importantes
    idx_code = cab.index("code")
    idx_r = cab.index("r")
    idx_g = cab.index("g")
    idx_b = cab.index("b")

    # 6.4) listas para armazenar
    codigos = []
    rgbs = []

    # 6.5) percorre linhas de dados
    for linha in linhas[1:]:
        # 6.6) pula linhas vazias
        if not linha.strip():
            continue

        # 6.7) separa colunas
        partes = linha.split(",")

        # 6.8) lê código e RGB
        codigo = partes[idx_code].strip()
        r = int(partes[idx_r])
        g = int(partes[idx_g])
        b = int(partes[idx_b])

        # 6.9) guarda
        codigos.append(codigo)
        rgbs.append([r, g, b])

    # 6.10) converte lista RGB para array NumPy (N,3)
    rgbs = np.array(rgbs, dtype=np.float32)

    # 6.11) retorna códigos e RGB
    return codigos, rgbs


# 7) Função: quantiza imagem RGB para a paleta RAL usando ΔE (Lab euclidiano)
def quantizar_para_ral(rgb_01, ral_rgb_255):
    # 7.1) converte tabela RAL RGB 0..255 para 0..1
    ral_rgb_01 = ral_rgb_255 / 255.0

    # 7.2) converte paleta RAL (N,3) para Lab (N,3)
    ral_lab = color.rgb2lab(ral_rgb_01.reshape(1, -1, 3)).reshape(-1, 3)

    # 7.3) converte imagem RGB (H,W,3) para Lab (H,W,3)
    img_lab = color.rgb2lab(rgb_01)

    # 7.4) achata imagem para (P,3), onde P = H*W
    H, W, _ = img_lab.shape
    img_lab_flat = img_lab.reshape(-1, 3)

    # 7.5) cria array de saída para o índice do RAL por pixel
    idx_out = np.empty((img_lab_flat.shape[0],), dtype=np.int32)

    # 7.6) processa em blocos para não estourar memória (ajuste se quiser)
    bloco = 200000  # 200k pixels por vez

    # 7.7) percorre a imagem em blocos
    for ini in range(0, img_lab_flat.shape[0], bloco):
        # 7.8) fim do bloco
        fim = min(ini + bloco, img_lab_flat.shape[0])

        # 7.9) pega o bloco de pixels Lab (B,3)
        px = img_lab_flat[ini:fim, :]

        # 7.10) calcula ΔE76: distância euclidiana até cada cor RAL
        #      resultado vira (B,N)
        dist = np.sqrt(
            (px[:, None, 0] - ral_lab[None, :, 0]) ** 2 +
            (px[:, None, 1] - ral_lab[None, :, 1]) ** 2 +
            (px[:, None, 2] - ral_lab[None, :, 2]) ** 2
        )

        # 7.11) escolhe o índice do menor ΔE em cada pixel
        idx_out[ini:fim] = np.argmin(dist, axis=1)

    # 7.12) remodela índices para (H,W)
    idx_img = idx_out.reshape(H, W)

    # 7.13) cria imagem RGB quantizada escolhendo a cor RAL do índice
    rgb_quant_255 = ral_rgb_255[idx_img]  # (H,W,3) em 0..255

    # 7.14) volta para 0..1 para visualizar com matplotlib
    rgb_quant_01 = rgb_quant_255 / 255.0

    # 7.15) retorna índice e imagem quantizada
    return idx_img, rgb_quant_01


# =========================
# 8) AJUSTE CAMINHOS AQUI
# =========================

# 8.2) caminho do CSV com a tabela RAL
caminho_ral_csv = r".\ral_classic_complete.csv"


# 9) lê tabela RAL (códigos e RGB)
codigos_ral, ral_rgb_255 = ler_ral_csv(caminho_ral_csv)

# 10) abre o TIFF
with rasterio.open(caminho_tiff) as src:
    # 10.1) lê bandas (bandas, H, W)
    arr = src.read()

    # 10.2) copia metadados para salvar GeoTIFF depois
    meta = src.meta.copy()

# 11) verifica se tem 3 bandas para RGB
if arr.shape[0] < 3:
    raise ValueError("Seu TIFF tem menos de 3 bandas. Não dá para formar RGB automaticamente.")

# 12) pega as 3 primeiras bandas como R,G,B (ajuste se sua ordem for diferente)
R_raw = arr[0, :, :]
G_raw = arr[1, :, :]
B_raw = arr[2, :, :]

# 13) normaliza cada banda para 0..1
R = normalizar_0_1(R_raw)
G = normalizar_0_1(G_raw)
B = normalizar_0_1(B_raw)

# 14) monta RGB (H,W,3)
rgb = np.dstack([R, G, B])

# 15) quantiza para RAL
idx_ral, rgb_ral = quantizar_para_ral(rgb, ral_rgb_255)

# 16) salva PNG para ver o resultado “RALizado”
plt.figure()
plt.imshow(np.clip(rgb_ral, 0.0, 1.0))
plt.title("Imagem quantizada para RAL (aproximação)")
plt.axis("off")
plt.savefig("imagem_ral_novo.png", dpi=200, bbox_inches="tight")
plt.close()

# 17) salva GeoTIFF com o índice RAL (um raster 1 banda)
meta_out = meta.copy()
meta_out.update(count=1, dtype=rasterio.int32)

with rasterio.open("indice_ral_novo.tif", "w", **meta_out) as dst:
    dst.write(idx_ral.astype(np.int32), 1)

# 18) cria um “dicionário” (arquivo texto) para interpretar os índices
with open("indice_ral_legenda_novo.txt", "w", encoding="utf-8") as f:
    for i, cod in enumerate(codigos_ral):
        f.write(f"{i} = {cod}\n")

# 19) confirma
print("Pronto!")
print("Arquivos gerados:")
print("- imagem_ral_novo.png (visualização)")
print("- indice_ral_novo.tif (raster com índice RAL por pixel)")
print("- indice_ral_legenda_novo.txt (mapa índice -> código RAL)")



with rasterio.open(caminho_tiff) as src:
    print("Color interpretation:", src.colorinterp)
    print("Descriptions:", src.descriptions)




with rasterio.open(caminho_tiff) as src:
    arr = src.read().astype(np.float32)

R = arr[0]
G = arr[1]
B = arr[2]

# diferença normalizada entre R e G (tipo um "pseudo-índice")
idx = (R - G) / (R + G + 1e-12)

plt.figure()
plt.imshow(idx, cmap="gray")
plt.title("(R - G) / (R + G)  (se vegetação ficar muito clara, R domina)")
plt.axis("off")
plt.show()