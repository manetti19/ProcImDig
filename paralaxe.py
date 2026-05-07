from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.feature import SIFT, match_descriptors

# 1. Parametros da camera
x0 = -3.24
y0 = 0.914
tp_mm = 3.3 / 1000.0

base_dir = Path(__file__).resolve().parent
saida_dir = base_dir / "saida_paralaxe"
saida_dir.mkdir(exist_ok=True)

# 2. Carregar as imagens com PIL
img1_path = base_dir / "praia1.jpeg"
img2_path = base_dir / "praia2.jpeg"

with Image.open(img1_path) as imagem1:
    img1 = np.array(imagem1.convert("L"), dtype=np.float32)

with Image.open(img2_path) as imagem2:
    img2 = np.array(imagem2.convert("L"), dtype=np.float32)

altura, largura = img1.shape
cx = largura / 2.0
cy = altura / 2.0

# Mapa esparso preenchido com NaN para separar fundo de valor valido.
mapa_paralaxe = np.full((altura, largura), np.nan, dtype=np.float32)

# 3. Inicializar SIFT e achar pontos homologos
sift1 = SIFT()
sift2 = SIFT()
sift1.detect_and_extract(img1)
sift2.detect_and_extract(img2)

if sift1.descriptors is None or sift2.descriptors is None:
    raise RuntimeError("Nao foi possivel extrair descritores SIFT das imagens.")

matches = match_descriptors(
    sift1.descriptors,
    sift2.descriptors,
    max_ratio=0.75,
    cross_check=True,
)

if len(matches) == 0:
    raise RuntimeError("Nenhuma correspondencia valida foi encontrada entre as imagens.")

# 4. Calcular a paralaxe e associar aos pixels
paralaxes = []

for indice1, indice2 in matches:
    v1, u1 = sift1.keypoints[indice1]
    v2, u2 = sift2.keypoints[indice2]

    # Conversao do sistema de imagem para o sistema fotogrametrico.
    dx1_px = u1 - cx
    dy1_px = cy - v1
    dx2_px = u2 - cx
    dy2_px = cy - v2

    x1_fisico = (dx1_px * tp_mm) - x0
    y1_fisico = (dy1_px * tp_mm) - y0
    x2_fisico = (dx2_px * tp_mm) - x0
    y2_fisico = (dy2_px * tp_mm) - y0

    px = x1_fisico - x2_fisico
    py = y1_fisico - y2_fisico
    p = np.sqrt((px ** 2) + (py ** 2))

    paralaxes.append(p)

    linha_pixel = int(round(v1))
    coluna_pixel = int(round(u1))

    if 0 <= linha_pixel < altura and 0 <= coluna_pixel < largura:
        valor_atual = mapa_paralaxe[linha_pixel, coluna_pixel]
        if np.isnan(valor_atual):
            mapa_paralaxe[linha_pixel, coluna_pixel] = p
        else:
            mapa_paralaxe[linha_pixel, coluna_pixel] = (valor_atual + p) / 2.0

# 5. Salvar o raster de paralaxe com PIL
mapa_para_salvar = np.nan_to_num(mapa_paralaxe, nan=0.0).astype(np.float32)
imagem_tif = Image.fromarray(mapa_para_salvar, mode="F")
imagem_tif.save(saida_dir / "mapa_paralaxe_pontos.tif")

# 6. Visualizar e salvar a figura final
plt.figure(figsize=(12, 9))
plt.imshow(mapa_paralaxe, cmap="jet")
plt.colorbar(label="Paralaxe Calculada (mm)")
plt.title(f"Mapa de Paralaxe Esparso ({len(matches)} pontos)")
plt.axis("off")
plt.tight_layout()
plt.savefig(saida_dir / "mapa_paralaxe_pontos.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Mapa TIFF salvo em: {saida_dir / 'mapa_paralaxe_pontos.tif'}")
print(f"Mapa PNG salvo em: {saida_dir / 'mapa_paralaxe_pontos.png'}")
