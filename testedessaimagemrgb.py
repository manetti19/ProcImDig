# 1) importa rasterio para ler o TIFF
import rasterio

# 2) importa numpy para manipular matrizes
import numpy as np

# 3) importa matplotlib para salvar imagem
import matplotlib.pyplot as plt

# 4) define o caminho do seu TIFF (ajuste se necessário)
caminho_tiff = r".\top_mosaic_09cm_area23.tif"

# 5) abre o TIFF
with rasterio.open(caminho_tiff) as src:
    # 6) lê as 3 bandas (formato: bandas, H, W)
    arr = src.read()

    # 7) imprime informações de “interpretação” das bandas (muito útil!)
    print("colorinterp:", src.colorinterp)
    print("descriptions:", src.descriptions)

# 8) pega R, G, B como estão no arquivo (bandas 1,2,3)
R = arr[0].astype(np.float32) / 255.0
G = arr[1].astype(np.float32) / 255.0
B = arr[2].astype(np.float32) / 255.0

# 9) empilha para RGB (H, W, 3)
rgb = np.dstack([R, G, B])

# 10) salva exatamente o RGB cru em PNG
plt.imsave("teste_rgb_cru.png", np.clip(rgb, 0.0, 1.0))

print("Salvei teste_rgb_cru.png")
