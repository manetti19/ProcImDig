
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    rasterio = None

def _to_uint8(img): #Serve para converter as informações numericas

    arr = img.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.uint8)
    p2, p98 = np.percentile(arr[finite], (2, 98))
    if p98 <= p2:  # evita divisão por zero
        p2, p98 = arr.min(), arr.max()
        if p98 <= p2:
            return np.zeros_like(arr, dtype=np.uint8)
    arr = np.clip((arr - p2) / (p98 - p2), 0, 1)
    return (arr * 255).astype(np.uint8)

def image(path, rgb_bands=(1, 2, 3), max_px=4096): #
    """
    Lê e apresenta uma imagem:
      - GIF ( 1º frame)
      - WebP
      - BMP
      - GeoTIFF (tenta RGB nas bandas 1-2-3)
    """
    ext = os.path.splitext(path)[1].lower()

    # ---------- GeoTIFF ----------

    if ext in [".tif", ".tiff"]:
        if rasterio is None:
            print("Para GeoTIFF, instale 'rasterio' (pip install rasterio). Tentando abrir com PIL...")
            img = Image.open(path)
            if img.mode not in ("RGB", "RGBA", "L"):
                img = img.convert("RGB")
            _show_pil(img, title=os.path.basename(path))
            return

        with rasterio.open(path) as src:
            count = src.count
            # Se muito grande, faz downsample rápido para exibição
            scale = max(src.width, src.height) / max_px if max(src.width, src.height) > max_px else 1
            if scale > 1:
                w = int(src.width / scale)
                h = int(src.height / scale)
                data = src.read(
                    indexes=list(range(1, min(count, 3) + 1)),
                    out_shape=(min(count, 3), h, w),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
            else:
                data = src.read(indexes=list(range(1, min(count, 3) + 1)))

            # Monta a imagem
            if data.shape[0] >= 3:
                img = np.dstack([data[0], data[1], data[2]])
                img = _to_uint8(img)
                left, bottom, right, top = src.bounds
                plt.figure(figsize=(8, 8))
                plt.imshow(img, extent=[left, right, bottom, top])
                plt.title(os.path.basename(path))
                plt.xlabel("X (proj.)")
                plt.ylabel("Y (proj.)")
                plt.tight_layout()
                plt.show()
            else:
                band1 = _to_uint8(data[0])
                left, bottom, right, top = src.bounds
                plt.figure(figsize=(8, 8))
                plt.imshow(band1, cmap="gray", extent=[left, right, bottom, top])
                plt.title(f"{os.path.basename(path)} (1 banda)")
                plt.xlabel("X (proj.)")
                plt.ylabel("Y (proj.)")
                plt.tight_layout()
                plt.show()
        return

    # ---------- GIF / WebP / BMP / outros raster comuns ----------
    img = Image.open(path)

    # GIF animado: mostra o 1º frame
    if getattr(img, "is_animated", False):
        img.seek(0)

    # Converte para modo exibível
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    # Reduz se for enorme (acelera renderização)
    img.thumbnail((max_px, max_px))
    _show_pil(img, title=os.path.basename(path))

def _show_pil(pil_image, title="Imagem"):
    plt.figure(figsize=(8, 8))
    if pil_image.mode == "L":
        plt.imshow(pil_image, cmap="gray")
    else:
        plt.imshow(pil_image)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def ver_filtros(caminho):
    # Lê imagem a partir do caminho recebido
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Não consegui abrir: {caminho}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Filtros passa-baixa por média
    filtro_3  = cv2.blur(img_rgb, (3, 3))
    filtro_9  = cv2.blur(img_rgb, (9, 9))
    filtro_25 = cv2.blur(img_rgb, (25, 25))

    # Mostrar resultados
    fig, axs = plt.subplots(1, 4, figsize=(16, 6))

    axs[0].imshow(img_rgb);   axs[0].set_title("Original");   axs[0].axis("off")
    axs[1].imshow(filtro_3);  axs[1].set_title("Média 3x3");  axs[1].axis("off")
    axs[2].imshow(filtro_9);  axs[2].set_title("Média 9x9");  axs[2].axis("off")
    axs[3].imshow(filtro_25); axs[3].set_title("Média 25x25");axs[3].axis("off")

    plt.tight_layout()
    plt.show()

def gradiente(image_path, show_sharpen=True, alpha=0.5):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.astype(np.uint8)


    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    #magnitude do gradiente = passa-alta
    mag = cv2.magnitude(gx, gy)          # sqrt(gx^2+gy^2)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_u8 = mag.astype(np.uint8)

    # 4) (opcional) realçar a original com o gradiente (unsharp-like)
    if show_sharpen:
        if img_bgr.ndim == 3:
            # equaliza para 3 canais para somar
            mag_rgb = cv2.cvtColor(mag_u8, cv2.COLOR_GRAY2BGR)
            sharp = cv2.addWeighted(img_bgr, 1.0, mag_rgb, alpha, 0)
        else:
            sharp = cv2.addWeighted(gray, 1.0, mag_u8, alpha, 0)

    # 5) exibir (matplotlib espera RGB)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim==3 else gray, cmap=None if img_bgr.ndim==3 else 'gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(mag_u8, cmap='gray'); plt.title('Passa‑alto (|∇I|)'); plt.axis('off')
    if show_sharpen:
        plt.subplot(1,3,3);
        plt.imshow(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB) if sharp.ndim==3 else sharp, cmap=None if sharp.ndim==3 else 'gray')
        plt.title(f'Sharpen (α={alpha})'); plt.axis('off')
    plt.tight_layout(); plt.show()

def laplaciano_cinza(path, alpha=0.5):
    # 1) Ler imagem em escala de cinza
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)

    # 2) Aplicar Laplaciano (2ª derivada)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    lap = cv2.convertScaleAbs(lap)  # converte para uint8 visível

    # 3) Sharpen: original + alpha * laplaciano
    sharp = cv2.addWeighted(img, 1.0, lap, alpha, 0)

    # 4) Mostrar resultados
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(lap, cmap='gray'); plt.title("Passa-alto (Laplaciano)"); plt.axis("off")
    #plt.subplot(1,3,3); plt.imshow(sharp, cmap='gray'); plt.title(f"Sharpen (α={alpha})"); plt.axis("off")
    plt.tight_layout(); plt.show()

def laplaciano_rgb(path, alpha=0.5, use_gray=False):
    # 1) Ler imagem
    if use_gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)  # BGR
    if img is None:
        raise FileNotFoundError(path)

    # 2) Aplicar Laplaciano
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
    lap = cv2.convertScaleAbs(lap)

    # 3) Sharpen = original + α * laplaciano
    sharp = cv2.addWeighted(img, 1.0, lap, alpha, 0)

    # 4) Converter para RGB se for colorido (pra matplotlib)
    if not use_gray:
        img_show   = cv2.cvtColor(img,   cv2.COLOR_BGR2RGB)
        lap_show   = cv2.cvtColor(lap,   cv2.COLOR_BGR2RGB)
        sharp_show = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)
    else:
        img_show, lap_show, sharp_show = img, lap, sharp

    # 5) Mostrar
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_show, cmap=None if not use_gray else 'gray');   plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(lap_show, cmap=None if not use_gray else 'gray');   plt.title("Passa-alto (Laplaciano)"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(sharp_show, cmap=None if not use_gray else 'gray'); plt.title(f"Sharpen (α={alpha})"); plt.axis("off")
    plt.tight_layout(); plt.show()



if __name__ == "__main__":
    # >>> edite o caminho do arquivo para testar <<<
    caminho = r"testando.webp"  # .gif, .webp, .bmp, .tif/.tiff
    #image(caminho)
    #ver_filtros(caminho)
    #gradiente(caminho)
    laplaciano_cinza(caminho)
    laplaciano_rgb(caminho)