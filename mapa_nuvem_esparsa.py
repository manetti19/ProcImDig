from pathlib import Path

import numpy as np
from PIL import Image
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import open3d as o3d
import laspy



# Parametros geometricos da estereoscopia e da camera.
# Valores observados no Metashape para a camera FC6310.
FOCAL_LENGTH_PIXELS = 3372.57719
PRINCIPAL_POINT_X = 0.0
PRINCIPAL_POINT_Y = 0.0
PIXEL_SIZE_MM = 0.00260928
BASELINE_MM = 27871.59
ALTURA_VOO_M = 119.068

# Reducao para a etapa de deteccao/matching, evitando explodir memoria
# nas fotos DJI em resolucao total.
DETECTION_MAX_DIM = 1200


def redimensionar_para_deteccao(imagem: Image.Image) -> tuple[np.ndarray, float, float]:
    largura_original, altura_original = imagem.size
    maior_dim = max(largura_original, altura_original)

    if maior_dim <= DETECTION_MAX_DIM:
        escala = 1.0
    else:
        escala = DETECTION_MAX_DIM / maior_dim

    nova_largura = max(1, int(round(largura_original * escala)))
    nova_altura = max(1, int(round(altura_original * escala)))
    imagem_redimensionada = imagem.resize((nova_largura, nova_altura), Image.Resampling.LANCZOS)

    escala_x = largura_original / nova_largura
    escala_y = altura_original / nova_altura
    return np.array(imagem_redimensionada.convert("L"), dtype=np.float32), escala_x, escala_y


def carregar_imagens(base_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    img1_path = base_dir / "DJI_0723.JPG"
    img2_path = base_dir / "DJI_0724.JPG"

    with Image.open(img1_path) as imagem1:
        img1_rgb = np.array(imagem1.convert("RGB"), dtype=np.uint8)
        img1_gray, escala_x, escala_y = redimensionar_para_deteccao(imagem1)

    with Image.open(img2_path) as imagem2:
        img2_gray, _, _ = redimensionar_para_deteccao(imagem2)

    return img1_rgb, img1_gray, img2_gray, escala_x, escala_y


def detectar_matches(
    img1_gray: np.ndarray,
    img2_gray: np.ndarray,
    escala_x: float,
    escala_y: float,
):
    sift1 = SIFT()
    sift2 = SIFT()
    sift1.detect_and_extract(img1_gray)
    sift2.detect_and_extract(img2_gray)

    if sift1.descriptors is None or sift2.descriptors is None:
        raise RuntimeError("Nao foi possivel extrair descritores SIFT das imagens.")

    matches = match_descriptors(
        sift1.descriptors,
        sift2.descriptors,
        max_ratio=0.80,
        cross_check=True,
    )

    if len(matches) < 8:
        raise RuntimeError("Quantidade insuficiente de matches para a etapa geometrica.")

    pontos1 = sift1.keypoints[matches[:, 0]][:, ::-1]
    pontos2 = sift2.keypoints[matches[:, 1]][:, ::-1]

    modelo, inliers = ransac(
        (pontos1, pontos2),
        FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=2.0,
        max_trials=5000,
    )

    if modelo is None or inliers is None or not np.any(inliers):
        raise RuntimeError("RANSAC nao encontrou uma geometria epipolar consistente.")

    pontos1_orig = pontos1[inliers].astype(np.float64, copy=True)
    pontos2_orig = pontos2[inliers].astype(np.float64, copy=True)
    pontos1_orig[:, 0] *= escala_x
    pontos1_orig[:, 1] *= escala_y
    pontos2_orig[:, 0] *= escala_x
    pontos2_orig[:, 1] *= escala_y

    return pontos1_orig, pontos2_orig


def reconstruir_pontos(
    pontos1: np.ndarray,
    pontos2: np.ndarray,
    img1_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    altura, largura = img1_rgb.shape[:2]
    cx = largura / 2.0
    cy = altura / 2.0
    baseline_m = BASELINE_MM / 1000.0
    pixel_size_m = PIXEL_SIZE_MM / 1000.0
    f_m = FOCAL_LENGTH_PIXELS * pixel_size_m

    pontos_3d = []
    cores = []

    for (u1, v1), (u2, v2) in zip(pontos1, pontos2):
        dx1_px = u1 - cx
        dy1_px = cy - v1
        dx2_px = u2 - cx
        dy2_px = cy - v2

        x1_mm = (dx1_px * PIXEL_SIZE_MM) - PRINCIPAL_POINT_X
        y1_mm = (dy1_px * PIXEL_SIZE_MM) - PRINCIPAL_POINT_Y
        x2_mm = (dx2_px * PIXEL_SIZE_MM) - PRINCIPAL_POINT_X
        y2_mm = (dy2_px * PIXEL_SIZE_MM) - PRINCIPAL_POINT_Y

        px_mm = x1_mm - x2_mm
        py_mm = y1_mm - y2_mm
        paralaxe_mm = np.sqrt((px_mm ** 2) + (py_mm ** 2))
        disparidade_px = abs(u1 - u2)

        if disparidade_px <= 1e-6 or paralaxe_mm <= 1e-9:
            continue

        # Profundidade relativa ao centro de perspectiva:
        # Z = (B * f) / p
        z = (baseline_m * f_m) / (paralaxe_mm / 1000.0)

        # Cota aproximada do ponto em relacao ao datum local da cena:
        # h = H - (B * f) / p
        h = ALTURA_VOO_M - z

        x_sensor_m = x1_mm / 1000.0
        y_sensor_m = y1_mm / 1000.0

        # Coordenadas do ponto no espaco usando a relacao de semelhanca.
        x = (x_sensor_m * baseline_m) / (paralaxe_mm / 1000.0)
        y = (y_sensor_m * baseline_m) / (paralaxe_mm / 1000.0)

        linha = int(round(v1))
        coluna = int(round(u1))
        if not (0 <= linha < altura and 0 <= coluna < largura):
            continue

        cor = img1_rgb[linha, coluna] / 255.0
        pontos_3d.append([x, y, h])
        cores.append(cor)

    if not pontos_3d:
        raise RuntimeError("Nenhum ponto 3D valido foi reconstruido.")

    return np.asarray(pontos_3d, dtype=np.float64), np.asarray(cores, dtype=np.float64)


def salvar_resultados(
    pontos_3d: np.ndarray,
    cores: np.ndarray,
    saida_dir: Path,
) -> None:
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pontos_3d)
    point_cloud.colors = o3d.utility.Vector3dVector(cores)

    ply_path = saida_dir / "nuvem_esparsa_paralaxe.ply"
    pcd_path = saida_dir / "nuvem_esparsa_paralaxe.pcd"
    txt_path = saida_dir / "nuvem_esparsa_paralaxe.xyz"
    las_path = saida_dir / "nuvem_esparsa_paralaxe.las"
    parametros_path = saida_dir / "parametros_nuvem_esparsa.txt"

    o3d.io.write_point_cloud(str(ply_path), point_cloud)
    o3d.io.write_point_cloud(str(pcd_path), point_cloud)
    np.savetxt(txt_path, pontos_3d, fmt="%.6f", header="X Y Z", comments="")

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = np.min(pontos_3d, axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = pontos_3d[:, 0]
    las.y = pontos_3d[:, 1]
    las.z = pontos_3d[:, 2]
    cores_uint16 = np.clip(cores * 65535.0, 0, 65535).astype(np.uint16)
    las.red = cores_uint16[:, 0]
    las.green = cores_uint16[:, 1]
    las.blue = cores_uint16[:, 2]
    las.write(str(las_path))

    parametros_path.write_text(
        "\n".join(
            [
                f"FOCAL_LENGTH_PIXELS={FOCAL_LENGTH_PIXELS}",
                f"PRINCIPAL_POINT_X={PRINCIPAL_POINT_X}",
                f"PRINCIPAL_POINT_Y={PRINCIPAL_POINT_Y}",
                f"PIXEL_SIZE_MM={PIXEL_SIZE_MM}",
                f"BASELINE_MM={BASELINE_MM}",
                f"ALTURA_VOO_M={ALTURA_VOO_M}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Nuvem PLY salva em: {ply_path}")
    print(f"Nuvem PCD salva em: {pcd_path}")
    print(f"Pontos XYZ salvos em: {txt_path}")
    print(f"Nuvem LAS salva em: {las_path}")
    print(f"Parametros salvos em: {parametros_path}")
    print(f"Quantidade de pontos reconstruidos: {len(pontos_3d)}")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    saida_dir = base_dir / "saida_paralaxe"
    saida_dir.mkdir(exist_ok=True)

    img1_rgb, img1_gray, img2_gray, escala_x, escala_y = carregar_imagens(base_dir)
    pontos1, pontos2 = detectar_matches(img1_gray, img2_gray, escala_x, escala_y)
    pontos_3d, cores = reconstruir_pontos(pontos1, pontos2, img1_rgb)
    salvar_resultados(pontos_3d, cores, saida_dir)


if __name__ == "__main__":
    main()
