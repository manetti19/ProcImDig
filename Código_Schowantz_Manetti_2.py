from PIL import Image
from collections import Counter
import heapq
from time import perf_counter
import math
import csv

# -----------------------------
# CONFIG
# -----------------------------
PATH_LENA = r".\lena-Color.png"
PATH_AREA23 = r".\top_mosaic_09cm_area23.tif"
THRESHOLD_PERCENT = 0.40  # 40%
# -----------------------------


# ============================================================
# 1) Preparar imagens (Lena RGB, cinza, binária)
# ============================================================

def load_rgb(path):
    return Image.open(path).convert("RGB")

def to_gray(img_rgb):
    return img_rgb.convert("L")

def to_binary(img_gray, thr_percent):
    T = int(round(thr_percent * 255))
    return img_gray.point(lambda p: 255 if p >= T else 0, mode="L")

def img_bytes(img):
    return img.tobytes()

def padroniza_area23(path):
    img = Image.open(path)
    # padroniza para 8 bits
    if img.mode == "RGB":
        return img.convert("RGB")
    return img.convert("L")


# ============================================================
# 2) Huffman (curto): tamanho comprimido em bits
# ============================================================

def huffman_code_lengths(freqs):
    heap = [[f, [sym, ""]] for sym, f in freqs.items()]
    heapq.heapify(heap)

    # caso: só 1 símbolo
    if len(heap) == 1:
        sym = heap[0][1][0]
        return {sym: 1}

    # ✅ caso: exatamente 2 símbolos (binária)
    if len(heap) == 2:
        sym0 = heap[0][1][0]
        sym1 = heap[1][1][0]
        return {sym0: 1, sym1: 1}

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)

        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]

        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    result = heap[0][1:]
    return {sym: len(code) for sym, code in result}

def huffman_compressed_bits(data):
    freqs = Counter(data)
    lengths = huffman_code_lengths(freqs)
    # total de bits = soma(freq * comprimento)
    return sum(freqs[sym] * lengths[sym] for sym in freqs)




# ============================================================
# 2*) Huffman
# ============================================================


def huffman_dahuffman_compressed_bits(data):
    """
    Compressão Huffman usando a biblioteca dahuffman.
    Retorna:
      - bits_reais (payload empacotado em bytes * 8)
      - codec (caso você queira inspecionar/guardar o modelo)
    """
    from dahuffman import HuffmanCodec

    codec = HuffmanCodec.from_data(data)   # cria árvore a partir dos dados
    encoded = codec.encode(data)           # payload comprimido (empacotado)
    decoded = codec.decode(encoded)        # decodifica para validar

    if decoded != data:
        raise ValueError("ERRO: dahuffman decodificou diferente do original!")

    bits_reais = len(encoded) * 8
    return bits_reais

# ============================================================
# 3) Aritmética (didática): bits necessários ~ -log2(tamanho do intervalo)
# ============================================================

def arithmetic_bits_estimate(data):
    freqs = Counter(data)
    N = len(data)

    # entropia (bits/símbolo)
    H = 0.0
    for sym, f in freqs.items():
        p = f / N
        H -= p * math.log2(p)

    # bits aproximados pela entropia (aritmética chega muito perto disso)
    return math.ceil(N * H)


# ============================================================
# 4) Rodar casos e imprimir
# ============================================================

def run_case(name, data):
    original_bits = len(data) * 8

    def ratio(orig, comp):
        return orig / comp if comp > 0 else float("inf")

    def reduction_percent(orig, comp):
        return (1.0 - (comp / orig)) * 100.0 if orig > 0 else 0.0

    rows = []

    # ---------------- Huffman (seu - teórico) ----------------
    t0 = perf_counter()
    huf_bits_teorico = huffman_compressed_bits(data)
    t1 = perf_counter()

    rows.append({
        "caso": name,
        "metodo": "Huffman (seu) - bits teóricos",
        "original_bits": original_bits,
        "comprimido_bits": huf_bits_teorico,
        "C": ratio(original_bits, huf_bits_teorico),
        "reducao_percent": reduction_percent(original_bits, huf_bits_teorico),
        "tempo_s": (t1 - t0),
        "entropia_bits_por_simbolo": ""
    })

    # ---------------- Huffman (dahuffman - real) ----------------
    t2 = perf_counter()
    huf_bits_real = huffman_dahuffman_compressed_bits(data)
    t3 = perf_counter()

    rows.append({
        "caso": name,
        "metodo": "Huffman (dahuffman) - bits reais",
        "original_bits": original_bits,
        "comprimido_bits": huf_bits_real,
        "C": ratio(original_bits, huf_bits_real),
        "reducao_percent": reduction_percent(original_bits, huf_bits_real),
        "tempo_s": (t3 - t2),
        "entropia_bits_por_simbolo": ""
    })

    # ---------------- Aritmética (entropia) ----------------
    t4 = perf_counter()
    ari_bits = arithmetic_bits_estimate(data)
    t5 = perf_counter()

    rows.append({
        "caso": name,
        "metodo": "Aritmética (entropia) - bits estimados",
        "original_bits": original_bits,
        "comprimido_bits": ari_bits,
        "C": ratio(original_bits, ari_bits),
        "reducao_percent": reduction_percent(original_bits, ari_bits),
        "tempo_s": (t5 - t4),
        "entropia_bits_por_simbolo": ""
    })

    return rows

def main():
    lena = load_rgb(PATH_LENA)
    lena_gray = to_gray(lena)
    lena_bin = to_binary(lena_gray, THRESHOLD_PERCENT)

    area23 = padroniza_area23(PATH_AREA23)

    # d.1 e d.2: salvar imagens
    lena_gray.save("lena_gray.png")
    lena_bin.save("lena_bin_40.png")

    # d.3 e d.4: rodar casos e acumular na tabela
    rows = []
    rows += run_case("Lena RGB", img_bytes(lena))
    rows += run_case("Lena Gray", img_bytes(lena_gray))
    rows += run_case("Lena Binary (40%)", img_bytes(lena_bin))
    rows += run_case("Area23", img_bytes(area23))

    # imprimir resumo (d.5)
    for r in rows:
        print(
            f"{r['caso']} | {r['metodo']} | "
            f"C={r['C']:.3f} | Redução={r['reducao_percent']:.2f}% | Tempo={r['tempo_s']:.4f}s"
        )

    # arredonda para CSV ficar bonito
    for r in rows:
        r["C"] = round(r["C"], 4)
        r["reducao_percent"] = round(r["reducao_percent"], 2)
        r["tempo_s"] = round(r["tempo_s"], 6)

    # salvar CSV
    with open("tabela_resultados_d.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
        f,
        fieldnames=[
            "caso",
            "metodo",
            "original_bits",
            "comprimido_bits",
            "C",
            "reducao_percent",
            "tempo_s",
            "entropia_bits_por_simbolo"
        ]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nArquivos salvos na pasta atual:")
    print("- lena_gray.png")
    print("- lena_bin_40.png")
    print("- tabela_resultados_d.csv")


if __name__ == "__main__":
    main()