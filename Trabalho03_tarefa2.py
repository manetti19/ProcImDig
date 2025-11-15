import numpy as np

def pedir_m_valido(n):
    while True:
        try:
            m = int(input(f"Digite a dimensão m do filtro (0 < m < {n}): "))

            if m <= 0:
                print("❌ m deve ser positivo. Tente novamente.")
                continue

            if m >= n:
                print(f"❌ m deve ser menor que n (m < {n}). Tente novamente.")
                continue

            return m  # valor válido encontrado

        except ValueError:
            print("❌ Entrada inválida. Digite apenas números inteiros.")



# =======================================================
# FUNÇÃO PARA CRIAR FILTRO DE MÉDIA m×m
# =======================================================
def criar_filtro_media(m):
    return np.ones((m, m)) / (m * m)

# =======================================================
# CONVOLUÇÃO COM TRATAMENTO DE BORDAS (zero padding)
# =======================================================
def aplicar_convolucao(matriz, filtro):
    n = matriz.shape[0]
    m = filtro.shape[0]
    pad = m // 2

    # bordas com zeros
    matriz_padded = np.pad(matriz, pad, mode="constant", constant_values=0)
    saida = np.zeros_like(matriz, dtype=float)

    # convolução manual
    for i in range(n):
        for j in range(n):
            janela = matriz_padded[i:i+m, j:j+m]
            saida[i, j] = np.sum(janela * filtro)

    return saida

# =======================================================
# PROGRAMA PRINCIPAL
# =======================================================




# -------- LE AS MATRIZES DA TAREFA 1 -----------------
arquivos = [
    ("matriz_10x10.txt", 10),
    ("matriz_100x100.txt", 100),
    ("matriz_1000x1000.txt", 1000),
    ]

for nome_arquivo, tamanho in arquivos:

    # -------- filtro ------------------------------------
    m = pedir_m_valido(tamanho)
    filtro = criar_filtro_media(m)

    print(f"\nProcessando {nome_arquivo}...")
    # Lê a matriz
    matriz = np.loadtxt(nome_arquivo)
    # Aplica o filtro
    matriz_filtrada = aplicar_convolucao(matriz, filtro)
    # Salva a saída
    saida_nome = nome_arquivo.replace(".txt", "_filtrada.txt")
    np.savetxt(saida_nome, matriz_filtrada, fmt="%.5f")
    print(f" -> Arquivo gerado: {saida_nome}")

# salvar o filtro também
np.savetxt("filtro_usado.txt", filtro, fmt="%.5f")
print("\nFiltro salvo como filtro_usado.txt")
