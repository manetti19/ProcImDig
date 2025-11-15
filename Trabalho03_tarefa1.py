import numpy as np

def gerar_matriz(n):
    return np.random.randint(1, n*n + 1, size=(n, n))

# -----------------------------
# GERAÇÃO DAS MATRIZES
# -----------------------------
matriz_10 = gerar_matriz(10)
matriz_100 = gerar_matriz(100)
matriz_1000 = gerar_matriz(1000)

# -----------------------------
# SALVAR ARQUIVOS NA MESMA PASTA
# -----------------------------
np.savetxt("matriz_10x10.txt", matriz_10, fmt="%d")
np.savetxt("matriz_100x100.txt", matriz_100, fmt="%d")
np.savetxt("matriz_1000x1000.txt", matriz_1000, fmt="%d")

print("Arquivos gerados:")
print(" - matriz_10x10.txt")
print(" - matriz_100x100.txt")
print(" - matriz_1000x1000.txt")
