import matplotlib.pyplot as plt
import numpy as np

# Criar o tabuleiro
n = 8
tabuleiro = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if (i + j) % 2 == 1:
            tabuleiro[i, j] = 1

# Posições iniciais das peças usando emojis
# Peças pretas (letras maiúsculas)
# Peças brancas (letras minúsculas)
pecas = [
    ["♜", "♞", "♝", "♛", "♚", "♝", "♞", "♜"],
    ["♟"] * 8,
    [""] * 8,
    [""] * 8,
    [""] * 8,
    [""] * 8,
    ["♙"] * 8,
    ["♖", "♘", "♗", "♕", "♔", "♗", "♘", "♖"]
]

# Plotar o tabuleiro
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(tabuleiro, cmap='gray', extent=[0, 8, 0, 8])

# Adicionar as peças
for i in range(n):
    for j in range(n):
        peca = pecas[i][j]
        if peca != "":
            ax.text(j + 0.5, 7.5 - i, peca, fontsize=32, ha='center', va='center')

# Adicionar grades e rótulos
for i in range(n + 1):
    ax.axhline(i, color='black', linewidth=1)
    ax.axvline(i, color='black', linewidth=1)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Tabuleiro de Xadrez do Rafael com Peças", fontsize=16)
plt.show()