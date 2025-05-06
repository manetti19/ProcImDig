import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Cria um tabuleiro 8x8 com padrão xadrez
board = np.zeros((8, 8))
board[1::2, ::2] = 1
board[::2, 1::2] = 1

# Letras das colunas (A-H) e números das linhas (1-8)
columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
rows = list(range(8, 0, -1))

# Mostra o tabuleiro
fig, ax = plt.subplots()
ax.imshow(board, cmap='gray', extent=[0, 8, 0, 8])

# Define os rótulos personalizados
ax.set_xticks(np.arange(0.5, 8.5, 1))
ax.set_yticks(np.arange(0.5, 8.5, 1))
ax.set_xticklabels(columns)
ax.set_yticklabels(rows)

# Move os ticks para o topo e esquerda
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Remove a grade e bordas padrões
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)

# Adiciona borda preta fina em volta do tabuleiro
borda = patches.Rectangle((0, 0), 8, 8, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(borda)

plt.show()
