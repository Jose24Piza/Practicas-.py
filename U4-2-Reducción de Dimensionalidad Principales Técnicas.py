import numpy as np
import matplotlib.pyplot as plt
import umap

# Cargar MNIST desde archivo local .npz
print("Cargando MNIST desde archivo local...")
with np.load('mnist.npz', allow_pickle=True) as f:
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']

# Combinar train y test
X = np.vstack((x_train, x_test))
y = np.hstack((y_train, y_test))

# Normalizar y reshape: de 28x28 a 784
X = X.reshape(X.shape[0], -1) / 255.0

# Tomar subconjunto para que sea más rápido
np.random.seed(42)
subset_idx = np.random.choice(X.shape[0], 10000, replace=False)
X_subset = X[subset_idx]
y_subset = y[subset_idx]

# Aplicar UMAP
print("Aplicando UMAP...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(X_subset)

# Visualizar
plt.figure(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], c=y_subset, cmap="Spectral", s=0.8, alpha=0.7)
plt.colorbar(label='Dígito')
plt.xticks([])
plt.yticks([])
plt.title("Proyección UMAP del Dataset MNIST")
plt.tight_layout()
plt.show()
