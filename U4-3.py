# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# Cargar datos
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Probar diferentes números de clusters
k_range = range(2, 7)
resultados = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    inercia = kmeans.inertia_
    
    resultados.append({
        'k': k,
        'Silhouette': silhouette,
        'Calinski-Harabasz': ch,
        'Davies-Bouldin': db,
        'Inercia': inercia
    })

# Tabla de resultados
df_resultados = pd.DataFrame(resultados)
print("Tabla comparativa de métricas:")
print(df_resultados.round(4))

# Gráficas
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Inercia
axes[0, 0].plot(k_range, df_resultados['Inercia'], marker='o')
axes[0, 0].set_title('Inercia vs Número de clusters')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('Inercia')

# Silhouette
axes[0, 1].plot(k_range, df_resultados['Silhouette'], marker='o', color='green')
axes[0, 1].set_title('Silhouette Score vs Número de clusters')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('Silhouette')

# Calinski-Harabasz
axes[1, 0].plot(k_range, df_resultados['Calinski-Harabasz'], marker='o', color='orange')
axes[1, 0].set_title('Calinski-Harabasz Index vs Número de clusters')
axes[1, 0].set_xlabel('k')
axes[1, 0].set_ylabel('CH Index')

# Davies-Bouldin
axes[1, 1].plot(k_range, df_resultados['Davies-Bouldin'], marker='o', color='red')
axes[1, 1].set_title('Davies-Bouldin Index vs Número de clusters')
axes[1, 1].set_xlabel('k')
axes[1, 1].set_ylabel('DB Index')

plt.tight_layout()
plt.show()

# Visualización de clusters para k=3 (valor real)
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init='auto')
labels_3 = kmeans_3.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels_3, cmap='viridis', edgecolor='k')
plt.scatter(kmeans_3.cluster_centers_[:, 0], kmeans_3.cluster_centers_[:, 1], 
            marker='x', s=200, linewidths=3, color='red')
plt.title('Clusters obtenidos con K-Means (k=3)')
plt.xlabel('Feature 1 (escalada)')
plt.ylabel('Feature 2 (escalada)')
plt.show()