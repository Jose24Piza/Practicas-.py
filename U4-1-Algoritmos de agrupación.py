import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== CREAR DATOS DE EJEMPLO ====================
# Simularemos datos de clientes con diferentes patrones de compra

np.random.seed(42)

# Grupo 1: Clientes de alto ingreso y alto gasto (40 clientes)
high_income_high_spend = np.random.normal([80, 85], [8, 5], (40, 2))

# Grupo 2: Clientes de ingreso medio y gasto medio (35 clientes)
medium_income_medium_spend = np.random.normal([50, 60], [6, 6], (35, 2))

# Grupo 3: Clientes de bajo ingreso pero alto gasto (25 clientes) - comportamiento interesante
low_income_high_spend = np.random.normal([25, 80], [4, 7], (25, 2))

# Grupo 4: Clientes de alto ingreso pero bajo gasto (20 clientes)
high_income_low_spend = np.random.normal([85, 25], [7, 5], (20, 2))

# Outliers: Clientes con comportamiento atípico (15 clientes)
outliers = np.random.uniform(low=[10, 10], high=[100, 100], size=(15, 2))

# Combinar todos los datos
X = np.vstack([high_income_high_spend, 
               medium_income_medium_spend,
               low_income_high_spend,
               high_income_low_spend,
               outliers])

# Crear DataFrame para mejor visualización
df = pd.DataFrame(X, columns=['Annual_Income', 'Spending_Score'])
print("Datos generados:")
print(f"Total de clientes: {len(df)}")
print(df.head())

# ==================== PREPROCESAMIENTO ====================
# Escalar los datos (importante para DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ==================== APLICAR DBSCAN ====================
# Parámetros: 
eps = 0.5  # En datos escalados
min_samples = 5

# Crear y entrenar modelo DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(X_scaled)

df['Cluster'] = clusters

# Estadísticas de clusters
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise = list(clusters).count(-1)

print("\n" + "="*50)
print("RESULTADOS DE DBSCAN")
print("="*50)
print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de puntos clasificados como ruido: {n_noise}")
print(f"Porcentaje de ruido: {(n_noise/len(df))*100:.2f}%")

# Mostrar distribución de clientes por cluster
print("\nDistribución de clientes por cluster:")
for cluster_id in sorted(set(clusters)):
    if cluster_id == -1:
        print(f"  Ruido: {list(clusters).count(cluster_id)} clientes")
    else:
        print(f"  Cluster {cluster_id}: {list(clusters).count(cluster_id)} clientes")

# ==================== VISUALIZACIÓN DE RESULTADOS ====================
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Datos originales coloreados por cluster
scatter1 = axes[0].scatter(df['Annual_Income'], 
                          df['Spending_Score'], 
                          c=df['Cluster'], 
                          cmap='viridis', 
                          s=50, 
                          alpha=0.7,
                          edgecolors='black', 
                          linewidth=0.5)
axes[0].set_xlabel('Ingreso Anual (miles $)', fontsize=12)
axes[0].set_ylabel('Puntaje de Gasto (1-100)', fontsize=12)
axes[0].set_title('Segmentación de Clientes con DBSCAN', fontsize=14, fontweight='bold')

# Agregar leyenda de clusters
legend_elements = []
for cluster_id in sorted(set(df['Cluster'])):
    if cluster_id == -1:
        label = 'Ruido'
        color = 'gray'
    else:
        label = f'Cluster {cluster_id}'
        # Obtener color del colormap
        color = plt.cm.viridis(cluster_id / max(1, max(df['Cluster'])))
    
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=color, 
                                       markersize=10, 
                                       label=label))
axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)

# Gráfico 2: Visualización destacando puntos de ruido
axes[1].scatter(df['Annual_Income'], 
               df['Spending_Score'], 
               c=df['Cluster'], 
               cmap='viridis', 
               s=50, 
               alpha=0.7,
               edgecolors='black', 
               linewidth=0.5)

# puntos de ruido
noise_points = df[df['Cluster'] == -1]
if len(noise_points) > 0:
    axes[1].scatter(noise_points['Annual_Income'], 
                   noise_points['Spending_Score'],
                   c='red', 
                   s=100, 
                   marker='x', 
                   alpha=0.8,
                   label='Ruido',
                   edgecolors='black',
                   linewidth=2)

axes[1].set_xlabel('Ingreso Anual (miles $)', fontsize=12)
axes[1].set_ylabel('Puntaje de Gasto (1-100)', fontsize=12)
axes[1].set_title('Identificación de Outliers (Ruido)', fontsize=14, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

# ==================== ANÁLISIS DESCRIPTIVO POR CLUSTER ====================
print("\n" + "="*50)
print("ANÁLISIS DESCRIPTIVO POR CLUSTER")
print("="*50)

for cluster_id in sorted(set(df['Cluster'])):
    if cluster_id == -1:
        print(f"\nCLIENTES RUIDO (Comportamiento Atípico):")
        cluster_data = df[df['Cluster'] == cluster_id]
    else:
        print(f"\nCLUSTER {cluster_id}:")
        cluster_data = df[df['Cluster'] == cluster_id]
    
    if len(cluster_data) > 0:
        print(f"  Número de clientes: {len(cluster_data)}")
        print(f"  Ingreso promedio: ${cluster_data['Annual_Income'].mean():.2f}k")
        print(f"  Gasto promedio: {cluster_data['Spending_Score'].mean():.2f}/100")
        print(f"  Rango de ingreso: ${cluster_data['Annual_Income'].min():.2f}k - ${cluster_data['Annual_Income'].max():.2f}k")
        print(f"  Rango de gasto: {cluster_data['Spending_Score'].min():.2f} - {cluster_data['Spending_Score'].max():.2f}")

print("\n✅ Análisis completado!")
