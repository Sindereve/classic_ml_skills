from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

def dbscan_analysis(X, eps_range=[0.3, 0.5, 0.7], min_samples_range=[5, 10, 15]):
    """Анализ DBSCAN с разными параметрами"""
    
    results = {}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                results[(eps, min_samples)] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_percentage': n_noise / len(labels) * 100
                }
                
            except Exception as e:
                print(f"ERROR: {e}")
                results[(eps, min_samples)] = None
    return results

# Автоматический подбор параметров
def find_optimal_dbscan_params(X, df):
    """Автоматический поиск оптимальных параметров DBSCAN"""
    
    # Метод локтя для выбора eps
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    distances = np.sort(distances[:, -1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('Метод локтя для выбора eps')
    plt.xlabel('Точки')
    plt.ylabel('Расстояние до 5-го соседа')
    plt.grid(True)
    plt.show()
    
    # Выбираем eps на основе "локтя" на графике
    recommended_eps = distances[len(distances) // 10]  # например, 10-й перцентиль
    print(f" Рекомендуемый eps: {recommended_eps:.3f}")
    
    return recommended_eps