from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

class kMeans:
    def clustering(self, X, categories):
        k = self.computeK(X, max_k=12, categoryName=categories)
        kmeans = KMeans(n_clusters=k, n_init=5, init='random')
        kmeans.fit(X)
        
        return kmeans
    
    def computeK(self, X, max_k, categoryName=""):
        distortions = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, n_init=5, init='random')
            kmeans.fit(X)
            distortions.append(kmeans.inertia_)

        optimal_k = KneeLocator(range(1, max_k + 1), distortions, curve="convex", direction="decreasing")
        kMeans.computePlot(max_k, distortions, optimal_k, categoryName)
        return optimal_k.knee

    def computePlot(max_k, distortions, optimal_k, category_name):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, max_k + 1), distortions, marker='o', linestyle='-',
                color='b', label='Varianza Intra-Cluster')
        plt.scatter(optimal_k.knee, distortions[optimal_k.knee - 1], c='red',
                    marker='o', s=200, label='Cluster Ottimale')
        plt.title(f'Elbow Method - {category_name}', fontsize=16)
        plt.xlabel('Numero di Cluster (k)', fontsize=14)
        plt.ylabel('Varianza Intra-Cluster', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(f'Evaluation/Elbow_Method_{category_name}.png')