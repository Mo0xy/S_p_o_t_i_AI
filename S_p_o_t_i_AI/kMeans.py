import os
import tkinter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.environ['TCL_LIBRARY'] = r'C:\Users\39324\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\39324\AppData\Local\Programs\Python\Python313\tcl\tk8.6'


class KMeansClustering:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()

    def fit(self, data):
        """
        Calcola il clustering KMeans su un dataset pandas.

        Parameters:
        - data (pd.DataFrame): dataset su cui calcolare il clustering.

        Returns:
        - pd.DataFrame: dataset con un'ulteriore colonna che indica il cluster di appartenenza.
        """
        # Eseguo la standardizzazione
        data_scaled = self.scaler.fit_transform(data)

        # Creo il modello KMeans
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        # Adatto il modello e ottengo le predizioni dei cluster
        clusters = self.kmeans.fit_predict(data_scaled)

        # Aggiungo i cluster al dataframe originale
        data['Cluster'] = clusters
        return data

    def get_centroids(self):
        """
        Restituisce i centroidi calcolati per ciascun cluster.

        Returns:
        - pd.DataFrame: dataframe con i centroidi per ciascun cluster.
        """
        if self.kmeans is None:
            raise ValueError("Il modello non è stato ancora addestrato. Chiamare prima il metodo fit().")

        # Restituisce i centroidi
        centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        return pd.DataFrame(centroids, columns=self.scaler.feature_names_in_)

    def predict(self, data):
        """
        Predice il cluster per nuovi dati.

        Parameters:
        - data (pd.DataFrame): dati da predire.

        Returns:
        - np.ndarray: array con i cluster predetti.
        """
        data_scaled = self.scaler.transform(data)
        return self.kmeans.predict(data_scaled)

    def get_tuples_by_cluster(self, data, cluster_label):
        """
        Estrae tutte le tuple del dataset appartenenti a un dato cluster.

        Parameters:
        - data (pd.DataFrame): Il dataset contenente i dati e i cluster.
        - cluster_label (int): L'etichetta del cluster da filtrare.

        Returns:
        - pd.DataFrame: Un nuovo DataFrame contenente solo le tuple del cluster specificato.
        """
        # Controlla se la colonna 'Cluster' esiste nel DataFrame
        if 'Cluster' not in data.columns:
            raise ValueError("La colonna 'Cluster' non è presente nel dataset.")

        # Filtra il dataset per ottenere solo le righe appartenenti al cluster specificato
        filtered_data = data[data['Cluster'] == cluster_label]

        return filtered_data

    def plot_clusters(self, data, x, y):
        """
        Visualizza un grafico a dispersione dei cluster.

        Parameters:
        - data (pd.DataFrame): dataset con i cluster.
        - x (str): nome della colonna da utilizzare sull'asse x.
        - y (str): nome della colonna da utilizzare sull'asse y.
        """
        plt.figure(figsize=(10, 6))
        for cluster in data['Cluster'].unique():
            plt.scatter(data[data['Cluster'] == cluster][x], data[data['Cluster'] == cluster][y],
                        label=f'Cluster {cluster}')

        plt.scatter(self.get_centroids()[x], self.get_centroids()[y], color='black', marker='x', s=100,
                    label='Centroidi')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title("Clustering KMeans")
        plt.legend()
        plt.grid()
        plt.show()

    def elbow_method(self, data, max_k=10):
        inertia = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertia.append(kmeans.inertia_)

        plt.plot(range(1, max_k + 1), inertia, marker='o')
        plt.xlabel("Numero di Cluster")
        plt.ylabel("Inertia (Distanza intra-cluster)")
        plt.title("Elbow Method")
        plt.show()
