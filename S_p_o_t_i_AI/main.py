import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv
from KNN import KNNTrainer
from accessAPI import SpotifyClient
from kMeans import KMeansClustering
import dataset

# Configurazione del backend di matplotlib e delle opzioni di visualizzazione
plt.switch_backend('TkAgg')
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)

# Carica le variabili d'ambiente
load_dotenv()


# Funzione per pulire le etichette rimuovendo caratteri speciali
def clean_label(label):
    return label.translate(str.maketrans('', '', '$#&@!'))


# Visualizza i dati utilizzando PCA
def visualize_data(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    X = data[numeric_columns]

    # Applicazione di PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['name'] = data['name']

    # Visualizzazione scatter plot con annotazioni
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'])
    for i in range(len(pca_df)):
        plt.annotate(clean_label(pca_df['name'].iloc[i]),
                     (pca_df['PC1'].iloc[i], pca_df['PC2'].iloc[i]),
                     fontsize=8, alpha=0.7)
    plt.title('Visualizzazione dei Dati con PCA')
    plt.xlabel('Componente Principale 1')
    plt.ylabel('Componente Principale 2')
    plt.grid()
    plt.show()


# Normalizza solo le colonne numeriche di un dataset
def get_normalized_data(ds):
    numeric_columns = [col for col in ds.getDataset().select_dtypes(include=['int64', 'float64']).columns
                       if col in ds.getDataset().columns]
    for column in numeric_columns:
        ds.normalizeColumn(column)
    return ds


if __name__ == "__main__":
    dropped_columns = ["release_date", "speechiness", "id", "year", "explicit", "mode", "name", "artists", "popularity"]

    # Carica e normalizza i dati
    data = dataset.Dataset("dataReduced.csv")
    normalized_data = get_normalized_data(data)
    print(normalized_data.getDataset())

    # --- KMeans Clustering ---
    kmeans_data = get_normalized_data(dataset.Dataset("dataReduced.csv"))
    kmeans_data.dropDatasetColumns(dropped_columns)
    kmeans = KMeansClustering(n_clusters=4)
    kmeans.fit(kmeans_data.getDataset())
    print("Centroidi dei cluster KMeans:", "\n", kmeans.get_centroids())

    # --- Esempio di ricerca e predizione con API Spotify ---
    api = SpotifyClient()
    example = api.create_training_example("The Spins", "Mac Miller")
    print("Esempio estratto:", example)



    # Normalizzazione e predizione del cluster

    tempData = dataset.Dataset("dataReduced.csv")
    tempData.dropDatasetColumns(dropped_columns)
    normalized_example = tempData.normalizeTuple(example)
    formatted_tuple = tuple(f"{x:.5f}" for x in normalized_example)

    print("Esempio normalizzato con cluster:", formatted_tuple)
    cluster = kmeans.predict(pd.DataFrame([formatted_tuple]))
    normalized_example += (0,)
    print("Predizione del cluster KMeans:", cluster[0])


    # Ottieni il dataset ridotto in base al cluster
    kmeans_data.addDatasetColumn("name", data.getColumn("name"))
    tuples = kmeans.get_tuples_by_cluster(kmeans_data.getDataset(), cluster[0])

    if not isinstance(tuples, pd.DataFrame):
        tuples = pd.DataFrame(tuples)

    # Crea il dataset ridotto
    reduced_df = dataset.Dataset.from_dataframe(tuples)

    # --- KNN Trainer per trovare gli esempi più vicini ---
    knn = KNNTrainer(target_column="name", drop_columns=["name"], dataset=reduced_df)
    knn.findBestParams()
    knn.run()

    # Trova i 3 esempi più vicini
    nearest_examples, distances = knn.findKNearestExamples(normalized_example, 3)
    print("Indici degli esempi più vicini:", nearest_examples.index)
    data = dataset.Dataset("dataReduced.csv")
    data = data.getDataFrame(["name", "artists"])
    print(data.iloc[nearest_examples.index])
