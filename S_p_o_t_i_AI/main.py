import os
import dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process
from KNN import KNNTrainer
from S_p_o_t_i_AI.DecisionTree import DecisionTreeTrainer
# from S_p_o_t_i_AI.DecisionTree import DecisionTreeTrainer
from kMeans import KMeansClustering
from RandomForest import RandomForestTrainer

# Configurazione del backend di matplotlib e delle opzioni di visualizzazione
plt.switch_backend('TkAgg')
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)


# Carica le variabili d'ambiente
# load_dotenv()


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


def visualize_data_3d(data):
    fig = px.scatter_3d(
        data, x='valence', y='energy', z='loudness',
        color='Cluster', title="3D Cluster Visualization"
    )
    fig.show()


# Normalizza solo le colonne numeriche di un dataset
def get_normalized_data(ds):
    numeric_columns = [col for col in ds.getDataset().select_dtypes(include=['int64', 'float64']).columns
                       if col in ds.getDataset().columns]
    for column in numeric_columns:
        ds.normalizeColumn(column)
    return ds


if __name__ == "__main__":
    # Carica e normalizza i dati
    """
    data = dataset.Dataset("dataReduced.csv")
    normalized_data = get_normalized_data(data)
    print(normalized_data.getDataset())

    #--- training dataset ---
    trainingData = dataset.Dataset("data.csv")
    #tempData.getReducedDataset(0.7)
    trainingData.dropDatasetColumns(dropped_columns)
    print("Dataset ridotto:\n", trainingData.getDataset())"""

    # Carica i due dataset
    """
    df1 = pd.read_csv("dataReduced.csv")
    df2 = pd.read_csv("data_w_genres.csv")

    df1 = df1.sample(10000)
    df2 = df2.sample(10000)

    print("Dataset 1:\n", df1)
    print("Dataset 2:\n", df2)

    # Unisci i due dataset su una colonna comune, ad esempio "id"
    # Rimuovi eventuali spazi in eccesso e porta i nomi a caratteri minuscoli per entrambe le colonne 'artists'
    df1['artists'] = df1['artists'].str.strip().str.lower()
    df2['artists'] = df2['artists'].str.strip().str.lower()

    # Ottieni una mappa di corrispondenze approssimate
    df2_artists_list = df2['artists'].tolist()
    df1['matched_artist'] = df1['artists'].apply(lambda x: process.extractOne(x, df2_artists_list)[0])

    # Esegui l'unione utilizzando la colonna 'matched_artist'
    merged_df = pd.merge(df1, df2[['artists', 'genres']], left_on='matched_artist', right_on='artists', how='left')
    # salva su file mergerd_df
    merged_df.to_csv("merged_data.csv")
    
    print("Dataset unito:\n", merged_df)

    missing_artists = df1[~df1['artists'].isin(df2['artists'])]
    print("Artisti non trovati:\n", missing_artists['artists'].unique())"""

    dropped_columns = ['year', 'duration_ms', 'instrumentalness', 'explicit', 'mode', 'key', 'speechiness'
        , 'liveness', 'tempo', 'acousticness', 'popularity', 'danceability']
    """
    
    merged_df = pd.read_csv("merged_data.csv")
    merged_df = merged_df.sample(1000)
    merged_df = dataset.Dataset.from_dataframe(merged_df)
    """
    merged_df = dataset.Dataset("merged_data.csv")
    merged_df = merged_df.remove_text_columns()

    kmeans_data = dataset.Dataset.from_dataframe(merged_df)

    kmeans_data.dropDatasetColumns(dropped_columns)
    # kmeans_data.dropDatasetColumns(dropped_columns)
    print("Dataset per KMeans:\n", kmeans_data.getDataset())

    # --- KMeans Clustering ---

    kmeans_data = get_normalized_data(kmeans_data)
    # merged_df['instrumentalness'] = df1['instrumentalness']
    # merged_df['key'] = df1['key']
    # kmeans_data.dropDatasetColumns(dropped_columns)
    kmeans = KMeansClustering(n_clusters=3)
    kmeans.fit(kmeans_data.getDataset())
    print("Centroidi dei cluster KMeans:", "\n", kmeans.get_centroids())
    # --- Visualizzazione dei cluster ---
    # kmeans.plot_clusters(kmeans_data.getDataset(), 'valence', 'energy')
    # visualize_data_3d(kmeans_data.getDataset())
    # 'valence', 'acousticness'
    # 'valence', 'speechiness'
    # 'valence', 'instrumentalness'

    # --- Esempio di ricerca e predizione con API Spotify ---
    """
    api = SpotifyClient()
    example = api.create_training_example(
        "Sonata No. 1 in F minor, OP. 2 No.1: I.Allegro",
        "Ludwig van Beethoven, Paul Lewis")
    print("Esempio estratto:", example)"""

    # Normalizzazione e predizione del cluster
    """
    dataf = dataset.Dataset("data.csv")
    dataf.dropDatasetColumns(dropped_columns)



    tempTuple = tuple(dataf.getDataset().iloc[140000])
    print("temp tuple:", tempTuple)

    normalized_example = trainingData.normalizeTuple(tempTuple)
    formatted_tuple = tuple(f"{x:.5f}" for x in normalized_example)

    print("Esempio normalizzato con cluster:", formatted_tuple)
    cluster = kmeans.predict(pd.DataFrame([formatted_tuple]))
    normalized_example += (0,)
    print("Predizione del cluster KMeans:", cluster[0])

    """
    # Ottieni il dataset ridotto in base al cluster
    kmeans_data.addDatasetColumn("genres", dataset.Dataset("merged_data.csv").getColumn("genres"))
    tuples = kmeans.get_tuples_by_cluster(kmeans_data.getDataset(), 1)

    if not isinstance(tuples, pd.DataFrame):
        tuples = pd.DataFrame(tuples)

    # Crea il dataset ridotto

    reduced_df = dataset.Dataset.from_dataframe(tuples)
    print("Dataset ridotto:", reduced_df.getDataset())
    restored_df = reduced_df.getDataset()
    """
    # --- KNN Trainer per trovare gli esempi più vicini ---
    knn = KNNTrainer(target_column="genres",
                     drop_columns=["genres"],
                     dataset=kmeans_data)
    knn.findBestParams()
    knn.run()
    print(knn.getMetrics())"""

    # Trova i 3 esempi più vicini
    """
    nearest_examples, distances = knn.findKNearestExamples(normalized_example, 3)
    print("Indici degli esempi più vicini:", nearest_examples.index)
    data = dataset.Dataset("dataReduced.csv")
    data = data.getDataFrame(["name", "artists"])
    print(data.iloc[nearest_examples.index])
    """
    #
    """
    # --- esecuzione Random Forest ---
    rf = RandomForestTrainer(target_column="genres",
                             drop_columns=["genres", "Cluster"],
                             dataset=dataset.Dataset.from_dataframe(restored_df))
    rf.findBestParams()
    rf.run()

    print(rf.getMetrics())
    """
    """
    dt = DecisionTreeTrainer(target_column="genres",
                             drop_columns=["genres", "Cluster"],
                             dataset=kmeans_data)
    dt.findBestParams()
    dt.run()
    print(dt.getMetrics())"""

    # --- esecuzione GNN ---

    from GNN import GNNClassifier  # Assicurati di avere GNNClassifier nel modulo GNN

    # Definisci i parametri iniziali per il modello
    params = {
        'input_dim': 3,  # Tre feature: valence, energy, loudness
        'hidden_units': 64,  # Un numero di unità nascoste di esempio
        'output_dim': None  # Sarà impostato automaticamente su `len(mlb.classes_)` in loadData()
    }

    # Inizializza e configura il modello
    gnn_classifier = GNNClassifier(params, kmeans_data.getDataset())

    # Carica i dati e aggiorna `output_dim` basato su `genres`
    # g, features = gnn_classifier.loadData()
    #params['output_dim'] = len(gnn_classifier.mlb.classes_)

    # Esegue la ricerca dei migliori iperparametri e addestra il modello
    gnn_classifier.train_model()

