import os
import GNN
import kMeans
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from S_p_o_t_i_AI import dataset
from S_p_o_t_i_AI.DecisionTree import DecisionTreeTrainer
from S_p_o_t_i_AI.KNN import KNNTrainer
from S_p_o_t_i_AI.RandomForest import RandomForestTrainer

LABEL_ENCODER_PATH = "label_encoder.pkl"
CLASSES_PATH = "classes.json"


def save_label_encoder(lbl_encoder):
    """Salva il LabelEncoder e la lista delle classi."""
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(lbl_encoder, f)

    with open(CLASSES_PATH, "w") as f:
        json.dump(lbl_encoder.classes_.tolist(), f)
    print("LabelEncoder salvato con classi:", lbl_encoder.classes_)


def load_label_encoder():
    """Carica il LabelEncoder e imposta le classi fisse."""
    with open(CLASSES_PATH, "r") as f:
        known_classes = json.load(f)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(known_classes)  # Imposta le classi fisse
    return lbl_encoder


def extract_main_genre(raw_df):
    """
    Seleziona il genere musicale principale per ogni traccia e assegna le etichette in modo coerente.
    """
    # Usa lo stesso LabelEncoder se esiste già
    label_encoder_path = "label_encoder.pkl"
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, "rb") as f:
            lbl_encoder = pickle.load(f)
    else:
        lbl_encoder = LabelEncoder()

    if "genres" not in raw_df.columns:
        raise ValueError("Il dataset non contiene una colonna 'genres'")

    raw_df["genres"] = raw_df["genres"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    all_genres = [genre for sublist in raw_df["genres"] for genre in sublist]
    genre_counts = Counter(all_genres)
    total_genres = sum(genre_counts.values())
    genre_probs = {genre: count / total_genres for genre, count in genre_counts.items()}

    def select_genre_probabilistic(genre_list):
        if not genre_list:
            return np.nan, np.nan
        probs = np.array([genre_probs[g] for g in genre_list])
        probs /= probs.sum()
        selected_genre = np.random.choice(genre_list, p=probs)
        selected_prob = genre_probs[selected_genre]
        return selected_genre, selected_prob

    raw_df["main_genre"], raw_df["main_genre_probability"] = zip(*raw_df["genres"].apply(select_genre_probabilistic))

    # Normalizza la probabilità
    raw_df["main_genre_probability"] = (raw_df["main_genre_probability"] - raw_df["main_genre_probability"].min()) / \
                                       (raw_df["main_genre_probability"].max() - raw_df["main_genre_probability"].min())

    raw_df["main_genre_encoded"] = lbl_encoder.fit_transform(raw_df["main_genre"])

    # Salva il LabelEncoder solo se non esiste già
    if not os.path.exists(LABEL_ENCODER_PATH):
        save_label_encoder(lbl_encoder)

    return raw_df, lbl_encoder


def cut_dataset(dataf, target_col, min_samples=50):
    """
    Taglia il dataset in modo che contenga solo le classi con almeno min_samples campioni.
    Applica la stessa maschera a 'main_genre' per evitare di perderla.
    """
    target = dataf[target_col]
    class_counts = target.value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    mask = target.isin(valid_classes)

    dataf = dataf.loc[mask].copy()  # Mantiene solo le righe necessarie
    target = target[mask]  # Aggiorna la target column

    return dataf, target


def preprocessing(path="merged_data.csv"):
    frame = pd.read_csv(path)
    encoder = OneHotEncoder(sparse_output=False)
    encoded_artists = encoder.fit_transform(frame[['matched_artist']])
    encoded_df = pd.DataFrame(encoded_artists, columns=encoder.get_feature_names_out(['matched_artist']))
    frame = pd.concat([frame, encoded_df], axis=1)
    frame.drop('matched_artist', axis=1, inplace=True)

    t_dataframe, label_encoder = extract_main_genre(frame)
    frame["main_genre_encoded"] = t_dataframe["main_genre_encoded"].copy()

    target = t_dataframe["main_genre_encoded"]

    def convert_explicit(value):
        if isinstance(value, list) and len(value) > 0:
            value = value[0]
        value = str(value).strip("[]").replace("'", "")
        return 1 if value.lower() in ['yes', 'true', '1'] else 0

    t_dataframe["explicit"] = t_dataframe["explicit"].apply(convert_explicit).astype(int)

    t_dataframe = t_dataframe.select_dtypes(include=['number'])

    t_dataframe = pd.DataFrame(MinMaxScaler().fit_transform(t_dataframe), columns=t_dataframe.columns)

    selected_features = t_dataframe[["duration_ms", "explicit", "year", "popularity"]]
    kmeans = kMeans.kMeans().clustering(selected_features, "Trend")
    t_dataframe["trend cluster"] = kmeans.fit_predict(selected_features)
    t_dataframe["trend cluster"] = MinMaxScaler().fit_transform(t_dataframe["trend cluster"].values.reshape(-1, 1))

    t_dataframe = t_dataframe.fillna(t_dataframe.mean())

    main_genre_encoded = target.values
    no_comp = 64
    pca = PCA(n_components=no_comp)
    df_pca = pca.fit_transform(t_dataframe)
    t_dataframe = pd.DataFrame(df_pca, columns=[f"pca_{i}" for i in range(no_comp)])

    t_dataframe["main_genre_encoded"] = main_genre_encoded
    print("Preprocessing completato con KMeans clustering e PCA. Il dataset è stato aggiornato e salvato.")

    return t_dataframe


if __name__ == "__main__":
    df = preprocessing("merged_data.csv")
    dataset = dataset.Dataset(df)

    # --- KNN Trainer ---
    knn = KNNTrainer(target_column="main_genre_encoded",
                     drop_columns=None,
                     dataset=dataset)
    knn.findBestParams()
    knn.run()

    # --- creazione del dataset filtrato ---

    proc_filt_dataset = df
    proc_filt_dataset, target = cut_dataset(proc_filt_dataset, "main_genre_encoded", min_samples=50)

    if not os.path.exists("processedDataset.csv"):
        print("Salvataggio del dataset filtrato...")
        proc_filt_dataset.to_csv("processedDataset.csv", index=False)  # Salva il dataset filtrato
    else:
        print("Dataset filtrato esistente.")

    proc_filt_dataset = dataset.from_dataframe(proc_filt_dataset)

    # --- Random Forest Trainer ---
    rf = RandomForestTrainer(target_column="main_genre_encoded",
                             drop_columns=None,
                             dataset=proc_filt_dataset)
    rf.findBestParams()
    rf.run()

    # --- DecisionTree Trainer ---
    dt = DecisionTreeTrainer(target_column="main_genre_encoded",
                             drop_columns=None,
                             dataset=dataset)
    dt.findBestParams()
    dt.run()

    # --- GNN Trainer ---
    df = pd.read_csv("processedDataset.csv")
    target_column = "main_genre_encoded"
    in_features = df.drop(columns=[target_column]).shape[1]
    num_classes = df[target_column].nunique()

    param_grid = {
        'hidden_units': [32, 64, 128],
        'lr': [0.001, 0.005],
        'epochs': [50, 100]
    }

    model = GNN.GNNClassifier(num_classes=num_classes, in_features=in_features)

    best_params = model.find_best_params(df, target_column, param_grid, k=5)
    model.run(df, target_column, best_params)
