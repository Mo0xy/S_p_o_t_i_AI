import warnings
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


class Dataset:

    def __init__(self, path=None, dataframe=None):
        if path is not None:
            self.dataset = pd.read_csv(path)
        elif dataframe is not None:
            self.dataset = dataframe
        else:
            raise ValueError("Devi fornire un path o un DataFrame.")

    @classmethod
    def from_dataframe(cls, dataframe):
        return cls(dataframe=dataframe)

    def getDataset(self):
        return self.dataset

    def getReducedDataset(self, percentage):
        self.dataset = self.dataset.sample(frac=percentage, random_state=1)
        self.dataset.to_csv('dataReduced.csv', index=False)
        return self

    def setDataset(self, dataset):
        self.dataset = dataset

    def dropDatasetColumns(self, columnsToRemove):
        self.dataset = self.dataset.drop(columns=columnsToRemove)

    def remove_text_columns(self):
        """
        Rimuove tutte le colonne contenenti dati testuali da un DataFrame.

        Parametri:
            df (pd.DataFrame): Il DataFrame da cui rimuovere le colonne.

        Ritorna:
            pd.DataFrame: Un nuovo DataFrame senza colonne testuali.
        """
        # Seleziona solo le colonne che non sono di tipo "object" (testuale)
        non_text_df = self.dataset.select_dtypes(exclude=['object'])
        return non_text_df

    def addDatasetColumn(self, column, value):
        self.dataset[column] = value

    def saveDataset(self, path):
        self.dataset.to_csv(path, index=False)

    def getColumn(self, column):
        return self.dataset[column]

    def getDataFrame(self, columns):
        return self.dataset[columns]

    def normalizeColumn(self, column):
        scaler = MinMaxScaler()
        self.dataset[column] = scaler.fit_transform(self.dataset[[column]])

    def replaceEmptyValues(self, column, toReplace):
        self.dataset[column] = self.dataset[column].replace(toReplace, 1).fillna(0)

    def EDA(self):
        # uniqueness analysis
        uniqueValues = self.dataset.nunique()
        length = len(self.dataset)
        uniqueValues = uniqueValues / length
        uniqueValues.to_csv("EDA/unique_values.csv")

        # null Values
        missingValues = self.dataset.isnull().sum()
        missingValues.to_csv("EDA/missing_values.csv")

    def normalizeTuple(self, t):

        # Creare un DataFrame dal dataset
        df = pd.DataFrame(self.getDataset())

        # Verifica che la tupla abbia la stessa lunghezza delle colonne del dataset
        if len(t) != len(df.columns):
            raise ValueError(f"La tupla fornita ha {len(t)} elementi, "
                             f"ma ci si aspettano {len(df.columns)} elementi.")

        # Crea e adatta un MinMaxScaler per ogni colonna numerica
        scalers = {col: MinMaxScaler().fit(df[[col]]) for col in df.columns}

        # Normalizza i valori della tupla
        normalized_tuple = []
        for i, col in enumerate(df.columns):
            normalized_value = scalers[col].transform([[t[i]]])[0][0]
            normalized_tuple.append(normalized_value)

        return tuple(normalized_tuple)  # Restituisce la tupla normalizzata

    def getColumnWithIndex(self, position):
        return self.dataset.iloc[:, position]
