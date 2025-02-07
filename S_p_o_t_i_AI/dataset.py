import pandas
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


class Dataset:
    def __init__(self, dataframe=None, path=None):
        if path is not None:
            self.dataset = pandas.read_csv(path)
        elif dataframe is not None:
            self.dataset = dataframe
        else:
            raise ValueError("Devi fornire un path o un DataFrame.")

    @classmethod
    def from_dataframe(cls, dataframe):
        return cls(dataframe=dataframe)

    def getDataset(self):
        return self.dataset

    def setDataset(self, dataset):
        self.dataset = dataset

    def dropDatasetColumns(self, columnsToRemove):
        if columnsToRemove is not None:
            self.dataset = self.dataset.drop(columns=columnsToRemove)

    def addDatasetColumn(self, column, value):
        self.dataset[column] = value

    def saveDataset(self, path):
        self.dataset.to_csv(path, index=False)

    def getColumn(self, column):
        return self.dataset[column]

    def getDataFrame(self, columns):
        return self.dataset[columns]

    def normalizeColumn(self, column):
        scalera = MinMaxScaler()
        self.dataset[column] = scalera.fit_transform(self.dataset[[column]])

    def emptyValues(self, column, toReplace):
        self.dataset[column] = self.dataset[column].replace(toReplace, 1).fillna(0)

    def getDummies(self, column):
        self.dataset = pandas.get_dummies(self.dataset, columns=[column], drop_first=True)

    def replaceBoolean(self, T1=True, T2=False):
        self.dataset = self.dataset.replace({T1: 1, T2: 0})

    def EDA(self):
        # uniqueness analysis
        uniqueValues = self.dataset.nunique()
        length = len(self.dataset)
        uniqueValues = uniqueValues / length
        uniqueValues.to_csv("EDA/unique_values.csv")

        # null Values
        missingValues = self.dataset.isnull().sum()
        missingValues.to_csv("EDA/missing_values.csv")
