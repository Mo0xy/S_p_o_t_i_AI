from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from collections import Counter
from os import path
import json

from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self, target_column, drop_columns, dataset):
        self.params = None
        self.metrics = None
        self.model = None
        self.dataset = dataset
        self.target_column = target_column
        self.drop_columns = drop_columns
        self.X, self.Y = self.loadData(dataset)

    def loadData(self, dataset):
        target = dataset.getColumn(self.target_column)
        dataset.dropDatasetColumns(self.drop_columns)
        X = dataset.getDataset()

        # Controlla la distribuzione delle classi
        class_counts = Counter(target)
        min_samples = 5  # numero minimo di campioni per classe da conservare

        # Filtra le classi con almeno min_samples campioni
        valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples]
        mask = target.isin(valid_classes)
        X, target = X[mask], target[mask]

        # Imposta k_neighbors come minimo tra 5 e il numero minimo di campioni meno 1, ma almeno 1
        k_neighbors = max(1, min(5, min(class_counts.values()) - 1))
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)

        # Applica SMOTE solo se ci sono abbastanza campioni per ogni classe rimanente
        X_resampled, y_resampled = smote.fit_resample(X, target)

        return X_resampled, y_resampled


    def evaluateModel(self, model, X_test, y_test):
        y_pred = model.predict(X_test)

        self.metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': float(precision_score(y_test, y_pred, average='micro')),
            'Recall': float(recall_score(y_test, y_pred, average='micro')),
            'F1_micro score': float(f1_score(y_test, y_pred, average='micro')),
            'F1_macro score': float(f1_score(y_test, y_pred, average='macro'))
        }

        self.X = X_test
        self.Y = y_test
        print("Model trained and evaluated\n")

    def saveBestParams(self, best_params, name):
        with open('best_params_' + name + '.json', 'w') as file:
            json.dump(best_params, file)

    def loadBestParams(self, name):
        filepath = 'best_params_' + name + '.json'
        if path.exists(filepath):
            with open(filepath, 'r') as file:
                return json.load(file)
        return None

    def featureSelection(self, X, y, k=10):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new


    def reduceNoise(self, X, y):
        # Rimuove gli outlier usando Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        yhat = iso_forest.fit_predict(X)

        # Seleziona solo i dati che non sono outlier
        mask = yhat != -1
        X, y = X[mask], y[mask]
        return X, y
    """
    def train_and_evaluate_with_hyperparams(self, model, param_grid, name):
        best_params = self.loadBestParams(name)
        if best_params:
            print(f'Using saved best parameters for {name}:', best_params)
            model.set_params(**best_params)
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            self.evaluateModel(model, X_test, y_test, name)
            return model"""

    """def train_model_KNN(self, name):
        model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7]
        }
        self.evaluateModel(model, self.X, self.Y)
        #self.train_and_evaluate_with_hyperparams(model, param_grid, name)"""

    def saveMetrics(self, file_path):
        with open(file_path, 'a') as file:
            file.write(str(self.metrics) + '\n')

    def run(self):
        self.model.set_params(**self.params)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.evaluateModel(self.model, X_test, y_test)
        self.saveMetrics('metrics.txt')

    def getMetrics(self):
        return self.metrics

    def getParams(self):
        return self.params

    def getModel(self):
        return self.model

    def getX(self):
        return self.X

    def getY(self):
        return self.Y
