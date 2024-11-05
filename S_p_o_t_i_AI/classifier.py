from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
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
        return X, target

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
