import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from os import path
import json


class Classifier:
    def __init__(self, target_column, drop_columns, dataset):
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


    def run(self):
        # Imposta i parametri del modello
        self.model.set_params(**self.params)

        # Definisce lo stratified k-fold (ad esempio, 5 fold)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Dizionario per salvare le metriche per ciascun fold
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_macro': [],
            'f1_micro': []
        }

        # Ciclo sui vari fold
        for train_index, test_index in skf.split(self.X, self.Y):
            # Se self.X e self.Y sono array NumPy, possiamo usare gli indici direttamente.
            # Se sono DataFrame, utilizzare .iloc[...]
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]


            # Addestra il modello sul fold corrente
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            # Calcola le metriche e le aggiunge al dizionario
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred, average='macro', zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, average='macro', zero_division=0))
            metrics['f1_macro'].append(f1_score(y_test, y_pred, average='macro', zero_division=0))
            metrics['f1_micro'].append(f1_score(y_test, y_pred, average='micro'))

        # Stampa le metriche con media e deviazione standard
        print("\nRisultati finali su dataset di test:\n")
        print("Accuracy: {:.6f} ± {:.6f}".format(np.mean(metrics['accuracy']), np.std(metrics['accuracy'])))
        print("Precision: {:.6f} ± {:.6f}".format(np.mean(metrics['precision']), np.std(metrics['precision'])))
        print("Recall: {:.6f} ± {:.6f}".format(np.mean(metrics['recall']), np.std(metrics['recall'])))
        print("F1_macro: {:.6f} ± {:.6f}".format(np.mean(metrics['f1_macro']), np.std(metrics['f1_macro'])))
        print("F1_micro: {:.6f} ± {:.6f}".format(np.mean(metrics['f1_micro']), np.std(metrics['f1_micro'])))


    def saveBestParams(self, best_params, name):
        # Define the directory and file path
        directory = 'ModelBestParams'
        filepath = os.path.join(directory, 'best_params_' + name + '.json')

        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)

        # Save the best parameters to the file
        with open(filepath, 'w') as file:
            json.dump(best_params, file)

    def loadBestParams(self, name):
        filepath = 'ModelBestParams/best_params_' + name + '.json'
        if path.exists(filepath):
            with open(filepath, 'r') as file:
                return json.load(file)
        return None

    def printMetrics(self):
        final_metrics = {
            "metrics": {metric: np.mean(values) for metric, values in self.metrics.items()},
            "std_dev": {metric: np.std(values) for metric, values in self.metrics.items()}
        }

        print("\nRisultati finali su dataset di test:\n")
        for metric in final_metrics["metrics"]:
            std_display = f" ± {final_metrics.get('std_dev')[metric]:.6f}" if final_metrics.get('std_dev')[metric] is not None else ""
            print(f"{metric.capitalize()}: {final_metrics.get('metrics')[metric]:.6f}{std_display}")

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
