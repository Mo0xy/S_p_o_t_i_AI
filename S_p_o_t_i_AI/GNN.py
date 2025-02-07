import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


class GNNClassifier(nn.Module):
    def __init__(self, num_classes, in_features=64, hidden_dim=64, dropout_rate=0.6):
        super(GNNClassifier, self).__init__()
        self.name = "GNNClassifier"
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.conv1 = pyg_nn.SAGEConv(in_features, hidden_dim)
        self.conv2 = pyg_nn.SAGEConv(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def train_model(self, data, params, patience=10):

        self.hidden_dim = params['hidden_units']
        epochs = params['epochs']
        lr = params['lr']
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = float('inf')
        patience_counter = 0

        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(data.y.cpu().numpy()),
                                             y=data.y.cpu().numpy())
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(data.y.device)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            logits = self(data)
            loss = self.loss_fn(logits, data.y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Loss = {loss.item()}')

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def evaluate_model(self, graph_data, multiple_runs, prev_metrics=None):
        self.eval()
        with torch.no_grad():
            logits = self(graph_data)
            preds = logits.argmax(dim=1)
            true_labels = graph_data.y.cpu().numpy()
            pred_labels = preds.cpu().numpy()

            accuracy = float(np.mean(true_labels == pred_labels))
            precision = float(precision_score(true_labels, pred_labels, average="macro", zero_division=0))
            recall = float(recall_score(true_labels, pred_labels, average="macro", zero_division=0))
            f1_macro = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
            f1_micro = float(f1_score(true_labels, pred_labels, average="micro", zero_division=0))

            metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro
            }

            # Calcolo della deviazione standard SOLO se abbiamo più esecuzioni
            if multiple_runs and prev_metrics is not None:
                std_dev = {metric: float(np.std(prev_metrics[metric] + [metrics[metric]])) for metric in metrics}
            else:
                std_dev = {metric: None for metric in metrics}  # Nessuna deviazione standard se esecuzione singola

            # Stampa delle metriche con deviazione standard
            for metric in metrics:
                std_display = f" ± {std_dev[metric]:.6f}" if std_dev[metric] is not None else ""
                print(f"{metric.capitalize()}: {metrics[metric]:.6f}{std_display}")

            return {"metrics": metrics, "std_dev": std_dev}

    def find_best_params(self, data, target_column, param_grid, k=5):

        best_params = self.load_best_params(self.name)
        if best_params:
            self.load_best_params(self.name)
            print(f'Using saved best parameters:', best_params)
            return best_params

        grid = ParameterGrid(param_grid)
        best_score = -np.inf
        best_params = {}

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        overall_metrics = {"accuracy": [], "precision": [], "recall": [], "f1_macro": [], "f1_micro": []}

        for params in grid:
            hidden_dim = params['hidden_units']
            lr = params['lr']
            epochs = params['epochs']

            self.__init__(self.num_classes, in_features=128, hidden_dim=hidden_dim)

            metrics_list = {"accuracy": [], "precision": [], "recall": [], "f1_macro": [], "f1_micro": []}

            for train_idx, val_idx in skf.split(data, data[target_column]):
                train_df, val_df = data.iloc[train_idx], data.iloc[val_idx]
                train_graph = self.build_graph_from_dataset(train_df, target_column)
                val_graph = self.build_graph_from_dataset(val_df, target_column)

                self.train_model(train_graph, params)
                metrics = self.evaluate_model(val_graph, multiple_runs=True, prev_metrics=metrics_list)

                for metric, value in metrics["metrics"].items():
                    metrics_list[metric].append(value)
                    overall_metrics[metric].append(value)

            avg_score = np.mean(metrics_list["f1_macro"]) if metrics_list["f1_macro"] else -np.inf

            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                self.save_best_params(self.name, best_params)

        # Calcolo della media e deviazione standard delle metriche sui vari fold

        print(f'Best params: {best_params}, Best score: {best_score}')
        print(f'Total evaluated models: {k * len(grid)}')
        return best_params
        # print("\nAverage metrics across folds:", final_metrics)

    def run(self, df, target_column, best_params, k_folds=5):
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=
        42)

        metrics_list = {"accuracy": [], "precision": [], "recall": [], "f1_macro": [], "f1_micro": []}

        for fold, (train_idx, test_idx) in enumerate(skf.split(df, df[target_column])):
            print(f"Fold {fold + 1}/{k_folds}")
            train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

            # Costruisci i grafi per training e test
            train_graph = self.build_graph_from_dataset(train_df, target_column)
            test_graph = self.build_graph_from_dataset(test_df, target_column)

            # Allena il modello
            self.train_model(train_graph, best_params)

            # Valuta il modello
            test_metrics = self.evaluate_model(test_graph, multiple_runs=True, prev_metrics=metrics_list)

            # Salva i risultati
            for metric, value in test_metrics["metrics"].items():
                metrics_list[metric].append(value)

        # Calcola media e deviazione standard delle metriche
        final_metrics = {
            "metrics": {metric: np.mean(values) for metric, values in metrics_list.items()},
            "std_dev": {metric: np.std(values) for metric, values in metrics_list.items()}
        }

        print("\nRisultati finali su dataset di test:\n")
        for metric in final_metrics["metrics"]:
            std_display = f" ± {final_metrics.get('std_dev')[metric]:.6f}" if final_metrics.get('std_dev')[
                                                                                  metric] is not None else ""
            print(f"{metric.capitalize()}: {final_metrics.get('metrics')[metric]:.6f}{std_display}")

        return final_metrics

    def load_best_params(self, name):
        path = f'ModelBestParams/best_params_{name}.json'
        try:
            with open(path, 'r') as f:
                save_dict = json.load(f)

            model_params = save_dict["model_params"]
            saved_num_classes = save_dict["num_classes"]

            # Controllo sul numero di classi
            if self.num_classes != saved_num_classes:
                print(f" Warning: Mismatch in num_classes (Saved: {saved_num_classes}, Current: {self.num_classes})")
                return None  # Evita di caricare parametri incompatibili

            if model_params is None:
                print(f" Errore: Nessun 'model_params' trovato in {path}")
                return None

            print(f" Best hyperparameters loaded from {path}: {model_params}")
            return model_params

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f" Error loading best parameters from {path}: {e}")
            return None

    def build_graph_from_dataset(self, dataframe, target_column, k=5):
        print("building graph..")
        df = dataframe
        label_encoder = LabelEncoder()
        df.loc[:, target_column] = label_encoder.fit_transform(df[target_column])

        features = df.drop(columns=[target_column])
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        distance_matrix = cdist(normalized_features, normalized_features, metric='euclidean')
        edge_index = []
        num_nodes = len(df)

        for i in range(num_nodes):
            nearest_indices = np.argsort(distance_matrix[i])[1:k + 1]
            for j in nearest_indices:
                edge_index.append((i, j))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.tensor(normalized_features, dtype=torch.float)
        y = torch.tensor(df[target_column].values, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)
