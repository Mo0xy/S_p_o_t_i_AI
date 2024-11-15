import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import kneighbors_graph
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import ParameterGrid
from torch_geometric.nn import SAGEConv  # Sostituisci GraphConv con SAGEConv


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_units, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_units)  # Sostituisci con SAGEConv
        self.conv2 = SAGEConv(hidden_units, output_dim)

    def forward(self, features, edge_index):
        features = features.float()
        edge_index = edge_index.long()  # Assicura che edge_index sia torch.long

        # Debug: Verifica i tipi prima della convoluzione
        # print("edge_index dtype:", edge_index.dtype)  # Deve essere torch.long
        # print("features dtype:", features.dtype)  # Deve essere torch.float

        x = self.conv1(features, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


class GNNClassifier:
    def __init__(self, params, dataset):
        self.trained_model = None
        self.params = params
        self.dataset = dataset
        self.mlb = MultiLabelBinarizer()

    def loadData(self):
        feature_columns = ['valence', 'energy', 'loudness']
        features = torch.tensor(self.dataset[feature_columns].values, dtype=torch.float)

        genres = self.dataset['genres'].apply(eval)
        self.Y = self.mlb.fit_transform(genres)

        knn_graph = kneighbors_graph(self.dataset[feature_columns], n_neighbors=5, mode='connectivity',
                                     include_self=False)

        # Converte `edge_index` in un tensore di tipo torch.long
        edge_index = torch.tensor(knn_graph.nonzero(), dtype=torch.long).view(2, -1)

        # Debug: verifica il tipo e la forma di edge_index
        # print("edge_index dtype:", edge_index.dtype)  # Deve essere torch.long
        # print("edge_index shape:", edge_index.shape)  # Deve essere [2, num_edges]

        self.g = Data(x=features, edge_index=edge_index)

        # Debug: verifica il tipo e la forma di features
        # print("features dtype:", features.dtype)  # Deve essere torch.float
        # print("features shape:", features.shape)  # Verifica che la forma sia corretta

        return features

    def train_model(self):
        features = self.loadData()
        self.params['output_dim'] = len(self.mlb.classes_)
        edge_index = self.g.edge_index  # Usa edge_index dell'intero grafo
        labels = torch.tensor(self.Y, dtype=torch.float)

        # Split train/test
        train_indices, test_indices, y_train, y_test = train_test_split(
            range(features.size(0)), labels, test_size=0.2, random_state=42
        )
        self.test_indices = test_indices  # Salva test_indices come attributo della classe

        train_loader = DataLoader(train_indices, batch_size=32, shuffle=True)

        model = GNNModel(self.params['input_dim'], self.params['hidden_units'], self.params['output_dim'])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 50
        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch_indices in train_loader:
                batch_features = features
                batch_labels = labels[batch_indices]
                optimizer.zero_grad()
                output = model(batch_features, edge_index)[batch_indices]

                loss = criterion(output, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')
                
        self.trained_model = model

        # Valutazione
        self.evaluate_model(model, features, labels, test_indices=self.test_indices)

    def evaluate_model(self, model, features, labels, test_indices):
        model.eval()
        edge_index = self.g.edge_index
        X_test = features[test_indices]
        y_test = labels[test_indices]

        with torch.no_grad():
            y_pred = model(features, edge_index)
            y_pred = y_pred[test_indices]  # Filtra i risultati solo per i test indices
            y_pred = (y_pred > 0.5).float()

            # Conversione a numpy
            y_pred_np = y_pred.cpu().numpy()
            y_test_np = y_test.cpu().numpy()

            metrics = {
                'Accuracy': accuracy_score(y_test_np, y_pred_np),
                'Precision': precision_score(y_test_np, y_pred_np, average='micro'),
                'Recall': recall_score(y_test_np, y_pred_np, average='micro'),
                'F1_micro score': f1_score(y_test_np, y_pred_np, average='micro'),
                'F1_macro score': f1_score(y_test_np, y_pred_np, average='macro')
            }

            print("Model Evaluation Results:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")
    """
    def find_best_parameters(self, param_grid):
        best_params = None
        best_score = 0

        # Genera combinazioni di parametri
        for params in ParameterGrid(param_grid):
            print(f"Testing parameters: {params}")
            self.params.update(params)

            # Calcola input_dim basato sul numero di feature
            if 'input_dim' not in self.params:
                self.params['input_dim'] = self.g.x.size(1)  # Numero di colonne delle feature

            self.train_model()

            # Valutazione usando le metriche di test
            model = self.trained_model  # Assumendo che train_model salvi il modello finale in self.trained_model
            features = self.g.x
            labels = torch.tensor(self.Y, dtype=torch.float)
            edge_index = self.g.edge_index

            model.eval()
            test_indices = self.test_indices
            X_test = features[test_indices]
            y_test = labels[test_indices]

            with torch.no_grad():
                y_pred = model(features, edge_index)
                y_pred = y_pred[test_indices]  # Filtra per i test indices
                y_pred = (y_pred > 0.5).float()

                # Conversione a numpy
                y_pred_np = y_pred.cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                score = accuracy_score(y_test_np, y_pred_np)
                print(f"Current Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_params = params

        print("Best Parameters Found:", best_params)
        print("Best Score:", best_score)
        self.params.update(best_params)"""

