from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from classifier import Classifier
import sys

sys.stdout.reconfigure(encoding='utf-8')


class KNNTrainer(Classifier):
    def findBestParams(self):
        name = "KNNClassifier"
        self.model = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'kd_tree', 'brute'],
            'metric': ['euclidean', 'chebyshev'],
        }

        best_params = self.loadBestParams(name)
        if best_params:
            print(f'Using saved best parameters for {name}:', best_params)
            self.params = best_params
            return None

        scorer = make_scorer(accuracy_score)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(self.model, param_grid, scoring=scorer, cv=cv, verbose=1)
        grid_search.fit(self.X, self.Y)

        best_params = grid_search.best_params_
        print(f'Best parameters for {name}:', best_params)

        self.saveBestParams(best_params, name)
        self.params = best_params
        return None

    def findKNearestExamples(self, example, k=3):
        distances, indices = self.model.kneighbors([example], n_neighbors=k)
        valid_indices = [i for i in indices[0] if i < len(self.X)]
        nearest_examples = self.X.iloc[valid_indices]
        print("Esempi più vicini:", "\n", nearest_examples)
        print("Distanze:", distances[0][:len(valid_indices)])
        return nearest_examples, distances[0][:len(valid_indices)]


    """
    def findKNearestExamples(self, example, k=3, tolerance=50):
        # Recupera k+1 vicini, considerando che k+1 include l'esempio stesso (distanza zero)
        distances, indices = self.model.kneighbors([example], n_neighbors=k + 1)
    
        valid_indices = []
        valid_distances = []
    
        # Filtra l'esempio identico (distanza zero), se esiste, e considera la tolleranza
        for i, dist in zip(indices[0], distances[0]):
            if dist > tolerance:  # Considera solo le distanze superiori alla tolleranza
                valid_indices.append(i)
                valid_distances.append(dist)
            if len(valid_indices) == k:  # Limita a k vicini
                break
    
        # Filtra gli indici per assicurarsi che siano entro i limiti di self.X
        max_index = len(self.X) - 1
        valid_indices = [i for i in valid_indices if i <= max_index]
    
        # Se non ci sono abbastanza vicini, restituisce solo quelli trovati
        if len(valid_indices) == 0:
            print("Nessun vicino valido trovato.")
            return None, None, None
        elif len(valid_indices) < k:
            print(f"Avviso: trovati solo {len(valid_indices)} vicini validi entro i limiti.")
    
        try:
            nearest_examples = self.X.iloc[valid_indices]
        except IndexError:
            print("Errore: indici fuori limite. Valid indices:", valid_indices)
            return None, None, None
    
        brani = nearest_examples["nome_brano"]  # Cambia "nome_brano" con il nome esatto della colonna
    
        print("Brani più vicini:", brani.tolist())
        print("Esempi più vicini:", nearest_examples)
        print("Distanze:", valid_distances)
    
        return brani.tolist(), nearest_examples, valid_distances """
