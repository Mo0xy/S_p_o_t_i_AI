from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from classifier import Classifier
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class DecisionTreeTrainer(Classifier):
    def findBestParams(self):
        name = "DecisionTreeClassifier"
        self.model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [None, 'sqrt', 'log2']
        }

        best_params = self.loadBestParams(name)
        if best_params:
            print(f'Using saved best parameters for {name}:', best_params)
            self.params = best_params
            return None

        scorer = make_scorer(accuracy_score)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        grid_search = GridSearchCV(self.model, param_grid, scoring=scorer, cv=cv, verbose=1)
        grid_search.fit(self.X, self.Y)

        best_params = grid_search.best_params_
        print(f'Best parameters for {name}:', best_params)

        self.saveBestParams(best_params, name)
        self.params = best_params
        return None
