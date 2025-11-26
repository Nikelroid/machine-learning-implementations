import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class AdaBoost:
    def __init__(self, n_estimators=60, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.betas = []
        self.models = []
    
    def fit(self, X, y):
        N = X.shape[0]
        y_new = 2 * y - 1
        weights = np.ones(N) / N
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y_new, sample_weight=weights)
            preds = tree.predict(X)
            incorrect = (preds != y_new).astype(float)
            error = np.sum(weights * incorrect)
            error = np.clip(error, 1e-10, 1 - 1e-10)
            beta = 0.5 * np.log((1 - error) / error)
            weights = weights * np.exp(-beta * y_new * preds)
            weights = weights / np.sum(weights)
            self.models.append(tree)
            self.betas.append(beta)
        return self
    
    def predict(self, X):
        N = X.shape[0]
        total = np.zeros(N)
        for beta, model in zip(self.betas, self.models):
            preds = model.predict(X)
            total += beta * preds
        return (total > 0).astype(int)
    
    def score(self, X, y):
        accuracy = accuracy_score(y, self.predict(X))
        return accuracy