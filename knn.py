import faiss
import numpy as np
from experiments import get_dataset
from sklearn.preprocessing import StandardScaler

class KNN:
    def __init__(self, k=100):
        self.k = k
        self.index = None

    def fit(self, X,y):
        self.index = faiss.index_factory(X.shape[1], "Flat")
        self.index.train(X)
        self.index.add(X)
        self.y = y
    def predict(self, X):
        _, neighbors = self.index.search(X, self.k)
        u, c = np.unique(self.y[neighbors], return_counts = True)
        return u[c == c.max()]

if __name__ == "__main__":
    X_train, y_train = get_dataset("data_for_classifier.csv", 0)
    X_val, y_val = get_dataset("data_for_classifier.csv", 1)
    X_test, y_test = get_dataset("data_for_classifier.csv", 2)

    Stadarizer = StandardScaler()
    X_train = Stadarizer.fit_transform(X_train)
    X_val = Stadarizer.transform(X_val)
    X_test = Stadarizer.transform(X_test)

    for k in [1,10,100,1000]:
        knn = KNN(k)
        knn.fit(X_train, y_train)
        print(k)
        print((knn.predict(X_val) == y_val).sum()/len(y_val))


    



