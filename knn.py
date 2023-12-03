import faiss
import numpy as np
from experiments import get_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
import time

class KNN:
    def __init__(self, k=100):
        self.k = k
        self.index = None

    def fit(self, X,y):
        num_centroids = len(np.unique(y))
        kmeans = faiss.Kmeans(X.shape[1], num_centroids, niter=20, verbose=False)
        kmeans.train(X)
        
    def predict(self, X):
        _, neighbors = self.index.search(X, self.k)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=self.y[neighbors])


if __name__ == "__main__":
    X_train, y_train = get_dataset("data_for_classifier.csv", 0)
    X_val, y_val = get_dataset("data_for_classifier.csv", 1)
    X_test, y_test = get_dataset("data_for_classifier.csv", 2)

    Stadarizer = StandardScaler()
    X_train = Stadarizer.fit_transform(X_train)
    X_val = Stadarizer.transform(X_val)
    X_test = Stadarizer.transform(X_test)
    """
    for k in [1,10,100,1000]:
        knn = KNN(k)
        knn.fit(X_train, y_train)
        print(k)
        print("Val Acc", accuracy_score(y_val, knn.predict(X_val)))
    """

    start_time = time.time()
    k =10
    knn = KNN(k)
    knn.fit(X_train, y_train)
    print("Time", time.time() - start_time)
    y_test_pred = knn.predict(X_test)
    print("Val Acc", accuracy_score(y_val, knn.predict(X_val)))
    print("Test Acc", accuracy_score(y_test, y_test_pred))
    print("Test F1", f1_score(y_test, y_test_pred, average="macro"))
    





    



