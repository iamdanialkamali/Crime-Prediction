import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.kernel_approximation import Nystroem  # For Spectral K-Means
import sklearn.model_selection
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_iris_dataset(test_size=0.2, val_size=0.25):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # First split to separate out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Second split to separate out the training and validation set
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size for the remaining dataset
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

def get_dataset(filename, split):
    y_column = "Primary Type"
    data_for_classifier = pd.read_csv(filename)
    x_columns = list(set(data_for_classifier.columns) - set([y_column, "split"]))
    data_for_classifier = data_for_classifier[data_for_classifier["split"] == split]
    return data_for_classifier[x_columns].to_numpy(), data_for_classifier[y_column].to_numpy()

def train(clf, X_train, y_train, X_val, y_val):
    logging.info("Training the model...")
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    return metrics(y_val, y_pred)

def metrics(y_val, y_pred):
    return {
        "f1_score": f1_score(y_val, y_pred, average='macro'),
        "accuracy": accuracy_score(y_val, y_pred)
    }

def get_params_from_args(args):
    params = {}
    if args.classifier == 'logistic_regression' and args.penalty in ['l1','l2']:
        params['penalty'] = args.penalty
    elif args.classifier == 'svm':
        params['kernel'] = args.kernel
    elif args.classifier == 'kmeans':
        params['mode'] = args.kmeans_mode
    return params


def perform_parameter_search(clf, X_train, y_train):
    logging.info("Performing parameter search...")

    param_grid = {}
    if isinstance(clf, RandomForestClassifier):
        param_grid = {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    elif isinstance(clf, SVC):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    elif isinstance(clf, LogisticRegression):
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        }
    elif isinstance(clf, MLPClassifier):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['tanh', 'relu']
        }

    if param_grid:
        grid_search = HalvingGridSearchCV(clf, param_grid, cv=5, factor=2, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    else:
        logging.warn("Parameter search for this model is not implemented! Actual model returned.")
        return clf


def get_model(args):
    clf_name = args.classifier
    params = get_params_from_args(args)

    if clf_name == 'random_forest':
        clf = RandomForestClassifier(**params)

    elif clf_name == 'svm':
        clf = SVC(**params)
    elif clf_name == 'logistic_regression':
        clf = LogisticRegression(**params)
    elif clf_name == 'kmeans':
        if params.get('mode') == 'spectral':
            transformer = Nystroem()
            clf = make_pipeline(transformer, KMeans(**params))
        else:
            clf = KMeans(**params)
    elif clf_name == 'mlp':
        clf = MLPClassifier(**params)
    else:
        raise ValueError("Invalid classifier name")
    return clf


def test(clf, X_test, y_test):
    logging.info("Testing the model...")
    print(clf)
    y_pred = clf.predict(X_test)
    return metrics(y_test, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--classifier', type=str, required=True, 
                        choices=['random_forest', 'svm', 'logistic_regression', 'kmeans', 'mlp'],
                        help='The type of classifier to use')
    parser.add_argument('--penalty', type=str, default='none', choices=['none','l1', 'l2'],
                        help='The norm used in penalization for Logistic Regression')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'linear'],
                        help='Kernel type for SVM')
    parser.add_argument('--kmeans_mode', type=str, default='normal', choices=['normal', 'spectral'],
                        help='Mode for K-Means clustering')
    parser.add_argument('--best_params', type=int, default=0, help='Whether to do the parameter search')
    args = parser.parse_args()

    # X_train, y_train = get_dataset("data_for_classifier.csv", 0)
    # X_val, y_val = get_dataset("data_for_classifier.csv", 1)
    # X_test, y_test = get_dataset("data_for_classifier.csv", 2)
    X_train, X_val, X_test, y_train, y_val, y_test = get_iris_dataset(test_size=0.2, val_size=0.25)


    model = get_model(args)

    if args.best_params:
        model = perform_parameter_search(model, X_train, y_train)

    train(model, X_train, y_train, X_val, y_val)
    test_results = test(model, X_test, y_test)
    print("Test results:", test_results)
>>>>>>> 760475466b633815509f5d173fd123765070059d
