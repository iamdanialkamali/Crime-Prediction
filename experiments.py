import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.kernel_approximation import Nystroem  # For Spectral K-Means
from sklearn.model_selection import PredefinedSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV,train_test_split
from scipy import sparse
import math
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.decomposition import TruncatedSVD
import time
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42)





def feature_selection_fsf(X_train, y_train, k=10):
    fsf = SelectKBest(score_func=f_classif, k=k)
    return fsf.fit_transform(X_train, y_train), fsf

def feature_selection_lr(X_train, y_train, threshold=0.01):
    class FeatureSelector:
        def __init__(self, selected_features_indices) -> None:
            self.selected_features_indices = selected_features_indices
        def transform(self, x):
            return x[:, self.selected_features_indices]


    lr = LogisticRegression(max_iter=10000,penalty='l2', C=1.0)
    lr.fit(X_train, y_train)
    coefficients = lr.coef_
    selected_features_indices = np.where(np.abs(coefficients) > threshold)
    print(selected_features_indices)
    X_train_selected = X_train[:, selected_features_indices]

    return X_train_selected, FeatureSelector(selected_features_indices)

def feature_selection_svd(X_train, n_components=10):
    svd = TruncatedSVD(n_components=min(n_components,X_train.shape[-1]))
    return svd.fit_transform(X_train), svd


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
    data_for_classifier = pd.read_csv(filename).dropna()
    x_columns = [x for x in data_for_classifier.columns if x in set(data_for_classifier.columns) - set([y_column, "split"])]
    data_for_classifier = data_for_classifier[data_for_classifier["split"] == split]
    return data_for_classifier[x_columns].to_numpy(), data_for_classifier[y_column].to_numpy()

def train(clf, X_train, y_train, X_val, y_val,chunks=1,recency = False,sampling="gm"):
    logging.info("Training the model...")

    weights = None

    if sampling == "":
        X_train,y_train = generate_data_for_imbalance(X_train,y_train)
    elif sampling == "over":
        X_train,y_train = RandomOverSampler().fit_resample(X_train,y_train)
    elif sampling == "under":
        X_train,y_train = RandomUnderSampler().fit_resample(X_train,y_train)
    if recency:
        weights = get_sample_weights(X_train, y_train)
    if chunks != 1:
        clf = ensemble_classifier(X_train, y_train, X_val, y_val,clf, num_subsets=chunks,sample_weights=weights)
    else:
        if isinstance(clf, MLPClassifier):
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train,sample_weight=weights)
    
    logging.info("Training complete, classifer is %s", clf)
    return (metrics(y_val, clf.predict(X_val)),metrics(y_train, clf.predict(X_train)))

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


def perform_parameter_search(clf, X_train, y_train,X_val, y_val):
    logging.info("Performing parameter search...")

    param_grid = {}
    if isinstance(clf, RandomForestClassifier):
        param_grid = {
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    elif isinstance(clf, lgb.LGBMClassifier):
        param_grid = {
            'num_leaves': [31, 50, 100],
            'learning_rate': [0.01, 0.1, 1],
            'n_estimators': [20, 40, 100]
        }
    
    elif isinstance(clf, SVC):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    elif isinstance(clf, LogisticRegression):
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1','l2']
        }
    elif isinstance(clf, MLPClassifier):
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01]
        }
    if param_grid:
        X = sparse.vstack((X_train, X_val))
        y = np.concatenate([y_train, y_val])
        test_fold = [-1 for _ in range(X_train.shape[0])] + [0 for _ in range(X_val.shape[0])]
        grid_search = HalvingGridSearchCV(clf, param_grid, factor=2, scoring='accuracy', cv = PredefinedSplit(test_fold))
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    else:
        logging.warn("Parameter search for this model is not implemented! Actual model returned.")
        return clf


def get_model(args,classes):
    clf_name = args.classifier
    params = get_params_from_args(args)

    if clf_name == 'random_forest':
        clf = RandomForestClassifier(**params)
    elif clf_name == 'lbgm':
        clf = lgb.LGBMClassifier(num_leaves = 100, learning_rate=0.01,**params)
    elif clf_name == 'svm':
        if "kernel" in params and params["kernel"] == 'linear':
            clf = SGDClassifier(max_iter=5000, loss='hinge', **params) #
        else:
            clf = SVC(**params)
    elif clf_name == 'logistic_regression':
        clf = SGDClassifier(max_iter=5000, **params) #LogisticRegression(max_iter=10000,**params)
    elif clf_name == 'kmeans':
        clf = KMeans(n_clusters=classes)
        if params.get('mode') == 'spectral':
            transformer = Nystroem()
            clf = make_pipeline(transformer, clf)
    elif clf_name == 'mlp':
        clf = MLPClassifier(alpha=0.001,**params)
    else:
        raise ValueError("Invalid classifier name")
    return clf


def test(clf, X_test, y_test):
    logging.info("Testing the model...")
    y_pred = clf.predict(X_test)
    return metrics(y_test, y_pred)


def ensemble_classifier(X_train, y_train, X_test, y_test, clf ,num_subsets=10,sample_weights=None):
    classifiers = []
    weights = []
    start = 0
    each_chunk = len(X_train) // num_subsets

    for i in range(num_subsets):
        subset_X,  subset_y, = X_train[start:start+each_chunk], y_train[start:start+each_chunk]
        if sample_weights is not None and not isinstance(clf, MLPClassifier):
            clf.fit(subset_X, subset_y,sample_weight=sample_weights[start:start+each_chunk])
        else:
            clf.fit(subset_X, subset_y)
        pred1 = clf.predict(X_test)
        start += each_chunk
        
    acc = accuracy_score(y_test, pred1)
    classifiers.extend([(f'rf_{i}', clf)])
    weights.extend([0.5*math.log((acc)/(1-acc))])
    
    weights = [x / sum(weights) for x in weights]
    voting_clf = VotingClassifier(estimators=classifiers, voting='soft', weights=weights)
    voting_clf.fit(X_train, y_train)

    ensemble_predictions = voting_clf.predict(X_test)

    print("Ensemble accuracy:", accuracy_score(y_test, ensemble_predictions))
    return voting_clf

def feature_selection(X_train, X_val, X_test, y_train, y_val, y_test):
    logging.info("Selecting features....")

    if args.feature_selection == 'FSF':
        X_train, selector = feature_selection_fsf(X_train, y_train, min(10, X_train.shape[-1] - 1))
    elif args.feature_selection == 'LR':
        X_train, selector = feature_selection_lr(X_train, y_train)
    elif args.feature_selection == 'SVD':
        X_train, selector = feature_selection_svd(X_train)

    X_val = selector.transform(X_val)
    X_test = selector.transform(X_test)
    return X_train, X_val, X_test, y_train, y_val, y_test

def pipeline(X_train, X_val, X_test, y_train, y_val, y_test, args):
    
    if args.feature_selection:
        X_train, X_val, X_test, y_train, y_val, y_test = feature_selection(X_train, X_val, X_test, y_train, y_val, y_test)
    
    model = get_model(args, classes=len(np.unique(y_train)))


    if args.best_params:
        time_start = time.time()
        model = perform_parameter_search(model, X_train, y_train, X_val, y_val)
        logging.info("Parameter search took %s seconds", time.time() - time_start)
    
    time_start = time.time()
    val_results,train_results = train(model, X_train, y_train, X_val, y_val,args.chunks,args.recency,args.sampling)
    logging.info("Training took %s seconds", time.time() - time_start)
    logging.info("Validation results: %s", val_results)
    logging.info("Train results: %s", train_results)

    time_start = time.time()
    test_results = test(model, X_test, y_test)
    logging.info("Testing took %s seconds", time.time() - time_start)
    logging.info("Test results: %s", test_results)


def get_sample_weights(X_train, y_train):
    #2023 samples have more weight compared to 2022 samples
    weights = np.ones(len(y_train))
    for i,year in enumerate(range(2023,2019,-1)):
        weights[X_train[:,1] <= year] = 1.0/(1+i)
    return weights

def generate_data_for_imbalance(X_train,y_train):
    """
    Assume all features are independent, and gaussian distributed
    """
    #fit gaussian distribution for each feature
    mean = np.mean(X_train,axis=0)
    std = np.std(X_train,axis=0)
    #generate data for each class
    
    #unique class counts 
    unq,unq_count = np.unique(y_train, return_counts=True)
    generate_data = [int(x) for x in (1- unq_count/len(y_train))*len(y_train)*0.1]

    X_train_new = []
    y_train_new = []
    for i,elem in enumerate(unq):
        X_train_new.append(np.random.normal(mean,std,size=(generate_data[i],X_train.shape[1])))
        y_train_new.append(np.ones(generate_data[i])*elem)
    
    X_train_new = np.concatenate(X_train_new,axis=0)
    y_train_new = np.concatenate(y_train_new,axis=0)


    train_size =  X_train.shape[0] / X_train_new.shape[0]
    X_train_new,_,y_train_new,_= train_test_split(X_train_new,y_train_new,train_size=train_size)

    return X_train_new,y_train_new





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a classifier')
    parser.add_argument('--classifier', type=str, required=True, 
                        choices=['random_forest','lbgm', 'svm', 'logistic_regression', 'kmeans', 'mlp'],
                        help='The type of classifier to use')
    parser.add_argument('--penalty', type=str, default='none', choices=['none','l1', 'l2'],
                        help='The norm used in penalization for Logistic Regression')
    parser.add_argument('--kernel', type=str, default='rbf', choices=['rbf', 'linear'],
                        help='Kernel type for SVM')
    parser.add_argument('--kmeans_mode', type=str, default='normal', choices=['normal', 'spectral'],
                        help='Mode for K-Means clustering')
    parser.add_argument('--feature_selection', type=str, default=None, choices=[None, 'FSF', 'LR', 'SVD'],
                    help='Method for feature selection')
    parser.add_argument('--best_params', type=int, default=0, help='Whether to do the parameter search')
    parser.add_argument('--chunks', type=int, default=1, help='Number of chunks we need to shard the data')
    parser.add_argument('--recency', type=bool, default=0, help='do we need to prioritize recent data')
    parser.add_argument('--sampling', type=str, default=None,choices=["over","under","gm"], help='do we need to prioritize recent data')


    args = parser.parse_args()

    X_train, y_train = get_dataset("data_for_classifier.csv", 0)
    X_val, y_val = get_dataset("data_for_classifier.csv", 1)
    X_test, y_test = get_dataset("data_for_classifier.csv", 2)
    pipeline(X_train, X_val, X_test, y_train, y_val, y_test, args)


