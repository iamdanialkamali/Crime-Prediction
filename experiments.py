import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import random

def k_fold_training(clf,X,Y,k=5,over_sampling=False):
    kf = KFold(n_splits=k)
    scores = []
    clf = make_pipeline(StandardScaler(),clf)
    for fold, (train_index, test_index) in tqdm(enumerate(kf.split(X), 1),total=k):
        X_train = X[train_index]
        Y_train = Y[train_index]  # Based on your code, you might need a ravel call here, but I would look into how you're generating your y
        X_test = X[test_index]
        Y_test = Y[test_index]  # See comment on ravel and  y_train
        if over_sampling:
            sm = RandomOverSampler(random_state=33)
            X_train, Y_train = sm.fit_resample(X_train, Y_train)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        f1_score_value = f1_score(Y_test, Y_pred,average='macro')
        print(f1_score_value)
        scores.append(f1_score_value)
    mean,variance = np.mean(scores),np.var(scores)
    return mean,variance

def get_dataset(data_for_classifier, split):
    y_column = "Primary Type"
    data_for_classifier = pd.read_csv("data_for_classifier.csv")
    print(data_for_classifier.head())
    x_columns = list(set(data_for_classifier.columns)-set([y_column,"split"]))
    data_for_classifier = data_for_classifier[data_for_classifier["split"]==split]
    return data_for_classifier[x_columns].to_numpy(), data_for_classifier[y_column].to_numpy()

def train(clf, X_train, y_train, X_val, y_val):
    clf = make_pipeline(StandardScaler(),clf)
    # if over_sampling:
    #     sm = RandomOverSampler(random_state=33)
    #     X_train, y_train = sm.fit_resample(X_train, Y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1_score_value = f1_score(y_val, y_pred,average='macro')
    return f1_score_value

def metrics(y_pred, y_test):
    metric_dict ={"f1_score":f1_score(y_test, y_pred,average='macro'),"accuracy":accuracy_score(y_test, y_pred)}
    

if __name__ == '__main__':
    X_train, y_train = get_dataset("data_for_classifier.csv", 0)
    X_val, y_val = get_dataset("data_for_classifier.csv", 1)
    X_test, y_test = get_dataset("data_for_classifier.csv", 2)
   

    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    train(clf, X_train, y_train, X_val, y_val)
