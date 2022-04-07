import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SequentialFeatureSelector
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from measure import measure_selected


def train_classifier():
    df = pd.read_csv("features/features_training.csv")
    features = ["asymmetry", "area", "perimeter", "color_dist_10_5", "color_score"]

    x = df[features].to_numpy()
    y = np.array(df["melanoma"])

    x = MinMaxScaler().fit_transform(x)
    clf = KNeighborsClassifier(n_neighbors=9).fit(x, y)
    with open("classifier.sav", "wb") as file:
        pickle.dump(clf, file)


def classify(img, seg):
    with open("classifier.sav", "rb") as file:
        clf = pickle.load(file)
    x = measure_selected(img, seg).reshape(1, 5)
    x = MinMaxScaler().fit_transform(x)
    pred_prob = clf.predict_proba(x)
    threshold = 0.2
    pred_label = int(pred_prob[0][1] > threshold)
    return pred_label, pred_prob