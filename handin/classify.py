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


def train_json_classifier():
    df = pd.read_csv("features/features_training.csv")
    features = ["asymmetry", "area", "perimeter", "color_dist_10_5", "color_score"]

    x = df[features].to_numpy()
    y = np.array(df["melanoma"])

    # JSON Data
    def to_numpy(arr):
        return np.array(arr[1:-1].split(), dtype=int)

    converters = {'pigment_network_hist': to_numpy, 'negative_network_hist': to_numpy, 'milia_like_hist': to_numpy, 'streaks_hist': to_numpy}
    jdf = pd.read_csv("features/json_training.csv", converters=converters)
    features = list(jdf.drop(columns=["image_id", "melanoma"]))
    xj = np.hstack([np.vstack(jdf[feat].to_numpy()) for feat in features])
    selector = SelectKBest(mutual_info_classif, k=5)
    selector.fit(xj, y)
    scores = selector.scores_
    selected_indices = np.argsort(selector.scores_)[-5:]
    with open("selected_json_features.txt") as f:
        f.write(str(selected_indices))
    xj = xj[:, selected_indices]
    x = np.c_[x, xj]

    x = MinMaxScaler().fit_transform(x)

    clf = KNeighborsClassifier(n_neighbors=9).fit(x, y)
    with open("classifier_json.sav", "wb") as file:
        pickle.dump(clf, file)



def classify_json(img, seg, spmask, json_df):
    # with open("classifier_json.sav", "rb") as file:
    #     clf = pickle.load(file)
    x = measure_selected(img, seg).reshape(1, 5)
    xj = measure_json(spmask, json_df)
    print(xj)
    
    # x = MinMaxScaler().fit_transform(x)
    # pred_prob = clf.predict_proba(x)
    # threshold = 0.2
    # pred_label = int(pred_prob[0][1] > threshold)
    # return pred_label, pred_prob