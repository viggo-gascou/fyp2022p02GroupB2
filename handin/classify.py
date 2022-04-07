import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from measure import measure_selected, measure_json


def train_classifier():
    df = pd.read_csv("features/features_training.csv")
    features = ["asymmetry", "area", "perimeter", "color_dist_10_5", "color_score"]

    x = df[features].to_numpy()
    y = np.array(df["melanoma"])

    scaler = MinMaxScaler().fit(x)
    x = scaler.transform(x)
    clf = KNeighborsClassifier(n_neighbors=9).fit(x, y)
    with open("classifier.sav", "wb") as file:
        pickle.dump(clf, file)
    with open("scaler.sav", "wb") as file:
        pickle.dump(scaler, file)


def classify(img, seg):
    with open("classifier.sav", "rb") as file:
        clf = pickle.load(file)
    with open("scaler.sav", "rb") as file:
        scaler = pickle.load(file)
    x = measure_selected(img, seg).reshape(1, 5)
    x = scaler.transform(x)
    pred_prob = clf.predict_proba(x)[0]
    threshold = 0.2
    pred_label = int(pred_prob[1] > threshold)
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
    selected_indices = np.argsort(selector.scores_)[-5:]
    with open("selected_json_features.txt", "w") as f:
        f.write(" ".join(selected_indices.astype(str).tolist()))
    xj = xj[:, selected_indices]
    x = np.c_[x, xj]

    scaler = MinMaxScaler().fit(x)
    x = scaler.transform(x)

    clf = KNeighborsClassifier(n_neighbors=9).fit(x, y)
    with open("classifier_json.sav", "wb") as file:
        pickle.dump(clf, file)
    with open("scaler_json.sav", "wb") as file:
        pickle.dump(scaler, file)


def classify_json(img, seg, spmask, json_df):
    with open("classifier_json.sav", "rb") as file:
        clf = pickle.load(file)
    with open("scaler_json.sav", "rb") as file:
        scaler = pickle.load(file)
    x = measure_selected(img, seg).reshape(1, 5)
    xj = measure_json(spmask, json_df)
    with open("selected_json_features.txt") as f:
        selected_indices = list(map(int, f.read().split()))
    xj = np.hstack(xj)[selected_indices].reshape(1, 5)
    x = np.c_[x, xj]

    x = scaler.transform(x)
    pred_prob = clf.predict_proba(x)[0]
    threshold = 0.2
    pred_label = int(pred_prob[1] > threshold)
    return pred_label, pred_prob
