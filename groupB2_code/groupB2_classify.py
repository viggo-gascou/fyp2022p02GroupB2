import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from groupB2_functions import measure_selected, measure_json


def train_classifier():
    """Trains ABC classifier on training data"""
    # Reads the csv with the measurements for the training set
    df = pd.read_csv("features/features_training.csv")
    # The 5 best features we have chosen
    features = ["asymmetry", "area", "perimeter", "color_dist_10_5", "color_score"]

    x = df[features].to_numpy()
    y = np.array(df["melanoma"])

    # Scales the features to values between 0 and 1
    scaler = MinMaxScaler().fit(x)
    x = scaler.transform(x)
    # Train the KNN model with k=9 and save the classifier and scaler
    clf = KNeighborsClassifier(n_neighbors=9).fit(x, y)
    with open("classifier.sav", "wb") as file:
        pickle.dump(clf, file)
    with open("scaler.sav", "wb") as file:
        pickle.dump(scaler, file)


def classify(img, seg):
    """Classifies image given an image array and segmentation array
       Image array must be a m x n x 3 integer array of RGB values between 0 and 255
       Segmentation array must be a 2D binary array with values 0 or 1
       Returns 2-length tuple with the predicted label and a 1 x 2 array with the probabilities"""
    # Loads the classifier and scaler
    with open("classifier.sav", "rb") as file:
        clf = pickle.load(file)
    with open("scaler.sav", "rb") as file:
        scaler = pickle.load(file)
    # Measures the selected features of the image and segmentation
    x = measure_selected(img, seg).reshape(1, 5)
    # Transforms the features and predicts probabilities
    x = scaler.transform(x)
    pred_prob = clf.predict_proba(x)[0]
    # With threshold 0.2 for melanoma, gets labels from probability
    threshold = 0.2
    pred_label = int(pred_prob[1] > threshold)
    return pred_label, pred_prob


def train_json_classifier():
    # Reads the csv with the measurements for the training set
    df = pd.read_csv("features/features_training.csv")
    # The 5 best features we have chosen
    features = ["asymmetry", "area", "perimeter", "color_dist_10_5", "color_score"]

    x = df[features].to_numpy()
    y = np.array(df["melanoma"])

    # JSON Data
    def to_numpy(arr):
        # Converts the csv columns to a numpy array
        return np.array(arr[1:-1].split(), dtype=int)

    # Converters for feature columns in csv
    converters = {'pigment_network_hist': to_numpy, 'negative_network_hist': to_numpy, 'milia_like_hist': to_numpy, 'streaks_hist': to_numpy}
    # Read the measured JSON features for the training image set
    jdf = pd.read_csv("features/json_training.csv", converters=converters)
    features = list(jdf.drop(columns=["image_id", "melanoma"]))
    # Stack the features in a single array of shape (2000, 20)
    xj = np.hstack([np.vstack(jdf[feat].to_numpy()) for feat in features])

    # Select the 5 best features and save the indices to use in the classification function
    selector = SelectKBest(mutual_info_classif, k=5)
    selector.fit(xj, y)
    selected_indices = np.argsort(selector.scores_)[-5:]
    with open("selected_json_features.txt", "w") as f:
        f.write(" ".join(selected_indices.astype(str).tolist()))
    xj = xj[:, selected_indices]
    # Add the JSON features to the ABC features
    x = np.c_[x, xj]

    # Scales the features to values between 0 and 1
    scaler = MinMaxScaler().fit(x)
    x = scaler.transform(x)

    # Train the KNN model with k=9 and save the classifier and scaler
    clf = KNeighborsClassifier(n_neighbors=9).fit(x, y)
    with open("classifier_json.sav", "wb") as file:
        pickle.dump(clf, file)
    with open("scaler_json.sav", "wb") as file:
        pickle.dump(scaler, file)


def classify_json(img, seg, spmask, json_df):
    """Classifies image given an image array and segmentation array
       Image array must be a m x n x 3 integer array of RGB values between 0 and 255
       Segmentation array must be a 2D binary array with values 0 or 1
       Returns 2-length tuple with the predicted label and a 1 x 2 array with the probabilities"""
    # Loads the classifier and scaler
    with open("classifier_json.sav", "rb") as file:
        clf = pickle.load(file)
    with open("scaler_json.sav", "rb") as file:
        scaler = pickle.load(file)
    # Measures the ABC and JSON features for the image
    x = measure_selected(img, seg).reshape(1, 5)
    xj = measure_json(spmask, json_df)
    # Read the indices of the best 5 JSON features selected during training
    with open("selected_json_features.txt") as f:
        selected_indices = list(map(int, f.read().split()))
    # Shape the features properly for classification
    xj = np.hstack(xj)[selected_indices].reshape(1, 5)
    # Concatenate ABC and JSON features
    x = np.c_[x, xj]

    # Scale features using the saved scaler
    x = scaler.transform(x)
    # Predict probabilities
    pred_prob = clf.predict_proba(x)[0]
    # With threshold 0.2 for melanoma, gets labels from probability
    threshold = 0.2
    pred_label = int(pred_prob[1] > threshold)
    return pred_label, pred_prob
