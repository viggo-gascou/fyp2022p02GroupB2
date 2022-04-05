import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map
from feature_functions import asymmetry


def measure_asymmetry(seg_file):
    seg = plt.imread(seg_file)
    return asymmetry(seg)


# Paths to the directories with the images and corresponding segmentations to measure
seg_path = "resized_data/training_segmentation_resized/"

if __name__ == "__main__":
    # Read the example ground thruth csv to get image ids and melanoma data
    df = pd.read_csv("features/features_train.csv")

    # Add paths and extensions to image ids
    seg_files = [seg_path + img + "_segmentation.png" for img in df["image_id"]]

    print("Measuring features...")
    # Measure features using custom functions for each image and segmentation file
    # tqdm and list for the zip() facilitates a progress bar
    features = np.vstack(process_map(measure_asymmetry, list(seg_files), chunksize=5))
    feature_names = [
        "asymmetry",
        "asymmetry_gauss",
    ]
    # Adding the columns of features to the data frame
    for col, feature in zip(features.T, feature_names):
        df[feature] = col
    df.to_csv("features/features_train.csv", index=False)
