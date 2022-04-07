import pandas as pd
import numpy as np
from PIL import Image
from groupB2_functions import label_json
from tqdm.contrib.concurrent import process_map


def wrapper(img):
    # Wrapper for multiprocessing
    # Reads superpixel mask and json metadata and gets the labels for connected components for json features
    spmask = Image.open(f"../data/training_superpixels/{img}_superpixels.png")
    spmask.thumbnail((100, 100), resample=False)
    spmask = np.array(spmask)
    json_df = pd.read_json(f"../data/training_json/{img}_features.json")
    return label_json(spmask, json_df)


if __name__ == "__main__":
    df = pd.read_csv("../data/training_ground_truth.csv").drop(columns="seborrheic_keratosis")
    # Get a list of size 4 with nested lists of size 2000 containing the label arrays for each feature
    # process_map enables progress bar and multiprocessing
    label_lists = list(zip(*process_map(wrapper, list(df["image_id"]), chunksize=5)))
    bin_list = []
    for labels in label_lists:
        # Get the sizes of the connected compunents in the label array
        counts = np.hstack([np.unique(arr[arr > 0], return_counts=True)[1] for arr in labels])
        # Compute 4 equipopulated bins and add them to a list
        bin_list.append([np.quantile(counts, n / 100) for n in range(0, 101, 25)])
    print(bin_list)
