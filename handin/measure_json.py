import pandas as pd
import numpy as np
from PIL import Image
from feature_functions import hist_json, percent_json
from tqdm import tqdm

df = pd.read_csv("../data/training_ground_truth.csv").drop(columns="seborrheic_keratosis")
img_names = list(df["image_id"])
features = {'pigment_network_percent': [], 'negative_network_percent': [], 'milia_like_cyst_percent': [], 'streaks_percent': [], 'pigment_network_hist': [], 'negative_network_hist': [], 'milia_like_cyst_hist': [], 'streaks_hist': []}
for img in tqdm(img_names):
    spmask = Image.open(f"../data/training_superpixels/{img}_superpixels.png")
    spmask.thumbnail((100, 100), resample=False)
    spmask = np.array(spmask)
    json_df = pd.read_json(f"../data/training_json/{img}_features.json")
    hists = hist_json(spmask, json_df)
    percentages = percent_json(json_df)
    for feat, hist, percent in zip(list(json_df), hists, percentages):
        features[feat + '_hist'].append(hist)
        features[feat + '_percent'].append(percent)
for feat, col in features.items():
    df[feat] = col
df.to_csv("../features/json_train.csv", index=False)
