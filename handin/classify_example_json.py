import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classify import classify_json
from tqdm import tqdm
from PIL import Image


df = pd.read_csv("../data/training_ground_truth.csv")[["image_id"]]
labels = []
for img_name in tqdm(list(df["image_id"])[:100]):
    img = plt.imread(f"../data_to_resize/training_image/{img_name}.jpg")
    seg = plt.imread(f"../data_to_resize/training_segmentation/{img_name}_segmentation.png")
    spmask = Image.open(f"../data/training_superpixels/{img_name}_superpixels.png")
    json_df = pd.read_json(f"../data/training_json/{img_name}_features.json")
    print(classify_json(img, seg, spmask, json_df))
