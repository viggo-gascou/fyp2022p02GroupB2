import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from classify import classify
from tqdm import tqdm


df = pd.read_csv("../data/example_ground_truth.csv")[["image_id"]]
labels = []
for img_name in tqdm(list(df["image_id"])):
    img = plt.imread(f"../data/example_image/{img_name}.jpg")
    seg = plt.imread(f"../data/example_segmentation/{img_name}_segmentation.png")
    labels.append(classify(img, seg))
df["label"] = labels[0]
df.to_csv("features/labels_example.csv")