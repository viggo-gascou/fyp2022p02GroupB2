import pandas as pd
from tqdm import tqdm
from functions.areaperimiter import measure
from functions.color_features import color_features
from functions.color_score import color_score
from math import pi


img_path = "resized_data/example_image_resized/"
seg_path = "resized_data/example_segmentation_resized/"
df = pd.read_csv("data/example_ground_truth.csv")[["image_id", "melanoma"]]

area = []
perimiter = []
print("Measuring area and perimiter...")
for img in tqdm(df["image_id"]):
    a, p = measure(seg_path + img + "_segmentation.png")
    area.append(a)
    perimiter.append(p)
df["area"], df["perimiter"] = area, perimiter
df["compactness"] = ((4 * pi) * df["area"]) / df["perimiter"] ** 2

color_dist = []
color_sd = []
color_scores = []
print("Measuring color features...")
for img in tqdm(df["image_id"]):
    img_file = img_path + img + ".jpg"
    seg_file = seg_path + img + "_segmentation.png"
    dist, sd = color_features(img_file, seg_file)
    color_dist.append(dist)
    color_sd.append(sd)
    color_scores.append(color_score(img_file, seg_file))
df["color_dist"], df["color_sd"], df["color_score"] = color_dist, color_sd, color_scores
df.to_csv("features/features.csv", index=False)
