import pandas as pd
from tqdm import tqdm
from feature_functions import measure, color_features, color_score
from math import pi
from PIL import Image


# Paths to the directories with the images and corresponding segmentations to measure
img_path = "resized_data/example_image_resized/"
seg_path = "resized_data/example_segmentation_resized/"
# Read the example ground thruth csv to get image ids and melanoma data
df = pd.read_csv("data/example_ground_truth.csv")[["image_id", "melanoma"]]

# Measure area and perimiter using custom function
area = []
perimeter = []
print("Measuring area and perimeter...")
# tqdm() enables a progress bar for the loop
for img in tqdm(df["image_id"]):
    a, p = measure(seg_path + img + "_segmentation.png")
    area.append(a)
    perimeter.append(p)
df["area"], df["perimeter"] = area, perimeter
# Calculate compactness from formula c = (4*pi*A)/p^2
df["compactness"] = ((4 * pi) * df["area"]) / df["perimeter"] ** 2

# Measure color features
color_dist = []
color_sd = []
color_scores = []
print("Measuring color features...")
for img in tqdm(df["image_id"]):
    # File paths for img and segmentation files
    img_file = img_path + img + ".jpg"
    seg_file = seg_path + img + "_segmentation.png"
    # Open the segmentation file as RGB and as bitmap
    img = Image.open(seg_file).convert("RGB")
    mask = Image.open(seg_file)
    # Paste the image onto the RGB mask using the bitmap mask as a mask
    # Resulting image is the masked lesion in RGB colours
    img.paste(Image.open(img_file).convert("RGB"), mask=mask)
    # Measure color dist and sd with custom functions
    dist, sd = color_features(img, mask)
    color_dist.append(dist)
    color_sd.append(sd)
    # Measure color score with custom function
    color_scores.append(color_score(img))
df["color_dist"], df["color_sd"], df["color_score"] = color_dist, color_sd, color_scores
df.to_csv("features/features.csv", index=False)
