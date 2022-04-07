import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


names = ["gustav", "magnus", "marie", "viggo", "frida"]
dfs = [pd.read_csv(f"classifications/{name}_classification.csv") for name in names]
full_df = pd.DataFrame({"image_id": [img[:-4] for img in dfs[0]["img"]]})
for i, (name, df) in enumerate(zip(names, dfs)):
    df = df.drop(columns=["benign", "keratosis"])
    df["image_id"] = full_df["image_id"]
    df = df[list(df)[-1:] + list(df)[1:-1]]
    df.rename(columns={"assymetry": "asymmetry"}, inplace=True)
    dfs[i] = df
    full_df = full_df.merge(df, on="image_id", suffixes=(None, f"_{name}"))
cols = list(full_df)[1:5]
renamed = {col: col + "_gustav" for col in cols}
full_df.rename(columns=renamed, inplace=True)
full_df.to_csv("classifications/manual_classification.csv", index=False)


columns = ["melanoma", "asymmetry", "border", "color"]
mean_cols = {}
for col in columns:
    mean_cols[col] = np.mean(
        [np.array(full_df[f"{col}_{name}"]) for name in names], axis=0
    )
mean_df = pd.DataFrame(mean_cols)

img_names = list(full_df["image_id"])
converter = {"probabilities": lambda x: np.array(x[1:-1].split(), dtype=float)}
class_df = pd.read_csv("classifications/classification_example.csv", converters=converter)
class_df = class_df.loc[[img in img_names for img in class_df["image_id"]]]
feature_df = pd.read_csv("features/features_example.csv")
feature_df = feature_df.loc[[img in img_names for img in feature_df["image_id"]]]
A = ["asymmetry", "asymmetry_gauss", "border_score"]
B = ["area", "perimeter", "compactness"]
C = [
    "color_dist_10_5",
    "color_sd_10_5",
    "color_dist_10_10",
    "color_sd_10_10",
    "color_dist_5_5",
    "color_sd_5_5",
    "color_dist_5_10",
    "color_sd_5_10",
    "color_score",
]

feature_names = [
    "asymmetry",
    "asymmetry_gauss",
    "area",
    "perimeter",
    "compactness",
    "color_dist_10_5",
    "color_sd_10_5",
    "color_dist_10_10",
    "color_sd_10_10",
    "color_dist_5_5",
    "color_sd_5_5",
    "color_dist_5_10",
    "color_sd_5_10",
    "color_score",
    "border_score",
]

A = [feature_names.index(s) for s in A]
B = [feature_names.index(s) for s in B]
C = [feature_names.index(s) for s in C]

x = np.array(feature_df[feature_names])
x = MinMaxScaler().fit_transform(x)

A = np.round(np.mean(x[:, A], axis=1) * 10, 1)
B = np.round(np.mean(x[:, B], axis=1) * 10, 1)
C = np.round(np.mean(x[:, C], axis=1) * 10, 1)

mean_df["asymmetry_class"] = A
mean_df["border_class"] = B
mean_df["color_class"] = C
mean_df["melanoma_class"] = class_df["label"]
mean_df["melanoma_prob_class"] = np.vstack(class_df["probabilities"])[:,1]
mean_df["melanoma_true"] = feature_df["melanoma"]
mean_df["image_id"] = img_names
mean_df["melanoma_prob"] = list(mean_df["melanoma"])
mean_df["melanoma"] = np.round(mean_df["melanoma"])
mean_df = mean_df[
    [
        "image_id",
        "melanoma_true",
        "melanoma",
        "melanoma_class",
        "melanoma_prob",
        "melanoma_prob_class",
        "asymmetry",
        "asymmetry_class",
        "border",
        "border_class",
        "color",
        "color_class",
    ]
]
mean_df.to_csv("classifications/mean_classification.csv")
