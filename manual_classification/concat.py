from os import listdir
import pandas as pd
from PIL import Image


# Member and feature names for accessing files and columns in dfs
names = ["gustav", "magnus", "marie", "viggo", "frida"]
features = ["benign", "melanoma", "keratosis"]
# Load all manual classifiaction csv files
name_dfs = [pd.read_csv(f"classifications/{name}_classification.csv") for name in names]
# List all 150 images and make a df with them
img_lists = [list(name_dfs[i]["img"]) for i in range(len(names))]
img_path = "../resized_data/example_image_resized/"
img_names = listdir(img_path)
full_df = pd.DataFrame(img_names, columns=["img"])
# Initialize columns {name}_{feature} for each name and feature to 0
for name in names:
    for feature in features:
        full_df[f"{name}_{feature}"] = 0
# # Initialize column for each feature to 0, for the result of the multiple classifications
# for feature in features:
#     full_df[feature] = 0

# Go through each image, find the dfs that contain it, then add the classification
# result to that person's column in full_df and find the highest count feature
# for each image and change that feature for the image to a 1.
for img in img_names:
    # Indices corresponding to names and name_dfs, where the image is in
    df_indices = [i for i in range(len(names)) if img in img_lists[i]]
    # Dictionary for counting the number of times each feature is chosen
    feature_counts = {feature: 0 for feature in features}
    # Counting features in the data frames containing the image
    for i in df_indices:
        for feature in features:
            # Value of the feature for the given image
            feature_val = int(name_dfs[i].loc[name_dfs[i]["img"] == img, feature])
            # Changing column name_feature in full_df to feature_val
            full_df.loc[full_df["img"] == img, f"{names[i]}_{feature}"] = feature_val
#             # Adding count of feature to feature_counts
#             feature_counts[feature] += feature_val
#     # If there is a disagreement (no definitive max feature occurence)
#     if list(feature_counts.values()).count(max(feature_counts.values())) > 1:
#         # Show the image on the screen
#         Image.open(img_path + img).show()
#         # Ask for a name and definitive feature classification
#         print([names[i] for i in df_indices])
#         print(feature_counts)
#         print("Names:\n1: Gustav\n2: Magnus\n3: Marie\n4: Viggo\n5: Frida")
#         classify_name = names[int(input("Name of decicion taker: ")) - 1]
#         print("1: Benign\n2: Melanoma\n3: Keratosis")
#         majority_feat = int(input("Type of skin lesion: ")) - 1
#         # Add classification of feature to that name
#         full_df.loc[full_df["img"] == img, f"{classify_name}_{majority_feat[0]}"] = 1
#     else:
#         # Find feature with highest count
#         majority_feat = [
#             k for k, v in feature_counts.items() if v == max(feature_counts.values())
#         ]
#     # Change highest count feature in full_df to 1
#     full_df.loc[full_df["img"] == img, majority_feat] = 1
# # Rearrange full_df columns
# columns = list(full_df)
# columns = columns[:1] + columns[-3:] + columns[1:-3]
# full_df = full_df[columns]
# Save data frame to csv
full_df.to_csv("full_classification.csv", index=False)
