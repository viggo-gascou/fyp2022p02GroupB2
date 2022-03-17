import pandas as pd
from PIL import Image


resized_folder = "./resized/"
df = pd.read_csv("../features/features.csv")
img_names = list(df["id"])

for img_name in img_names:
    img = Image.open("./example_image/" + img_name + ".jpg")
    img.thumbnail((600, 600))
    img.save(resized_folder + img_name + ".jpg")
