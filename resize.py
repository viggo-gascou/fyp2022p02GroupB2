from os import listdir, makedirs
from shutil import rmtree
from PIL import Image

try:
    rmtree("resized_data")
except FileNotFoundError:
    pass

resized_img = "resized_data/example_image/"
makedirs(resized_img, exist_ok=True)
resized_seg = "resized_data/example_segmentation/"
makedirs(resized_seg, exist_ok=True)
img_names = listdir("fyp2022-imaging/data/example_image")
seg_names = listdir("fyp2022-imaging/data/example_segmentation")

for img_name in img_names:
    img = Image.open("fyp2022-imaging/data/example_image/" + img_name)
    img.thumbnail((600, 600))
    img.save(resized_img + img_name)

for seg_name in seg_names:
    img = Image.open("fyp2022-imaging/data/example_segmentation/" + seg_name)
    img.thumbnail((600, 600))
    img.save(resized_seg + seg_name)