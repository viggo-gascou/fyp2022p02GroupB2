import numpy as np
from PIL import Image


def color_score(img_file, seg_file):
    img = Image.open(seg_file).convert("RGB")
    mask = Image.open(seg_file)
    img.paste(Image.open(img_file).convert("RGB"), mask=mask)
    img_arr = np.array(img) / 255
    lesion = img_arr[img_arr != 0]
    pixels = len(lesion)
    color_count = 0
    masks = {
        "white": (lesion[:,:,0] >= 0.8) & (lesion[:,:,1] >= 0.8 & lesion[:,:,2] >= 0.8),
        "red": (lesion[:,:,0] >= 0.588) & (lesion[:,:,1] >= 0.2 & lesion[:,:,2] >= 0.2),
        "lightbrown": (lesion[:,:,0] >= 0.588) & (lesion[:,:,1] > 0.2) & (lesion[:,:,2] > 0) & (lesion[:,:,0] <= 0.94) & (lesion[:,:,1] <= 0.588) & (lesion[:,:,2] < 0.392),
        "darkbrown": (lesion[:,:,0] > 0.243) & (lesion[:,:,1] >= 0) & (lesion[:,:,2] > 0) & (lesion[:,:,0] < 0.56) & (lesion[:,:,1] < 0.392) & (lesion[:,:,2] < 0.392),
        "blue-gray": (lesion[:,:,0] >= 0) & (lesion[:,:,1] >= 0.392) & (lesion[:,:,2] >= 0.490) & (lesion[:,:,0] <= 0.588) & (lesion[:,:,1] <= 0.588) & (lesion[:,:,2] <= 0.588),
        "black": (lesion[:,:,0] <= 0.243) & (lesion[:,:,1] <= 0.243) & (lesion[:,:,2] <= 0.243)
    }
    for mask in masks.values():
        if len(lesion[mask]) / pixels > 0.05:
            color_count += 1
    return color_count
